"""Classify 3D assets into body-plan categories with a local Qwen3.5 / Qwen3-VL model.

Per asset the model sees three kinds of evidence:

  * the rest-pose render — the 2x2 T-pose grid PNG split into its four views
    (``--no-split_grid`` passes the grid as one image);
  * the first frame of one animation clip from the same four cameras
    (``--motion_render_root``, i.e. ``dataset/render/<dataset>``): the
    character in its natural stance, which matters because a rest pose can
    be lying flat on the ground, floating or rotated;
  * skeleton facts derived from the export (``--export_dir``): joint count,
    a histogram of the cleaned joint labels (``Wing x8``, ``Pectoral Fin x2``,
    ``Thigh x2``), the body extents and which labelled parts exist;
  * up to ``--max_captions`` motion captions of the asset's clips
    (``<export_dir>/motion_captions.json``).

It answers with a small JSON object (evidence fields + category +
confidence). ``--votes N`` samples N answers with the official non-thinking
sampling parameters and takes the majority; a tie, a low-confidence majority
or the model's own ``uncertain`` verdict lands in the ``uncertain`` bucket
so it can be reviewed instead of silently mis-filed.

Two render layouts are supported and auto-detected:
  Flat (objaverse, truebones):  <render_root>/<asset_id>.png
  Nested:                       <render_root>/<NAME-MOTION>/tpose_grid.png
      Object types are grouped by the segment before '-'; one
      classification per type is then propagated to every clip of that type.

Output:
    * ``category_groups.json``  — {category: [asset_id, ...]} for assets
                                  that classified (``uncertain`` included).
    * ``category_groups_review.json`` (sibling) — per asset: final category,
                                  confidence, the votes and the parsed
                                  evidence of every answer, for manual review.
    * ``category_groups_errors.json`` (sibling) — {asset_id: error_message}
                                  for assets whose retries were exhausted.
                                  ERROR entries are NOT in category_groups.json
                                  so a re-run retries them automatically.

Resumability: assets already in ``category_groups.json`` are skipped, except
those under ``unknown`` (always re-attempted) and, with ``--retry_uncertain``,
those under ``uncertain``.

Usage:
    python -m data_process.vlm_caption.classify_category \\
        --render_root dataset/render/objaverse_tpose \\
        --export_dir dataset/export/objaverse \\
        --category_groups_json dataset/export/objaverse/category_groups.json
"""

import argparse
import glob
import json
import os
import re
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from loguru import logger
from PIL import Image
from tqdm import tqdm

from data_process.vlm_caption.prompts import CATEGORY_PROMPT
from data_process.utils.vlm import (
    add_qwen_model_args,
    flatten_rgba,
    load_json,
    load_qwen_model,
    qwen_generate,
    save_json,
)

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

VALID_CATEGORIES = [
    "bipedal", "quadrupedal", "insectoid", "avian", "marine",
    "serpentine", "articulated_rigid",
]
UNCERTAIN_CATEGORY = "uncertain"  # model / vote could not decide — kept for review
UNKNOWN_CATEGORY = "unknown"      # unparseable output — re-attempted on resume
CONFIDENCES = ("high", "medium", "low")

# Per-asset retry policy: up to NUM_RETRIES additional attempts after the
# initial try (5 total). Retries cover exceptions (image I/O, CUDA OOM,
# generation failure) and rounds where too few votes could be parsed.
NUM_RETRIES = 4
RETRY_BACKOFF_SECONDS = 2.0
SAVE_EVERY = 20  # assets per incremental JSON flush (all output files)

# Labels that define limbs: always listed in the histogram shown to the model.
LIMB_WORDS = ("Leg", "Thigh", "Shin", "Foot", "Toe", "Arm", "Forearm", "Hand", "Wing",
              "Fin", "Flipper", "Tail", "Hip", "Claw", "Paw", "Hoof", "Pincer", "Mandible",
              "Antenna", "Tentacle", "Knee", "Elbow", "Shoulder")
MAX_LABELS = 20
# Joint labels that carry body-plan evidence, reported as flags (whole words).
LABEL_FLAGS = {
    "wings": ("Wing", "Feather"),
    "fins": ("Fin", "Flipper"),
    "tail": ("Tail",),
    "antennae_or_mandibles": ("Antenna", "Mandible", "Pincer", "Fang", "Stinger"),
    "tentacles": ("Tentacle",),
}


def _has_word(label, words):
    return any(re.search(r"\b{}\b".format(w), label) for w in words)

_JSON_RE = re.compile(r"\{.*?\}", re.DOTALL)
_CATEGORY_RE = re.compile(r'"?category"?\s*[:=]\s*"?([A-Za-z_ ]+)"?', re.IGNORECASE)
_CONFIDENCE_RE = re.compile(r'"?confidence"?\s*[:=]\s*"?(high|medium|low)', re.IGNORECASE)


# ─────────────────────────────────────────────────────────────────────────────
# Evidence: skeleton facts and captions from the export
# ─────────────────────────────────────────────────────────────────────────────

class ExportEvidence:
    """Lazy reader of the stage-1/2/3 sidecars used as text evidence."""

    def __init__(self, export_dir, max_captions=6, motion_render_root=None):
        self.export_dir = export_dir
        self.max_captions = max_captions
        self.motion_render_root = motion_render_root
        self.joint_names = load_json(os.path.join(export_dir, "joint_names.json")) if export_dir else {}
        self.clean_names = load_json(os.path.join(export_dir, "clean_joint_names.json")) if export_dir else {}
        captions = load_json(os.path.join(export_dir, "motion_captions.json")) if export_dir else {}
        self.captions_by_asset = defaultdict(list)  # type: dict
        for clip, text in captions.items():
            self.captions_by_asset[clip.split("-", 1)[0]].append(text)
        if export_dir:
            logger.info("Export evidence: {} rigs with joint names, {} with clean labels, "
                        "captions for {} assets".format(
                            len(self.joint_names), len(self.clean_names),
                            len(self.captions_by_asset)))

    # -- skeleton --------------------------------------------------------
    def _rest_positions(self, asset_id):
        paths = sorted(glob.glob(os.path.join(self.export_dir, "motions", asset_id + "-*.npz")))
        if not paths:
            paths = [p for p in [os.path.join(self.export_dir, "motions", asset_id + ".npz")]
                     if os.path.isfile(p)]
        if not paths:
            return None, None
        from Animation import Animation, Quaternions, positions_global
        d = np.load(paths[0], allow_pickle=True)
        parents = d["parents"]
        anim = Animation(
            rotations=Quaternions(d["rest_local_rot"][None]),
            positions=d["rest_local_pos"][None],
            orients=Quaternions.id(len(parents)),
            offsets=d["rest_local_pos"],
            parents=parents,
        )
        return positions_global(anim)[0], parents

    def skeleton_facts(self, asset_id):
        """Return (dict, text) or (None, "") when the export has no such rig."""
        names = self.joint_names.get(asset_id)
        if not names:
            return None, ""
        clean = self.clean_names.get(asset_id) or []
        facts = {"joints": len(names)}
        base = [re.sub(r"^(Left|Right) ", "", c) for c in clean]
        hist = Counter(b for b in base if b not in ("Bone", "Bone End", "Root", "Root End"))
        # Limb-defining labels always make the list (a T-rex's two Thigh
        # joints matter more than its twelve tail joints); the rest by count.
        limb = [(lab, n) for lab, n in hist.most_common() if _has_word(lab, LIMB_WORDS)]
        rest = [(lab, n) for lab, n in hist.most_common() if not _has_word(lab, LIMB_WORDS)]
        facts["labels"] = (limb + rest)[:MAX_LABELS]
        for flag, keys in LABEL_FLAGS.items():
            facts[flag] = any(_has_word(b, keys) for b in base)
        try:
            g, _ = self._rest_positions(asset_id)
        except Exception as e:  # noqa: BLE001 — evidence is optional
            logger.warning("rest pose FK failed for {}: {}".format(asset_id, e))
            g = None
        if g is not None:
            height = float(np.ptp(g[:, 1])) or 1.0
            ext = sorted(float(v) / height for v in (np.ptp(g[:, 0]), np.ptp(g[:, 2])))
            facts["horizontal_extents"] = [round(ext[1], 2), round(ext[0], 2)]
        lines = ["Skeleton facts (from the rig itself):",
                 "- {} joints".format(facts["joints"])]
        if "horizontal_extents" in facts:
            lines.append("- horizontal extents {} and {} (relative to height 1.0)".format(
                *facts["horizontal_extents"]))
        if facts["labels"]:
            lines.append("- joint labels: " + ", ".join(
                "{} x{}".format(lab, n) for lab, n in facts["labels"]))
        else:
            lines.append("- joint names are opaque (no anatomical labels)")
        flags = [f.replace("_", " ") for f in LABEL_FLAGS if facts.get(f)]
        if flags:
            lines.append("- labelled parts present: " + ", ".join(flags))
        return facts, "\n".join(lines)

    # -- first animation frame, four cameras ------------------------------
    def first_frame_views(self, asset_id, num_views=4):
        """First rendered frame of the asset's first clip from every camera
        (``<root>/<asset_id>-<clip>/v00k/0000.png``); [] when unavailable."""
        if not self.motion_render_root:
            return []
        clip_dirs = sorted(glob.glob(os.path.join(self.motion_render_root, asset_id + "-*")))
        if not clip_dirs and os.path.isdir(os.path.join(self.motion_render_root, asset_id)):
            clip_dirs = [os.path.join(self.motion_render_root, asset_id)]
        for clip_dir in clip_dirs:
            frames = []
            for k in range(num_views):
                pngs = sorted(glob.glob(os.path.join(clip_dir, "v{:03d}".format(k), "*.png")))
                if not pngs:
                    break
                frames.append(flatten_rgba(Image.open(pngs[0])))
            if len(frames) == num_views:
                return frames
        return []

    # -- captions --------------------------------------------------------
    def captions(self, asset_id):
        seen, out = set(), []
        for text in self.captions_by_asset.get(asset_id, []):
            if text not in seen:
                seen.add(text)
                out.append(text)
            if len(out) >= self.max_captions:
                break
        if not out:
            return ""
        return ("Motion captions of this asset's clips (written from rendered videos "
                "by a separate pass):\n" + "\n".join("- " + c for c in out))


# ─────────────────────────────────────────────────────────────────────────────
# Answer parsing and voting
# ─────────────────────────────────────────────────────────────────────────────

def _normalize_category(text):
    t = text.lower().replace(" ", "_").strip("_*`'\".,;:!?()[]\n\t ")
    if t in VALID_CATEGORIES or t == UNCERTAIN_CATEGORY:
        return t
    if t in ("quadruped", "quadrupeds"):
        return "quadrupedal"
    if t in ("biped", "bipeds", "humanoid"):
        return "bipedal"
    if t in ("insect", "arthropod"):
        return "insectoid"
    if t in ("flying", "winged", "bird"):
        return "avian"
    if t in ("aquatic", "fish"):
        return "marine"
    if t in ("snake", "limbless"):
        return "serpentine"
    if t in ("rigid", "mechanical", "articulated", "articulated-rigid"):
        return "articulated_rigid"
    return None


def parse_answer(raw):
    """Parse one model answer into {category, confidence, evidence, raw}.

    Accepts the requested JSON object, a JSON object embedded in prose, or a
    bare category token. ``category`` is None when nothing parses.
    """
    answer = {"category": None, "confidence": None, "evidence": {}, "raw": raw}
    m = _JSON_RE.search(raw)
    if m:
        try:
            obj = json.loads(m.group(0))
        except ValueError:
            obj = None
        if isinstance(obj, dict):
            cat = obj.get("category")
            answer["category"] = _normalize_category(str(cat)) if cat is not None else None
            conf = str(obj.get("confidence", "")).lower()
            answer["confidence"] = conf if conf in CONFIDENCES else None
            answer["evidence"] = {k: v for k, v in obj.items()
                                  if k not in ("category", "confidence")}
    if answer["category"] is None:
        m = _CATEGORY_RE.search(raw)
        if m:
            answer["category"] = _normalize_category(m.group(1))
    if answer["category"] is None:
        # Bare token / prose fallback: earliest valid category mentioned.
        low = raw.lower().replace(" ", "_")
        hits = sorted((low.find(v), v) for v in VALID_CATEGORIES + [UNCERTAIN_CATEGORY] if v in low)
        if hits:
            answer["category"] = hits[0][1]
    if answer["confidence"] is None:
        m = _CONFIDENCE_RE.search(raw)
        answer["confidence"] = m.group(1).lower() if m else None
    return answer


def aggregate_votes(answers, votes):
    """Majority vote over parsed answers → (category, confidence).

    * unanimous valid votes → that category, ``high``
    * a strict majority → that category, ``medium`` (``low`` if every
      majority vote reported low confidence)
    * otherwise, or a majority for the model's own ``uncertain`` → the
      ``uncertain`` bucket, ``low``
    """
    cats = [a["category"] for a in answers if a["category"]]
    if not cats:
        return None, None
    top, n_top = Counter(cats).most_common(1)[0]
    if top == UNCERTAIN_CATEGORY or n_top * 2 <= votes:
        return UNCERTAIN_CATEGORY, "low"
    majority_conf = [a["confidence"] for a in answers if a["category"] == top]
    if votes == 1:
        conf = majority_conf[0] or "medium"   # single answer: the model's own confidence
    elif n_top == votes:
        conf = "high"
    else:
        conf = "medium"
    if majority_conf and all(c == "low" for c in majority_conf):
        conf = "low"
    if conf == "low":
        return UNCERTAIN_CATEGORY, "low"
    return top, conf


# ─────────────────────────────────────────────────────────────────────────────
# Core classification
# ─────────────────────────────────────────────────────────────────────────────

def split_grid(image):
    """Split a 2x2 grid into its four tiles (TL, TR, BL, BR)."""
    w, h = image.size
    boxes = [(0, 0, w // 2, h // 2), (w // 2, 0, w, h // 2),
             (0, h // 2, w // 2, h), (w // 2, h // 2, w, h)]
    return [image.crop(b) for b in boxes]


def build_messages(images, asset_id, facts_text, captions_text, system_prompt,
                   motion_images=()):
    n = len(images)
    intro = ("Classify this 3D asset (id: '{}'). ".format(asset_id) +
             ("The next {} images are rest-pose (T-pose) views of the same asset "
              "from four cameras 90 degrees apart (nominally the asset's front, "
              "back, left and right; the character itself may face any "
              "direction).".format(n)
              if n > 1 else
              "The next image is a 2x2 grid of four canonical rest-pose views."))
    content = [{"type": "text", "text": intro}]
    for img in images:
        content.append({"type": "image", "image": img})
    if motion_images:
        content.append({"type": "text", "text": (
            "The next {} images are the FIRST FRAME of one of this asset's "
            "animation clips, seen from four static cameras 90 degrees apart. "
            "They show the character in its natural stance and orientation; "
            "the rest pose above may be lying flat on the ground, floating or "
            "rotated, so judge upright vs. horizontal and which limbs touch "
            "the ground from these frames.".format(len(motion_images)))})
        for img in motion_images:
            content.append({"type": "image", "image": img})
    extra = "\n\n".join(t for t in (facts_text, captions_text) if t)
    if extra:
        content.append({"type": "text", "text": extra})
    content.append({"type": "text", "text": "Answer with the JSON object only."})
    return [
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
        {"role": "user", "content": content},
    ]


def classify_asset(images, asset_id, model, processor, facts_text="",
                   captions_text="", max_tokens=96, votes=3,
                   system_prompt=CATEGORY_PROMPT, motion_images=()):
    """Classify one asset. Returns (category, review_record).

    ``category`` is one of VALID_CATEGORIES, ``uncertain``, or None when
    fewer than half of the votes could be parsed (caller retries).
    """
    messages = build_messages(images, asset_id, facts_text, captions_text, system_prompt,
                              motion_images=motion_images)
    answers = []
    for v in range(votes):
        raw = qwen_generate(model, processor, messages, max_tokens=max_tokens,
                            do_sample=votes > 1)
        answers.append(parse_answer(raw))
    n_parsed = sum(1 for a in answers if a["category"])
    if n_parsed * 2 <= votes and n_parsed < votes:
        logger.warning("Only {}/{} votes parsed for '{}': {}".format(
            n_parsed, votes, asset_id, [a["raw"][:80] for a in answers]))
        return None, None
    category, confidence = aggregate_votes(answers, votes)
    record = {
        "category": category,
        "confidence": confidence,
        "votes": [a["category"] for a in answers],
        "answers": [{"category": a["category"], "confidence": a["confidence"],
                     **a["evidence"]} for a in answers],
    }
    logger.info("Category for '{}': {} ({}; votes={})".format(
        asset_id, category, confidence, record["votes"]))
    return category, record


# ─────────────────────────────────────────────────────────────────────────────
# Asset discovery + batch processing
# ─────────────────────────────────────────────────────────────────────────────

def _discover_assets(render_root):
    """Locate (asset_id, tpose_png) pairs under ``render_root``.

    Auto-detects the layout:
      * flat   — ``<render_root>/<asset_id>.png`` (one render per asset).
      * nested — ``<render_root>/<NAME-MOTION>/tpose_grid.png`` (multi-clip;
                 grouped by the segment before '-' so every object type is
                 classified exactly once via its first available render).
    Returns ``(layout, items)`` where ``items`` is sorted by asset_id.
    """
    flat_pngs = sorted(p for p in render_root.glob("*.png") if p.is_file())
    if flat_pngs:
        return "flat", [(p.stem, p) for p in flat_pngs]
    char_dirs = sorted(
        d for d in render_root.iterdir()
        if d.is_dir() and (d / "tpose_grid.png").exists()
    )
    type_to_png = {}
    for d in char_dirs:
        obj_type = d.name.split("-", 1)[0] if "-" in d.name else d.name
        type_to_png.setdefault(obj_type, d / "tpose_grid.png")
    return "nested", sorted(type_to_png.items())


def _sibling(category_groups_json, suffix):
    p = Path(category_groups_json)
    return str(p.with_name(p.stem + suffix + ".json"))


def classify_batch(
    render_root,
    model,
    processor,
    category_groups_json=None,
    evidence=None,
    max_tokens=96,
    votes=3,
    split=True,
    retry_uncertain=False,
    system_prompt=CATEGORY_PROMPT,
):
    """Classify every asset under ``render_root`` (see module docstring)."""
    render_root = Path(render_root)
    if category_groups_json is None:
        category_groups_json = str(render_root / "category_groups.json")
    error_log_path = _sibling(category_groups_json, "_errors")
    review_path = _sibling(category_groups_json, "_review")

    existing_groups = load_json(category_groups_json)
    already_classified = {}
    for cat, members in existing_groups.items():
        if cat == UNKNOWN_CATEGORY or (retry_uncertain and cat == UNCERTAIN_CATEGORY):
            continue
        for asset_id in members:
            already_classified[asset_id] = cat
    existing_errors = load_json(error_log_path)
    asset_errors = {aid: err for aid, err in existing_errors.items()
                    if aid not in already_classified}
    review = {aid: rec for aid, rec in load_json(review_path).items()
              if aid in already_classified}

    layout, items = _discover_assets(render_root)
    logger.info("Layout: {} | Found {} assets in {}".format(layout, len(items), render_root))
    asset_categories = dict(already_classified)
    to_classify = [(aid, p) for aid, p in items if aid not in already_classified]
    logger.info("{} already classified, {} remaining ({} previously errored)".format(
        len(already_classified), len(to_classify), len(asset_errors)))

    evidence = evidence or ExportEvidence(None)
    pbar = tqdm(to_classify, desc="Classifying", unit="asset", dynamic_ncols=True)
    total_attempts = NUM_RETRIES + 1
    n_unsaved = 0

    def _flush():
        _save_category_groups(asset_categories, category_groups_json)
        save_json(asset_errors, error_log_path)
        save_json(review, review_path)

    try:
        for asset_id, png_path in pbar:
            pbar.set_postfix_str(asset_id, refresh=False)
            facts, facts_text = evidence.skeleton_facts(asset_id)
            captions_text = evidence.captions(asset_id)
            motion_images = evidence.first_frame_views(asset_id)
            result, record, last_error = None, None, None
            for attempt in range(1, total_attempts + 1):
                try:
                    image = flatten_rgba(Image.open(str(png_path)))
                    images = split_grid(image) if split else [image]
                    result, record = classify_asset(
                        images, asset_id, model, processor,
                        facts_text=facts_text, captions_text=captions_text,
                        max_tokens=max_tokens, votes=votes, system_prompt=system_prompt,
                        motion_images=motion_images)
                    if result is not None:
                        break
                    last_error = ValueError("too few parseable votes")
                except Exception as e:  # noqa: BLE001 — retried below
                    last_error = e
                    logger.warning("Attempt {}/{} failed for {}: {}".format(
                        attempt, total_attempts, asset_id, e))
                if attempt < total_attempts:
                    time.sleep(RETRY_BACKOFF_SECONDS)
            if result is not None:
                asset_categories[asset_id] = result
                asset_errors.pop(asset_id, None)
                if facts:
                    record["skeleton"] = {k: v for k, v in facts.items() if k != "labels"}
                record["motion_frames"] = len(motion_images)
                review[asset_id] = record
            else:
                err_msg = "All {} attempts failed: {}".format(total_attempts, last_error)
                logger.error("[{}] {}".format(asset_id, err_msg))
                asset_categories[asset_id] = "ERROR: {}".format(last_error)
                asset_errors[asset_id] = err_msg
                review.pop(asset_id, None)
            n_unsaved += 1
            if n_unsaved >= SAVE_EVERY:
                _flush()
                n_unsaved = 0
    finally:
        _flush()

    counts = Counter(c for c in asset_categories.values()
                     if isinstance(c, str) and not c.startswith("ERROR:"))
    logger.info("Done. {} classified {}, {} errored (logged to {})".format(
        sum(counts.values()), dict(counts), len(asset_errors), error_log_path))
    return asset_categories


def _save_category_groups(asset_categories, output_path):
    """Invert {asset_id: category} into {category: [asset_id, ...]} and save.

    ERROR entries are dropped so a re-run will retry those assets.
    """
    assets_by_category = defaultdict(list)
    for asset_id, cat in sorted(asset_categories.items()):
        if isinstance(cat, str) and cat.startswith("ERROR:"):
            continue
        assets_by_category[cat].append(asset_id)
    save_json({cat: sorted(a) for cat, a in sorted(assets_by_category.items())}, output_path)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Classify 3D assets into body-plan categories with a local "
                    "Qwen3.5 / Qwen3-VL model (T-pose views + skeleton facts + captions)."
    )
    parser.add_argument('--render_root', type=str, required=True,
                        help='Root directory holding T-pose 2x2 grid PNGs '
                             '(flat <asset_id>.png or nested '
                             '<NAME-MOTION>/tpose_grid.png — auto-detected).')
    parser.add_argument('--export_dir', type=str, default=None,
                        help='Stage-1/2/3 export folder of the dataset '
                             '(joint_names.json, clean_joint_names.json, '
                             'motion_captions.json, motions/). Optional but '
                             'strongly recommended: supplies the skeleton facts '
                             'and captions.')
    parser.add_argument('--category_groups_json', type=str, default=None,
                        help='Output JSON {category: [asset_id, ...]} '
                             '(default <render_root>/category_groups.json). '
                             'Siblings <stem>_review.json and <stem>_errors.json '
                             'are written next to it; all are re-read on entry.')
    parser.add_argument('--motion_render_root', type=str, default=None,
                        help='Stage-2a motion renders (dataset/render/<dataset>): the '
                             'first frame of the asset\'s first clip from all four '
                             'cameras is shown as the natural-stance reference. '
                             'Optional but recommended: rest poses can lie flat.')
    parser.add_argument('--votes', type=int, default=3,
                        help='Sampled answers per asset; majority wins (default 3; '
                             '1 = single greedy answer).')
    parser.add_argument('--max_captions', type=int, default=6,
                        help='Captions of the asset shown to the model (default 6).')
    parser.add_argument('--split_grid', action=argparse.BooleanOptionalAction, default=True,
                        help='Pass the four views of the grid as separate images '
                             '(default) or the grid as one image.')
    parser.add_argument('--retry_uncertain', action='store_true',
                        help='Also re-attempt assets currently under "uncertain".')
    add_qwen_model_args(parser)
    # The captioner defaults (256*28*28 both ways) are far below the
    # 256–1280 visual tokens the Qwen3-VL guide recommends per image; the
    # 512x512 tiles of the grid pass through untouched with these bounds.
    parser.set_defaults(min_pixels=256 * 32 * 32, max_pixels=1280 * 32 * 32)
    parser.add_argument('--max_tokens', type=int, default=96,
                        help='Max new tokens per answer (default 96: one JSON object).')
    return parser.parse_args()


def main():
    args = parse_args()
    qwen_model, qwen_processor = load_qwen_model(
        args.model,
        device_map=args.device_map,
        torch_dtype=args.torch_dtype,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
    )
    evidence = ExportEvidence(args.export_dir, max_captions=args.max_captions,
                              motion_render_root=args.motion_render_root)
    classify_batch(
        render_root=args.render_root,
        model=qwen_model, processor=qwen_processor,
        category_groups_json=args.category_groups_json,
        evidence=evidence,
        max_tokens=args.max_tokens,
        votes=args.votes,
        split=args.split_grid,
        retry_uncertain=args.retry_uncertain,
    )


if __name__ == "__main__":
    main()
