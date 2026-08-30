"""Classify 3D assets into body-plan categories using a local Qwen3-VL model.

Input is a 2x2 multi-view grid PNG per asset (front, back, plus two side
views, T-pose). Two render layouts are supported and auto-detected:

  Flat (e.g. objaverse):     <render_root>/<asset_id>.png
  Nested (e.g. truebones):   <render_root>/<NAME-MOTION>/tpose_grid.png
      Object types are grouped by the segment before '-'; one
      classification per type is then propagated to every clip of that
      type.

Output:
    * ``category_groups.json``  — {category: [asset_id, ...]} for assets
                                  that classified successfully.
    * ``category_groups_errors.json`` (sibling, same directory)
                                — {asset_id: error_message} for assets
                                  whose retries were exhausted. ERROR
                                  entries are NOT in category_groups.json
                                  so a re-run will retry them automatically;
                                  the error log persists the last reason.

Usage:
    # Objaverse (flat layout)
    python -m data_process.vlm_caption.classify_category \
        --render_root dataset/render/objaverse_tpose

    # Truebones (nested layout)
    python -m data_process.vlm_caption.classify_category \
        --render_root dataset/render/truebones_tpose
"""

import argparse
import time
from collections import defaultdict
from pathlib import Path

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
UNKNOWN_CATEGORY = "unknown"  # fallback when the model output matches no valid category

# Per-asset retry policy: up to NUM_RETRIES additional attempts after the
# initial try (5 total). Retries cover both exceptions (image I/O, CUDA OOM,
# generation failure) and unparseable model outputs (UNKNOWN_CATEGORY) so
# stochastic decoding gets a second chance to land on a valid category.
NUM_RETRIES = 4
RETRY_BACKOFF_SECONDS = 2.0

SAVE_EVERY = 20  # assets per incremental JSON flush (both output files)


# ─────────────────────────────────────────────────────────────────────────────
# Core classification
# ─────────────────────────────────────────────────────────────────────────────

def classify_asset(
    tpose_image,
    asset_id,
    model,
    processor,
    max_tokens=10,
    system_prompt=CATEGORY_PROMPT,
    do_sample=False,
):
    # type: (Image.Image, str, object, object, int, str, bool) -> str
    """Classify one asset from its 2x2 T-pose grid PNG.

    Returns one of ``VALID_CATEGORIES`` or ``UNKNOWN_CATEGORY`` if the
    model output cannot be parsed into a known category.
    """
    messages = [
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
        {"role": "user", "content": [
            {"type": "text", "text": (
                "Classify this 3D asset (id: '{}'). The image is a 2x2 grid "
                "of four canonical T-pose views.".format(asset_id)
            )},
            {"type": "image", "image": tpose_image},
        ]},
    ]

    raw = qwen_generate(model, processor, messages, max_tokens=max_tokens,
                        do_sample=do_sample).lower()
    # Accept "articulated rigid" (space) as well as "articulated_rigid" (underscore).
    raw_normalized = raw.replace(" ", "_").strip()
    # Bare token after stripping quotes/punctuation — the expected reply.
    bare = raw_normalized.strip("_*`'\".,;:!?()[]\n\t ")

    if bare in VALID_CATEGORIES:
        category = bare
    else:
        # Take the category that appears EARLIEST in the response, not the
        # first one in VALID_CATEGORIES order: "quadrupedal (not bipedal)"
        # must resolve to quadrupedal.
        hits = sorted((raw_normalized.find(v), v)
                      for v in VALID_CATEGORIES if v in raw_normalized)
        if len(hits) > 1:
            logger.warning("Ambiguous category response '{}' for '{}': {} — "
                           "taking the earliest match '{}'".format(
                               raw, asset_id,
                               ", ".join(v for _, v in hits), hits[0][1]))
        category = hits[0][1] if hits else None
    if category is None:
        logger.warning("Unexpected category '{}' for '{}', mapping to '{}'".format(
            raw, asset_id, UNKNOWN_CATEGORY))
        category = UNKNOWN_CATEGORY

    logger.info("Category for '{}': {}".format(asset_id, category))
    return category


# ─────────────────────────────────────────────────────────────────────────────
# Asset discovery + batch processing
# ─────────────────────────────────────────────────────────────────────────────

def _discover_assets(render_root):
    # type: (Path) -> Tuple[str, List[Tuple[str, Path]]]
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
        items = [(p.stem, p) for p in flat_pngs]
        return "flat", items

    char_dirs = sorted(
        d for d in render_root.iterdir()
        if d.is_dir() and (d / "tpose_grid.png").exists()
    )
    type_to_png = {}  # type: Dict[str, Path]
    for d in char_dirs:
        obj_type = d.name.split("-", 1)[0] if "-" in d.name else d.name
        type_to_png.setdefault(obj_type, d / "tpose_grid.png")
    items = sorted(type_to_png.items())
    return "nested", items


def _error_log_path_for(category_groups_json):
    # type: (str) -> str
    """Sibling error-log path next to ``category_groups.json``."""
    p = Path(category_groups_json)
    return str(p.with_name(p.stem + "_errors.json"))


def classify_batch(
    render_root,
    model,
    processor,
    category_groups_json=None,
    max_tokens=10,
    system_prompt=CATEGORY_PROMPT,
):
    """Classify every asset under ``render_root``.

    Outputs:
        * ``category_groups.json`` — {category: [asset_id, ...]} for the
          assets that classified successfully.
        * ``<stem>_errors.json``   — {asset_id: error_message} for assets
          whose retries were exhausted (siblings of the groups JSON).

    Both files are read on entry for resumability — successful asset_ids
    are skipped, and stale error entries (now-successful assets) are
    pruned. Re-runs only re-attempt assets in the error log plus any
    assets the discovery pass found that aren't yet classified.
    """
    render_root = Path(render_root)

    if category_groups_json is None:
        category_groups_json = str(render_root / "category_groups.json")
    error_log_path = _error_log_path_for(category_groups_json)

    existing_groups = load_json(category_groups_json)  # type: Dict[str, List[str]]
    already_classified = {}  # type: Dict[str, str]
    for cat, members in existing_groups.items():
        for asset_id in members:
            already_classified[asset_id] = cat

    # Prune stale errors (assets now successfully classified) and retain
    # the rest as the running error log for this run.
    existing_errors = load_json(error_log_path)  # type: Dict[str, str]
    asset_errors = {
        aid: err for aid, err in existing_errors.items()
        if aid not in already_classified
    }  # type: Dict[str, str]

    layout, items = _discover_assets(render_root)
    logger.info("Layout: {} | Found {} assets in {}".format(
        layout, len(items), render_root))

    asset_categories = dict(already_classified)  # type: Dict[str, str]
    to_classify = [(aid, p) for aid, p in items if aid not in already_classified]
    logger.info("{} already classified, {} remaining ({} previously errored)".format(
        len(already_classified), len(to_classify), len(asset_errors)))

    pbar = tqdm(to_classify, desc="Classifying", unit="asset",
                dynamic_ncols=True)
    total_attempts = NUM_RETRIES + 1
    n_unsaved = 0

    def _flush():
        """Write both output files (called every SAVE_EVERY assets)."""
        _save_category_groups(asset_categories, category_groups_json)
        save_json(asset_errors, error_log_path)

    try:
        for asset_id, png_path in pbar:
            pbar.set_postfix_str(asset_id, refresh=False)
            result = None
            last_error = None
            for attempt in range(1, total_attempts + 1):
                try:
                    tpose_image = flatten_rgba(Image.open(str(png_path)))
                    # Greedy first, sampled on retries — a greedy retry would
                    # deterministically repeat the same unparseable output.
                    category = classify_asset(
                        tpose_image, asset_id,
                        model=model, processor=processor,
                        max_tokens=max_tokens, system_prompt=system_prompt,
                        do_sample=attempt > 1,
                    )
                    if category != UNKNOWN_CATEGORY:
                        result = category
                        break
                    last_error = ValueError(
                        "model output did not match any valid category")
                except Exception as e:
                    last_error = e
                    logger.warning("Attempt {}/{} failed for {}: {}".format(
                        attempt, total_attempts, asset_id, e))

                if attempt < total_attempts:
                    time.sleep(RETRY_BACKOFF_SECONDS)

            if result is not None:
                asset_categories[asset_id] = result
                asset_errors.pop(asset_id, None)  # success clears any prior error
            else:
                # Exhausted retries on either exceptions or unparseable output.
                # Mark as ERROR so _save_category_groups drops it from the JSON
                # and a later re-run will retry from scratch. Record the reason
                # in the sibling error log for visibility.
                if isinstance(last_error, ValueError):
                    err_msg = "All {} attempts produced unparseable output".format(
                        total_attempts)
                else:
                    err_msg = "All {} attempts failed: {}".format(
                        total_attempts, last_error)
                logger.error("[{}] {}".format(asset_id, err_msg))
                asset_categories[asset_id] = "ERROR: {}".format(last_error)
                asset_errors[asset_id] = err_msg

            # Rewriting both JSONs per asset is O(n^2) over a large run;
            # batch it (a crash loses at most SAVE_EVERY classifications).
            n_unsaved += 1
            if n_unsaved >= SAVE_EVERY:
                _flush()
                n_unsaved = 0
    finally:
        _flush()
    n_ok = sum(
        1 for c in asset_categories.values()
        if isinstance(c, str) and not c.startswith("ERROR:")
    )
    logger.info("Done. {} classified, {} errored (logged to {})".format(
        n_ok, len(asset_errors), error_log_path))
    return asset_categories


def _save_category_groups(asset_categories, output_path):
    # type: (Dict[str, str], str) -> None
    """Invert {asset_id: category} into {category: [asset_id, ...]} and save.

    ERROR entries are dropped so a re-run will retry those assets. After
    retries are exhausted (in classify_batch) the asset is marked ERROR
    and recorded in the sibling error log instead of leaking into the
    groups JSON.
    """
    assets_by_category = defaultdict(list)  # type: Dict[str, List[str]]
    for asset_id, cat in sorted(asset_categories.items()):
        if isinstance(cat, str) and cat.startswith("ERROR:"):
            continue
        assets_by_category[cat].append(asset_id)

    category_assets = {
        cat: sorted(assets) for cat, assets in sorted(assets_by_category.items())
    }  # type: Dict[str, List[str]]
    save_json(category_assets, output_path)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Classify 3D assets into body-plan categories "
                    "using a local Qwen3-VL model."
    )

    parser.add_argument('--render_root', type=str, required=True,
                        help='Root directory holding T-pose 2x2 grid PNGs '
                             '(flat <asset_id>.png or nested '
                             '<NAME-MOTION>/tpose_grid.png — auto-detected).')

    parser.add_argument('--category_groups_json', type=str, default=None,
                        help='Output JSON file mapping '
                             '{category: [asset_id, ...]}. '
                             'Defaults to <render_root>/category_groups.json. '
                             'A sibling <stem>_errors.json records assets '
                             'whose retries were exhausted. Both files are '
                             're-read on entry for resumability.')
    add_qwen_model_args(parser)
    parser.add_argument('--max_tokens', type=int, default=10,
                        help='Max new tokens to generate (default: 10).')

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

    classify_batch(
        render_root=args.render_root,
        model=qwen_model, processor=qwen_processor,
        category_groups_json=args.category_groups_json,
        max_tokens=args.max_tokens,
    )


if __name__ == "__main__":
    main()
