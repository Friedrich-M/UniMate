"""Inference entry point: generate motion samples from a trained model."""
import dataclasses
import gc
import glob
import json
import os
import re
import time
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import tyro
from accelerate.utils import set_seed

warnings.filterwarnings("ignore")

from unimate.configs.schema import MainConfig
from unimate.dataset.conditioning import create_sample_condition
from unimate.dataset.factory import create_dataset
from unimate.inference.motion_inbetweening import build_keep_mask, parse_keep_frames
from unimate.inference.motion_editing import (
    build_joint_keep_mask,
    get_joint_names,
    parse_keep_joints,
)
from unimate.inference.motion_expansion import expand_motion_chain
from unimate.models.flow.transport import Sampler
from unimate.models.factory import create_diffusion, create_model, create_transport
from unimate.training.ema import EMAModel
from unimate.models.text_encoder.factory import create_text_encoder
from unimate.utils.logger import get_logger
from unimate.inference.generate import generate_samples
from unimate.utils.text_emb_cache import pool, sequences_from_hidden
from unimate.utils.visualization import visualize_and_save_motions

logger = get_logger(file_name=__file__)

# (case_id_for_filename, object_type, caption_text, caption_enc, clip_name).
# ``caption_enc`` is the encoded prompt as ``{'caption_emb': (D,),
# 'caption_tokens': (T, D)}`` — the pooled vector text_cond='adaln' reads and
# the token sequence text_cond='cross_attn' attends, mirroring what the data
# loader hands the model at training time. ``clip_name`` is None unless an
# in-betweening / motion-editing run pins a specific clip, whose GT motion is
# then clamped during sampling.
CaptionEnc = Dict[str, np.ndarray]
TestCase = Tuple[str, str, str, CaptionEnc, Optional[str]]


@dataclasses.dataclass
class InferenceArgs:
    """Command-line arguments for inference (parsed by tyro)."""
    exp_dir: str
    model_path: Optional[str] = None
    seed: Optional[int] = None
    output_dir: Optional[str] = None
    num_repetitions: int = 1
    cfg_scale: Optional[float] = None
    # JSON of {"<object_type>-<case_id>": caption}; used when cfg_scale > 1.0.
    # When None, every clip in the dataset is enumerated as a test case.
    test_cases_json: Optional[str] = None
    # Plain-text object_types (one per line) for unconditional sampling at
    # cfg_scale == 1.0. '#' starts a comment.
    test_cases_txt: Optional[str] = None
    # Per-chunk inference batch size — caps GPU memory regardless of total count.
    batch_size: int = 64
    # Skip mp4/PNG renders; write only the .npy motion features.
    only_save_motion: bool = False
    # Skip the RIC-recovered mp4 (FK render + T-pose still saved).
    # Defaults to False at inference: RIC is largely redundant with FK.
    save_ric: bool = False
    # Motion in-betweening: clamp the specified frames to ground truth and
    # let the flow ODE denoise only the rest. Requires either
    # --test_cases_json (with keys '<object_type>-<clip_id>' resolvable in
    # train or eval motion_dict) or the dataset's eval split.
    inbetween: bool = False
    # Comma-separated signed temporal indices to hold clean. Negatives count
    # from the per-clip valid length. Default keeps first + last frame.
    keep_frames: str = "0,-1"
    # Text-guided motion editing: clamp the listed joints to GT for all
    # frames and let the model denoise the rest under a new caption (e.g.,
    # fix the lower body, change the upper body action). Same clip-pinning
    # requirement as --inbetween; mutually exclusive with it.
    motion_edit: bool = False
    # Comma-separated joint names to hold clean (matched case-insensitive
    # against the per-skeleton ``clean_joint_names`` / ``joint_names``).
    keep_joints: str = ""
    # When set, the GT clip is cropped starting at this absolute frame index
    # (instead of the default 'tpos' random window). Up to max_motion_length
    # frames are taken from start_idx; if fewer remain, the GT is trimmed
    # and the model still generates over the full ODE window but saved
    # outputs (and the GT-clamp signal) are trimmed back via valid_lengths.
    # Most useful with --motion_edit / --inbetween for deterministic, user-
    # chosen edit windows.
    gt_start_frame: Optional[int] = None
    # Motion expansion: chain multiple text-conditioned generations into one
    # long motion. Requires --test_cases_json whose values are *lists* of
    # prompts (one segment per prompt). Each segment after the first pins
    # its first `expand_overlap` frames to the previous segment's last
    # `expand_overlap` frames via replacement-style sampling. Mutually
    # exclusive with --inbetween / --motion_edit.
    motion_expand: bool = False
    # Number of frames overlapped (clamped) between consecutive expansion
    # segments. Must satisfy 0 < expand_overlap < max_motion_length.
    expand_overlap: int = 10


class _UnconditionalWrapper(nn.Module):
    """Force ``force_mask=True`` so the caption embedding is always zeroed.

    Used at cfg_scale == 1.0 to mirror training-time CFG dropout while
    preserving topology conditioning.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x, timesteps, cond=None):
        return self.model(x, timesteps, cond, force_mask=True)


# ---------------------------------------------------------------------------
# Setup helpers
# ---------------------------------------------------------------------------


def _resolve_exp_paths(exp_dir: str, model_path: Optional[str]) -> Tuple[str, str]:
    """Locate config + highest-step checkpoint inside ``exp_dir``."""
    config_path = os.path.join(exp_dir, "config.json")
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"No config.json in {exp_dir!r}")

    if model_path is not None:
        return config_path, model_path

    ckpt_dir = os.path.join(exp_dir, "checkpoints")
    candidates = glob.glob(os.path.join(ckpt_dir, "checkpoint_step_*.pt"))
    if not candidates:
        raise FileNotFoundError(f"No checkpoint_step_*.pt files in {ckpt_dir!r}")

    step_re = re.compile(r"checkpoint_step_(\d+)\.pt$")

    def _step(p):
        m = step_re.search(os.path.basename(p))
        return int(m.group(1)) if m else -1

    return config_path, max(candidates, key=_step)


def _build_diffusion(config: MainConfig):
    """Return ``(diffusion, gen_diffusion)`` based on ``training.diff_model``."""
    if config.training.diff_model == 'flow':
        diffusion = create_transport(training_config=config.training)
        return diffusion, Sampler(diffusion)
    if config.training.diff_model == 'diffusion':
        diffusion = create_diffusion(
            scheduler_config=config.scheduler,
            training_config=config.training,
        )
        return diffusion, None
    raise ValueError(f"Unknown diff_model: {config.training.diff_model!r}")


def _load_checkpoint(model, model_path: str, config: MainConfig):
    """Load weights into ``model`` in place; apply EMA shadow if available."""
    state_dict = torch.load(model_path, map_location='cpu')
    model.load_state_dict(state_dict.get('model_state_dict', state_dict))

    if config.training.use_ema and 'ema_state_dict' in state_dict:
        logger.info("Loading EMA weights into model parameters.")
        ema_model = EMAModel(
            parameters=model.parameters(),
            decay=config.training.ema_decay,
            use_ema_warmup=True,
        )
        ema_model.load_state_dict(state_dict['ema_state_dict'])
        ema_model.copy_to(model.parameters())
    else:
        logger.info("Using standard model weights (no EMA).")


# ---------------------------------------------------------------------------
# Test-case loaders
# ---------------------------------------------------------------------------


def _known_object_types(dataset) -> set:
    """Object types with at least one train or eval clip — only these can
    realize a test case (otherwise no reference clip is available)."""
    md = dataset.motion_dataset
    return {
        ot for ot in md.cond_dict.keys()
        if md.train_object_motions_map.get(ot) or md.eval_object_motions_map.get(ot)
    }


def _make_text_encoder(config: MainConfig, device: torch.device):
    # pool=False so a prompt yields its token sequence; ``_encode_prompt``
    # derives the pooled vector from it, the same rule the data loader uses.
    return create_text_encoder(
        encoder_type=config.model.text_encoder_type,
        encoder_version=config.model.text_encoder_version,
        device=str(device),
        pool=False,
    )


def _encode_prompt(encoder, text: str) -> CaptionEnc:
    """Encode one prompt -> ``{'caption_emb': (D,), 'caption_tokens': (T, D)}``.

    Padding is dropped by the attention mask, so the mean over the kept rows
    is exactly the encoder's own pooled output — including for the empty
    prompt, whose mask is all zeros and which therefore pools to the zero
    vector the unconditional reference has always been.
    """
    with torch.no_grad():
        inputs = encoder.tokenize(text)
        hidden = encoder(inputs)                                   # (1, T, D)
    tokens = sequences_from_hidden(hidden.detach().cpu(),
                                   inputs['attention_mask'].cpu())[0]
    return {'caption_emb': pool(tokens), 'caption_tokens': tokens}


def _release_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _resolve_inference_scope(args: InferenceArgs) -> Tuple[Optional[set], Optional[set]]:
    """Return ``(target_object_types, target_clip_stems)`` extracted from the
    test-case spec, or ``(None, None)`` when neither file is supplied.

    ``target_clip_stems`` is populated only for in-betweening / motion
    editing — they pin specific GT clips that must actually be on disk.
    """
    object_types: Optional[set] = None
    clip_stems: Optional[set] = None

    if args.test_cases_json is not None:
        with open(args.test_cases_json) as f:
            raw_cases = json.load(f)
        object_types = set()
        if args.inbetween or args.motion_edit:
            clip_stems = set()
        for case_key in raw_cases:
            obj_type, _, clip_id = case_key.partition('-')
            if obj_type:
                object_types.add(obj_type)
            if clip_stems is not None and clip_id:
                # On-disk file stem is "{obj_type}-{clip_id}" for
                # Truebones/Objaverse and "{obj_type}_{clip_id}" for Mixamo
                # (prefixed at load). Also keep the literal clip_id for
                # users who already supply the full stem in case_key.
                clip_stems.add(clip_id)
                clip_stems.add(f"{obj_type}-{clip_id}")
                clip_stems.add(f"{obj_type}_{clip_id}")
        logger.info(
            f"Inference scope from {args.test_cases_json}: "
            f"{len(object_types)} object types"
            + (f", {len(clip_stems)} pinned clip stems" if clip_stems else '')
        )

    if args.test_cases_txt is not None:
        if object_types is None:
            object_types = set()
        with open(args.test_cases_txt) as f:
            for line in f:
                s = line.strip()
                if s and not s.startswith('#'):
                    object_types.add(s)
        logger.info(
            f"Inference scope augmented from {args.test_cases_txt}: "
            f"{len(object_types)} object types total."
        )

    return object_types, clip_stems


def _resolve_stats_path(exp_dir: str) -> Optional[str]:
    """Return path to ``dataset_stats.npy`` in ``exp_dir``, or ``None``."""
    candidate = os.path.join(exp_dir, "dataset_stats.npy")
    if os.path.isfile(candidate):
        return candidate
    logger.warning(
        f"No dataset_stats.npy at {candidate} — stats will be recomputed from "
        f"the loaded clips, which may not match training-time normalization."
    )
    return None


def _resolve_clip_name(dataset, obj_type: str, clip_id: str) -> Optional[str]:
    """Find the dataset key whose stem matches ``clip_id`` for ``obj_type``.

    Motion-dict keys are filenames (typically with a ``.npz`` extension);
    user-supplied case ids omit the extension. We search train then eval —
    same priority as :func:`create_sample_condition` — and verify the
    ``object_type`` so a clip-id collision across types can't pin the wrong
    skeleton.
    """
    md = dataset.motion_dataset
    # Accept either the suffix-only form (``clip_id``) or one of the prefixed
    # forms that match motion_dict keys: Truebones/Objaverse stems are
    # "{obj_type}-{clip_id}", Mixamo prefixed stems are "{obj_type}_{clip_id}".
    candidates = {clip_id, f"{obj_type}-{clip_id}", f"{obj_type}_{clip_id}"}
    for source in (md.train_motion_dict, md.eval_motion_dict):
        for clip_name in source:
            stem = os.path.splitext(clip_name)[0]
            if stem in candidates and source[clip_name].get('object_type') == obj_type:
                return clip_name
    return None


def _load_test_cases_json(
    json_path: str,
    config: MainConfig,
    dataset,
    device: torch.device,
    resolve_clip: bool = False,
) -> List[TestCase]:
    """Caption-conditioned cases from {"<object_type>-<id>": caption} JSON.

    When ``resolve_clip`` is True (in-betweening), the ``<id>`` portion is
    treated as a dataset clip stem and must resolve to an actual clip in
    train or eval motion_dict — otherwise the case is dropped, since
    in-betweening has nothing to clamp against. When False, ``<id>`` is a
    free-form tag and only ``<object_type>`` needs to exist.
    """
    with open(json_path) as f:
        raw_cases: Dict[str, str] = json.load(f)

    known = _known_object_types(dataset)
    cases: List[Tuple[str, str, str, Optional[str]]] = []
    for case_key, caption in raw_cases.items():
        obj_type, _, clip_id = case_key.partition('-')
        if obj_type not in known:
            logger.warning(
                f"Test case {case_key!r}: object_type={obj_type!r} not in dataset; skipping."
            )
            continue
        # An empty caption at cfg>1 collapses to ~unconditional output and
        # is almost always a data-entry mistake; skip rather than silently
        # generating garbage.
        if not caption or not caption.strip():
            logger.warning(f"Test case {case_key!r}: empty caption; skipping.")
            continue

        resolved_clip: Optional[str] = None
        if resolve_clip:
            if not clip_id:
                logger.warning(
                    f"Test case {case_key!r}: in-betweening requires "
                    f"'<object_type>-<clip_id>' format with a non-empty id; skipping."
                )
                continue
            resolved_clip = _resolve_clip_name(dataset, obj_type, clip_id)
            if resolved_clip is None:
                logger.warning(
                    f"Test case {case_key!r}: clip_id={clip_id!r} not found in "
                    f"train or eval motion_dict for object_type={obj_type!r}; skipping."
                )
                continue

        cases.append((case_key, obj_type, caption, resolved_clip))

    if not cases:
        raise ValueError(
            f"No usable test cases in {json_path}: object_types not present in dataset."
        )
    logger.info(f"Loaded {len(cases)} test cases from {json_path}")

    encoder = _make_text_encoder(config, device)
    encoded: List[TestCase] = []
    for case_key, obj_type, caption, clip_name in cases:
        encoded.append((case_key, obj_type, caption,
                        _encode_prompt(encoder, caption), clip_name))
    del encoder
    _release_gpu()
    return encoded


def _load_test_cases_txt(
    txt_path: str,
    config: MainConfig,
    dataset,
    device: torch.device,
) -> List[TestCase]:
    """Unconditional cases — one object_type per line.

    The empty-string embedding is a shape-only placeholder; ``_UnconditionalWrapper``
    zeroes it inside the model at sample time.
    """
    with open(txt_path) as f:
        object_types = [
            line.strip() for line in f
            if line.strip() and not line.lstrip().startswith('#')
        ]

    known = _known_object_types(dataset)
    cases = []
    for ot in object_types:
        if ot not in known:
            logger.warning(f"object_type={ot!r} not in dataset; skipping.")
            continue
        cases.append(ot)

    if not cases:
        raise ValueError(
            f"No usable test cases in {txt_path}: object_types not present in dataset."
        )
    logger.info(f"Loaded {len(cases)} unconditional test cases from {txt_path}")

    encoder = _make_text_encoder(config, device)
    null_emb = _encode_prompt(encoder, "")
    del encoder
    _release_gpu()

    # case_id == object_type since txt rows have no per-case suffix.
    return [(ot, ot, "", null_emb, None) for ot in cases]


def _load_test_cases_from_dataset(dataset) -> List[TestCase]:
    """Enumerate test cases from the dataset, prioritised by data-loader source.

    Three regimes, in priority order:

    1. **eval split populated by ``test_objects.txt``** (data loader saw an
       explicit object-type holdout list): per-clip enumeration, no dedup.
       Every clip in the listed object_types becomes its own test case.
       This is the default for quantitative evaluation.

    2. **eval split populated by ``test_split_ratio > 0``** (no
       ``test_objects.txt``, but a random clip-level split exists):
       per-clip enumeration, no dedup. Same as (1) but the eval-set
       composition came from the random-split logic in the dataloader.

    3. **no eval split** (``test_objects.txt`` absent and ratio == 0):
       dedup'd unique ``(object_type, caption)`` enumeration on the train
       split — a visualization sweep with no held-out set.

    Captions reuse pre-encoded embeddings — no text encoder needed.
    """
    md = dataset.motion_dataset

    if md.eval_motion_dict:
        source, split, dedup = md.eval_motion_dict, 'eval', False
        if any(v for v in md.explicit_eval_objects.values()):
            origin = "test_objects.txt (explicit object_type holdout)"
        else:
            origin = "test_split_ratio (random clip holdout)"
    else:
        source, split, dedup = md.train_motion_dict, 'train', True
        origin = "fallback — no eval split (test_objects.txt absent, ratio=0)"

    seen = set()
    encoded: List[TestCase] = []
    per_object_count: Dict[str, int] = {}
    for clip_name in sorted(source.keys()):
        entry = source[clip_name]
        if 'caption_emb' not in entry:
            continue
        ot = entry['object_type']
        caption = entry['caption']
        if dedup:
            key = (ot, caption)
            if key in seen:
                continue
            seen.add(key)
        per_object_count[ot] = per_object_count.get(ot, 0) + 1
        case_id = os.path.splitext(clip_name)[0]
        # Dataset entries carry both views already (the loader built them).
        enc = {'caption_emb': entry['caption_emb']}
        if 'caption_tokens' in entry:
            enc['caption_tokens'] = entry['caption_tokens']
        encoded.append((case_id, ot, caption, enc, clip_name))

    if not encoded:
        raise ValueError(
            "No usable test cases in dataset: no clips have a pre-encoded caption."
        )
    mode = 'unique (object_type, caption)' if dedup else 'per-clip'
    summary = ', '.join(f'{ot}={n}' for ot, n in sorted(per_object_count.items()))
    logger.info(
        f"Auto-enumerated {len(encoded)} {mode} test cases from {split} split "
        f"[source: {origin}] [{summary}]."
    )
    return encoded


def _load_expand_test_cases(
    json_path: str,
    config: MainConfig,
    dataset,
    device: torch.device,
) -> List[Tuple[str, str, List[str], List[np.ndarray]]]:
    """Motion-expand cases from {"<object_type>-<label>": [prompt, ...]} JSON.

    Each value must be a non-empty list of non-empty strings. The list
    order is the segment order: prompt[0] drives the free first segment,
    prompt[i>0] drives a follow-up segment whose first ``expand_overlap``
    frames are clamped to the previous segment's last ``expand_overlap``
    frames.

    Returns ``[(case_id, object_type, [prompts], [caption_embs])]``.
    Unlike motion-edit / in-betweening, no GT clip is pinned — only the
    skeleton from ``<object_type>`` is needed.
    """
    with open(json_path) as f:
        raw_cases = json.load(f)

    known = _known_object_types(dataset)
    cases: List[Tuple[str, str, List[str]]] = []
    for case_key, prompts in raw_cases.items():
        obj_type, _, _ = case_key.partition('-')
        if obj_type not in known:
            logger.warning(
                f"Test case {case_key!r}: object_type={obj_type!r} not in dataset; skipping."
            )
            continue
        if not isinstance(prompts, list):
            logger.warning(
                f"Test case {case_key!r}: value must be a list of prompts for "
                f"--motion_expand (got {type(prompts).__name__}); skipping."
            )
            continue
        valid = [p.strip() for p in prompts if isinstance(p, str) and p.strip()]
        if len(valid) != len(prompts):
            logger.warning(
                f"Test case {case_key!r}: {len(prompts) - len(valid)} prompt(s) "
                f"empty or non-string; keeping {len(valid)}/{len(prompts)}."
            )
        if not valid:
            logger.warning(f"Test case {case_key!r}: no usable prompts; skipping.")
            continue
        cases.append((case_key, obj_type, valid))

    if not cases:
        raise ValueError(
            f"No usable motion-expand test cases in {json_path}."
        )
    logger.info(f"Loaded {len(cases)} motion-expand test cases from {json_path}.")

    encoder = _make_text_encoder(config, device)
    encoded: List[Tuple[str, str, List[str], List[np.ndarray]]] = []
    for case_key, obj_type, prompts in cases:
        embs = [_encode_prompt(encoder, p) for p in prompts]
        encoded.append((case_key, obj_type, prompts, embs))
    del encoder
    _release_gpu()
    return encoded


def _select_test_cases(
    args: InferenceArgs,
    cfg_scale: float,
    config: MainConfig,
    dataset,
    device: torch.device,
    resolve_clip: bool = False,
) -> List[TestCase]:
    """Test-case selection at cfg > 1.0, in priority order:

    1. ``--test_cases_json`` provided
       → use the JSON's ``{object_type-id: prompt}`` map verbatim
         (curated prompts).
    2. dataset's eval split populated by ``test_objects.txt``
       → per-clip enumeration over the listed object_types (eval mode).
    3. dataset's eval split populated by ``test_split_ratio > 0``
       → per-clip enumeration over the random eval clips (eval mode).
    4. no JSON, no eval split
       → dedup'd unique ``(object_type, caption)`` prompts from the train
         split (visualization sweep).

    Cases 2/3/4 are all handled by ``_load_test_cases_from_dataset``, which
    introspects the dataloader's split state and logs which source it used.

    At cfg == 1.0 the function always returns unconditional test cases from
    ``--test_cases_txt`` (one ``object_type`` per line, null caption emb).
    """
    if cfg_scale > 1.0:
        if args.test_cases_json is not None:
            return _load_test_cases_json(
                args.test_cases_json, config, dataset, device,
                resolve_clip=resolve_clip,
            )
        return _load_test_cases_from_dataset(dataset)
    return _load_test_cases_txt(args.test_cases_txt, config, dataset, device)


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------


def _validate_cfg_inputs(cfg_scale: float, args: InferenceArgs):
    if cfg_scale < 1.0:
        raise ValueError(f"cfg_scale must be >= 1.0, got {cfg_scale}.")
    if args.inbetween and args.motion_edit:
        raise ValueError(
            "--inbetween and --motion_edit are mutually exclusive (different "
            "mask axes — frames vs. joints)."
        )
    if args.motion_expand and (args.inbetween or args.motion_edit):
        raise ValueError(
            "--motion_expand is mutually exclusive with --inbetween / --motion_edit."
        )
    if (args.inbetween or args.motion_edit) and cfg_scale <= 1.0:
        # cfg==1.0 uses --test_cases_txt (no clip pinning), so there's no
        # GT motion to clamp known frames/joints against.
        mode = '--inbetween' if args.inbetween else '--motion_edit'
        raise ValueError(
            f"{mode} requires cfg_scale > 1.0 so test cases come from "
            f"--test_cases_json or the dataset eval split (both pin a clip)."
        )
    if args.motion_expand and cfg_scale <= 1.0:
        raise ValueError(
            "--motion_expand requires cfg_scale > 1.0 and a --test_cases_json "
            "whose values are lists of prompts (one per segment)."
        )
    if args.motion_expand and args.test_cases_json is None:
        raise ValueError(
            "--motion_expand requires --test_cases_json with per-case prompt lists."
        )
    if args.motion_edit and not args.keep_joints.strip():
        raise ValueError(
            "--motion_edit requires --keep_joints (comma-separated joint names)."
        )
    if cfg_scale > 1.0:
        if args.test_cases_json is None:
            logger.info(
                f"cfg_scale={cfg_scale} > 1.0 with no --test_cases_json: "
                f"enumerating the dataset's test split (per-clip if eval split "
                f"is non-empty, else dedup-fallback on train)."
            )
        if args.test_cases_txt is not None:
            logger.warning(
                "--test_cases_txt is ignored at cfg > 1.0 — the dataset's "
                "test_objects.txt drives the eval split at data-loading time."
            )
    else:  # cfg_scale == 1.0
        if args.test_cases_txt is None:
            raise ValueError("cfg_scale=1.0 requires --test_cases_txt.")
        if args.test_cases_json is not None:
            logger.warning("--test_cases_json ignored because cfg_scale == 1.0.")


def _wrap_for_cfg(model, cfg_scale: float):
    """Pass-through at cfg > 1.0; force unconditional inference at cfg == 1.0."""
    if cfg_scale > 1.0:
        return model
    if not (getattr(model, 'cond_mask_prob', 0) > 0):
        logger.warning(
            "Unconditional sampling requested but model was trained with "
            "cond_mask_prob == 0; sample quality may be poor."
        )
    logger.info("Wrapping model in _UnconditionalWrapper (force_mask=True).")
    return _UnconditionalWrapper(model)


def _gt_valid_lengths(dataset, clip_names, gt_offset: int, max_T: int,
                      device: torch.device) -> torch.Tensor:
    """Per-clip usable GT frame counts, as a ``(B,)`` long tensor.

    Lengths come from the source motion_dict (``cond["motion_length"]``
    reports the padded length at inference, which would resolve ``-1`` to a
    zero-padded frame instead of the clip's true last frame). When the GT was
    cropped from ``gt_offset`` (``--gt_start_frame``), the usable length is
    ``n - gt_offset``; everything is clamped to ``[0, max_T]``.
    """
    md = dataset.motion_dataset
    return torch.tensor(
        [
            max(0, min(
                (md.train_motion_dict.get(cn)
                 or md.eval_motion_dict[cn])['motion'].shape[0] - gt_offset,
                max_T,
            ))
            for cn in clip_names
        ],
        dtype=torch.long, device=device,
    )


def _run_sampling(
    config: MainConfig,
    sample_model,
    dataset,
    test_cases: List[TestCase],
    diffusion,
    gen_diffusion,
    cfg_scale: float,
    device: torch.device,
    args: InferenceArgs,
    output_dir: str,
):
    """Generate ``num_repetitions`` samples per test case, in chunks of
    ``args.batch_size``. ``no_grad`` keeps the ODE rollout from accumulating
    autograd graphs across denoising steps.

    When ``args.inbetween`` is set, ``create_sample_condition`` returns the
    normalized GT motion (because ``test_case_captions`` now includes the
    pinned clip_name). That GT plus a keep_mask are passed to
    ``generate_samples`` which routes to the replacement-style sampler.
    Each case's GT is also saved as ``<case_id>-gt.npy`` for side-by-side
    rendering.
    """
    # Only pin the reference clip when in-betweening or motion-editing (the
    # clip's GT motion is what the sampler clamps against). Other runs let
    # create_sample_condition pick a random clip of the same object_type,
    # since only the skeleton is needed.
    needs_gt = args.inbetween or args.motion_edit
    if needs_gt:
        test_case_captions = [
            (ot, cap, emb, clip) for _, ot, cap, emb, clip in test_cases
        ]
    else:
        test_case_captions = [
            (ot, cap, emb) for _, ot, cap, emb, _ in test_cases
        ]
    case_ids = [case_id for case_id, *_ in test_cases]
    captions_text = [caption for _, _, caption, _, _ in test_cases]
    total = len(test_cases)
    chunk_size = max(1, args.batch_size)
    use_cuda_sync = device.type == 'cuda'

    keep_frame_indices = parse_keep_frames(args.keep_frames) if args.inbetween else None
    keep_joint_names = parse_keep_joints(args.keep_joints) if args.motion_edit else None
    if args.inbetween:
        logger.info(f"In-betweening: keep_frames={keep_frame_indices}.")
    if args.motion_edit:
        logger.info(f"Motion editing: keep_joints={keep_joint_names}.")

    # ``captions.json`` mirrors the motion-feature filenames written by
    # ``visualize_and_save_motions``: keys are ``<case_id>-rep_<rep>-<idx>.npy``,
    # values are the prompt used to condition each sample. Rewritten after
    # every chunk so a partial run still leaves a valid index on disk.
    captions_path = os.path.join(output_dir, 'captions.json')
    captions_map: Dict[str, str] = {}

    logger.info(
        f"Starting sampling: {total} test cases × {args.num_repetitions} reps "
        f"in chunks of {chunk_size}."
    )

    with torch.no_grad():
        for rep_i in range(args.num_repetitions):
            logger.info(f'--- rep #{rep_i} ---')
            for chunk_start in range(0, total, chunk_size):
                chunk_end = min(chunk_start + chunk_size, total)
                # test_case_captions drives selection here, one reference
                # clip per case, so create_sample_condition's own sampling
                # (num_samples) never applies.
                gt_motion, cond = create_sample_condition(
                    config=config,
                    data=dataset,
                    test_case_captions=test_case_captions[chunk_start:chunk_end],
                    gt_start_frame=args.gt_start_frame,
                )
                cond = {
                    k: v.to(device) if torch.is_tensor(v) else v
                    for k, v in cond.items()
                }
                bsz = cond["n_joints"].shape[0]
                motion_shape = (
                    bsz,
                    config.dataset.max_joints,
                    config.dataset.feature_len,
                    config.dataset.max_motion_length,
                )

                # Replacement-style sampling for in-betweening or motion
                # editing. Both branches share the same GT-clamping sampler
                # (only the mask differs); ``generate_samples`` routes on the
                # presence of x1_known + keep_mask.
                x1_known = None
                keep_mask = None
                gt_valid_lengths = None  # per-sample lengths for trimming saved GT
                sample_valid_lengths = None  # same, for the generated sample (motion_edit only)
                if args.inbetween:
                    chunk_clip_names = [
                        c for _, _, _, _, c in test_cases[chunk_start:chunk_end]
                    ]
                    max_T = config.dataset.max_motion_length
                    valid_lengths = _gt_valid_lengths(
                        dataset, chunk_clip_names, args.gt_start_frame or 0,
                        max_T, device,
                    )
                    keep_mask, resolved_keep = build_keep_mask(
                        valid_lengths, keep_frame_indices, max_T, device,
                        labels=chunk_clip_names,
                    )
                    x1_known = gt_motion.to(device)
                    # For clips shorter than the padded ODE window, copy the
                    # clip's actual last frame into any keep slot beyond T_i
                    # so the generation endpoint (e.g. frame 59) clamps to a
                    # real GT pose instead of the zero-padded slot.
                    for i in range(bsz):
                        T_i = int(valid_lengths[i].item())
                        if T_i >= max_T:
                            continue
                        last_gt = x1_known[i, :, :, T_i - 1].clone()
                        for idx in resolved_keep:
                            if idx >= T_i:
                                x1_known[i, :, :, idx] = last_gt
                    gt_valid_lengths = valid_lengths
                elif args.motion_edit:
                    # Per-sample joint names come from the (already-loaded)
                    # cond_dict — same source the model used at training to
                    # build joint_names_emb, so indices align.
                    chunk_object_types = cond["object_type"]
                    joint_names_per_sample = [
                        get_joint_names(dataset, ot) for ot in chunk_object_types
                    ]
                    keep_mask = build_joint_keep_mask(
                        joint_names_per_sample,
                        keep_joint_names,
                        config.dataset.max_joints,
                        device,
                    )
                    x1_known = gt_motion.to(device)
                    # Motion-edit operates on the clip's actual frames only;
                    # the model still generates 60 padded frames, but anything
                    # past T_i is meaningless (no GT to compare against), so
                    # trim both the sample and the GT to the per-clip length
                    # at save time.
                    chunk_clip_names = [
                        c for _, _, _, _, c in test_cases[chunk_start:chunk_end]
                    ]
                    max_T = config.dataset.max_motion_length
                    valid_lengths = _gt_valid_lengths(
                        dataset, chunk_clip_names, args.gt_start_frame or 0,
                        max_T, device,
                    )
                    # The joint-keep mask is (B, J, 1, 1) — broadcasts over
                    # the temporal axis — so kept joints get clamped to GT
                    # at every frame. For samples whose cropped GT is shorter
                    # than max_T (short clips, or ``--gt_start_frame`` near
                    # the clip's end), the zero-padded tail would clamp kept
                    # joints to a zero pose during denoising. Hold the last
                    # real frame instead so the constraint stays
                    # geometrically sensible; the tail is trimmed at save
                    # time via ``valid_lengths``.
                    for i in range(bsz):
                        T_i = int(valid_lengths[i].item())
                        if 0 < T_i < max_T:
                            x1_known[i, :, :, T_i:] = x1_known[i, :, :, T_i - 1:T_i]
                    gt_valid_lengths = valid_lengths
                    sample_valid_lengths = valid_lengths
                    for cn, T_i in zip(chunk_clip_names, valid_lengths.tolist()):
                        logger.info(
                            f"motion_edit: {cn} → valid_length={T_i} "
                            f"(sample + GT will be saved at this length)."
                        )

                if use_cuda_sync:
                    torch.cuda.synchronize(device)
                t_start = time.perf_counter()

                samples = generate_samples(
                    model=sample_model,
                    cond=cond,
                    motion_shape=motion_shape,
                    diff_model=config.training.diff_model,
                    diffusion=diffusion,
                    gen_diffusion=gen_diffusion,
                    device=device,
                    cfg_scale=cfg_scale,
                    x1_known=x1_known,
                    keep_mask=keep_mask,
                )

                if use_cuda_sync:
                    torch.cuda.synchronize(device)
                elapsed = time.perf_counter() - t_start
                logger.info(
                    f'rep#{rep_i} chunk[{chunk_start}:{chunk_end}] '
                    f'{elapsed:.2f}s for batch={bsz} ({elapsed / bsz:.3f}s/motion).'
                )

                # ``rep_{rep_i}`` prefixes every saved file so repetitions of
                # the same test case never collide.
                chunk_case_ids = case_ids[chunk_start:chunk_end]
                visualize_and_save_motions(
                    config=config,
                    cond=cond,
                    samples=samples,
                    save_dir=output_dir,
                    prefix=f'rep_{rep_i}',
                    case_ids=chunk_case_ids,
                    only_save_motion=args.only_save_motion,
                    save_ric=args.save_ric,
                    valid_lengths=sample_valid_lengths,
                )

                # Save GT under a per-rep "gt_rep_<i>" prefix. Per-rep rather
                # than once-only because ``apply_cropping`` picks a random
                # start_idx for ``topology_condition_type='tpos'``, so each
                # rep clamps against a different windowed GT.
                # NOTE the two files can differ in length: motion_edit trims
                # sample and GT alike (``sample_valid_lengths``), while
                # in-betweening trims only the GT — the model is asked for the
                # whole ODE window there, so its output stays max_motion_length
                # even when the reference clip is shorter. Align on frame 0
                # before diffing them.
                # ``gt_valid_lengths`` (set for inbetween and motion_edit)
                # trims the saved GT to the clip's true length rather than
                # the padded ODE window.
                if needs_gt:
                    visualize_and_save_motions(
                        config=config,
                        cond=cond,
                        samples=x1_known,
                        save_dir=output_dir,
                        prefix=f'gt_rep_{rep_i}',
                        case_ids=chunk_case_ids,
                        only_save_motion=args.only_save_motion,
                        save_ric=args.save_ric,
                        valid_lengths=gt_valid_lengths,
                    )

                for object_idx, case_id in enumerate(chunk_case_ids):
                    npy_name = f'{case_id}-rep_{rep_i}-{object_idx}.npy'
                    captions_map[npy_name] = captions_text[chunk_start + object_idx]
                with open(captions_path, 'w') as f:
                    json.dump(captions_map, f, indent=2, ensure_ascii=False)

                # Drop chunk-scoped tensors before the next batch so peak GPU
                # memory tracks per-batch, not per-run, usage.
                del cond, samples
                if x1_known is not None:
                    del x1_known
                _release_gpu()


def _run_expansion_sampling(
    config: MainConfig,
    sample_model,
    dataset,
    expand_cases: List[Tuple[str, str, List[str], List[np.ndarray]]],
    diffusion,
    gen_diffusion,
    cfg_scale: float,
    device: torch.device,
    args: InferenceArgs,
    output_dir: str,
):
    """Per-case chain generation: each case produces one concatenated motion
    of length ``max_T + (max_T - overlap) * (N - 1)``.

    Each case is processed independently (no cross-case batching) because
    prompt-list lengths can differ. Within a case, the cond dict is set up
    once via ``create_sample_condition`` and the per-segment cond is built
    by swapping in the segment's pre-encoded ``caption_emb`` — ``caption_emb``
    is a fixed-shape ``(B, text_dim)`` tensor (see ``mixture_batch_collate``)
    so the swap is a single tensor assignment.
    """
    overlap = args.expand_overlap
    max_T = config.dataset.max_motion_length
    if overlap <= 0 or overlap >= max_T:
        raise ValueError(
            f"--expand_overlap must be in (0, max_motion_length={max_T}), "
            f"got {overlap}."
        )

    captions_path = os.path.join(output_dir, 'captions.json')
    captions_map: Dict[str, str] = {}

    logger.info(
        f"Starting motion-expand sampling: {len(expand_cases)} case(s) × "
        f"{args.num_repetitions} rep(s); overlap={overlap}, max_T={max_T}."
    )

    motion_shape = (
        1,
        config.dataset.max_joints,
        config.dataset.feature_len,
        max_T,
    )

    with torch.no_grad():
        for rep_i in range(args.num_repetitions):
            logger.info(f'--- rep #{rep_i} ---')
            for case_idx, (case_id, obj_type, prompts, embs) in enumerate(expand_cases):
                # Bootstrap the skeleton cond using the first prompt; the
                # caption is overwritten per segment below.
                _, cond = create_sample_condition(
                    config=config,
                    data=dataset,
                    test_case_captions=[(obj_type, prompts[0], embs[0])],
                )
                cond = {
                    k: v.to(device) if torch.is_tensor(v) else v
                    for k, v in cond.items()
                }
                if 'caption_emb' not in cond:
                    raise RuntimeError(
                        "caption_emb missing from cond — --motion_expand requires "
                        "a text-conditioned model (cond_mode='text')."
                    )

                # Per-segment cond dicts share every skeleton field; only the
                # caption and its encodings change. Shallow copy is enough —
                # the tensor reassignment doesn't leak across segments.
                cond_per_segment: List[Dict] = []
                dtype = cond['caption_emb'].dtype
                for prompt, enc in zip(prompts, embs):
                    seg_cond = dict(cond)
                    seg_cond['caption_emb'] = torch.from_numpy(
                        enc['caption_emb']).to(device=device, dtype=dtype).unsqueeze(0)
                    if 'caption_tokens' in enc:
                        toks = torch.from_numpy(enc['caption_tokens']).to(
                            device=device, dtype=dtype).unsqueeze(0)   # (1, T, D)
                        seg_cond['caption_tokens'] = toks
                        seg_cond['caption_mask'] = torch.ones(
                            toks.shape[:2], dtype=torch.bool, device=device)
                    seg_cond['caption'] = [prompt]
                    cond_per_segment.append(seg_cond)

                t_start = time.perf_counter()
                chain = expand_motion_chain(
                    cond_per_segment=cond_per_segment,
                    motion_shape=motion_shape,
                    overlap=overlap,
                    sample_model=sample_model,
                    diff_model=config.training.diff_model,
                    diffusion=diffusion,
                    gen_diffusion=gen_diffusion,
                    cfg_scale=cfg_scale,
                    device=device,
                )
                elapsed = time.perf_counter() - t_start
                logger.info(
                    f'rep#{rep_i} case[{case_idx}] {case_id!r}: '
                    f'{len(prompts)} segments → T_total={chain.shape[-1]} '
                    f'({elapsed:.2f}s).'
                )

                # Stitch the prompts into a single caption for the saved
                # video title; the model never sees this — segment captions
                # were used individually during generation.
                joined_caption = " | ".join(prompts)
                viz_cond = dict(cond)
                viz_cond['caption'] = [joined_caption]

                visualize_and_save_motions(
                    config=config,
                    cond=viz_cond,
                    samples=chain,
                    save_dir=output_dir,
                    prefix=f'rep_{rep_i}',
                    case_ids=[case_id],
                    only_save_motion=args.only_save_motion,
                    save_ric=args.save_ric,
                )

                npy_name = f'{case_id}-rep_{rep_i}-0.npy'
                captions_map[npy_name] = joined_caption
                with open(captions_path, 'w') as f:
                    json.dump(captions_map, f, indent=2, ensure_ascii=False)

                del cond, cond_per_segment, chain
                _release_gpu()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(args: InferenceArgs):
    # Read config first so CLI-omitted fields can fall back to the saved
    # ``sampling`` block (mirrors the cfg_scale fallback below).
    config_path = os.path.join(args.exp_dir, "config.json")
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"No config.json in {args.exp_dir!r}")
    config = MainConfig.from_json(config_path)

    model_path_arg = args.model_path or config.sampling.model_path
    config_path, model_path = _resolve_exp_paths(args.exp_dir, model_path_arg)
    logger.info(f"Using config:     [{config_path}]")
    logger.info(f"Using checkpoint: [{model_path}]")

    cfg_scale = args.cfg_scale if args.cfg_scale is not None else config.sampling.cfg_scale
    _validate_cfg_inputs(cfg_scale, args)

    base_output = args.output_dir or os.path.join(args.exp_dir, "samples")
    # Inbetween / motion-edit / motion-expand write to a sibling subdir so
    # GT/output pairs don't collide with vanilla sampling artifacts from
    # the same exp_dir.
    if args.inbetween:
        output_dir = os.path.join(base_output, "inbetween")
    elif args.motion_edit:
        output_dir = os.path.join(base_output, "motion_edit")
    elif args.motion_expand:
        output_dir = os.path.join(base_output, "motion_expand")
    else:
        output_dir = base_output
    os.makedirs(output_dir, exist_ok=True)

    if args.inbetween:
        keep_indices = parse_keep_frames(args.keep_frames)
        ledger_path = os.path.join(output_dir, "inbetween_keep.json")
        with open(ledger_path, "w") as f:
            json.dump({"keep_frames": keep_indices}, f, indent=2)
    elif args.motion_edit:
        # Record the user-supplied names; the actual per-skeleton matches
        # (and any misses) are logged by build_joint_keep_mask at run time.
        keep_names = parse_keep_joints(args.keep_joints)
        ledger_path = os.path.join(output_dir, "motion_edit_keep.json")
        with open(ledger_path, "w") as f:
            json.dump({"keep_joints": keep_names}, f, indent=2)
    elif args.motion_expand:
        ledger_path = os.path.join(output_dir, "motion_expand.json")
        with open(ledger_path, "w") as f:
            json.dump({"expand_overlap": args.expand_overlap}, f, indent=2)

    if args.seed is not None:
        set_seed(args.seed)
        logger.info(f"Set random seed to [{args.seed}]")

    target_object_types, target_clip_stems = _resolve_inference_scope(args)
    stats_path = _resolve_stats_path(args.exp_dir)

    dataset = create_dataset(
        dataset_config=config.dataset,
        model_config=config.model,
        inference=True,
        target_object_types=target_object_types,
        target_clip_stems=target_clip_stems,
        stats_path=stats_path,
    )

    logger.info("Creating model and diffusion...")
    model = create_model(
        dataset_config=config.dataset,
        model_config=config.model,
    )
    diffusion, gen_diffusion = _build_diffusion(config)

    logger.info(f"Loading checkpoints from [{model_path}]...")
    _load_checkpoint(model, model_path, config)

    device = torch.device(config.sampling.device)

    # Encode test cases before moving the diffusion model to device, so the
    # text encoder and diffusion model don't coexist on GPU. Motion-expand
    # has a distinct test-case structure (per-case prompt lists), so it
    # uses its own loader + runner.
    if args.motion_expand:
        expand_cases = _load_expand_test_cases(
            args.test_cases_json, config, dataset, device,
        )
        test_cases = None
    else:
        expand_cases = None
        test_cases = _select_test_cases(
            args, cfg_scale, config, dataset, device,
            resolve_clip=args.inbetween or args.motion_edit,
        )

    model.to(device)
    model.eval()

    logger.info(f"Using cfg_scale={cfg_scale}")
    sample_model = _wrap_for_cfg(model, cfg_scale)

    if args.motion_expand:
        _run_expansion_sampling(
            config=config,
            sample_model=sample_model,
            dataset=dataset,
            expand_cases=expand_cases,
            diffusion=diffusion,
            gen_diffusion=gen_diffusion,
            cfg_scale=cfg_scale,
            device=device,
            args=args,
            output_dir=output_dir,
        )
    else:
        _run_sampling(
            config=config,
            sample_model=sample_model,
            dataset=dataset,
            test_cases=test_cases,
            diffusion=diffusion,
            gen_diffusion=gen_diffusion,
            cfg_scale=cfg_scale,
            device=device,
            args=args,
            output_dir=output_dir,
        )


if __name__ == "__main__":
    main(tyro.cli(InferenceArgs))
