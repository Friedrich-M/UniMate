"""Text-guided motion editing via replacement-style flow-matching inference.

Given a clean ground-truth motion ``x1_known`` and a list of joint names to
hold fixed, clamp those joints to GT for all frames while letting the flow
ODE denoise the remaining joints under a new text prompt. Example: fix the
lower body of a walking clip and ask the model to add an upper-body action
("walking while waving").

Mask shape is ``(B, J, 1, 1)`` — broadcasts uniformly over feature dim D and
time T, since whole joints are kept across the entire clip. The underlying
ODE sampler (``inbetween_sample_ode``) is shape-agnostic, so the replacement
math is the same as in-betweening; only the mask differs.

Joint names are resolved per-case against
``motion_dataset.cond_dict[object_type]``, matching **either** vocabulary:
the rig's own bone names (``Bip01_L_Thigh``) or the cleaned ones stage 3
produces (``Left Thigh``). Cleaning is a per-element string transform that
preserves joint order, so an index found through either list points at the
same joint row the model was conditioned on via ``joint_names_emb`` (which
is built from the cleaned names — see ``dataset.model_joint_names``).
"""

from typing import List, Sequence

import torch

from unimate.utils.logger import get_logger

logger = get_logger(file_name=__file__)


# ---------------------------------------------------------------------------
# Keep-joint spec parsing
# ---------------------------------------------------------------------------


def parse_keep_joints(spec: str) -> List[str]:
    """Parse a comma-separated list of joint name strings.

    Whitespace around each name is stripped; empty tokens are skipped so
    ``"Hips, LeftUpLeg ,LeftLeg"`` and ``"Hips,LeftUpLeg,LeftLeg"`` are
    equivalent. Resolution against per-skeleton joint lists is deferred to
    :func:`build_joint_keep_mask`.
    """
    if not spec.strip():
        raise ValueError("keep_joints spec is empty.")
    out: List[str] = []
    for tok in spec.split(','):
        tok = tok.strip()
        if tok:
            out.append(tok)
    if not out:
        raise ValueError(f"keep_joints spec parsed to empty list: {spec!r}.")
    return out


# ---------------------------------------------------------------------------
# Joint-name lookup
# ---------------------------------------------------------------------------


def get_joint_names(dataset, object_type: str) -> List[str]:
    """Return per-joint name *aliases* for ``object_type``, in model-index order.

    Each entry is ``"<raw>|<clean>"`` when the two differ, so ``--keep_joints``
    matches either vocabulary: the source rig's own names (``Bip01_Pelvis``,
    ``Wing1.L``), which is what someone looking at the asset knows, and the
    canonicalized ones this repo produces in stage 3 (``Hips``), which is what
    the T-pose visualizations, the captions and the model's own
    ``joint_names_emb`` all use. Matching only the raw list made every cleaned
    name silently miss.

    Safe to conflate because cleaning is a per-element string transform: it
    preserves joint *order*, so index ``j`` is the same joint row either way.
    """
    md = dataset.motion_dataset
    cond_entry = md.cond_dict[object_type]
    raw = cond_entry.get('joint_names')
    clean = cond_entry.get('clean_joint_names')
    if (raw is None or len(raw) == 0) and (clean is None or len(clean) == 0):
        raise KeyError(
            f"cond_dict[{object_type!r}] has neither 'joint_names' nor "
            f"'clean_joint_names'."
        )
    # ``cond_entry`` values are sometimes numpy arrays of strings; coerce so
    # downstream membership checks behave like plain Python strings.
    raw = [str(n) for n in (raw if raw is not None else [])]
    clean = [str(n) for n in (clean if clean is not None else [])]
    if not raw:
        return clean
    if len(clean) != len(raw):
        return raw
    return [r if r == c else f'{r}|{c}' for r, c in zip(raw, clean)]


# ---------------------------------------------------------------------------
# Spatial keep-mask
# ---------------------------------------------------------------------------


def build_joint_keep_mask(
    joint_names_per_sample: Sequence[Sequence[str]],
    keep_names: List[str],
    max_joints: int,
    device: torch.device,
) -> torch.Tensor:
    """Build a ``(B, J, 1, 1)`` bool mask: True for joints whose name appears
    in ``keep_names`` (case-insensitive). Padding joints past the per-sample
    joint count are left False.

    A joint entry may carry several ``|``-separated aliases (see
    :func:`get_joint_names`); matching any one of them selects the joint.

    Logs a warning per sample for any ``keep_names`` entries that don't
    match a joint in that skeleton — a single keep list is typically applied
    across heterogeneous skeletons (different topology, different naming),
    so partial matches are expected and shouldn't fail the run.

    Args:
        joint_names_per_sample: per-sample lists of joint name strings,
            ordered to match the model's joint-index axis.
        keep_names: user-supplied target joint names.
        max_joints: padded joint-axis size (``cond["joint_mask"]`` width).
        device: target device.
    """
    keep_lower = {n.lower() for n in keep_names}
    B = len(joint_names_per_sample)
    mask = torch.zeros((B, max_joints, 1, 1), dtype=torch.bool, device=device)

    total_hits = 0
    for i, names in enumerate(joint_names_per_sample):
        if len(names) > max_joints:
            # Should not happen — ``max_joints`` is derived from the loaded
            # skeletons — but be defensive: truncating silently here would
            # clamp the wrong indices.
            raise ValueError(
                f"Sample {i}: skeleton has {len(names)} joints > max_joints={max_joints}."
            )
        hit_lower = set()
        for j, nm in enumerate(names):
            aliases = {a.strip().lower() for a in str(nm).split('|') if a.strip()}
            matched = aliases & keep_lower
            if matched:
                mask[i, j, 0, 0] = True
                hit_lower |= matched
                total_hits += 1
        misses = sorted(keep_lower - hit_lower)
        if misses:
            logger.warning(
                f"Sample {i} ({len(names)} joints): {len(misses)} keep_joints "
                f"unmatched — {misses}"
            )

    if total_hits == 0:
        raise ValueError(
            f"build_joint_keep_mask: no keep_joints matched any skeleton in "
            f"the batch (keep_names={keep_names}). Check spelling against the "
            f"clip's joint_names or clean_joint_names."
        )
    return mask
