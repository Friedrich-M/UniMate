"""Motion expansion: chain text-conditioned generations into a long motion.

Given a list of N text prompts and a target skeleton, generate:

  * ``seg_1``: free generation under ``prompt_1`` (no GT clamp).
  * ``seg_i`` for ``i in [2, N]``: replacement-style sampling under
    ``prompt_i`` with ``x1_known[:, :, :, :overlap] = seg_{i-1}[:, :, :, -overlap:]``
    and a ``(B, 1, 1, T)`` keep_mask True for the first ``overlap`` frames.

The concatenated output stitches segments at the overlap seam:

  ``chain = [seg_1, seg_2[overlap:], seg_3[overlap:], ..., seg_N[overlap:]]``

so the seam frames appear once and motion continuity is enforced by the
flow ODE's clamping (same machinery as in-betweening — only the mask
construction differs).

Total length: ``max_T + (max_T - overlap) * (N - 1)``.
"""

from typing import List

import torch

from unimate.utils.logger import get_logger
from unimate.inference.generate import generate_samples

logger = get_logger(file_name=__file__)


# ---------------------------------------------------------------------------
# Mask + seed helpers
# ---------------------------------------------------------------------------


def build_overlap_keep_mask(
    batch_size: int, overlap: int, max_T: int, device: torch.device,
) -> torch.Tensor:
    """``(B, 1, 1, T)`` bool mask: True for the first ``overlap`` frames.

    Broadcasts uniformly over the joint and feature axes — same shape
    convention as :func:`unimate.inference.motion_inbetweening.build_keep_mask`, so the
    shared ``inbetween_sample_ode`` clamps without per-axis logic.
    """
    mask = torch.zeros((batch_size, 1, 1, max_T), dtype=torch.bool, device=device)
    mask[:, :, :, :overlap] = True
    return mask


def seed_expansion_x1(
    prev_seq: torch.Tensor, overlap: int, motion_shape, device: torch.device,
) -> torch.Tensor:
    """Build ``x1_known`` for the next segment.

    Frames ``[0, overlap)`` are copied from ``prev_seq[..., -overlap:]``;
    the remaining slots are zero. Only the kept slice is read by the
    sampler, so the zero tail is inert.
    """
    x1 = torch.zeros(motion_shape, device=device)
    x1[:, :, :, :overlap] = prev_seq[:, :, :, -overlap:]
    return x1


# ---------------------------------------------------------------------------
# Chain generation
# ---------------------------------------------------------------------------


def expand_motion_chain(
    cond_per_segment: List[dict],
    motion_shape,
    overlap: int,
    sample_model,
    diff_model: str,
    diffusion,
    gen_diffusion,
    cfg_scale: float,
    device: torch.device,
) -> torch.Tensor:
    """Generate a chain of motion segments and concatenate at the seam.

    Args:
        cond_per_segment: per-segment conditioning dicts (already on
            device). All entries must share the same skeleton fields
            (``tpos_first_frame``, ``parents``, ``mean``/``std``, etc.) —
            only ``caption_emb`` (and ``caption`` for logging) should
            differ across entries.
        motion_shape: ``(B, J, D, max_T)`` for each segment generation.
        overlap: number of frames pinned between consecutive segments.
            Must be in ``(0, max_T)``.
        sample_model: denoising network. CFG-wrapping handled inside
            ``generate_samples`` based on ``cfg_scale``.
        diff_model, diffusion, gen_diffusion, cfg_scale, device: forwarded
            to ``generate_samples``.

    Returns:
        ``(B, J, D, T_total)`` concatenated chain, where
        ``T_total = max_T + (max_T - overlap) * (N - 1)``.
    """
    max_T = motion_shape[3]
    if overlap <= 0 or overlap >= max_T:
        raise ValueError(
            f"overlap must be in (0, max_T={max_T}), got {overlap}."
        )
    if not cond_per_segment:
        raise ValueError("cond_per_segment is empty — need at least one prompt.")

    segments: List[torch.Tensor] = []
    prev_seq = None
    for i, cond in enumerate(cond_per_segment):
        if prev_seq is None:
            x1_known = None
            keep_mask = None
        else:
            x1_known = seed_expansion_x1(prev_seq, overlap, motion_shape, device)
            keep_mask = build_overlap_keep_mask(
                motion_shape[0], overlap, max_T, device,
            )

        sample = generate_samples(
            model=sample_model,
            cond=cond,
            motion_shape=motion_shape,
            diff_model=diff_model,
            diffusion=diffusion,
            gen_diffusion=gen_diffusion,
            device=device,
            cfg_scale=cfg_scale,
            x1_known=x1_known,
            keep_mask=keep_mask,
        )
        segments.append(sample)
        prev_seq = sample
        logger.info(
            f"motion_expansion: segment {i + 1}/{len(cond_per_segment)} done "
            f"(T={sample.shape[-1]})."
        )

    parts = [segments[0]]
    for seg in segments[1:]:
        parts.append(seg[:, :, :, overlap:])
    chain = torch.cat(parts, dim=-1)
    logger.info(
        f"motion_expand: chained {len(segments)} segments → total T={chain.shape[-1]}."
    )
    return chain
