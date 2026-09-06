"""Motion in-betweening via replacement-style flow-matching inference.

Given a clean ground-truth motion ``x1_known`` and a temporal ``keep_mask``
marking which frames to hold fixed, run the flow ODE on the remaining frames
while pinning the kept slice to the analytical interpolant
``(1 - t) * eps + t * x1_known`` at every step. With a linear interpolant
(``path.ICPlan``) and a velocity-prediction model this is exact: at ``t = 1``
the kept frames equal ``x1_known``.

Mirrors RePaint (Lugmayr et al. 2022) for diffusion, but for flow matching
the analytic interpolant removes the need for forward/back resampling loops —
one pass of fixed-step Euler with replacement is sufficient.

Only supports ``diff_model == "flow"`` with ``ModelType.VELOCITY`` and
``PathType.LINEAR`` — i.e. every current config in this repo. Diffusion-model
in-betweening would need a separate RePaint variant.
"""

from typing import List, Optional, Sequence, Tuple

import torch

from unimate.utils.logger import get_logger

logger = get_logger(file_name=__file__)


# ---------------------------------------------------------------------------
# Keep-frame spec parsing
# ---------------------------------------------------------------------------


def parse_keep_frames(spec: str) -> List[int]:
    """Parse a comma-separated list of signed integer indices.

    Negative indices are kept as-is here and resolved against the per-clip
    valid frame count in :func:`build_keep_mask`. Duplicates are de-duplicated
    after resolution, not at parse time (a user listing ``0,-1`` for a 1-frame
    clip should still produce a single kept index, not an error).

    Examples:
        ``"0,-1"``        → ``[0, -1]``        (first + last frame)
        ``"0,1,-2,-1"``   → ``[0, 1, -2, -1]`` (first two + last two)
    """
    if not spec.strip():
        raise ValueError("keep_frames spec is empty.")
    out: List[int] = []
    for tok in spec.split(','):
        tok = tok.strip()
        if not tok:
            continue
        try:
            out.append(int(tok))
        except ValueError as e:
            raise ValueError(
                f"keep_frames: cannot parse {tok!r} as integer ({spec!r})."
            ) from e
    if not out:
        raise ValueError(f"keep_frames spec parsed to empty list: {spec!r}.")
    return out


def build_keep_mask(
    motion_lengths: torch.Tensor,
    keep_indices: List[int],
    max_T: int,
    device: torch.device,
    labels: Optional[Sequence[str]] = None,
) -> Tuple[torch.Tensor, List[int]]:
    """Build a ``(B, 1, 1, T)`` boolean keep-mask plus the resolved indices.

    Indices resolve against ``max_T`` (so ``-1`` is always the last
    generation slot, e.g. frame 59 when ``max_T = 60``) and clamp to
    ``[0, max_T - 1]``. The mask is identical across the batch — the only
    per-sample concern is that some clips are shorter than ``max_T`` and
    need their ``x1_known`` patched so the kept slot ≥ ``T_i`` holds the
    clip's last real frame; that patch lives in the caller (sample.py).

    Args:
        motion_lengths: ``(B,)`` long tensor of per-sample valid frame counts.
            Only used here for logging — the resolution itself uses ``max_T``.
        keep_indices: signed indices; negatives count from ``max_T``.
        max_T: padded temporal length (``cond["lengths_mask"]`` width).
        device: target device.
        labels: optional per-sample labels (clip names) for logging.

    Returns:
        ``(mask, resolved_indices)``. ``resolved_indices`` is the sorted list
        of absolute positions to clamp; callers use it to fix up ``x1_known``
        for short clips.
    """
    B = motion_lengths.shape[0]
    resolved = sorted({
        max(0, min(k if k >= 0 else max_T + k, max_T - 1))
        for k in keep_indices
    })
    mask = torch.zeros((B, 1, 1, max_T), dtype=torch.bool, device=device)
    for idx in resolved:
        mask[:, 0, 0, idx] = True

    for i in range(B):
        T_i = int(motion_lengths[i].item())
        out_of_gt = [idx for idx in resolved if idx >= T_i]
        label = labels[i] if labels is not None else f"sample[{i}]"
        note = (f" (slots {out_of_gt} ≥ T_i clamp to GT's last frame)"
                if out_of_gt else "")
        logger.info(
            f"build_keep_mask: {label} → valid_length={T_i}, "
            f"keep_indices={resolved}{note}"
        )
    return mask, resolved


# ---------------------------------------------------------------------------
# Replacement-style ODE sampler
# ---------------------------------------------------------------------------


def inbetween_sample_ode(
    sample_model,
    transport,
    cond: dict,
    x1_known: torch.Tensor,
    keep_mask: torch.Tensor,
    motion_shape,
    *,
    num_steps: int = 50,
    device: torch.device,
) -> torch.Tensor:
    """Fixed-step Euler ODE with per-step replacement of the kept slice.

    Mask-shape-agnostic: the sampler clamps whatever ``keep_mask`` selects,
    so the same routine handles temporal in-betweening (``(B, 1, 1, T)``
    mask), spatial motion editing (``(B, J, 1, 1)`` mask), or any
    combination as long as the mask broadcasts to ``(B, J, D, T)``.

    Args:
        sample_model: denoising network (already CFG-wrapped if cfg > 1).
        transport: the ``Transport`` instance from the ``Sampler`` (needed
            for the integration interval ``[t0, t1]``).
        cond: conditioning dict (tensors already on device). Forwarded to
            ``sample_model`` as the ``cond=`` keyword at every step.
        x1_known: ``(B, J, D, T)`` normalized ground-truth motion. Values
            outside ``keep_mask`` are ignored.
        keep_mask: bool tensor broadcastable to ``(B, J, D, T)``; True where
            the GT is pinned. Common shapes:
              * ``(B, 1, 1, T)`` — temporal in-betweening (keep frames).
              * ``(B, J, 1, 1)`` — spatial motion editing (keep joints).
        motion_shape: ``(B, J, D, T)`` — output shape (used to sample noise).
        num_steps: number of Euler steps. Replacement needs a fixed-step
            solver — the kept slice has to be re-pinned at each known ``t``,
            which an adaptive solver's internal stages do not expose — so
            this path does NOT use ``Sampler.sample_ode``'s default
            ``dopri5``. The three application modes therefore integrate the
            same ODE with a different (lower-order, fixed) discretization
            than plain text-conditioned sampling does.
        device: target device.

    Returns:
        ``(B, J, D, T)`` final sample. Positions selected by ``keep_mask``
        equal ``x1_known`` (modulo the epsilon end-stop).
    """
    t0, t1 = transport.check_interval(
        transport.train_eps,
        transport.sample_eps,
        sde=False, eval=True, reverse=False, last_step_size=0.0,
    )

    # Fixed noise — both the initial state and the per-step interpolant
    # reuse the same eps so kept frames trace the exact training-time path
    # x_t = (1-t) eps + t x_1.
    eps = torch.randn(motion_shape, device=device)
    x = eps.clone()

    # Mask must broadcast to (B, J, D, T). The replacement math is identical
    # whichever axes the mask collapses on — see inbetween (keep frames) vs.
    # motion editing (keep joints) for the two common cases.
    mask = keep_mask

    def _replace(x_t, t_scalar):
        t_blend = (1.0 - t_scalar) * eps + t_scalar * x1_known
        return torch.where(mask, t_blend, x_t)

    # Initial state: pin the kept frames to their t=t0 interpolant value.
    x = _replace(x, t0)

    ts = torch.linspace(t0, t1, num_steps + 1, device=device)

    for i in range(num_steps):
        t_cur = ts[i]
        t_next = ts[i + 1]
        dt = t_next - t_cur

        t_batch = torch.full((x.size(0),), t_cur.item(), device=device, dtype=x.dtype)
        v = sample_model(x, t_batch, cond=cond)
        x = x + dt * v
        x = _replace(x, t_next)

    return x
