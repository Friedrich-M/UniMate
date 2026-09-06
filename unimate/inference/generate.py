"""Sampling utilities for diffusion / flow-matching generation."""

import torch
import torch.nn as nn

from unimate.utils.logger import get_logger

logger = get_logger(file_name=__file__)


class ClassifierFreeSampleModel(nn.Module):
    """Wrapper that applies classifier-free guidance during sampling.

    Runs the model twice (conditional + unconditional) and interpolates
    outputs using the guidance scale: out = out_uncond + scale * (out_cond - out_uncond).

    Reference: Ho & Salimans, "Classifier-Free Diffusion Guidance", 2022.
    https://arxiv.org/abs/2207.12598

    Args:
        model: Diffusion model (must have cond_mask_prob > 0 from training).
        cfg_scale: Classifier-free guidance scale (1.0 = no guidance).
    """

    def __init__(self, model, cfg_scale=1.0):
        super().__init__()
        self.model = model
        self.cfg_scale = cfg_scale

        assert self.model.cond_mask_prob > 0, \
            'Cannot run guided diffusion on a model not trained with conditional dropout'

    def forward(self, x, timesteps, cond=None):
        out = self.model(x, timesteps, cond)
        out_uncond = self.model(x, timesteps, cond, force_mask=True)

        return out_uncond + (self.cfg_scale * (out - out_uncond))


def generate_samples(
    model,
    cond,
    motion_shape,
    diff_model,
    diffusion,
    gen_diffusion=None,
    device=None,
    cfg_scale=1.0,
    x1_known=None,
    keep_mask=None,
):
    """Run the generative model to produce motion samples.

    Shared between training-time visualization and standalone inference.

    Args:
        model: The denoising network (unwrapped, in eval mode).
        cond: Conditioning dict (tensors already on device).
        motion_shape: Tuple (B, J, D, T) for the output motion.
        diff_model: ``"flow"`` or ``"diffusion"``.
        diffusion: The diffusion schedule (Gaussian DDPM or flow transport).
        gen_diffusion: ``Sampler`` wrapper for flow transport (required when
            ``diff_model == "flow"``).
        device: Target device for noise tensor. Inferred from model if None.
        cfg_scale: Classifier-free guidance scale (1.0 = no guidance).
        x1_known: optional ``(B, J, D, T)`` normalized clean motion for
            in-betweening. When set, ``keep_mask`` must also be set, and
            ``diff_model`` must be ``"flow"``.
        keep_mask: optional ``(B, 1, 1, T)`` bool mask — True for frames
            held fixed at ``x1_known`` during sampling.

    Returns:
        samples: Tensor of shape ``motion_shape``.
    """
    if device is None:
        device = next(model.parameters()).device

    # Optional classifier-free guidance wrapper. The cfg_scale is logged once
    # by the caller (sample.py); skip a per-chunk log here to avoid flooding
    # the output during multi-rep / multi-chunk runs.
    if cfg_scale > 1.0:
        sample_model = ClassifierFreeSampleModel(model, cfg_scale=cfg_scale)
    else:
        sample_model = model

    inbetween = x1_known is not None or keep_mask is not None
    if inbetween:
        if x1_known is None or keep_mask is None:
            raise ValueError("x1_known and keep_mask must both be set for in-betweening.")
        if diff_model != "flow":
            raise NotImplementedError(
                "In-betweening currently supports diff_model='flow' only."
            )
        # Imported lazily so non-inbetween code paths don't pull torchdiffeq
        # dependencies twice or fail on import errors here.
        from unimate.inference.motion_inbetweening import inbetween_sample_ode
        assert gen_diffusion is not None, "gen_diffusion (Sampler) required for flow sampling"
        return inbetween_sample_ode(
            sample_model=sample_model,
            transport=gen_diffusion.transport,
            cond=cond,
            x1_known=x1_known,
            keep_mask=keep_mask,
            motion_shape=motion_shape,
            device=device,
        )

    model_kwargs = dict(cond=cond)
    noise = torch.randn(motion_shape, device=device)

    if diff_model == "flow":
        assert gen_diffusion is not None, "gen_diffusion (Sampler) required for flow sampling"
        model_fn = gen_diffusion.sample_ode()
        samples = model_fn(noise, sample_model, **model_kwargs)[-1]
    elif diff_model == "diffusion":
        samples = diffusion.p_sample_loop(
            model=sample_model,
            shape=motion_shape,
            noise=noise,
            clip_denoised=False,
            model_kwargs=model_kwargs,
            progress=False,
        )
    else:
        raise ValueError(f"Unknown diff_model: {diff_model!r}")

    return samples
