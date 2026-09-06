"""Factories for the denoising model and its generation schedule.

``create_model``     — builds the denoiser variant selected by the two
                       orthogonal axes ``model_config.attention``
                       (``'full'`` / ``'graph'``) and ``model_config.text_cond``
                       (``'adaln'`` / ``'cross_attn'``).
``create_diffusion`` — Gaussian DDPM schedule (``training.diff_model == 'diffusion'``).
``create_transport`` — flow-matching transport (``training.diff_model == 'flow'``,
                       used by all current configs).
"""

from unimate.models.denoiser import (UniMateFullAdaLN, UniMateFullCrossAttn,
                                     UniMateGraphAdaLN, UniMateGraphCrossAttn)
from unimate.models.diffusion import gaussian_diffusion as gd
from unimate.models.diffusion.respace import SpacedDiffusion, space_timesteps
from unimate.models.flow.transport import Transport, ModelType, WeightType, PathType
from unimate.models.text_encoder.factory import get_text_encoder_dim


# (attention, text_cond) -> model class. All four pairs are implemented; the
# two axes are genuinely independent.
_MODEL_BY_AXES = {
    ('full', 'adaln'): UniMateFullAdaLN,
    ('full', 'cross_attn'): UniMateFullCrossAttn,
    ('graph', 'adaln'): UniMateGraphAdaLN,
    ('graph', 'cross_attn'): UniMateGraphCrossAttn,
}


def create_model(dataset_config, model_config):
    """Build the denoiser for a config.

    Selected by ``attention`` ('full' | 'graph') x ``text_cond``
    ('adaln' | 'cross_attn'); see :data:`_MODEL_BY_AXES`.
    """
    axes = (model_config.attention, model_config.text_cond)
    model_cls = _MODEL_BY_AXES.get(axes)
    if model_cls is None:
        raise ValueError(
            f"Unsupported model axes: attention={model_config.attention!r}, "
            f"text_cond={model_config.text_cond!r}. Available: "
            + ', '.join(f'{a}+{t}' for a, t in sorted(_MODEL_BY_AXES)))

    text_dim = get_text_encoder_dim(
        model_config.text_encoder_type,
        model_config.text_encoder_version,
    )

    # Shared kwargs for all variants, grouped by purpose.
    model_kwargs = dict(
        # Dataset-derived shape
        feature_len=dataset_config.feature_len,
        max_motion_length=dataset_config.max_motion_length,
        max_joints=dataset_config.max_joints,
        max_depth=dataset_config.max_depth,
        # Architecture
        latent_dim=model_config.latent_dim,
        ff_size=model_config.ff_size,
        num_layers=model_config.num_layers,
        num_heads=model_config.num_heads,
        dropout=model_config.dropout,
        # Positional encoding
        use_spectral_rope=model_config.use_spectral_rope,
        max_freqs=model_config.max_freqs,
        use_signnet=model_config.use_signnet,
        # Conditioning
        cond_mode=model_config.cond_mode,
        cond_mask_prob=model_config.cond_mask_prob,
        text_dim=text_dim,
        # Embeddings
        use_joint_name_emb=model_config.use_joint_name_emb,
        use_graph_emb=model_config.use_graph_emb,
        use_depth_emb=model_config.use_depth_emb,
        concat_parent_features=model_config.concat_parent_features,
        num_tpos_queries=model_config.num_tpos_queries,
        inject_tpos_to_adaln=model_config.inject_tpos_to_adaln,
    )

    # Knobs only the factored (attention='graph') variants accept. The
    # cross_attn one always uses the graph bias, so it takes the sharing
    # switch but not the on/off ablation.
    if model_cls is UniMateGraphAdaLN:
        model_kwargs.update(
            use_graph_attn_bias=model_config.use_graph_attn_bias,
            share_graph_attn_bias=model_config.share_graph_attn_bias,
            gradient_checkpointing=model_config.gradient_checkpointing,
        )
    elif model_cls is UniMateGraphCrossAttn:
        model_kwargs.update(
            share_graph_attn_bias=model_config.share_graph_attn_bias,
            gradient_checkpointing=model_config.gradient_checkpointing,
        )

    return model_cls(**model_kwargs)


def create_diffusion(scheduler_config, training_config):
    betas = gd.get_named_beta_schedule(
        scheduler_config.noise_schedule,
        scheduler_config.diffusion_steps,
        scheduler_config.scale_beta
    )
    loss_type = gd.LossType.MSE
    timestep_respacing = scheduler_config.timestep_respacing or [scheduler_config.diffusion_steps]

    diffusion = SpacedDiffusion(
        use_timesteps=space_timesteps(
            scheduler_config.diffusion_steps,
            timestep_respacing
        ),
        betas=betas,
        model_mean_type=(
            gd.ModelMeanType.EPSILON if not scheduler_config.predict_xstart else gd.ModelMeanType.START_X
        ),
        model_var_type=(
            (
                gd.ModelVarType.FIXED_LARGE
                if not scheduler_config.sigma_small
                else gd.ModelVarType.FIXED_SMALL
            )
            if not scheduler_config.learn_sigma
            else gd.ModelVarType.LEARNED_RANGE
        ),
        loss_type=loss_type,
        rescale_timesteps=scheduler_config.rescale_timesteps,
        lambda_geo=training_config.lambda_geo,
    )

    return diffusion


def create_transport(
    path_type='Linear',
    prediction="velocity",
    loss_weight=None,
    train_eps=None,
    sample_eps=None,
    training_config=None,
):
    """Create a flow-matching :class:`Transport`.

    Args:
        path_type: interpolant family — 'Linear' (default), 'GVP', or 'VP'.
        prediction: model parameterization — 'velocity' (default), 'noise',
            or 'score'.
        loss_weight: loss weighting — None (default), 'velocity', or
            'likelihood'. Only affects noise/score prediction.
        train_eps / sample_eps: interval-clipping epsilons; None selects a
            path-dependent default (0 for the stable velocity + Linear/GVP
            combination used by all current configs).
        training_config: supplies ``lambda_geo`` / ``lambda_smooth`` for the
            auxiliary losses.
    """

    if prediction == "noise":
        model_type = ModelType.NOISE
    elif prediction == "score":
        model_type = ModelType.SCORE
    else:
        model_type = ModelType.VELOCITY

    if loss_weight == "velocity":
        loss_type = WeightType.VELOCITY
    elif loss_weight == "likelihood":
        loss_type = WeightType.LIKELIHOOD
    else:
        loss_type = WeightType.NONE

    path_choice = {
        "Linear": PathType.LINEAR,
        "GVP": PathType.GVP,
        "VP": PathType.VP,
    }

    path_type = path_choice[path_type]

    if path_type in [PathType.VP]:
        train_eps = 1e-5 if train_eps is None else train_eps
        sample_eps = 1e-3 if sample_eps is None else sample_eps
    elif path_type in [PathType.GVP, PathType.LINEAR] and model_type != ModelType.VELOCITY:
        train_eps = 1e-3 if train_eps is None else train_eps
        sample_eps = 1e-3 if sample_eps is None else sample_eps
    else:  # velocity & [GVP, LINEAR] is stable everywhere
        train_eps = 0
        sample_eps = 0

    state = Transport(
        model_type=model_type,
        path_type=path_type,
        loss_type=loss_type,
        train_eps=train_eps,
        sample_eps=sample_eps,
        lambda_geo=training_config.lambda_geo,
        lambda_smooth=training_config.lambda_smooth,
    )

    return state