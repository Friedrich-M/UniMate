# Adapted from SiT: https://github.com/willisma/SiT (MIT License)
"""Flow-matching transport for training and sampling generative models.

Supports three model parameterizations (velocity, noise, score) and three
interpolant paths (linear, GVP, VP).  The primary training path for this
project uses ``ModelType.VELOCITY`` with ``PathType.LINEAR``.
"""

import enum

import numpy as np
import torch as th

from . import path
from .integrators import ode, sde

from unimate.models.diffusion.nn import mean_flat, sum_flat
from unimate.utils.rotation_conversions import rotation_6d_to_matrix_safe
from unimate.utils.geo_utils import geodesic_distance


# ------------------------------------------------------------------
# Enums
# ------------------------------------------------------------------

class ModelType(enum.Enum):
    """What the model predicts."""
    NOISE = enum.auto()     # epsilon
    SCORE = enum.auto()     # nabla log p(x)
    VELOCITY = enum.auto()  # v(x)


class PathType(enum.Enum):
    """Interpolant path family."""
    LINEAR = enum.auto()
    GVP = enum.auto()
    VP = enum.auto()


class WeightType(enum.Enum):
    """Loss weighting strategy."""
    NONE = enum.auto()
    VELOCITY = enum.auto()
    LIKELIHOOD = enum.auto()


# ------------------------------------------------------------------
# Transport
# ------------------------------------------------------------------

class Transport:
    """Core transport object pairing a path sampler with a model type.

    Args:
        model_type: prediction parameterization (velocity / noise / score).
        path_type: interpolant family (linear / GVP / VP).
        loss_type: loss weighting strategy.
        train_eps / sample_eps: epsilon for numerical stability.
        lambda_geo: weight for the auxiliary geodesic rotation loss.
        lambda_smooth: weight for the auxiliary velocity smoothness loss
            (penalizes Δ on the local-velocity channels of the predicted x1).
    """

    def __init__(
        self,
        *,
        model_type,
        path_type,
        loss_type,
        train_eps,
        sample_eps,
        lambda_geo=0.0,
        lambda_smooth=0.0,
    ):
        path_options = {
            PathType.LINEAR: path.ICPlan,
            PathType.GVP: path.GVPCPlan,
            PathType.VP: path.VPCPlan,
        }

        self.loss_type = loss_type
        self.model_type = model_type
        self.path_sampler = path_options[path_type]()
        self.train_eps = train_eps
        self.sample_eps = sample_eps
        self.lambda_geo = lambda_geo
        self.lambda_smooth = lambda_smooth

        self.l2_loss = lambda a, b: (a - b) ** 2

    # ------------------------------------------------------------------
    # Loss helpers
    # ------------------------------------------------------------------

    def temporal_spatial_masked_l2(self, target, pred, temp_mask, spat_mask, lengths, n_joints):
        """Masked L2 loss over temporal and spatial dimensions.

        Args:
            target, pred: (B, J, D, T)
            temp_mask:    (B, 1, 1, T)
            spat_mask:    (B, 1, 1, J)
            lengths:      (B,) valid frame counts
            n_joints:     (B,) valid joint counts
        Returns:
            (B,) per-sample MSE values.
        """
        loss = self.l2_loss(target, pred)
        temp_masked_loss = loss * temp_mask.float()
        spat_temp_masked_loss = temp_masked_loss * spat_mask.float().transpose(1, 3)
        loss = sum_flat(spat_temp_masked_loss)
        non_zero_elements = lengths * n_joints * target.size(2)
        return loss / non_zero_elements

    def geodesic_loss(self, target, pred, temp_mask, spat_mask, lengths, n_joints):
        """Masked geodesic (rotation angle) loss.

        Extracts 6D rotation features (channels 3–9), converts to matrices,
        then computes the angular distance.

        Args/Returns: same shape convention as :meth:`temporal_spatial_masked_l2`.
        """
        # (B, J, D, T) → (B, T, J, D) → extract 6D → (B, T, J, 3, 3)
        rots_target = rotation_6d_to_matrix_safe(target.permute(0, 3, 1, 2)[..., 3:9])
        rots_pred = rotation_6d_to_matrix_safe(pred.permute(0, 3, 1, 2)[..., 3:9])

        loss = geodesic_distance(rots_pred, rots_target).permute(0, 2, 3, 1)  # (B, J, 1, T)
        temp_masked_loss = loss * temp_mask.float()
        spat_temp_masked_loss = temp_masked_loss * spat_mask.float().transpose(1, 3)
        loss = sum_flat(spat_temp_masked_loss)
        non_zero_elements = lengths * n_joints
        return loss / non_zero_elements

    def velocity_smoothness_loss(self, pred, temp_mask, spat_mask, lengths, n_joints):
        """Masked smoothness penalty on the local-velocity channels.

        Penalizes per-frame finite differences on channels [-3:] of the
        predicted clean motion x1 — i.e., regularizes the acceleration of
        the UniMate ``local_velocity`` block (channels [9:12] in the
        12-dim layout). Pure regularization on the prediction; no GT.

        Args:
            pred:      (B, J, D, T) predicted clean motion (denormalized).
            temp_mask: (B, 1, 1, T)
            spat_mask: (B, 1, 1, J)
            lengths:   (B,) valid frame counts.
            n_joints:  (B,) valid joint counts.
        Returns:
            (B,) per-sample smoothness penalty.
        """
        v = pred[:, :, -3:, :]                                          # (B, J, 3, T)
        dv = v[..., 1:] - v[..., :-1]                                   # (B, J, 3, T-1)
        pair_mask = temp_mask[..., 1:].float() * temp_mask[..., :-1].float()  # (B,1,1,T-1)
        loss = (dv ** 2) * pair_mask
        loss = loss * spat_mask.float().transpose(1, 3)
        loss = sum_flat(loss)
        denom = th.clamp(lengths - 1, min=1) * n_joints * 3
        return loss / denom

    # ------------------------------------------------------------------
    # Sampling & interval helpers
    # ------------------------------------------------------------------

    def prior_logp(self, z):
        """Log-probability under standard multivariate normal."""
        shape = th.tensor(z.size())
        N = th.prod(shape[1:])
        _fn = lambda x: -N / 2.0 * np.log(2 * np.pi) - th.sum(x ** 2) / 2.0
        return th.vmap(_fn)(z)

    def check_interval(
        self,
        train_eps,
        sample_eps,
        *,
        diffusion_form="SBDM",
        sde=False,
        reverse=False,
        eval=False,
        last_step_size=0.0,
    ):
        """Compute integration interval [t0, t1] with epsilon clipping."""
        t0 = 0
        t1 = 1
        eps = train_eps if not eval else sample_eps

        if type(self.path_sampler) in [path.VPCPlan]:
            t1 = 1 - eps if (not sde or last_step_size == 0) else 1 - last_step_size

        elif (type(self.path_sampler) in [path.ICPlan, path.GVPCPlan]
              and (self.model_type != ModelType.VELOCITY or sde)):
            # Avoid numerical issues with a semi-implicit first step
            t0 = eps if (diffusion_form == "SBDM" and sde) or self.model_type != ModelType.VELOCITY else 0
            t1 = 1 - eps if (not sde or last_step_size == 0) else 1 - last_step_size

        if reverse:
            t0, t1 = 1 - t0, 1 - t1

        return t0, t1

    def sample(self, x1):
        """Sample noise x0 and time t for training.

        Args:
            x1: data point, shape (B, *dim).
        Returns:
            (t, x0, x1) where x0 ~ N(0, I), t ~ U[t0, t1].
        """
        x0 = th.randn_like(x1)
        t0, t1 = self.check_interval(self.train_eps, self.sample_eps)
        t = th.rand((x1.shape[0],)) * (t1 - t0) + t0
        t = t.to(x1)
        return t, x0, x1

    # ------------------------------------------------------------------
    # Training loss
    # ------------------------------------------------------------------

    def training_losses(self, model, x1, model_kwargs=None):
        """Compute training loss for the flow-matching model.

        Args:
            model: denoising network (velocity / noise / score predictor).
            x1: clean data, shape (B, J, D, T).
            model_kwargs: dict with ``cond`` key containing masks, lengths, etc.
        Returns:
            dict of per-sample loss tensors keyed by name.
        """
        if model_kwargs is None:
            model_kwargs = {}

        # Extract masks from conditioning
        cond = model_kwargs.get("cond", {})
        temp_mask = cond["lengths_mask"]    # (B, 1, 1, T)
        gt_n_frames = cond["motion_length"] # (B,)
        joint_mask = cond["joint_mask"]   # (B, 1, 1, J)
        gt_n_joints = cond["n_joints"]      # (B,)

        # Sample noise and interpolate
        t, x0, x1 = self.sample(x1)
        t, xt, ut = self.path_sampler.plan(t, x0, x1)

        model_output = model(xt, t, **model_kwargs)
        B, *_, C = xt.shape
        assert model_output.size() == (B, *xt.size()[1:-1], C)

        terms = {}

        if self.model_type == ModelType.VELOCITY:
            # --- Velocity prediction loss ---
            terms["diff_loss"] = self.temporal_spatial_masked_l2(
                target=ut, pred=model_output,
                temp_mask=temp_mask, spat_mask=joint_mask,
                lengths=gt_n_frames, n_joints=gt_n_joints,
            )
            terms["loss"] = terms["diff_loss"].clone()

            # Reconstruct x1 from velocity (then denormalize) for any
            # auxiliary loss that operates in data space.
            if self.lambda_geo > 0 or self.lambda_smooth > 0:
                # x1 = xt + (1-t)·v holds for the linear interpolant only
                # (xt = t·x1 + (1-t)·x0, ut = x1 - x0).
                assert type(self.path_sampler) is path.ICPlan, (
                    "geo/smooth aux losses assume PathType.LINEAR")
                pred_x1 = model_output * (1 - t).view(-1, 1, 1, 1) + xt
                mean = cond["mean"][..., None]  # (B, J, D, 1)
                std = cond["std"][..., None]
                pred_x1_denorm = pred_x1 * std + mean

            # Optional geodesic rotation loss
            if self.lambda_geo > 0:
                target_x1 = x1 * std + mean

                terms["geo_loss"] = self.geodesic_loss(
                    target=target_x1, pred=pred_x1_denorm,
                    temp_mask=temp_mask, spat_mask=joint_mask,
                    lengths=gt_n_frames, n_joints=gt_n_joints,
                )
                terms["loss"] = terms["loss"] + self.lambda_geo * terms["geo_loss"]

            # Optional velocity smoothness regularizer
            if self.lambda_smooth > 0:
                terms["smooth_loss"] = self.velocity_smoothness_loss(
                    pred=pred_x1_denorm,
                    temp_mask=temp_mask, spat_mask=joint_mask,
                    lengths=gt_n_frames, n_joints=gt_n_joints,
                )
                terms["loss"] = terms["loss"] + self.lambda_smooth * terms["smooth_loss"]

        else:
            # --- Noise / score prediction loss (weighted) ---
            _, drift_var = self.path_sampler.compute_drift(xt, t)
            sigma_t, _ = self.path_sampler.compute_sigma_t(path.expand_t_like_x(t, xt))

            if self.loss_type in [WeightType.VELOCITY]:
                weight = (drift_var / sigma_t) ** 2
            elif self.loss_type in [WeightType.LIKELIHOOD]:
                weight = drift_var / (sigma_t ** 2)
            elif self.loss_type in [WeightType.NONE]:
                weight = 1
            else:
                raise NotImplementedError(f"Unknown loss type: {self.loss_type}")

            if self.model_type == ModelType.NOISE:
                terms["loss"] = mean_flat(weight * ((model_output - x0) ** 2))
            else:
                terms["loss"] = mean_flat(weight * ((model_output * sigma_t + x0) ** 2))

        return terms

    # ------------------------------------------------------------------
    # Drift & score for ODE/SDE sampling
    # ------------------------------------------------------------------

    def get_drift(self):
        """Return the drift function for the probability flow ODE."""

        def score_ode(x, t, model, **model_kwargs):
            drift_mean, drift_var = self.path_sampler.compute_drift(x, t)
            model_output = model(x, t, **model_kwargs)
            return -drift_mean + drift_var * model_output

        def noise_ode(x, t, model, **model_kwargs):
            drift_mean, drift_var = self.path_sampler.compute_drift(x, t)
            sigma_t, _ = self.path_sampler.compute_sigma_t(path.expand_t_like_x(t, x))
            model_output = model(x, t, **model_kwargs)
            score = model_output / -sigma_t
            return -drift_mean + drift_var * score

        def velocity_ode(x, t, model, **model_kwargs):
            return model(x, t, **model_kwargs)

        if self.model_type == ModelType.NOISE:
            drift_fn = noise_ode
        elif self.model_type == ModelType.SCORE:
            drift_fn = score_ode
        else:
            drift_fn = velocity_ode

        def body_fn(x, t, model, **model_kwargs):
            model_output = drift_fn(x, t, model, **model_kwargs)
            assert model_output.shape == x.shape, "Output shape from ODE solver must match input shape"
            return model_output

        return body_fn

    def get_score(self):
        """Return the score function: nabla log p_t(x_t)."""
        if self.model_type == ModelType.NOISE:
            score_fn = lambda x, t, model, **kwargs: (
                model(x, t, **kwargs) / -self.path_sampler.compute_sigma_t(path.expand_t_like_x(t, x))[0]
            )
        elif self.model_type == ModelType.SCORE:
            score_fn = lambda x, t, model, **kwargs: model(x, t, **kwargs)
        elif self.model_type == ModelType.VELOCITY:
            score_fn = lambda x, t, model, **kwargs: (
                self.path_sampler.get_score_from_velocity(model(x, t, **kwargs), x, t)
            )
        else:
            raise NotImplementedError()

        return score_fn


# ------------------------------------------------------------------
# Sampler
# ------------------------------------------------------------------

class Sampler:
    """Wraps a :class:`Transport` to provide ODE and SDE sampling methods.

    Args:
        transport: a Transport object specifying model type and interpolant.
    """

    def __init__(self, transport):
        self.transport = transport
        self.drift = self.transport.get_drift()
        self.score = self.transport.get_score()

    # ------------------------------------------------------------------
    # SDE sampling
    # ------------------------------------------------------------------

    def __get_sde_diffusion_and_drift(self, *, diffusion_form="SBDM", diffusion_norm=1.0):
        """Build SDE drift and diffusion coefficient functions."""

        def diffusion_fn(x, t):
            return self.transport.path_sampler.compute_diffusion(
                x, t, form=diffusion_form, norm=diffusion_norm
            )

        sde_drift = lambda x, t, model, **kwargs: (
            self.drift(x, t, model, **kwargs)
            + diffusion_fn(x, t) * self.score(x, t, model, **kwargs)
        )
        return sde_drift, diffusion_fn

    def __get_last_step(self, sde_drift, *, last_step, last_step_size):
        """Build the final denoising step function for SDE sampling."""
        if last_step is None:
            return lambda x, t, model, **model_kwargs: x
        elif last_step == "Mean":
            return lambda x, t, model, **model_kwargs: (
                x + sde_drift(x, t, model, **model_kwargs) * last_step_size
            )
        elif last_step == "Tweedie":
            alpha = self.transport.path_sampler.compute_alpha_t
            sigma = self.transport.path_sampler.compute_sigma_t
            return lambda x, t, model, **model_kwargs: (
                x / alpha(t)[0][0]
                + (sigma(t)[0][0] ** 2) / alpha(t)[0][0] * self.score(x, t, model, **model_kwargs)
            )
        elif last_step == "Euler":
            return lambda x, t, model, **model_kwargs: (
                x + self.drift(x, t, model, **model_kwargs) * last_step_size
            )
        else:
            raise NotImplementedError(f"Unknown last_step: {last_step}")

    def sample_sde(
        self,
        *,
        sampling_method="Euler",
        diffusion_form="SBDM",
        diffusion_norm=1.0,
        last_step="Mean",
        last_step_size=0.04,
        num_steps=250,
    ):
        """Return a sampling function using SDE integration.

        Args:
            sampling_method: "Euler" or "Heun".
            diffusion_form: diffusion coefficient form.
            diffusion_norm: diffusion coefficient magnitude.
            last_step: final step type ("Mean", "Tweedie", "Euler", or None).
            last_step_size: step size for the final step.
            num_steps: total integration steps.
        """
        if last_step is None:
            last_step_size = 0.0

        sde_drift, sde_diffusion = self.__get_sde_diffusion_and_drift(
            diffusion_form=diffusion_form, diffusion_norm=diffusion_norm,
        )

        t0, t1 = self.transport.check_interval(
            self.transport.train_eps, self.transport.sample_eps,
            diffusion_form=diffusion_form,
            sde=True, eval=True, reverse=False,
            last_step_size=last_step_size,
        )

        _sde = sde(
            sde_drift, sde_diffusion,
            t0=t0, t1=t1, num_steps=num_steps, sampler_type=sampling_method,
        )
        last_step_fn = self.__get_last_step(
            sde_drift, last_step=last_step, last_step_size=last_step_size,
        )

        def _sample(init, model, **model_kwargs):
            xs = _sde.sample(init, model, **model_kwargs)
            ts = th.ones(init.size(0), device=init.device) * t1
            x = last_step_fn(xs[-1], ts, model, **model_kwargs)
            xs.append(x)
            assert len(xs) == num_steps, "Samples does not match the number of steps"
            return xs

        return _sample

    # ------------------------------------------------------------------
    # ODE sampling
    # ------------------------------------------------------------------

    def sample_ode(
        self,
        *,
        sampling_method="dopri5",
        num_steps=50,
        atol=1e-6,
        rtol=1e-3,
        reverse=False,
    ):
        """Return a sampling function using ODE integration.

        Args:
            sampling_method: "dopri5" (adaptive) or "euler"/"heun" (fixed).
            num_steps: integration steps (fixed) or saved datapoints (adaptive).
            atol / rtol: error tolerances for adaptive solvers.
            reverse: if True, integrate from data to noise.
        """
        if reverse:
            drift = lambda x, t, model, **kwargs: self.drift(x, th.ones_like(t) * (1 - t), model, **kwargs)
        else:
            drift = self.drift

        t0, t1 = self.transport.check_interval(
            self.transport.train_eps, self.transport.sample_eps,
            sde=False, eval=True, reverse=reverse, last_step_size=0.0,
        )

        _ode = ode(
            drift=drift, t0=t0, t1=t1,
            sampler_type=sampling_method, num_steps=num_steps,
            atol=atol, rtol=rtol,
        )
        return _ode.sample

    # ------------------------------------------------------------------
    # Likelihood
    # ------------------------------------------------------------------

    def sample_ode_likelihood(
        self,
        *,
        sampling_method="dopri5",
        num_steps=50,
        atol=1e-6,
        rtol=1e-3,
    ):
        """Return a function that computes log-likelihood via the instantaneous
        change-of-variables formula (Hutchinson trace estimator)."""

        def _likelihood_drift(x, t, model, **model_kwargs):
            x, _ = x
            eps = th.randint(2, x.size(), dtype=th.float, device=x.device) * 2 - 1
            t = th.ones_like(t) * (1 - t)
            with th.enable_grad():
                x.requires_grad = True
                grad = th.autograd.grad(th.sum(self.drift(x, t, model, **model_kwargs) * eps), x)[0]
                logp_grad = th.sum(grad * eps, dim=tuple(range(1, len(x.size()))))
                drift = self.drift(x, t, model, **model_kwargs)
            return (-drift, logp_grad)

        t0, t1 = self.transport.check_interval(
            self.transport.train_eps, self.transport.sample_eps,
            sde=False, eval=True, reverse=False, last_step_size=0.0,
        )

        _ode = ode(
            drift=_likelihood_drift, t0=t0, t1=t1,
            sampler_type=sampling_method, num_steps=num_steps,
            atol=atol, rtol=rtol,
        )

        def _sample_fn(x, model, **model_kwargs):
            init_logp = th.zeros(x.size(0)).to(x)
            input = (x, init_logp)
            drift, delta_logp = _ode.sample(input, model, **model_kwargs)
            drift, delta_logp = drift[-1], delta_logp[-1]
            prior_logp = self.transport.prior_logp(drift)
            logp = prior_logp - delta_logp
            return logp, drift

        return _sample_fn