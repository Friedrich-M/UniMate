# Adapted from SiT: https://github.com/willisma/SiT (MIT License)
"""Interpolant paths (coupling plans) for flow matching.

Each plan defines how to interpolate between noise x0 and data x1 over time,
providing coefficients alpha_t (data) and sigma_t (noise), along with drift
and diffusion terms for the corresponding SDE.

Supported plans:
  - ``ICPlan``   — Linear interpolation (identity coupling)
  - ``VPCPlan``  — Variance-preserving path
  - ``GVPCPlan`` — Geodesic variance-preserving path
"""

import numpy as np
import torch as th


def expand_t_like_x(t, x):
    """Reshape time vector ``t`` to be broadcastable with data ``x``.

    Args:
        t: (B,) time vector.
        x: (B, ...) data tensor.
    """
    dims = [1] * (len(x.size()) - 1)
    t = t.view(t.size(0), *dims)
    return t


# ------------------------------------------------------------------
# Linear Coupling Plan (base class)
# ------------------------------------------------------------------

class ICPlan:
    """Linear (identity coupling) interpolant: x_t = t * x1 + (1-t) * x0."""

    def __init__(self, sigma=0.0):
        self.sigma = sigma

    def compute_alpha_t(self, t):
        """Data coefficient alpha_t and its time derivative."""
        return t, 1

    def compute_sigma_t(self, t):
        """Noise coefficient sigma_t and its time derivative."""
        return 1 - t, -1

    def compute_d_alpha_alpha_ratio_t(self, t):
        """d(alpha_t)/dt / alpha_t, numerically stable form."""
        return 1 / t

    def compute_drift(self, x, t):
        """SDE drift (score parameterization). Returns (-drift, diffusion)."""
        t = expand_t_like_x(t, x)
        alpha_ratio = self.compute_d_alpha_alpha_ratio_t(t)
        sigma_t, d_sigma_t = self.compute_sigma_t(t)
        drift = alpha_ratio * x
        diffusion = alpha_ratio * (sigma_t ** 2) - sigma_t * d_sigma_t
        return -drift, diffusion

    def compute_diffusion(self, x, t, form="constant", norm=1.0):
        """SDE diffusion coefficient under various functional forms."""
        t = expand_t_like_x(t, x)
        choices = {
            "constant": norm,
            "SBDM": norm * self.compute_drift(x, t)[1],
            "sigma": norm * self.compute_sigma_t(t)[0],
            "linear": norm * (1 - t),
            "decreasing": 0.25 * (norm * th.cos(np.pi * t) + 1) ** 2,
            "increasing-decreasing": norm * th.sin(np.pi * t) ** 2,
        }

        try:
            diffusion = choices[form]
        except KeyError:
            raise NotImplementedError(f"Diffusion form {form} not implemented")

        return diffusion

    # --- Conversion between model parameterizations ---

    def get_score_from_velocity(self, velocity, x, t):
        """Convert velocity prediction to score."""
        t = expand_t_like_x(t, x)
        alpha_t, d_alpha_t = self.compute_alpha_t(t)
        sigma_t, d_sigma_t = self.compute_sigma_t(t)
        mean = x
        reverse_alpha_ratio = alpha_t / d_alpha_t
        var = sigma_t ** 2 - reverse_alpha_ratio * d_sigma_t * sigma_t
        score = (reverse_alpha_ratio * velocity - mean) / var
        return score

    def get_noise_from_velocity(self, velocity, x, t):
        """Convert velocity prediction to noise (denoiser)."""
        t = expand_t_like_x(t, x)
        alpha_t, d_alpha_t = self.compute_alpha_t(t)
        sigma_t, d_sigma_t = self.compute_sigma_t(t)
        mean = x
        reverse_alpha_ratio = alpha_t / d_alpha_t
        var = reverse_alpha_ratio * d_sigma_t - sigma_t
        noise = (reverse_alpha_ratio * velocity - mean) / var
        return noise

    def get_velocity_from_score(self, score, x, t):
        """Convert score prediction to velocity."""
        t = expand_t_like_x(t, x)
        drift, var = self.compute_drift(x, t)
        velocity = var * score - drift
        return velocity

    # --- Interpolation and planning ---

    def compute_mu_t(self, t, x0, x1):
        """Mean of the time-dependent density p_t."""
        t = expand_t_like_x(t, x1)
        alpha_t, _ = self.compute_alpha_t(t)
        sigma_t, _ = self.compute_sigma_t(t)
        return alpha_t * x1 + sigma_t * x0

    def compute_xt(self, t, x0, x1):
        """Sample x_t from p_t (deterministic for zero sigma)."""
        return self.compute_mu_t(t, x0, x1)

    def compute_ut(self, t, x0, x1, xt):
        """Conditional vector field u_t corresponding to p_t."""
        t = expand_t_like_x(t, x1)
        _, d_alpha_t = self.compute_alpha_t(t)
        _, d_sigma_t = self.compute_sigma_t(t)
        return d_alpha_t * x1 + d_sigma_t * x0

    def plan(self, t, x0, x1):
        """Compute (t, x_t, u_t) for a batch of (noise, data) pairs."""
        xt = self.compute_xt(t, x0, x1)
        ut = self.compute_ut(t, x0, x1, xt)
        return t, xt, ut


# ------------------------------------------------------------------
# Variance-Preserving Coupling Plan
# ------------------------------------------------------------------

class VPCPlan(ICPlan):
    """VP (variance-preserving) path for flow matching."""

    def __init__(self, sigma_min=0.1, sigma_max=20.0):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.log_mean_coeff = (
            lambda t: -0.25 * ((1 - t) ** 2) * (self.sigma_max - self.sigma_min)
            - 0.5 * (1 - t) * self.sigma_min
        )
        self.d_log_mean_coeff = (
            lambda t: 0.5 * (1 - t) * (self.sigma_max - self.sigma_min)
            + 0.5 * self.sigma_min
        )

    def compute_alpha_t(self, t):
        alpha_t = th.exp(self.log_mean_coeff(t))
        d_alpha_t = alpha_t * self.d_log_mean_coeff(t)
        return alpha_t, d_alpha_t

    def compute_sigma_t(self, t):
        p_sigma_t = 2 * self.log_mean_coeff(t)
        sigma_t = th.sqrt(1 - th.exp(p_sigma_t))
        d_sigma_t = th.exp(p_sigma_t) * (2 * self.d_log_mean_coeff(t)) / (-2 * sigma_t)
        return sigma_t, d_sigma_t

    def compute_d_alpha_alpha_ratio_t(self, t):
        return self.d_log_mean_coeff(t)

    def compute_drift(self, x, t):
        t = expand_t_like_x(t, x)
        beta_t = self.sigma_min + (1 - t) * (self.sigma_max - self.sigma_min)
        return -0.5 * beta_t * x, beta_t / 2


# ------------------------------------------------------------------
# Geodesic Variance-Preserving Coupling Plan
# ------------------------------------------------------------------

class GVPCPlan(ICPlan):
    """GVP (geodesic variance-preserving) path with trigonometric coefficients."""

    def __init__(self, sigma=0.0):
        super().__init__(sigma)

    def compute_alpha_t(self, t):
        alpha_t = th.sin(t * np.pi / 2)
        d_alpha_t = np.pi / 2 * th.cos(t * np.pi / 2)
        return alpha_t, d_alpha_t

    def compute_sigma_t(self, t):
        sigma_t = th.cos(t * np.pi / 2)
        d_sigma_t = -np.pi / 2 * th.sin(t * np.pi / 2)
        return sigma_t, d_sigma_t

    def compute_d_alpha_alpha_ratio_t(self, t):
        return np.pi / (2 * th.tan(t * np.pi / 2))