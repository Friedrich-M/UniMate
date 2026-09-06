# Adapted from SiT: https://github.com/willisma/SiT (MIT License)
"""Numerical integrators for SDE and ODE sampling.

Provides ``sde`` (Euler-Maruyama / Heun) and ``ode`` (via ``torchdiffeq``)
solvers used by :class:`transport.Sampler`.
"""

import torch as th
from torchdiffeq import odeint


# ------------------------------------------------------------------
# SDE Solver
# ------------------------------------------------------------------

class sde:
    """Forward-time SDE integrator (Euler-Maruyama or Heun)."""

    def __init__(self, drift, diffusion, *, t0, t1, num_steps, sampler_type):
        assert t0 < t1, "SDE sampler requires forward time (t0 < t1)"

        self.num_timesteps = num_steps
        self.t = th.linspace(t0, t1, num_steps)
        self.dt = self.t[1] - self.t[0]
        self.drift = drift
        self.diffusion = diffusion
        self.sampler_type = sampler_type

    def __Euler_Maruyama_step(self, x, mean_x, t, model, **model_kwargs):
        w_cur = th.randn(x.size()).to(x)
        t = th.ones(x.size(0)).to(x) * t
        dw = w_cur * th.sqrt(self.dt)
        drift = self.drift(x, t, model, **model_kwargs)
        diffusion = self.diffusion(x, t)
        mean_x = x + drift * self.dt
        x = mean_x + th.sqrt(2 * diffusion) * dw
        return x, mean_x

    def __Heun_step(self, x, _, t, model, **model_kwargs):
        w_cur = th.randn(x.size()).to(x)
        dw = w_cur * th.sqrt(self.dt)
        t_cur = th.ones(x.size(0)).to(x) * t
        diffusion = self.diffusion(x, t_cur)
        xhat = x + th.sqrt(2 * diffusion) * dw
        K1 = self.drift(xhat, t_cur, model, **model_kwargs)
        xp = xhat + self.dt * K1
        K2 = self.drift(xp, t_cur + self.dt, model, **model_kwargs)
        # At the last time point we skip the Heun correction
        return xhat + 0.5 * self.dt * (K1 + K2), xhat

    def __forward_fn(self):
        """Select the step function based on ``sampler_type``."""
        sampler_dict = {
            "Euler": self.__Euler_Maruyama_step,
            "Heun": self.__Heun_step,
        }
        try:
            sampler = sampler_dict[self.sampler_type]
        except KeyError:
            raise NotImplementedError(f"Sampler type '{self.sampler_type}' not implemented.")
        return sampler

    def sample(self, init, model, **model_kwargs):
        """Run the forward SDE loop and collect samples at each step."""
        x = init
        mean_x = init
        samples = []
        sampler = self.__forward_fn()
        for ti in self.t[:-1]:
            with th.no_grad():
                x, mean_x = sampler(x, mean_x, ti, model, **model_kwargs)
                samples.append(x)
        return samples


# ------------------------------------------------------------------
# ODE Solver
# ------------------------------------------------------------------

class ode:
    """ODE integrator wrapping ``torchdiffeq.odeint``."""

    def __init__(self, drift, *, t0, t1, sampler_type, num_steps, atol, rtol):
        assert t0 < t1, "ODE sampler requires forward time (t0 < t1)"

        self.drift = drift
        self.t = th.linspace(t0, t1, num_steps)
        self.atol = atol
        self.rtol = rtol
        self.sampler_type = sampler_type

    def sample(self, x, model, **model_kwargs):
        """Integrate the ODE from t0 to t1."""
        device = x[0].device if isinstance(x, tuple) else x.device

        def _fn(t, x):
            batch_t = (
                th.ones(x[0].size(0)).to(device) * t
                if isinstance(x, tuple)
                else th.ones(x.size(0)).to(device) * t
            )
            return self.drift(x, batch_t, model, **model_kwargs)

        t = self.t.to(device)
        atol = [self.atol] * len(x) if isinstance(x, tuple) else [self.atol]
        rtol = [self.rtol] * len(x) if isinstance(x, tuple) else [self.rtol]
        samples = odeint(
            _fn, x, t,
            method=self.sampler_type, atol=atol, rtol=rtol,
        )
        return samples