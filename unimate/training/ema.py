"""Exponential Moving Average (EMA) of model weights."""

import torch
from typing import Iterator


class EMAModel:
    """Exponential Moving Average of model weights.

    With ``use_ema_warmup`` the decay follows the standard warmup schedule
    ``decay = (1 + n) / (10 + n)``, where *n* counts EMA updates in units of
    100 optimizer steps; each optimizer step applies the 100th root of that
    value so the smoothing is spread evenly across steps while preserving
    the same half-life. The result is clamped to ``[min_decay, decay]``.
    """

    def __init__(
        self,
        parameters: Iterator[torch.nn.Parameter],
        decay: float = 0.9999,
        min_decay: float = 0.0,
        update_after_step: int = 0,
        use_ema_warmup: bool = False,
    ):
        parameters = list(parameters)
        self.shadow_params = [p.clone().detach() for p in parameters if p.requires_grad]

        self.decay = decay
        self.min_decay = min_decay
        self.update_after_step = update_after_step
        self.use_ema_warmup = use_ema_warmup
        self.optimization_step = 0
        self.cur_decay_value = None

    # ------------------------------------------------------------------
    # Decay schedule
    # ------------------------------------------------------------------

    def get_decay(self, optimization_step: int) -> float:
        """Compute the EMA decay factor for the current step."""
        step = max(0, optimization_step - self.update_after_step - 1)
        if step <= 0:
            return 0.0

        if self.use_ema_warmup:
            # Per-step equivalent of an every-100-steps update with
            # decay = (1 + n) / (10 + n): take the 100th root.
            n = step / 100.0
            cur_decay_value = ((1 + n) / (10 + n)) ** 0.01
        else:
            cur_decay_value = (1 + step) / (10 + step)

        return max(min(cur_decay_value, self.decay), self.min_decay)

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------

    @torch.no_grad()
    def step(self, parameters: Iterator[torch.nn.Parameter]) -> None:
        """Update shadow params toward *parameters* with current decay."""
        parameters = [p for p in parameters if p.requires_grad]
        self.optimization_step += 1

        decay = self.get_decay(self.optimization_step)
        self.cur_decay_value = decay
        one_minus_decay = 1 - decay

        for s_param, param in zip(self.shadow_params, parameters):
            if param.dtype != s_param.dtype:
                param = param.to(s_param.dtype)
            s_param.sub_(one_minus_decay * (s_param - param))

    # ------------------------------------------------------------------
    # Copy / store / restore
    # ------------------------------------------------------------------

    def copy_to(self, parameters: Iterator[torch.nn.Parameter]) -> None:
        """Copy shadow (averaged) params into *parameters*."""
        parameters = [p for p in parameters if p.requires_grad]
        for s_param, param in zip(self.shadow_params, parameters):
            param.data.copy_(s_param.data)

    def store(self, parameters: Iterator[torch.nn.Parameter]) -> None:
        """Snapshot current model params so they can be restored later."""
        self.collected_params = [p.clone() for p in parameters if p.requires_grad]

    def restore(self, parameters: Iterator[torch.nn.Parameter]) -> None:
        """Restore params saved by :meth:`store`."""
        parameters = [p for p in parameters if p.requires_grad]
        for c_param, param in zip(self.collected_params, parameters):
            param.data.copy_(c_param.data)

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def state_dict(self) -> dict:
        return {
            "decay": self.decay,
            "min_decay": self.min_decay,
            "optimization_step": self.optimization_step,
            "update_after_step": self.update_after_step,
            "use_ema_warmup": self.use_ema_warmup,
            "shadow_params": self.shadow_params,
        }

    def load_state_dict(self, state_dict: dict) -> None:
        self.decay = state_dict["decay"]
        self.min_decay = state_dict.get("min_decay", 0.0)
        self.optimization_step = state_dict["optimization_step"]
        self.update_after_step = state_dict["update_after_step"]
        self.use_ema_warmup = state_dict.get("use_ema_warmup", False)
        self.shadow_params = state_dict["shadow_params"]

    def to(self, device=None, dtype=None) -> None:
        """Move shadow params to the given device / dtype."""
        self.shadow_params = [p.to(device=device, dtype=dtype) for p in self.shadow_params]