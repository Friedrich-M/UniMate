"""Diffusion / flow-matching trainer with EMA and visualization support.

Handles two generation paradigms selected via ``training.diff_model``:
  * ``"flow"``      — flow matching (ODE-based sampling)
  * ``"diffusion"`` — Gaussian DDPM (iterative denoising)
"""

import os
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from unimate.configs.schema import MainConfig
from unimate.dataset.conditioning import create_sample_condition
from unimate.models.flow.transport import Sampler
from unimate.models.factory import create_diffusion, create_transport
from unimate.training.ema import EMAModel
from unimate.training.base_trainer import BaseTrainer
from unimate.utils.logger import get_logger
from unimate.inference.generate import generate_samples
from unimate.utils.visualization import visualize_and_save_motions

logger = get_logger(file_name=__file__)


class DiffusionTrainer(BaseTrainer):
    """Trainer that pairs a denoising model with a diffusion/flow schedule.

    Owns:
      - ``model``        — the trainable denoising network
      - ``diffusion``    — the noise schedule (Gaussian or flow transport)
      - ``ema_model``    — optional exponential moving average of model weights
      - ``data``         — reference to the dataloader (used for visualization)
    """

    def __init__(
        self,
        config: MainConfig,
        model: nn.Module,
        data: torch.utils.data.DataLoader,
        checkpoint_dir: Optional[str] = None,
    ):
        super().__init__(checkpoint_dir)

        self.config = config
        self.model = model
        self.data = data

        self._setup_sdpa_backends()
        self._setup_diffusion(config)
        self._setup_ema_config(config)

        logger.info(
            f"DiffusionTrainer initialized"
            f" | Parameters: {self.get_num_parameters() / 1e6:.2f}M"
            f" | EMA: {self.use_ema} (decay={self.ema_decay})"
        )

    # ------------------------------------------------------------------
    # Initialization helpers
    # ------------------------------------------------------------------

    def _setup_diffusion(self, config: MainConfig) -> None:
        """Create the diffusion or flow-matching schedule."""
        if config.training.diff_model == "flow":
            self.diffusion = create_transport(training_config=config.training)
            self.gen_diffusion = Sampler(self.diffusion)
        elif config.training.diff_model == "diffusion":
            self.diffusion = create_diffusion(
                scheduler_config=config.scheduler,
                training_config=config.training,
            )
        else:
            raise ValueError(
                f"Unknown diff_model: {config.training.diff_model!r}"
            )

    def _setup_ema_config(self, config: MainConfig) -> None:
        """Store EMA configuration; actual EMA model is created later via
        :meth:`initialize_ema` (after ``accelerator.prepare``)."""
        self.use_ema = config.training.use_ema
        self.ema_decay = config.training.ema_decay
        self.ema_model: Optional[EMAModel] = None

    @staticmethod
    def _setup_sdpa_backends() -> None:
        """Enable Flash Attention and memory-efficient SDPA backends."""
        logger.info(
            f"SDPA backends — Flash: {torch.backends.cuda.flash_sdp_enabled()}"
            f", MemEfficient: {torch.backends.cuda.mem_efficient_sdp_enabled()}"
            f", Math: {torch.backends.cuda.math_sdp_enabled()}"
        )
        if torch.cuda.is_available():
            try:
                torch.backends.cuda.enable_flash_sdp(True)
                torch.backends.cuda.enable_mem_efficient_sdp(True)
                logger.info("Enabled Flash Attention and Memory Efficient backends")
            except Exception as e:
                logger.warning(f"Failed to enable SDPA backends: {e}")

    def initialize_ema(self, accelerator) -> None:
        """Create the EMA model. Call *after* ``accelerator.prepare``.

        Uses the warmup schedule in :class:`EMAModel`: the every-100-steps
        form ``decay = (1 + n) / (10 + n)`` spread over single steps by its
        100th root, so the decay starts near 0.98 (not 0.09 — that figure
        describes the un-rooted form) and creeps toward the ``ema_decay`` cap,
        reaching 0.9999 at roughly 90k optimizer steps.
        """
        self.accelerator = accelerator
        if not self.use_ema:
            return
        self.ema_model = EMAModel(
            parameters=self._unwrapped_params(),
            decay=self.ema_decay,
            use_ema_warmup=True,
        )
        self.ema_model.to(accelerator.device, dtype=torch.float32)
        logger.info(f"EMA initialized with decay={self.ema_decay}, warmup=True")

    # ------------------------------------------------------------------
    # Properties & mode switching
    # ------------------------------------------------------------------

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    def train(self, mode: bool = True):
        """Delegates to the wrapped ``model`` (the trainer itself has no
        trainable parameters)."""
        self.model.train(mode)
        return self

    def eval(self):
        return self.train(False)

    # ------------------------------------------------------------------
    # Loss computation
    # ------------------------------------------------------------------

    def compute_loss(
        self, batch
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute diffusion/flow training loss.

        Returns:
            (loss, loss_dict) — scalar for backprop and per-component floats.
        """
        motion, cond = batch
        cond = self._to_device(cond)
        model_kwargs = dict(cond=cond)

        if self.config.training.diff_model == "flow":
            loss_dict = self.diffusion.training_losses(
                self.model, motion, model_kwargs
            )
        else:  # "diffusion"
            t = torch.randint(
                0, self.diffusion.num_timesteps,
                (motion.size(0),), device=motion.device,
            )
            loss_dict = self.diffusion.training_losses(
                self.model, motion, t, model_kwargs
            )

        loss = loss_dict["loss"].mean()
        loss_dict = {k: v.mean().item() for k, v in loss_dict.items()}
        return loss, loss_dict

    # ------------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------------

    @torch.no_grad()
    def visualize_samples(
        self,
        save_dir: str,
        prefix: str,
        num_samples: int = 3,
        cfg_scale: float = 1.0,
    ) -> None:
        """Generate and save sample motions. Uses EMA weights when available.

        Temporarily swaps in EMA parameters for generation, then restores
        the original weights so training can continue unaffected.
        """
        use_ema = self.use_ema and self.ema_model is not None
        # Use the unwrapped model for visualization to avoid DDP
        # communication that would cause NCCL timeouts on other ranks.
        # Bound before the try/EMA-swap so the finally block can always
        # restore train mode regardless of where a later error fires.
        unwrapped_model = self.accelerator.unwrap_model(self.model)

        if use_ema:
            logger.info("Using EMA model for sample visualization.")
            self.ema_model.store(self._unwrapped_params())
            self.ema_model.copy_to(self._unwrapped_params())

        try:
            unwrapped_model.eval()
            os.makedirs(save_dir, exist_ok=True)

            # Build conditioning from both train and eval pools so each viz
            # batch shows in-distribution and held-out behaviour side-by-side.
            # Routing per dataset (handled inside create_sample_condition):
            #   * 'object_type' (truebones, objaverse) → num_samples train
            #     types + num_samples eval types, one random clip each.
            #   * 'clip' (mixamo) → num_samples random train clips + same
            #     for eval, all from the single skeleton (caption varies).
            motion_dataset = self.data.dataset.motion_dataset
            n_eval_types = len(motion_dataset.eval_object_motions_map)
            logger.info(
                f"Visualizing {num_samples} train + {num_samples} eval samples "
                f"per dataset ({n_eval_types} eval object types loaded)"
            )
            sample_motion, cond = create_sample_condition(
                config=self.config, data=self.data,
                num_samples=num_samples,
            )

            # -- Sample from the generative model --
            logger.info("Generating samples for visualization...")
            cond = self._to_device(cond)
            motion_shape = (
                sample_motion.size(0),
                self.config.dataset.max_joints,
                self.config.dataset.feature_len,
                self.config.dataset.max_motion_length,
            )

            samples = generate_samples(
                model=unwrapped_model,
                cond=cond,
                motion_shape=motion_shape,
                diff_model=self.config.training.diff_model,
                diffusion=self.diffusion,
                gen_diffusion=getattr(self, 'gen_diffusion', None),
                device=self.device,
                cfg_scale=cfg_scale,
            )

            # -- Save visualizations --
            visualize_and_save_motions(
                config=self.config, cond=cond,
                samples=samples, save_dir=save_dir, prefix=prefix,
            )
        finally:
            unwrapped_model.train()
            if use_ema:
                self.ema_model.restore(self._unwrapped_params())
            logger.info("Sample visualization completed.")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _unwrapped_params(self):
        """Return parameters from the unwrapped (non-DDP) model."""
        return self.accelerator.unwrap_model(self.model).parameters()

    def _to_device(self, cond: dict) -> dict:
        """Move all tensor values in a condition dict to ``self.device``."""
        return {
            k: v.to(self.device) if torch.is_tensor(v) else v
            for k, v in cond.items()
        }
