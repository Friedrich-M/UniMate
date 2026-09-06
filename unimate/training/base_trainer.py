"""Base trainer class providing a loss interface.

Subclasses must implement :meth:`compute_loss`.  Checkpoint I/O is handled
directly via ``accelerator.save`` in the training loop for distributed
compatibility.
"""

from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn


class BaseTrainer(ABC, nn.Module):
    """Abstract base trainer wrapping a generative model.

    Inherits ``nn.Module`` so that subclasses can register buffers or
    sub-modules (e.g. an EMA model) if needed.
    """

    def __init__(self, checkpoint_dir: Optional[str] = None):
        super().__init__()
        self.checkpoint_dir = checkpoint_dir

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def compute_loss(
        self, batch: Dict[str, torch.Tensor], **kwargs
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute training loss.

        Returns:
            (total_loss, loss_dict) — scalar for backprop and a dict of
            named component values (floats) for logging.
        """

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def get_num_parameters(self) -> int:
        """Total number of trainable parameters in the wrapped model.

        Subclasses that store the model as an attribute (rather than as
        a registered sub-module) should override this.
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
