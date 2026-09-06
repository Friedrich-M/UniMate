"""Lightweight tracker for step-based or epoch-based training loops.

Provides a single object that tracks current step/epoch, determines when
training is finished, and formats progress descriptions for ``tqdm``.

Usage::

    tracker = TrainingTracker(max_steps=100_000, steps_per_epoch=500)
    while not tracker.training_finished():
        for batch in dataloader:
            ...
            tracker.step()
        tracker.next_epoch()
"""


class TrainingTracker:
    """Track training progress for step-based or epoch-based training.

    Exactly one of *max_steps* or *max_epochs* must be provided.

    Args:
        max_epochs: Stop after this many full passes through the dataset.
        max_steps: Stop after this many optimizer steps.
        steps_per_epoch: Number of batches per epoch (``len(dataloader)``).
            Used to estimate epochs from steps (or vice versa).
    """

    def __init__(self, max_epochs=None, max_steps=None, steps_per_epoch=None):
        if max_steps is None and max_epochs is None:
            raise ValueError("Either max_epochs or max_steps must be specified")
        if max_steps is not None and max_epochs is not None:
            raise ValueError("Only one of max_epochs or max_steps should be specified")

        self.steps_per_epoch = steps_per_epoch
        self.current_step = 0
        self.current_epoch = 0

        if max_steps is not None:
            self.mode = "steps"
            self.max_steps = max_steps
            self.max_epochs = None
            self.estimated_epochs = (
                (max_steps + steps_per_epoch - 1) // steps_per_epoch
                if steps_per_epoch else None
            )
        else:
            self.mode = "epochs"
            self.max_epochs = max_epochs
            self.max_steps = (
                max_epochs * steps_per_epoch if steps_per_epoch else None
            )
            self.estimated_epochs = max_epochs

    # -------------------------------------------------------------------
    # Progress
    # -------------------------------------------------------------------

    @property
    def total_steps(self):
        """Total number of optimizer steps (exact or estimated)."""
        if self.mode == "steps":
            return self.max_steps
        return self.max_epochs * self.steps_per_epoch

    def step(self):
        """Record one optimizer step."""
        self.current_step += 1

    def next_epoch(self):
        """Record the start of a new epoch."""
        self.current_epoch += 1

    def training_finished(self):
        """Return True when the training budget is exhausted."""
        if self.mode == "steps":
            return self.current_step >= self.max_steps
        return self.current_epoch >= self.max_epochs

    # -------------------------------------------------------------------
    # Display helpers
    # -------------------------------------------------------------------

    def get_progress_desc(self):
        """Return a short string for ``tqdm.set_description``."""
        if self.mode == "steps":
            return f"Step {self.current_step}/{self.max_steps} (Epoch {self.current_epoch})"
        return f"Epoch {self.current_epoch}/{self.max_epochs}"

    def get_duration_str(self):
        """Return a human-readable summary of the training duration."""
        spe = self.steps_per_epoch
        if self.mode == "steps":
            return f"Steps: {self.max_steps:,} ({spe} steps/epoch)"
        total = self.max_epochs * spe
        return f"Epochs: {self.max_epochs} ({spe} steps/epoch, {total:,} total steps)"
