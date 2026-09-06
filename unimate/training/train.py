"""Train a diffusion model on motions for any skeleton topology.

Entry point (usually via ``accelerate launch``, see scripts/)::

    python -m unimate.training.train --config configs/<config>.json

CLI overrides (via tyro): ``--output_dir``, ``--batch_size``, ``--num_workers``,
``--resume``.
"""

import os
import warnings
from dataclasses import dataclass
from typing import Optional

import torch
import tyro
from accelerate import Accelerator, DataLoaderConfiguration
from accelerate.utils import set_seed
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers.optimization import get_cosine_with_min_lr_schedule_with_warmup

from unimate.configs.schema import MainConfig
from unimate.dataset.factory import create_train_dataloader
from unimate.models.factory import create_model
from unimate.training.trainer import DiffusionTrainer
from unimate.training.tracker import TrainingTracker
from unimate.utils.logger import get_logger

warnings.filterwarnings("ignore")
logger = get_logger(file_name=__file__)


# ===================================================================
# CLI arguments
# ===================================================================

@dataclass
class TrainingArgs:
    """CLI arguments parsed by tyro (override config values)."""
    config: str
    output_dir: Optional[str] = None
    batch_size: Optional[int] = None
    num_workers: Optional[int] = None
    resume: Optional[str] = None  # path to checkpoint .pt file to resume from


# ===================================================================
# Training loop
# ===================================================================

def train_diffusion(args: TrainingArgs, config: MainConfig,
                    accelerator: Accelerator):
    """End-to-end training: setup, loop, checkpoint, visualize."""

    is_main = accelerator.is_main_process

    # ---- Output directories ----
    output_dir = args.output_dir or config.experiment.output_dir
    checkpoint_dir = os.path.join(output_dir, "checkpoints")
    log_dir = os.path.join(output_dir, "logs")
    debug_dir = os.path.join(output_dir, "debug")
    for d in [checkpoint_dir, log_dir, debug_dir]:
        os.makedirs(d, exist_ok=True)

    # ---- Reproducibility ----
    if config.training.seed is not None:
        # device_specific: seed += process_index. Without it every rank draws
        # the SAME flow-matching t and the same x0 noise (both come from the
        # global RNG), so an 8-GPU step sees one batch's worth of distinct
        # (t, noise) pairs instead of eight — and the same CFG caption-dropout
        # pattern on every rank. Model init may now differ per rank, which is
        # harmless: DDP broadcasts rank 0's parameters when it wraps the model.
        set_seed(config.training.seed, device_specific=True)
        logger.info(
            f"Set random seed to {config.training.seed} (device-specific)")

    # ---- Dataset ----
    logger.info("Creating dataset loader...")
    batch_size = args.batch_size or config.training.batch_size
    num_workers = args.num_workers or config.training.num_workers
    dataloader = create_train_dataloader(
        dataset_config=config.dataset,
        model_config=config.model,
        balanced=config.training.balanced,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    # ---- Persist config (after dataset so auto-computed max_joints /
    # max_depth are written; inference reads these back to rebuild the
    # exact same model architecture). ----
    if is_main:
        config_path = os.path.join(output_dir, "config.json")
        config.to_json(config_path)
        logger.info(f"Saved config to {config_path}")

        # Persist normalization stats so inference normalizes identically.
        stats_path = os.path.join(output_dir, "dataset_stats.npy")
        dataloader.dataset.save_dataset_stats(stats_path)

    # ---- Model ----
    logger.info("Creating model and diffusion...")
    model = create_model(
        dataset_config=config.dataset,
        model_config=config.model,
    )

    # ---- Trainer (wraps model + diffusion/flow for loss & visualization) ----
    trainer = DiffusionTrainer(
        config=config,
        model=model,
        data=dataloader,
        checkpoint_dir=checkpoint_dir,
    )

    # ---- Optimizer & LR schedule ----
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
        betas=(config.training.adam_beta1, config.training.adam_beta2),
    )
    lr_scheduler = get_cosine_with_min_lr_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(config.training.warmup_ratio * config.training.num_steps),
        num_training_steps=config.training.num_steps,
        min_lr_rate=config.training.min_lr_ratio,
        num_cycles=0.5,
    )

    # ---- Accelerate: wrap model, optimizer, dataloader for distributed ----
    # NOTE: lr_scheduler is NOT wrapped by accelerator. AcceleratedScheduler
    # with split_batches=False steps num_processes times per call to compensate
    # for data sharding, but num_training_steps already equals the target number
    # of optimizer steps, so wrapping would advance the schedule num_gpus× too fast.
    model, optimizer, dataloader = accelerator.prepare(
        model, optimizer, dataloader,
    )
    trainer.model = model  # point trainer at the wrapped model
    trainer.initialize_ema(accelerator)

    # ---- TensorBoard ----
    writer = SummaryWriter(log_dir) if is_main else None

    # ---- Training tracker ----
    tracker = TrainingTracker(
        max_epochs=config.training.num_epochs,
        max_steps=config.training.num_steps,
        steps_per_epoch=len(dataloader),
    )

    # ---- Resume from checkpoint (if requested) ----
    if args.resume:
        _resume_from_checkpoint(
            args.resume, accelerator, model, optimizer, lr_scheduler,
            trainer, tracker,
        )

    if is_main:
        logger.info(f"Training for {tracker.get_duration_str()}")

    # ---- Initial visualization (skip if resuming) ----
    if not args.resume:
        trainer.eval()
        if is_main:
            trainer.visualize_samples(
                save_dir=debug_dir,
                prefix="initial_before_training",
                num_samples=config.sampling.num_samples,
                cfg_scale=config.sampling.cfg_scale,
            )
            torch.cuda.empty_cache()

    trainer.train()

    # ---- Main training loop ----
    pbar = tqdm(
        range(tracker.total_steps),
        desc=tracker.get_progress_desc(),
        initial=tracker.current_step,
        disable=not is_main,
    )
    grad_norm = None

    while not tracker.training_finished():
        pbar.set_description(tracker.get_progress_desc())

        # Update sampler epoch so each epoch gets different random indices in DDP
        _set_sampler_epoch(dataloader, tracker.current_epoch)

        for batch in dataloader:
            with accelerator.accumulate(model):
                # Forward pass (mixed precision)
                with accelerator.autocast():
                    loss, loss_dict = trainer.compute_loss(batch)

                # NaN guard. The decision is reduced across ranks first, so
                # every rank takes the same branch — a rank-local skip would
                # desync DDP. A bad micro-batch is dropped by skipping its
                # backward: earlier micro-batches' gradients survive and are
                # all-reduced by the group's final backward as usual. Only
                # when the bad batch IS that final one is there no all-reduce,
                # and then the group's (rank-local) gradients must be thrown
                # away rather than stepped on. Backward on a detached zero
                # would raise 'does not require grad', so never do that.
                bad_local = (~torch.isfinite(loss)).to(torch.int32)
                skip_batch = accelerator.reduce(bad_local, reduction="max").item() == 1
                if skip_batch and is_main:
                    logger.warning(
                        f"[Global skip] non-finite loss at step {tracker.current_step}"
                    )

                # Backward + optimizer step
                if not skip_batch:
                    accelerator.backward(loss)

                if accelerator.sync_gradients and skip_batch:
                    # No all-reduce ran for this accumulation group.
                    optimizer.zero_grad(set_to_none=True)
                elif accelerator.sync_gradients:
                    if config.training.max_grad_norm is not None:
                        grad_norm = accelerator.clip_grad_norm_(
                            accelerator.unwrap_model(model).parameters(),
                            config.training.max_grad_norm,
                        )

                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()

                    # EMA update
                    if trainer.use_ema and trainer.ema_model is not None:
                        trainer.ema_model.step(
                            accelerator.unwrap_model(model).parameters()
                        )

                    pbar.update(1)
                    tracker.step()

                    # --- Per-effective-step side effects ---
                    step = tracker.current_step

                    # Progress bar postfix
                    if is_main:
                        pbar.set_postfix(
                            loss=f"{loss.item():.4f}",
                            lr=f"{lr_scheduler.get_last_lr()[0]:.6f}",
                        )

                    # Periodic logging
                    if step > 0 and step % config.training.log_interval == 0 and is_main:
                        _log_metrics(writer, trainer, lr_scheduler, loss_dict,
                                     grad_norm, config, step)

                    # Periodic checkpoint + visualization
                    if step > 0 and (step % config.training.save_interval == 0
                                     or tracker.training_finished()):
                        # Sync all ranks before checkpoint so rank 1+ don't timeout
                        # waiting during rank 0's lengthy ODE visualization.
                        accelerator.wait_for_everyone()
                        if is_main:
                            _save_checkpoint_and_visualize(
                                trainer, accelerator, model, optimizer, lr_scheduler,
                                tracker, config, checkpoint_dir, debug_dir,
                            )
                        accelerator.wait_for_everyone()

            # Sync all GPUs after the very first step
            if tracker.current_step == 1:
                accelerator.wait_for_everyone()

            if tracker.training_finished():
                break

        if not tracker.training_finished():
            tracker.next_epoch()

    # ---- Cleanup ----
    if is_main:
        writer.close()
        pbar.close()

    accelerator.wait_for_everyone()
    accelerator.end_training()
    logger.info("Training completed.")


# ===================================================================
# Helper functions
# ===================================================================

def _set_sampler_epoch(dataloader, epoch: int):
    """Call set_epoch on the underlying sampler if it supports it (for DDP).

    After accelerator.prepare(), the chain is:
      dataloader.batch_sampler → BatchSamplerShard
        .batch_sampler → original BatchSampler
          .sampler → MixtureSampler (or DistributedSampler)
    We walk the chain until we find a sampler with set_epoch.
    """
    # Walk batch_sampler chain (Accelerate's BatchSamplerShard wraps the original)
    bs = getattr(dataloader, 'batch_sampler', None)
    while bs is not None:
        sampler = getattr(bs, 'sampler', None)
        if sampler is not None and hasattr(sampler, 'set_epoch'):
            sampler.set_epoch(epoch)
            return
        # Accelerate's BatchSamplerShard stores the original as .batch_sampler
        bs = getattr(bs, 'batch_sampler', None)

    # Fallback: direct sampler attribute
    sampler = getattr(dataloader, 'sampler', None)
    if hasattr(sampler, 'set_epoch'):
        sampler.set_epoch(epoch)


def _log_metrics(writer, trainer, lr_scheduler, loss_dict, grad_norm,
                 config, step):
    """Write training scalars to TensorBoard."""
    for key, value in loss_dict.items():
        writer.add_scalar(f"train/{key}", value, step)

    if (trainer.use_ema and trainer.ema_model is not None
            and trainer.ema_model.cur_decay_value is not None):
        writer.add_scalar("train/ema_decay",
                          trainer.ema_model.cur_decay_value, step)

    writer.add_scalar("train/learning_rate",
                      lr_scheduler.get_last_lr()[0], step)

    if config.training.max_grad_norm is not None and grad_norm is not None:
        writer.add_scalar("train/grad_norm", grad_norm, step)


def _save_checkpoint_and_visualize(trainer, accelerator, model, optimizer,
                                   lr_scheduler, tracker, config,
                                   checkpoint_dir, debug_dir):
    """Generate sample visualizations and persist a checkpoint to disk."""
    trainer.visualize_samples(
        save_dir=debug_dir,
        prefix=f"step_{tracker.current_step}",
        num_samples=config.sampling.num_samples,
        cfg_scale=config.sampling.cfg_scale,
    )
    torch.cuda.empty_cache()

    checkpoint_dict = {
        "model_state_dict": accelerator.get_state_dict(model),
        "optimizer_state_dict": optimizer.state_dict(),
        "lr_scheduler_state_dict": lr_scheduler.state_dict(),
        "step": tracker.current_step,
        "epoch": tracker.current_epoch,
    }
    if trainer.use_ema and trainer.ema_model is not None:
        checkpoint_dict["ema_state_dict"] = trainer.ema_model.state_dict()

    checkpoint_path = os.path.join(
        checkpoint_dir, f"checkpoint_step_{tracker.current_step}.pt"
    )
    accelerator.save(checkpoint_dict, checkpoint_path)
    logger.info(f"Saved checkpoint to {checkpoint_path}")


def _resume_from_checkpoint(
    resume_path, accelerator, model, optimizer, lr_scheduler, trainer, tracker,
):
    """Restore all training state from a checkpoint file.

    Loads model weights, optimizer, lr_scheduler, EMA, and fast-forwards the
    tracker to the saved step/epoch so training continues seamlessly.
    """
    logger.info(f"Resuming from checkpoint: {resume_path}")
    checkpoint = torch.load(resume_path, map_location="cpu")

    # Model weights
    unwrapped = accelerator.unwrap_model(model)
    unwrapped.load_state_dict(checkpoint["model_state_dict"])

    # Optimizer
    if "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    # LR scheduler
    if "lr_scheduler_state_dict" in checkpoint:
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler_state_dict"])
    else:
        # Older checkpoint without scheduler state — fast-forward manually
        for _ in range(checkpoint.get("step", 0)):
            lr_scheduler.step()

    # EMA
    if trainer.use_ema and trainer.ema_model is not None:
        if "ema_state_dict" in checkpoint:
            trainer.ema_model.load_state_dict(checkpoint["ema_state_dict"])
            trainer.ema_model.to(accelerator.device)
            logger.info("Restored EMA state from checkpoint.")
        else:
            # No EMA state in checkpoint — rebuild EMA from the just-loaded
            # model weights so it doesn't lag behind with the pre-resume random init.
            trainer.initialize_ema(accelerator)
            logger.warning("No EMA state in checkpoint — EMA re-initialized from loaded model weights.")

    # Tracker
    resumed_step = checkpoint.get("step", 0)
    resumed_epoch = checkpoint.get("epoch", 0)
    tracker.current_step = resumed_step
    tracker.current_epoch = resumed_epoch

    logger.info(f"Resumed at step={resumed_step}, epoch={resumed_epoch}")


# ===================================================================
# Entry point
# ===================================================================

def main(args: TrainingArgs):
    """Parse config and launch training."""
    config = MainConfig.from_json(args.config)

    accelerator = Accelerator(
        dataloader_config=DataLoaderConfiguration(
            use_seedable_sampler=False,
            dispatch_batches=False,
        ),
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
    )

    if accelerator.is_main_process:
        logger.info("Training Pipeline")
        logger.info(f"  Config: {args.config}")
        logger.info(f"  Mixed Precision: {accelerator.mixed_precision}")
        logger.info(f"  Gradient Accumulation Steps: {accelerator.gradient_accumulation_steps}")

    train_diffusion(args, config, accelerator)


if __name__ == "__main__":
    args = tyro.cli(TrainingArgs)
    main(args)
