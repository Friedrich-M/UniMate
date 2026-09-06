"""Visualization utilities for generated motion samples."""

import os

import numpy as np
import torch
from typing import Dict, List, Optional
from PIL import Image

from unimate.configs.schema import MainConfig
from unimate.utils.motion_utils import (
    recover_unimate_joint_pos_from_ric,
    recover_unimate_joint_pos_from_rot,
)
# Skeleton rendering lives in the (public, self-contained) data_process
# package; unimate/ may depend on data_process but never the other way around.
from data_process.utils.plotting import (
    save_skeleton_motion as plot_motion,
    save_skeleton_motion_spectral as plot_motion_spectral,
    render_skeleton_tpose,
    render_skeleton_tpos_spectral,
)
from unimate.utils.logger import get_logger

logger = get_logger(file_name=__file__)

FPS = 30


def visualize_and_save_motions(
    config: MainConfig,
    cond: Dict[str, torch.Tensor],
    samples: torch.Tensor,
    save_dir: str,
    prefix: str,
    case_ids: Optional[List[str]] = None,
    only_save_motion: bool = False,
    save_ric: bool = True,
    valid_lengths: Optional[torch.Tensor] = None,
):
    """Visualize and save generated motion samples as MP4 videos and T-pose images.

    Args:
        config: Main training/inference config.
        cond: Conditioning dict from the dataloader (tensors).
        samples: Generated motion samples (B, J, D, T).
        save_dir: Output directory for saved files.
        prefix: Filename prefix for saved files.
        case_ids: Optional per-sample identifiers (e.g. test-case keys). When
            given, output files use them as the per-sample suffix and the
            ``[train]/[eval]`` split tag is suppressed in titles/filenames.
        only_save_motion: If True, only the denormalized motion-feature
            ``.npy`` is written under ``<save_dir>/motions/``; the FK / RIC
            mp4 renders, the T-pose PNG, and the position recoveries that
            feed them are all skipped.
        save_ric: If False, skip the RIC-recovered ``_ric.mp4`` render (FK
            and T-pose are still saved). Inference defaults to False.
        valid_lengths: optional ``(B,)`` tensor of per-sample valid frame
            counts. When provided, the saved motion (and downstream FK / RIC
            renders) is trimmed to the per-sample valid length instead of
            keeping the padded ``T = max_motion_length`` window — useful for
            saving GT clips whose true length is shorter than the padded
            ODE window.
    """
    motions_dir = os.path.join(save_dir, 'motions')
    os.makedirs(motions_dir, exist_ok=True)
    if not only_save_motion:
        animations_dir = os.path.join(save_dir, 'animations')
        os.makedirs(animations_dir, exist_ok=True)

    for object_idx, sample_motion in enumerate(samples):
        object_type = cond["object_type"][object_idx]
        n_joints = cond["n_joints"][object_idx].item()

        motion = sample_motion[:n_joints].cpu().permute(2, 0, 1).numpy().copy()  # (F, J, D)
        if valid_lengths is not None:
            valid_T = int(valid_lengths[object_idx].item())
            motion = motion[:valid_T]
        tpos_first_frame = cond["tpos_first_frame"][object_idx][:n_joints][None, ...].cpu().numpy().copy()

        mean = cond["mean"][object_idx][:n_joints][None, ...].cpu().numpy().copy()
        std = cond["std"][object_idx][:n_joints][None, ...].cpu().numpy().copy()
        motion = motion * std + mean
        tpos_first_frame = tpos_first_frame * std + mean

        parents = cond["parents"][object_idx]
        offsets = cond["offsets"][object_idx][:n_joints].cpu().numpy().copy()

        # None falls back to the red/blue joint coloring in plotting.py.
        spectral_feats = None
        if "spectral_feats" in cond and cond["spectral_feats"] is not None:
            spectral_feats = (
                cond["spectral_feats"][object_idx][:n_joints].cpu().numpy().copy()
            )

        recover_gl_pos = None
        recover_gl_pos_fk = None
        if not only_save_motion:
            if save_ric:
                recover_gl_pos = recover_unimate_joint_pos_from_ric(motion)
            recover_gl_pos_fk = recover_unimate_joint_pos_from_rot(motion, parents, offsets)

        case_id = case_ids[object_idx] if case_ids is not None else None

        # case_ids encode user-supplied test setups; train/eval origin
        # is meaningless there, so suppress it in titles + filenames.
        if case_id is not None:
            split_tag = None
        else:
            split_tags = cond.get("split_tag")
            split_tag = split_tags[object_idx] if split_tags is not None else None

        title = case_id if case_id else object_type
        if split_tag:
            title += f" [{split_tag}]"

        if config.model.cond_mode == 'text':
            caption = cond.get("caption", [None] * len(samples))
            caption_text = caption[object_idx] if caption[object_idx] else None
            if caption_text:
                title += f"\n{caption_text}"

        # .npy filenames follow the training-data convention
        # ``{object_type}-{motion}[-{clip}]`` so downstream tools (e.g. the
        # mesh-animation stage in ``data_process/``) can recover the
        # object_type by splitting on the first '-'. All separators are '-'
        # (never '_') to stay aligned with that template.
        # ``prefix`` encodes the training iteration and ``object_idx`` is
        # unique within a batch, so ``{lead}-{prefix}-{object_idx}`` is
        # unique per (object_type, iteration, sample).
        split_suffix = f'-{split_tag}' if split_tag else ''
        if case_id is not None:
            npy_lead = case_id
            base_name = f'{case_id}-{prefix}-sample{object_idx}'
        else:
            npy_lead = f'{object_type}{split_suffix}'
            base_name = f'{prefix}-{object_type}{split_suffix}-{object_idx}'

        motion_npy_name = f'{npy_lead}-{prefix}-{object_idx}'

        motion_npy_path = os.path.join(motions_dir, f'{motion_npy_name}.npy')
        np.save(motion_npy_path, motion)
        logger.info(f"Saved motion features at [{motion_npy_path}]")

        if only_save_motion:
            continue

        tag_str = f", {case_id}" if case_id else (f", {split_tag}" if split_tag else "")

        if save_ric:
            save_path = os.path.join(animations_dir, f'{base_name}_ric.mp4')
            if spectral_feats is not None:
                plot_motion_spectral(
                    save_path=save_path, parents=parents, positions=recover_gl_pos,
                    spectral_feats=spectral_feats, fps=FPS, title=title,
                    figsize=(6.4, 6.4), dpi=100
                )
            else:
                plot_motion(
                    save_path=save_path, parents=parents,
                    positions=recover_gl_pos, fps=FPS, title=title,
                    figsize=(6.4, 6.4), dpi=100
                )
            logger.info(f"Saved sample for object [{object_type}, {object_idx}{tag_str}] at [{save_path}]")

        fk_save_path = os.path.join(animations_dir, f'{base_name}_fk.mp4')
        if spectral_feats is not None:
            plot_motion_spectral(
                save_path=fk_save_path, parents=parents, positions=recover_gl_pos_fk,
                spectral_feats=spectral_feats, fps=FPS, title=title,
                figsize=(6.4, 6.4), dpi=100
            )
        else:
            plot_motion(
                save_path=fk_save_path, parents=parents,
                positions=recover_gl_pos_fk, fps=FPS, title=title,
                figsize=(6.4, 6.4), dpi=100
            )
        logger.info(f"Saved FK results for object [{object_type}, {object_idx}{tag_str}] at [{fk_save_path}]")

        # T-pose is a per-object_type property — render it once per type
        # (skip when the stable-named file already exists from a prior call).
        tpos_save_path = os.path.join(animations_dir, f'{object_type}_tpos.png')
        if os.path.exists(tpos_save_path):
            continue
        # tpos is (1, J, 12) after feature-dim padding; positions live in [:3].
        recover_tpos_gl_pos = tpos_first_frame[0, :, :3]
        if spectral_feats is not None:
            vis_tpos = render_skeleton_tpos_spectral(
                parents=parents, positions=recover_tpos_gl_pos,
                spectral_feats=spectral_feats, title=object_type,
                figsize=(6.4, 6.4), dpi=100
            )
        else:
            vis_tpos = render_skeleton_tpose(
                parents=parents, positions=recover_tpos_gl_pos, title=object_type,
                figsize=(6.4, 6.4), dpi=100
            )
        Image.fromarray(vis_tpos).save(tpos_save_path)
        logger.info(f"Saved tpose visualization at [{tpos_save_path}]")
