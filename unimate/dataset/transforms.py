"""Shared per-sample processing for training and inference.

Cropping, condition extraction, normalization, padding, and parent-feature
construction used by both ``mixture/dataset.py`` (training samples) and
``conditioning.py`` (sampling / inference batches).
"""

import random

import numpy as np


# ---------------------------------------------------------------------------
# Cropping
# ---------------------------------------------------------------------------

def apply_cropping(
    augmented_motion, topology_condition_type, max_motion_length, start_idx=None,
):
    """Crop augmented_motion before condition extraction.

    For 'first_frame' the first frame is split off as the condition, so the
    crop length is ``max_motion_length + 1``. For 'tpos' the full sequence is
    the motion, so the crop length is ``max_motion_length``.

    Start-index policy:
      * ``start_idx`` provided: slice ``[start_idx, start_idx + target_len]``
        verbatim. When fewer than ``target_len`` frames remain after
        ``start_idx`` the returned tensor is shorter — downstream
        ``apply_padding`` zero-fills the tail, and the caller's
        valid-length signal trims the saved output back.
      * ``start_idx is None``: 'first_frame' uses 0 (continuity with the
        global first frame); 'tpos' picks a uniform random start.

    Returns:
        (cropped_motion, start_idx)
    """
    n_frames = augmented_motion.shape[0]

    if topology_condition_type == 'first_frame':
        target_len = max_motion_length + 1  # +1 for the first-frame condition
    else:  # 'tpos'
        target_len = max_motion_length

    if start_idx is not None:
        if start_idx < 0:
            raise ValueError(f"start_idx must be >= 0, got {start_idx}.")
        if start_idx >= n_frames:
            raise ValueError(
                f"start_idx={start_idx} is >= clip length {n_frames}."
            )
        return augmented_motion[start_idx:start_idx + target_len], start_idx

    start_idx = 0
    if n_frames > target_len:
        if topology_condition_type == 'first_frame':
            start_idx = 0
        else:  # 'tpos'
            start_idx = random.randint(0, n_frames - target_len)
        augmented_motion = augmented_motion[start_idx:start_idx + target_len]

    return augmented_motion, start_idx


# ---------------------------------------------------------------------------
# Condition extraction
# ---------------------------------------------------------------------------

def extract_conditions(augmented_motion, tpos, topology_condition_type):
    """Split augmented_motion into motion and condition arrays.

    Returns a dict with keys:
        - ``motion``: the motion frames (first frame removed for 'first_frame')
        - ``tpos_first_frame``: T-pose or first-frame condition (J, D)
    """
    if topology_condition_type == 'first_frame':
        return {
            'tpos_first_frame': augmented_motion[0].copy(),
            'motion': augmented_motion[1:].copy(),
        }
    elif topology_condition_type == 'tpos':
        return {
            'tpos_first_frame': tpos.copy(),
            'motion': augmented_motion.copy(),
        }
    else:
        raise ValueError(f"Invalid topology_condition_type: {topology_condition_type}")


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

def apply_normalization(motion, mean, std):
    """Normalize motion and replace NaN with zero.

    Assumes ``std`` is strictly positive (enforced upstream by a variance
    floor in ``_calculate_dataset_stats``), so no epsilon is added here —
    keeping the forward op exact lets denormalization invert it cleanly.
    """
    if motion.ndim == 3:
        motion = (motion - mean[None, :]) / std[None, :]
    elif motion.ndim == 2:
        motion = (motion - mean) / std
    else:
        raise ValueError("Motion data must be 2D or 3D.")
    return np.nan_to_num(motion)


# ---------------------------------------------------------------------------
# Padding
# ---------------------------------------------------------------------------

def apply_padding(motion, max_motion_length):
    """Pad motion to max_motion_length if shorter.

    Returns:
        (padded_motion, original_motion_length)
    """
    motion_length = motion.shape[0]
    if motion_length < max_motion_length:
        pad = np.zeros((max_motion_length - motion_length, *motion.shape[1:]))
        motion = np.concatenate([motion, pad], axis=0)
    return motion, motion_length


# ---------------------------------------------------------------------------
# Parent-feature maps
# ---------------------------------------------------------------------------

def build_parent_features(tpos_first_frame, parents):
    """Compute parent-copied feature array for tpos_first_frame.

    For each joint j with parent p (!= -1), copies the parent's features.

    Returns a dict with:
        - ``tpos_first_frame_parents``: (J, D)
    """
    tpos_first_frame_parents = tpos_first_frame.copy()
    for j_idx, p_idx in enumerate(parents):
        if p_idx != -1:
            tpos_first_frame_parents[j_idx] = tpos_first_frame[p_idx]

    return {'tpos_first_frame_parents': tpos_first_frame_parents}
