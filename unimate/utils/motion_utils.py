"""Construction and recovery of the UniMate (12-dim) motion representation.

Per joint: ``RIFKE_pos(3) | rotation_6d(6) | local_velocity(3)``. Rotations
are T-pose-relative (see :func:`compute_rots_from_tpos`) and HML-reordered
(see :func:`compute_cont6d_params`). The root slot uses a special encoding:
position is ``[0, Y_height, 0]``, the 6D channels store the facing-direction
quaternion, and velocity is ``[XZ_vel_x, Y_vel, XZ_vel_z]`` (the XZ
trajectory is recovered via cumulative integration).
"""

import numpy as np

from Quaternions import Quaternions

from unimate.utils.rotation_conversions import rotation_6d_to_matrix_np


# ===========================================================================
# UniMate 12-dim
# ===========================================================================
#
# Layout: ``[RIFKE_pos(3) | rot6d(6) | local_velocity(3)]`` per joint.
# Root is special (see ``compute_cont6d_params`` / ``compute_rifke``):
#   pos = ``[0, Y_height, 0]``,  rot6d = facing quat,
#   vel = ``[XZ_vel_x, Y_vel, XZ_vel_z]``.
#
# Sections below:
#   1. Forward feature construction
#   2. Clip alignment
#   3. HML ↔ BVH and Animation FK helpers (shared by recovery / augmentation)
#   4. Recovery from motion features
# ---------------------------------------------------------------------------
# 1. Forward feature construction
# ---------------------------------------------------------------------------

def compute_rots_from_tpos(
    tpos_quats: Quaternions,
    anim_quats: Quaternions,
    parents: np.ndarray,
) -> Quaternions:
    """Re-express animation rotations relative to a canonical T-pose.

    For every joint, the transformed rotation represents the delta from the
    T-pose orientation, conjugated into the parent's cumulative T-pose frame.
    This makes rotations skeleton-agnostic (invariant to rest-pose choice).

    Exported clip NPZs store raw BVH local rotations (see
    ``data_process/utils/motion_features.py::save_clip``); the dataset loader
    applies this rebase before feature extraction so the model always sees
    T-pose-relative rotations.

    Args:
        tpos_quats: T-pose rotations (F, J) as a Quaternions array.
        anim_quats: Animation rotations (F, J) as a Quaternions array.
        parents: Parent indices (J,).

    Returns:
        New (F, J) Quaternions expressed relative to *tpos_quats*.
    """
    new_rots = anim_quats.copy()
    # Root: cancel the T-pose root orientation from the animation root.
    new_rots[:, 0] = new_rots[:, 0] * -tpos_quats[:, 0]

    # Non-root joints: propagate cumulative T-pose rotations down the chain
    # and re-express each joint rotation in the parent's cumulative frame.
    cum_rots = tpos_quats.copy()
    for j, p in enumerate(parents[1:], start=1):
        cum_rots[:, j] = cum_rots[:, p] * tpos_quats[:, j]
        new_rots[:, j] = (
            cum_rots[:, p] * anim_quats[:, j] * -tpos_quats[:, j] * -cum_rots[:, p]
        )

    return new_rots


def compute_cont6d_params(global_positions, local_rotations, parents,
                          root_facing_quat):
    """Extract continuous 6D rotation parameters with HML joint reordering.

    BVH convention stores parent-to-child rotation at the parent joint.
    HML reordering shifts each child's rotation to the child index, so
    joint j stores the rotation that was at its parent. The root stores
    the facing-direction quaternion in 6D form instead.

    Also computes root linear velocity (in facing frame) and root angular
    velocity as quaternion differences.

    Args:
        global_positions: Global joint positions (T, J, 3).
        local_rotations: Parent-relative joint rotations (T, J) as Quaternions.
        parents: Parent joint indices (J,).
        root_facing_quat: Per-frame root facing quaternion (T,) as Quaternions.

    Returns:
        Tuple of (cont_6d_reordered (T, J, 6), r_velocity (T-1, 4),
        velocity (T-1, 3)).
    """
    cont_6d_params = local_rotations.copy().rotation_matrix(cont6d=True)  # (T, J, 6)

    # Reorder: each joint stores its own rotation (parent -> self)
    cont_6d_params_reordered = np.zeros_like(cont_6d_params)
    for j, p in enumerate(parents[1:], 1):
        cont_6d_params_reordered[:, j] = cont_6d_params[:, p]
    cont_6d_params_reordered[:, 0] = root_facing_quat.copy().rotation_matrix(cont6d=True)

    # Root linear velocity (T-1, 3)
    velocity = (global_positions[1:, 0] - global_positions[:-1, 0]).copy()
    velocity = root_facing_quat[1:] * velocity  # rotate into root-facing frame

    # Root angular velocity (T-1, 4)
    r_velocity = root_facing_quat[1:] * -root_facing_quat[:-1]

    return cont_6d_params_reordered, r_velocity, velocity


def compute_rifke(global_positions, root_facing_quat):
    """Compute Root-Invariant Forward Kinematics (RIFKE) positions.

    Subtracts root XZ from all joints (making root XZ always zero), then
    rotates into the root-facing frame. Y (height) is kept as absolute value.

    Args:
        global_positions: Global joint positions (T, J, 3).
        root_facing_quat: Per-frame root facing quaternion (T,) as Quaternions.

    Returns:
        RIFKE positions (T, J, 3).
    """
    positions = global_positions.copy()
    positions[..., 0] -= positions[:, 0:1, 0]
    positions[..., 2] -= positions[:, 0:1, 2]
    positions = np.repeat(root_facing_quat[:, None], positions.shape[1], axis=1) * positions
    return positions


def compute_local_velocity(global_positions, root_facing_quat):
    """Compute per-joint local velocity in the root-facing frame.

    For each frame, the global position difference is rotated into the
    root coordinate system of the *destination* frame.

    Args:
        global_positions: Global joint positions (T, J, 3).
        root_facing_quat: Per-frame root facing quaternion (T,) as Quaternions.

    Returns:
        Local velocity (T-1, J, 3).
    """
    velocity = global_positions[1:] - global_positions[:-1]
    velocity = np.repeat(root_facing_quat[1:, None], velocity.shape[1], axis=1) * velocity
    return velocity


def compute_unimate_motion_feats(
    global_positions: np.ndarray,
    local_rotations: Quaternions,
    parents: np.ndarray,
    root_facing_quat: Quaternions,
):
    """Extract 12-dim UniMate motion features.

    Per joint: RIFKE_pos(3) | rotation_6d(6) | local_velocity(3).

    Args:
        global_positions: Global joint positions (T, J, 3).
        local_rotations: Parent-relative joint rotations (T, J) as Quaternions.
        parents: Parent joint indices (J,).
        root_facing_quat: Per-frame root facing quaternion (T,) as Quaternions.

    Returns:
        Motion features (F-1, J, 12).
    """
    cont_6d_params, _, _ = compute_cont6d_params(
        global_positions, local_rotations, parents, root_facing_quat)

    positions = compute_rifke(global_positions, root_facing_quat)  # (T, J, 3)
    local_vel = compute_local_velocity(global_positions, root_facing_quat)  # (T-1, J, 3)

    return np.concatenate([positions[:-1], cont_6d_params[:-1], local_vel], axis=-1)  # (T-1, J, 12)


# ---------------------------------------------------------------------------
# 2. Clip alignment
# ---------------------------------------------------------------------------

def realign_unimate_clip(clip_feats, parents):
    """Re-align a clip so the root facing at frame 0 is identity (faces Z+).

    The root facing quaternion (joint 0, channels 3:9) is the rotation that
    maps the current facing direction to Z+, not the facing direction itself.
    To make frame 0 identity, we right-multiply by the inverse of the
    first-frame facing quat: ``q_t * q_0^{-1}``.

    Direct children of the root store the root's local rotation (HML
    reordering). These need the facing rotation applied: ``q_0 * q_child``.

    RIFKE positions and velocities are per-frame root-relative and invariant
    to global Y rotation, so only the 6D rotation channels need adjustment.

    Args:
        clip_feats: Clip motion features (T, J, 12).
        parents: Parent index list (J,).

    Returns:
        Re-aligned clip features (T, J, 12).
    """
    clip = clip_feats.copy()

    # Decode first-frame root facing quaternion
    q0 = Quaternions.from_transforms(
        rotation_6d_to_matrix_np(clip[0, 0, 3:9][None]))[0]

    # Root facing (joint 0): make relative to first frame
    root_6d = clip[:, 0, 3:9]
    q_root = Quaternions.from_transforms(rotation_6d_to_matrix_np(root_6d))
    clip[:, 0, 3:9] = (q_root * -q0).rotation_matrix(cont6d=True)

    # Direct children of root: apply facing rotation to root's local rotation
    for j, p in enumerate(parents):
        if p == 0:
            child_6d = clip[:, j, 3:9]
            q_child = Quaternions.from_transforms(rotation_6d_to_matrix_np(child_6d))
            clip[:, j, 3:9] = (q0 * q_child).rotation_matrix(cont6d=True)

    return clip


# ---------------------------------------------------------------------------
# 3. HML ↔ BVH and Animation FK helpers
# ---------------------------------------------------------------------------

def hml_rotations_to_bvh_quaternions(motion_rot6d_hml, parents):
    """Invert the HML joint re-ordering back to BVH-ordered rotations.

    HML convention (see :func:`compute_cont6d_params`):
    ``hml[j] = bvh[parents[j]]`` for ``j >= 1``. Slot 0 holds the root
    facing quaternion — it's ignored here, since the root's global rotation
    is recovered from any child-of-root slot (which stores ``bvh[0]``).

    Args:
        motion_rot6d_hml: 6D rotations in HML order (T, J, 6).
        parents: Parent indices (J,).

    Returns:
        BVH-ordered rotations as ``Quaternions`` (T, J).
    """
    T, J, _ = motion_rot6d_hml.shape
    bvh_rot_mat = np.tile(np.eye(3), (T, J, 1, 1))  # identity defaults
    hml_mat = rotation_6d_to_matrix_np(motion_rot6d_hml)
    for j, p in enumerate(parents[1:], 1):
        bvh_rot_mat[:, p] = hml_mat[:, j]
    return Quaternions.from_transforms(bvh_rot_mat)


def fk_global_positions(rotations_bvh, parents, offsets, root_positions):
    """Run FK via the Animation library with BVH-ordered rotations.

    Mirrors the FK call used during feature extraction (``positions_global``
    in ``data_process/utils/motion_features.py``), so positions produced here
    match the training data exactly.

    Args:
        rotations_bvh: Quaternions (T, J) in BVH order.
        parents: (J,) parent indices.
        offsets: (J, 3) rest-pose bone offsets.
        root_positions: (T, 3) per-frame root world positions. Pass zeros to
            obtain positions with the root pinned at the origin.

    Returns:
        Global joint positions (T, J, 3).
    """
    from Animation import Animation, positions_global
    T = rotations_bvh.shape[0]
    anim_positions = offsets[None].repeat(T, axis=0)  # (T, J, 3)
    anim_positions[:, 0] = root_positions
    anim = Animation(
        rotations=rotations_bvh,
        positions=anim_positions,
        offsets=offsets,
        parents=parents,
        orients=Quaternions.id(0),
    )
    return positions_global(anim)


# ---------------------------------------------------------------------------
# 4. Recovery from motion features
# ---------------------------------------------------------------------------

def recover_unimate_root_quat_and_pos(data):
    """Recover root quaternion and global root position from 12-dim root features.

    Root rotation is decoded from the 6D channels [3:9].
    XZ trajectory is recovered by cumulatively integrating velocity channels
    [9] and [11] (rotated back to world frame). Y is read directly from
    the absolute height channel [1].

    Args:
        data: Root joint features, shape (..., T, 12).

    Returns:
        (r_rot_quat (..., T), r_pos (..., T, 3)).
    """
    r_rot_quat = Quaternions.from_transforms(rotation_6d_to_matrix_np(data[:, 3:9]))

    r_pos = np.zeros(data.shape[:-1] + (3,))  # (T, 3)
    r_pos[..., 1:, [0, 2]] = data[..., :-1, [9, 11]]  # XZ velocity
    r_pos = -r_rot_quat * r_pos  # Rotate back to world frame
    r_pos = np.cumsum(r_pos, axis=-2)
    r_pos[..., 1] = data[..., 1]

    return r_rot_quat, r_pos


def recover_unimate_joint_pos_from_ric(data):
    """Recover global joint positions from 12-dim features using RIFKE positions.

    Inverts compute_rifke: un-rotates joint positions from the facing frame back to
    world frame, then adds back the root XZ trajectory.

    Args:
        data: 12-dim features, shape (..., J, 12).

    Returns:
        Global positions, shape (..., J, 3).
    """
    r_rot_quat, r_pos = recover_unimate_root_quat_and_pos(data[..., 0, :])

    positions = data[..., 1:, :3]
    positions = np.repeat(-r_rot_quat[..., None, :], positions.shape[-2], axis=-2) * positions

    positions[..., 0] += r_pos[..., 0:1]
    positions[..., 2] += r_pos[..., 2:3]

    positions = np.concatenate([r_pos[..., np.newaxis, :], positions], axis=-2)
    return positions


def recover_unimate_joint_pos_from_rot(data, parents, offsets):
    """Recover global positions via FK from 12-dim rotation channels.

    Reverses HML joint reordering, integrates the root trajectory from the
    velocity channels, and runs FK via the Animation library.

    Args:
        data: 12-dim features (T, J, 12).
        parents: Parent indices (J,).
        offsets: Rest-pose bone offsets (J, 3).

    Returns:
        Global joint positions (T, J, 3).
    """
    _, r_pos = recover_unimate_root_quat_and_pos(data[:, 0])
    bvh_rots = hml_rotations_to_bvh_quaternions(data[:, :, 3:9], parents)
    return fk_global_positions(bvh_rots, parents, offsets, r_pos)


def recover_unimate_anim_from_rot(data, parents, offsets):
    """Recover an Animation object from 12-dim rotation channels.

    Similar to ``recover_unimate_joint_pos_from_rot`` but returns the decoded
    ``Animation`` (BVH-ordered local rotations, integrated root trajectory,
    facing quaternions as orients) instead of running FK to positions — used
    to re-apply generated motion to rigged characters.

    Args:
        data: 12-dim features (T, J, 12).
        parents: Parent indices (J,).
        offsets: Rest-pose bone offsets (J, 3).

    Returns:
        The recovered ``Animation`` object.
    """
    nframes, njoints = data.shape[:2]
    r_rot_quat, r_pos = recover_unimate_root_quat_and_pos(data[:, 0])
    bvh_rots = hml_rotations_to_bvh_quaternions(data[:, :, 3:9], parents)

    anim_positions = offsets[None].repeat(nframes, axis=0)  # (T, J, 3)
    anim_positions[:, 0] = r_pos

    from Animation import Animation
    anim = Animation(
        rotations=bvh_rots,
        positions=anim_positions,
        offsets=offsets,
        parents=parents,
        orients=r_rot_quat,
    )
    return anim
