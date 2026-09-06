"""Skeleton- and joint-level augmentations for the Mixture dataset.

Pure functions mutating (and returning) a standard ``aug`` dict; ``max_freqs``
is passed explicitly, no class state is required. Every aug function leaves
the aug dict in a valid state (all derived arrays recomputed) ready for the
next stage. All ops assume the UniMate 12-dim representation (HML-reordered
rotation slots, identity rest rotations — see ``unimate/utils/motion_utils.py``).

Standard aug-dict keys: ``motion``, ``parents``, ``edge_indexs``,
``joint_graph_dist``, ``joint_relations``, ``joint_depths``, ``spectral_feats``,
``tpos``, ``offsets``, ``joint_names_emb``, ``mean``, ``std``.
"""

import random

import numpy as np

from Quaternions import Quaternions

from unimate.utils.topology_utils import (
    compute_edge_relations_and_distances,
    compute_joint_depths,
    compute_edge_indexs,
    compute_laplacian_eigenvectors,
)
from unimate.utils.rotation_conversions import rotation_6d_to_matrix_np
from unimate.utils.motion_utils import (
    compute_rifke,
    hml_rotations_to_bvh_quaternions,
    fk_global_positions,
)


# ---------------------------------------------------------------------------
# Low-level skeleton helpers
# ---------------------------------------------------------------------------

def build_children_map(parents):
    """Build a list-of-lists mapping each joint to its children."""
    children = [[] for _ in range(len(parents))]
    for c, p in enumerate(parents):
        if p >= 0 and c != p:
            children[p].append(c)
    return children


def bone_length_weighted_sample(candidates, offsets, n_select, alpha=0.5):
    """Sample joints weighted by inverse bone length (shorter = more likely).

    Per-joint weight ∝ bone_length^(-alpha); ``alpha=0`` uniform,
    ``alpha=1`` proportional to ``1/length``.
    """
    bone_lengths = np.array([np.linalg.norm(offsets[j]) for j in candidates])
    bone_lengths = np.maximum(bone_lengths, 1e-8)
    weights = bone_lengths ** (-alpha)
    weights /= weights.sum()
    n_select = min(n_select, len(candidates))
    selected = np.random.choice(
        candidates, size=n_select, replace=False, p=weights,
    ).tolist()
    return selected


def delete_joint(aug, j):
    """Delete joint *j* from all per-joint arrays and fix parent indices."""
    aug['motion'] = np.delete(aug['motion'], j, axis=1)
    aug['parents'] = np.delete(aug['parents'], j, axis=0)
    aug['parents'][aug['parents'] > j] -= 1
    for key in ('tpos', 'offsets', 'joint_names_emb', 'joint_depths', 'mean', 'std'):
        aug[key] = np.delete(aug[key], j, axis=0)


def collapse_single_child_joint(aug, j, c, p):
    """Collapse a single-child interior joint *j* into its child *c*.

    Merges offsets and rotation channels, then re-parents *c* to grandparent
    *p*. In the HML layout, joint k's slot stores ``bvh[parents[k]]``: after
    collapse, c's parent becomes p, so ``motion[c, 3:9]`` must hold
    ``bvh[p]`` — which is exactly what ``motion[j, 3:9]`` stored pre-collapse.
    Grandchildren's slots hold ``bvh[c]`` which must remain ``bvh[c]_old``
    for FK below c to be preserved, so they are left untouched. Offsets are
    summed in the rest frame (identity rest rotations).
    """
    aug['motion'][:, c, 3:9] = aug['motion'][:, j, 3:9]
    aug['offsets'][c] = aug['offsets'][j] + aug['offsets'][c]
    aug['parents'][c] = p


# ---------------------------------------------------------------------------
# Derived-array recomputation
# ---------------------------------------------------------------------------

def _augmented_global_positions(aug):
    """Run FK on the augmented skeleton via :func:`fk_global_positions`.

    Root is pinned at ``(0, Y_height, 0)`` where ``Y_height`` is read from
    the stored RIFKE Y channel (``motion[:, 0, 1]``). This keeps the
    resulting Y consistent with the stored feature and zeroes out XZ — both
    properties required by :func:`compute_rifke` (which subtracts root XZ
    and preserves Y).
    """
    motion = aug['motion']
    T = motion.shape[0]
    root_positions = np.zeros((T, 3))
    root_positions[:, 1] = motion[:, 0, 1]  # world Y from RIFKE root channel
    bvh_rots = hml_rotations_to_bvh_quaternions(motion[:, :, 3:9], aug['parents'])
    return fk_global_positions(
        bvh_rots, aug['parents'], aug['offsets'], root_positions,
    )


def recompute_position_features(aug):
    """Recompute RIFKE position features (0:3) for non-root joints via FK.

    Re-runs the same FK pipeline used at data-processing time
    (``positions_global`` + ``compute_rifke``) so the augmented clip is
    consistent with the feature-extraction convention.
    """
    global_positions = _augmented_global_positions(aug)
    root_facing_quat = Quaternions.from_transforms(
        rotation_6d_to_matrix_np(aug['motion'][:, 0, 3:9])
    )
    rifke = compute_rifke(global_positions, root_facing_quat)
    # Root RIFKE channels [0, Y, 0] are invariant to offset changes.
    aug['motion'][:, 1:, :3] = rifke[:, 1:]


def recompute_tpos_positions(aug):
    """Recompute tpos global positions ``(J, 3)`` from offsets.

    Rest rotations are identity, so global positions are the running sum of
    ``offsets`` along the kinematic chain.
    """
    njoints = len(aug['parents'])
    global_pos = np.zeros((njoints, 3))
    for j, p in enumerate(aug['parents']):
        if p == -1:
            global_pos[j] = aug['offsets'][j]
        else:
            global_pos[j] = global_pos[p] + aug['offsets'][j]
    aug['tpos'][:] = global_pos


def recompute_topology_arrays(aug, max_freqs, max_path_len=5):
    """Recompute topology-derived arrays from ``aug['parents']``."""
    parents_list = aug['parents'].tolist()
    aug['edge_indexs'] = compute_edge_indexs(aug['parents'])
    aug['joint_relations'], aug['joint_graph_dist'] = \
        compute_edge_relations_and_distances(parents_list, max_path_len=max_path_len)
    aug['joint_depths'] = compute_joint_depths(parents_list)
    aug['spectral_feats'], _ = compute_laplacian_eigenvectors(
        parents_list, max_freqs=max_freqs,
    )


def _recompute_all(aug, max_freqs, max_path_len=5):
    """Recompute position features + tpos + topology arrays in one call."""
    recompute_position_features(aug)
    recompute_tpos_positions(aug)
    recompute_topology_arrays(aug, max_freqs, max_path_len=max_path_len)


# ---------------------------------------------------------------------------
# Joint removal
# ---------------------------------------------------------------------------

def apply_joint_removal(aug, removal_rate, max_freqs):
    """Remove end-effector (leaf) joints from the skeleton.

    Only leaves are removed; deletion requires no rotation composition or
    offset merging. Processed in descending index order so deletions don't
    shift later indices.
    """
    parents = aug['parents']
    n_joints = len(parents)
    children = build_children_map(parents)

    removable = [j for j in range(1, n_joints) if len(children[j]) == 0]
    if not removable:
        return aug

    n_remove = min(int(round(len(removable) * removal_rate)), 3)
    if n_remove == 0:
        return aug
    to_remove = bone_length_weighted_sample(removable, aug['offsets'], n_remove)
    to_remove.sort(reverse=True)

    for j in to_remove:
        # Re-check in case prior deletions turned a leaf into something else
        children = build_children_map(aug['parents'])
        if len(children[j]) != 0:
            continue
        delete_joint(aug, j)

    _recompute_all(aug, max_freqs)
    return aug


# ---------------------------------------------------------------------------
# Joint addition (linear split along the bone)
# ---------------------------------------------------------------------------

def apply_joint_addition_linear(aug, max_freqs, max_path_len=5.):
    """Insert a kinematically neutral joint along a bone.

    New joint has identity local rotation and splits the parent→child offset,
    so FK output is exactly preserved.

    Not wired to any config flag — the default addition aug is the ellipsoid
    variant below; this simpler on-the-bone split is kept as an ablation
    alternative.
    """
    motion = aug['motion']
    parents = aug['parents']
    offsets = aug['offsets']
    tpos = aug['tpos']
    joint_names_emb = aug['joint_names_emb']
    mean, std = aug['mean'], aug['std']

    n_frames, n_joints, D = motion.shape

    candidates = list(range(1, n_joints))
    if not candidates:
        return aug

    ji = random.choice(candidates)
    pi = int(parents[ji])

    alpha = random.uniform(0.3, 0.7)
    offset_new = offsets[ji] * alpha
    offset_ji_rem = offsets[ji] * (1 - alpha)

    # HML: the new joint's slot stores bvh[parents[new]] = bvh[pi],
    # which is exactly what old motion[ji, 3:9] already holds. With
    # identity local rotation at the new joint, bvh[new] = bvh[pi],
    # so after concat motion[ji+1, 3:9] (old-ji's shifted slot, which
    # must store bvh[new]) is already correct — leave it unchanged.
    new_motion = np.zeros((n_frames, 1, D), dtype=motion.dtype)
    new_motion[:, 0, 3:9] = motion[:, ji, 3:9]

    aug['motion'] = np.concatenate(
        [motion[:, :ji], new_motion, motion[:, ji:]], axis=1,
    )
    aug['offsets'] = np.concatenate(
        [offsets[:ji], offset_new[None], offsets[ji:]], axis=0,
    )
    aug['offsets'][ji + 1] = offset_ji_rem

    tpos_new = (tpos[pi] + offset_new).copy()
    aug['tpos'] = np.concatenate([tpos[:ji], tpos_new[None], tpos[ji:]], axis=0)

    name_new = (joint_names_emb[pi] + joint_names_emb[ji]) / 2
    aug['joint_names_emb'] = np.concatenate(
        [joint_names_emb[:ji], name_new[None], joint_names_emb[ji:]], axis=0,
    )

    mean_new = mean[ji].copy()
    aug['mean'] = np.concatenate([mean[:ji], mean_new[None], mean[ji:]], axis=0)
    aug['std'] = np.concatenate([std[:ji], std[ji:ji + 1], std[ji:]], axis=0)

    new_parents = parents.copy()
    new_parents[new_parents >= ji] += 1
    aug['parents'] = np.concatenate([new_parents[:ji], [pi], new_parents[ji:]])
    aug['parents'][ji + 1] = ji

    _recompute_all(aug, max_freqs, max_path_len=max_path_len)
    return aug


# ---------------------------------------------------------------------------
# Joint addition (ellipsoid sampling)
# ---------------------------------------------------------------------------

def sample_ellipsoid_gaussian(bone_offset, sigma=0.5, lateral_ratio=0.5):
    """Sample a 3D displacement inside a bone-aligned ellipsoid with Gaussian falloff.

    Long axis = bone direction (semi-axis = bone_length); perpendicular
    semi-axes = ``bone_length * lateral_ratio``; radial density is a
    truncated half-normal.
    """
    bone_length = np.linalg.norm(bone_offset)
    if bone_length < 1e-8:
        return np.zeros(3)

    # Direction — rejection-sample a uniform unit vector in [-1,1]^3.
    while True:
        d = np.random.uniform(-1, 1, size=3)
        norm = np.linalg.norm(d)
        if 0 < norm <= 1:
            d /= norm
            break
    # Radius — truncated half-normal.
    while True:
        r = abs(np.random.normal(0, sigma))
        if r <= 1:
            break
    sphere_point = r * d

    a = bone_length
    b = bone_length * lateral_ratio
    scaled = sphere_point * np.array([a, b, b])

    e1 = bone_offset / bone_length
    ref = np.array([0.0, 1.0, 0.0])
    if abs(np.dot(e1, ref)) > 0.9:
        ref = np.array([1.0, 0.0, 0.0])
    e2 = np.cross(e1, ref)
    e2 /= np.linalg.norm(e2)
    e3 = np.cross(e1, e2)

    R = np.column_stack([e1, e2, e3])
    return R @ scaled


def apply_joint_addition_ellipsoid(aug, max_freqs,
                                   max_path_len=5., sigma=0.5, lateral_ratio=0.5):
    """Insert a joint between a joint and its parent via ellipsoid sampling.

    The new joint's offset is displaced from the linear split point by a
    sample inside a bone-aligned ellipsoid with Gaussian falloff.

    FK preservation: the rest skeleton shape is the running sum of
    ``offsets`` (identity rest rotations), so all offsets stay in the rest
    frame (``offset_ji_rem = d_rem_parent``) and the rest-pose position of
    every node in ji's subtree is preserved. At every frame,
    ``pi_world + bvh[pi]·offsets[ji]
    = pi_world + bvh[pi]·offset_new + bvh[new]·offset_ji_rem`` requires
    ``bvh[new]·d_rem_parent = bvh[pi]·d_rem_parent``; choosing identity
    local rotation at the new joint satisfies this exactly and leaves every
    global rotation in the subtree unchanged — so neither ji's HML slot
    (now at ji+1, storing ``bvh[new] = bvh[pi]``, already the pre-existing
    value) nor any descendant slot needs to be modified.
    """
    motion = aug['motion']
    parents = aug['parents']
    offsets = aug['offsets']
    tpos = aug['tpos']
    joint_names_emb = aug['joint_names_emb']
    mean, std = aug['mean'], aug['std']

    n_frames, n_joints, D = motion.shape

    candidates = list(range(1, n_joints))
    if not candidates:
        return aug

    ji = random.choice(candidates)
    pi = int(parents[ji])

    alpha = random.uniform(0.3, 0.7)
    split_offset = offsets[ji] * alpha
    displacement = sample_ellipsoid_gaussian(
        offsets[ji], sigma=sigma, lateral_ratio=lateral_ratio,
    )
    offset_new = split_offset + displacement
    offset_ji_rem = offsets[ji] - offset_new

    new_motion = np.zeros((n_frames, 1, D), dtype=motion.dtype)
    new_motion[:, 0, 3:9] = motion[:, ji, 3:9]

    aug['motion'] = np.concatenate(
        [motion[:, :ji], new_motion, motion[:, ji:]], axis=1,
    )
    aug['offsets'] = np.concatenate(
        [offsets[:ji], offset_new[None], offsets[ji:]], axis=0,
    )
    aug['offsets'][ji + 1] = offset_ji_rem

    tpos_new = (tpos[pi] + offset_new).copy()
    aug['tpos'] = np.concatenate([tpos[:ji], tpos_new[None], tpos[ji:]], axis=0)

    name_new = (joint_names_emb[pi] + joint_names_emb[ji]) / 2
    aug['joint_names_emb'] = np.concatenate(
        [joint_names_emb[:ji], name_new[None], joint_names_emb[ji:]], axis=0,
    )

    mean_new = mean[ji].copy()
    aug['mean'] = np.concatenate([mean[:ji], mean_new[None], mean[ji:]], axis=0)
    aug['std'] = np.concatenate([std[:ji], std[ji:ji + 1], std[ji:]], axis=0)

    new_parents = parents.copy()
    new_parents[new_parents >= ji] += 1
    aug['parents'] = np.concatenate([new_parents[:ji], [pi], new_parents[ji:]])
    aug['parents'][ji + 1] = ji

    _recompute_all(aug, max_freqs, max_path_len=max_path_len)
    return aug


# ---------------------------------------------------------------------------
# Skeleton pooling — reduce toward minimal graph
# ---------------------------------------------------------------------------

def apply_skeleton_pooling(aug, pool_rate, max_freqs):
    """Collapse degree-2 joints to reduce the skeleton toward its minimal graph.

    Teaches the model that skeletons differing only in linear-chain length
    are topologically equivalent. Rotation slots are re-wired into the
    child and offsets are merged so the FK chain is preserved.
    """
    parents = aug['parents']
    n_joints = len(parents)
    children = build_children_map(parents)

    degree2 = [j for j in range(1, n_joints) if len(children[j]) == 1]
    if not degree2:
        return aug

    n_pool = max(1, int(len(degree2) * pool_rate))
    to_pool = bone_length_weighted_sample(degree2, aug['offsets'], n_pool)
    to_pool.sort(reverse=True)

    for j in to_pool:
        children = build_children_map(aug['parents'])
        if len(children[j]) != 1:
            continue
        collapse_single_child_joint(
            aug, j, children[j][0], int(aug['parents'][j]),
        )
        delete_joint(aug, j)

    _recompute_all(aug, max_freqs)
    return aug


# ---------------------------------------------------------------------------
# Joint perturbation
# ---------------------------------------------------------------------------

def apply_joint_perturbation(aug, max_scale=0.1):
    """Apply random scale noise to bone lengths, then recompute positions via FK.

    Each non-root joint gets an independent uniform scale factor in
    ``[1 - max_scale, 1 + max_scale]`` so perturbations are not correlated
    across the skeleton. Rotations are left unchanged. Only position features
    are recomputed (topology is preserved).
    """
    offsets = aug['offsets']
    parents = aug['parents']
    J = len(parents)

    scales = np.random.uniform(1.0 - max_scale, 1.0 + max_scale, size=J)

    for j in range(1, J):
        offsets[j] *= scales[j]

    recompute_position_features(aug)
    recompute_tpos_positions(aug)
    return aug
