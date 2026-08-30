"""Skeleton processing utilities.

Topology-agnostic helpers that operate on ``Animation`` objects: BFS joint
reordering, geometric measurements, uniform scaling, root alignment,
facing-axis alignment and motion activity metrics.

Naming conventions:
    get_*    — pure computations that return a value (no mutation).
    scale_*  — uniform scaling transforms returning a new Animation.
    align_*  — positional / rotational alignment transforms returning a
               new Animation (together with the offset/quaternion applied).
"""

import statistics
from collections import deque

import numpy as np
from Quaternions import Quaternions
from Animation import Animation, positions_global, offset_lengths


# ---------------------------------------------------------------------------
# BFS joint reordering
# ---------------------------------------------------------------------------

def bfs_reorder_joints(parents, offsets=None):
    """Compute a canonical breadth-first ordering of joints starting from the root.

    Children sorted by subtree size (descending), bone length as tiebreaker (ascending).
    """
    njoints = len(parents)

    if offsets is not None:
        bone_lengths = np.linalg.norm(offsets, axis=-1)
    else:
        bone_lengths = None

    # Build children adjacency and find root.
    children = [[] for _ in range(njoints)]
    root = None
    for j, p in enumerate(parents):
        if p == -1:
            root = j
        else:
            children[p].append(j)
    assert root is not None, "No root joint found (parent == -1)"

    # Compute subtree sizes via BFS topological order (bottom-up accumulation).
    subtree_size = [1] * njoints
    topo = []
    q = deque([root])
    while q:
        node = q.popleft()
        topo.append(node)
        q.extend(children[node])
    for node in reversed(topo):
        for child in children[node]:
            subtree_size[node] += subtree_size[child]

    # Sort children: largest subtree first, shortest bone as tiebreaker.
    for node in range(njoints):
        if children[node]:
            children[node].sort(
                key=lambda idx: (-subtree_size[idx],
                                 bone_lengths[idx] if bone_lengths is not None else 0))

    # Final BFS traversal with sorted children → canonical ordering.
    bfs_order = []
    q = deque([root])
    while q:
        node = q.popleft()
        bfs_order.append(node)
        q.extend(children[node])

    assert len(bfs_order) == njoints

    # Build old→new index mapping and remap parents.
    old_to_new = [0] * njoints
    for new_idx, old_idx in enumerate(bfs_order):
        old_to_new[old_idx] = new_idx

    new_parents = []
    for new_idx, old_idx in enumerate(bfs_order):
        old_parent = parents[old_idx]
        new_parents.append(-1 if old_parent == -1 else old_to_new[old_parent])

    return bfs_order, new_parents


def reorder_anim_bfs(anim):
    """Reorder an Animation's joints into BFS order."""
    parents_list = anim.parents.tolist() if hasattr(anim.parents, 'tolist') else list(anim.parents)
    offsets = np.array(anim.offsets) if hasattr(anim, 'offsets') else None
    bfs_order, new_parents = bfs_reorder_joints(parents_list, offsets=offsets)

    if bfs_order == list(range(len(parents_list))):
        return anim, bfs_order, parents_list

    reordered = anim[:, bfs_order]
    return reordered, bfs_order, new_parents


# ---------------------------------------------------------------------------
# Geometry: skeleton diameter
# ---------------------------------------------------------------------------

def get_skeleton_diameter(anim):
    """Return the leaf-to-leaf geodesic diameter of a skeleton.

    Uses the classic two-pass BFS trick on the bone-length-weighted skeleton
    tree:
        1. BFS from the root to find an extreme leaf ``u``.
        2. BFS from ``u`` to find the farthest node — that distance is the
           tree diameter.

    The BFS formulation is robust to arbitrary root placements and
    non-humanoid topologies (multi-limb creatures, snakes, etc.).

    Args:
        anim: Animation object whose ``parents`` / ``offsets`` define the
            rest-pose skeleton.

    Returns:
        Diameter as a float. Returns 0.0 for degenerate single-joint skeletons.
    """
    parents = anim.parents
    offsets = anim.offsets
    n_joints = len(parents)

    # Build a bidirectional, bone-length-weighted adjacency list.
    adj = {i: [] for i in range(n_joints)}
    bone_lengths = np.linalg.norm(offsets, axis=-1)
    for i, p in enumerate(parents):
        if p != -1:
            adj[p].append((i, bone_lengths[i]))
            adj[i].append((p, bone_lengths[i]))

    def _bfs_farthest(start):
        """BFS returning (farthest_node, distance) from ``start``."""
        distances = {start: 0.0}
        visited = {start}
        queue = [start]
        farthest_node = start
        max_dist = 0.0

        ptr = 0
        while ptr < len(queue):
            u = queue[ptr]
            ptr += 1
            for v, weight in adj[u]:
                if v in visited:
                    continue
                visited.add(v)
                dist = distances[u] + weight
                distances[v] = dist
                queue.append(v)
                if dist > max_dist:
                    max_dist = dist
                    farthest_node = v
        return farthest_node, max_dist

    u, _ = _bfs_farthest(0)          # pass 1: one extreme leaf
    _, diameter = _bfs_farthest(u)   # pass 2: farthest from u → diameter
    return diameter


# ---------------------------------------------------------------------------
# Uniform scaling
# ---------------------------------------------------------------------------

def scale_anim(anim, scale_factor):
    """Return a copy of *anim* with positions and offsets scaled uniformly."""
    return Animation(
        rotations=anim.rotations.copy(),
        positions=anim.positions.copy() * scale_factor,
        orients=anim.orients.copy(),
        offsets=anim.offsets.copy() * scale_factor,
        parents=anim.parents.copy(),
    )


def scale_by_diameter(anim, scale_factor=None, target_diameter=2.0):
    """Scale an animation so that its leaf-to-leaf diameter matches a target.

    Args:
        anim: Animation to scale.
        scale_factor: Explicit scale. If None, computed as
            ``target_diameter / get_skeleton_diameter(anim)``.
        target_diameter: Desired diameter (default 2.0).

    Returns:
        Tuple of (scaled Animation, scale_factor applied).
    """
    if scale_factor is None:
        scale_factor = target_diameter / get_skeleton_diameter(anim)
    return scale_anim(anim, scale_factor), scale_factor


def scale_by_bone_length(anim, scale_factor=None, target_bone_len=None):
    """Scale an animation so that the mean bone length matches a target.

    If ``scale_factor`` is not provided it is derived as
    ``target_bone_len / mean(offset_lengths(anim))``. Rotations are untouched.

    Args:
        anim: Animation to scale.
        scale_factor: Pre-computed scale. If None, derived from *target_bone_len*.
        target_bone_len: Desired mean bone length (used only when *scale_factor*
            is None).

    Returns:
        Tuple of (scaled Animation, scale_factor applied).
    """
    if scale_factor is None:
        mean_len = statistics.mean(offset_lengths(anim))
        scale_factor = target_bone_len / mean_len
    return scale_anim(anim, scale_factor), scale_factor


# ---------------------------------------------------------------------------
# Root alignment (translation)
# ---------------------------------------------------------------------------

def _translated_anim(anim, new_positions, new_offsets):
    """Return a copy of *anim* with replacement positions/offsets."""
    return Animation(
        rotations=anim.rotations.copy(),
        positions=new_positions,
        orients=anim.orients.copy(),
        offsets=new_offsets,
        parents=anim.parents.copy(),
    )


def align_root_xz_to_origin(anim, root_pose_init_xz=None):
    """Translate the skeleton so the root's XZ position is at the origin at frame 0.

    Only the horizontal (XZ) components are subtracted — the absolute Y height
    is preserved.

    Args:
        anim: Animation to translate.
        root_pose_init_xz: Pre-computed XZ offset (3,) with Y component zero.
            If None, auto-detected from the frame-0 root position.

    Returns:
        Tuple of (translated Animation, root_pose_init_xz actually subtracted).
    """
    if root_pose_init_xz is None:
        # Zero-out the Y component so only XZ is subtracted.
        root_pose_init_xz = anim.positions[0, 0] * np.array([1, 0, 1])

    new_positions = anim.positions.copy()
    new_positions[:, 0] -= root_pose_init_xz
    new_offsets = anim.offsets.copy()
    new_offsets[0] -= root_pose_init_xz

    return _translated_anim(anim, new_positions, new_offsets), root_pose_init_xz


def align_root_to_ground(anim, ground_height=None):
    """Translate the skeleton vertically so the lowest joint sits on Y=0.

    If ``ground_height`` is not provided, it is the global minimum Y across
    all joints and frames (computed via forward kinematics).

    Args:
        anim: Animation to translate.
        ground_height: Pre-computed Y offset. If None, auto-detected.

    Returns:
        Tuple of (translated Animation, ground_height subtracted).
    """
    if ground_height is None:
        global_positions = positions_global(anim)
        ground_height = global_positions.min(axis=0).min(axis=0)[1]

    new_positions = anim.positions.copy()
    new_positions[:, 0, 1] -= ground_height
    new_offsets = anim.offsets.copy()
    new_offsets[0, 1] -= ground_height

    return _translated_anim(anim, new_positions, new_offsets), ground_height


# ---------------------------------------------------------------------------
# Root orientation alignment (facing axis)
# ---------------------------------------------------------------------------

def _normalize(v, eps=1e-8):
    """Return ``v`` normalized along its last axis with a numerical floor."""
    norm = np.sqrt((v ** 2).sum(axis=-1, keepdims=True))
    return v / np.maximum(norm, eps)


def get_root_facing_quat(joints, face_joint_idxs, body_axis=False):
    """Compute per-frame root quaternion that rotates the skeleton to face Z+.

    The left-right axis ``across`` is assembled from the given face joint
    indices:

    - 2 indices ``[r_hip, l_hip]``: ``joints[r_hip] - joints[l_hip]``.
    - 4 indices ``[r_hip, l_hip, r_shoulder, l_shoulder]``: the sum of the
      hip and shoulder left-right vectors.

    Forward direction is ``Y_up x across``, and the returned quaternion
    rotates that forward vector onto Z+. When *body_axis* is True (e.g.
    snakes), ``across`` is treated as a body-axis (forward) vector rather
    than a lateral vector: it is rotated by -90° Y to convert it into a
    lateral axis before the cross product.

    If ``face_joint_idxs`` is None or the sentinel ``[-1, -1]`` (face joint
    names unavailable for this skeleton), returns per-frame identity
    quaternions so the caller can skip facing alignment.

    Args:
        joints: Global joint positions (F, J, 3).
        face_joint_idxs: Sequence of 2 or 4 joint indices, or ``[-1, -1]``
            sentinel. For body-axis creatures, convention is r_hip=head/tongue,
            l_hip=tail.
        body_axis: If True, treat the joint vector as body-axis (forward)
            rather than lateral.

    Returns:
        Quaternions of shape (F,) aligning the skeleton to face Z+.
    """
    if face_joint_idxs is None or (
        len(face_joint_idxs) >= 2
        and face_joint_idxs[0] == -1
        and face_joint_idxs[1] == -1
    ):
        return Quaternions.id(len(joints))

    if len(face_joint_idxs) == 4:
        r_hip, l_hip, sdr_r, sdr_l = face_joint_idxs
        across = (joints[:, r_hip] - joints[:, l_hip]) + (
            joints[:, sdr_r] - joints[:, sdr_l]
        )
    elif len(face_joint_idxs) == 2:
        r_hip, l_hip = face_joint_idxs
        across = joints[:, r_hip] - joints[:, l_hip]
    else:
        raise ValueError(
            f"face_joint_idxs must have 2 or 4 entries, got {len(face_joint_idxs)}"
        )

    across = _normalize(across)
    forward = _normalize(np.cross(np.array([[0, 1, 0]]), across, axis=-1))

    if body_axis:
        rot_correction = Quaternions.from_euler(np.array([0, -np.pi / 2, 0]), "xyz")
        forward = rot_correction * forward

    target = np.array([[0, 0, 1]]).repeat(len(forward), axis=0)
    root_quat = Quaternions.between(forward, target)

    return root_quat


def align_initial_facing(anim, face_joint_idxs, body_axis=False):
    """Rotate the entire motion so the skeleton faces Z+ at the first frame.

    Computes the facing-direction quaternion from the first frame using
    ``get_root_facing_quat`` and applies it to the root joint across all
    frames. Non-root joints are unaffected at the rotation level (the
    rotation propagates through the root's orientation at render time).

    When ``face_joint_idxs`` is None or the sentinel ``[-1, -1]`` (face
    joint names unavailable), the animation is returned unchanged
    (identity facing rotation).

    Args:
        anim: Animation object.
        face_joint_idxs: Indices [r_hip, l_hip] or
            [r_hip, l_hip, r_shoulder, l_shoulder], or ``[-1, -1]`` sentinel.
        body_axis: If True, apply an extra -90° Y correction (e.g. snakes).

    Returns:
        New Animation with rotated root.
    """
    if face_joint_idxs is None or (
        len(face_joint_idxs) >= 2
        and face_joint_idxs[0] == -1
        and face_joint_idxs[1] == -1
    ):
        return Animation(
            rotations=anim.rotations.copy(),
            positions=anim.positions.copy(),
            orients=anim.orients.copy(),
            offsets=anim.offsets.copy(),
            parents=anim.parents.copy(),
        )

    global_pos = positions_global(anim)
    quat = get_root_facing_quat(
        global_pos, face_joint_idxs, body_axis=body_axis,
    )[0]

    nframes = anim.shape[0]
    quat_tiled = quat.repeat(nframes, axis=0)

    new_rots = anim.rotations.copy()
    new_pos = anim.positions.copy()
    new_offsets = anim.offsets.copy()
    new_rots[:, 0] = quat_tiled * new_rots[:, 0]
    new_pos[:, 0] = quat_tiled * new_pos[:, 0]
    new_offsets[0] = quat * new_offsets[0]

    return Animation(
        rotations=new_rots,
        positions=new_pos,
        orients=anim.orients.copy(),
        offsets=new_offsets,
        parents=anim.parents.copy(),
    )


# ---------------------------------------------------------------------------
# Full alignment pipeline
# ---------------------------------------------------------------------------

def canonicalize_anim(anim,
                        face_joint_idxs, body_axis=False,
                        root_pose_init_xz=None, ground_height=None,
                        scale_method="diameter", scale_factor=None, target_scale_metric=None):
    """Full alignment pipeline: facing -> center XZ -> scale -> ground.

    Each step accepts an optional pre-computed parameter so T-pose calibration
    values can be reused for motion clips.

    Args:
        anim: Raw Animation object.
        face_joint_idxs: Joint indices defining the facing direction.
        body_axis: If True, apply an extra -90° Y correction (e.g. snakes).
        root_pose_init_xz: XZ offset for centering (from T-pose calibration).
        ground_height: Vertical ground offset (from T-pose calibration).
        scale_method: ``"diameter"`` (default) or ``"bone_length"``.
        scale_factor: Uniform scale (from T-pose calibration).
        target_scale_metric: Target value for the chosen scale method. Interpreted
            as target diameter when ``scale_method="diameter"`` or target mean bone
            length when ``scale_method="bone_length"``. Only used when
            *scale_factor* is None.

    Returns:
        (aligned_anim, root_pose_init_xz, ground_height, scale_factor).
    """
    rotated = align_initial_facing(anim, face_joint_idxs, body_axis=body_axis)
    centered, root_pose_init_xz = align_root_xz_to_origin(rotated, root_pose_init_xz)

    if scale_method == "diameter":
        scaled, scale_factor = scale_by_diameter(
            centered, scale_factor=scale_factor, target_diameter=target_scale_metric or 2.0)
    elif scale_method == "bone_length":
        scaled, scale_factor = scale_by_bone_length(
            centered, scale_factor=scale_factor, target_bone_len=target_scale_metric)
    else:
        raise ValueError(f"Unknown scale_method: {scale_method!r}")

    grounded, ground_height = align_root_to_ground(scaled, ground_height)
    return grounded, root_pose_init_xz, ground_height, scale_factor


# ---------------------------------------------------------------------------
# Motion activity metrics
# ---------------------------------------------------------------------------


def get_motion_activity_metrics(positions, top_k_ratio=0.25, top_k_min=2):
    """Compute motion activity metrics from joint positions.

    Activity is measured as the **temporal spread** (std over frames) of each
    joint's position. Non-root joint activity is computed from *root-relative*
    positions, so it isolates articulation from global translation. Root
    activity uses the global root position.

    Assumes positions are already at a canonical scale (e.g. via
    ``scale_by_diameter``), so no bone-length normalization is applied.

    This spread-based formulation is robust to frame rate and to motions that
    have tiny jitter but no real movement: a jittery-but-static motion has
    near-zero std even if its frame-to-frame step is nonzero, so thresholding
    on these metrics reliably filters out inactive clips.

    In addition to the mean over all non-root joints, a top-k mean is returned
    — the average spread of the most active joints. This catches long-tailed /
    multi-limb skeletons where only a few joints move (e.g. snake tail tip,
    single-limb gestures); the mean gets diluted by the many static joints,
    but the top-k mean stays high. ``k`` scales with skeleton size as
    ``max(top_k_min, round(top_k_ratio * (J - 1)))`` so it is meaningful for
    both small skeletons (~10 joints) and large ones (100+ joints).

    Args:
        positions: Joint global positions (F, J, 3), at canonical scale.
        top_k_ratio: Fraction of non-root joints to include in the top-k mean
            (default 0.25 → top quartile of most-active joints).
        top_k_min: Minimum number of joints in the top-k window, so tiny
            skeletons still get a meaningful average (default 2).

    Returns:
        Dict with keys ``joint_activity``, ``joint_activity_topk`` and
        ``root_activity`` (floats).
    """
    assert positions.ndim == 3, "positions must be (F, J, 3)"

    # Root activity: temporal spread of the global root trajectory.
    root_pos = positions[:, 0, :]                        # (F, 3)
    root_activity = float(np.linalg.norm(root_pos.std(axis=0)))

    # Joint activity: temporal spread of *root-relative* joint positions,
    # which isolates articulation from global translation.
    if positions.shape[1] > 1:
        rel_pos = positions[:, 1:, :] - positions[:, 0:1, :]  # (F, J-1, 3)
        per_joint_std = rel_pos.std(axis=0)                    # (J-1, 3)
        per_joint_spread = np.linalg.norm(per_joint_std, axis=-1)  # (J-1,)
        joint_activity = float(per_joint_spread.mean())

        # Top-k mean: robust to long-tailed skeletons where the mean is
        # diluted by many static joints but a few limbs carry the motion.
        # k scales with joint count so it stays meaningful across skeleton sizes.
        n_non_root = per_joint_spread.shape[0]
        k = max(top_k_min, int(round(top_k_ratio * n_non_root)))
        k = min(k, n_non_root)
        top_k_vals = np.partition(per_joint_spread, -k)[-k:]
        joint_activity_topk = float(top_k_vals.mean())
    else:
        joint_activity = 0.0
        joint_activity_topk = 0.0

    return {
        "joint_activity": joint_activity,
        "joint_activity_topk": joint_activity_topk,
        "root_activity": root_activity,
    }
