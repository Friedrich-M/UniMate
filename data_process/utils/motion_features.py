"""Per-object motion feature extraction (export NPZ → canonical clips).

Pipeline:
    1. ``process_tpose``            — load / canonicalize the T-pose, compute
                                      scale + rest offsets.
    2. ``build_topology_cond``      — build the per-object topology condition
                                      dict (joint graph, spectral feats, …).
    3. ``load_and_canonicalize_motion`` + ``trim_static_ends``
                                    — load / normalize each motion, strip
                                      static pre/post-roll.
    4. ``extract_clip_feats`` + ``save_clip``
                                    — slice into clips, dump per-frame arrays
                                      (positions / rotations / facing) and an
                                      MP4 visualization.

``process_object`` orchestrates 1–4 for one object type and returns an
``ObjectResult`` (see below) summarizing the saved artifacts.

Output layout under ``save_dir``:
    motions/{object}-{motion}-{clip_idx}.npz   # per-clip per-frame arrays
    videos/{object}-{motion}-{clip_idx}.mp4
    tpose/{object}.png                          # T-pose still
    (driver-level) cond.npy, captions.json, filtered_clips.json, metadata.txt

The driver is :mod:`data_process.feature_extraction.extract_features`.
"""

import os
from os.path import join as pjoin
from pathlib import Path

import numpy as np
from loguru import logger

from Animation import (
    Animation,
    positions_global,
    rotations_global,
    offsets_from_positions
)
from Quaternions import Quaternions

from data_process.utils.topology import (
    compute_edge_relations_and_distances,
    compute_kinematic_chains,
    compute_joint_depths,
    compute_edge_indexs,
    compute_laplacian_eigenvectors,
)
from data_process.utils.skeleton import (
    reorder_anim_bfs,
    get_root_facing_quat,
    canonicalize_anim,
    get_motion_activity_metrics,
    get_skeleton_diameter,
)
from data_process.utils.plotting import (
    save_skeleton_motion,
    save_skeleton_motion_ground,
    save_skeleton_motion_spectral,
    save_skeleton_tpose,
    save_skeleton_tpos_spectral,
)
from data_process.feature_extraction.metadata import (
    atomic_output_path,
    get_motion_clip_windows,
)


# Smallest leaf-to-leaf diameter a skeleton may have and still be scaled to
# the canonical target diameter (below this the scale factor blows up / divides
# by zero).
MIN_SKELETON_DIAMETER = 1e-8


class DegenerateSkeletonError(ValueError):
    """Raised when a T-pose cannot be canonicalized (e.g. zero-length bones).

    Callers treat this as a skip-with-reason for the whole object type rather
    than a hard failure.
    """


# ===========================================================================
# Joint-name resolution
# ===========================================================================

def resolve_face_joint_idxs(face_joints, names):
    """Resolve face joint names to indices from a face_joints dict.

    Returns the sentinel ``[-1, -1]`` when the entry is missing, empty, or
    its names don't appear in the skeleton — downstream code treats that as
    identity facing rather than skipping the object.

    Returns:
        Tuple of (face_joint_idxs, body_axis).
    """
    if not face_joints or 'r_hip' not in face_joints or 'l_hip' not in face_joints:
        return [-1, -1], False
    r_hip_name = face_joints['r_hip']['raw']
    l_hip_name = face_joints['l_hip']['raw']
    if r_hip_name in names and l_hip_name in names:
        idxs = [names.index(r_hip_name), names.index(l_hip_name)]
        return idxs, face_joints.get('body_axis', False)
    return [-1, -1], False


def resolve_joint_names(object_type, tpos_data, corps_names=None,
                        clean_names=None, expected_names=None):
    """Resolve joint names and corps subset indices for an object type.

    Returns:
        Tuple of (corps_idxs, corps_names, clean_names_list, raw_parents,
                  raw_njoints).
    """
    raw_parents = tpos_data['parents'].tolist()
    raw_njoints = len(raw_parents)
    raw_names = tpos_data['names'].tolist()

    # clean_names comes from clean_joint_names.json, whose entries are
    # positionally aligned with joint_names.json — NOT necessarily with this
    # NPZ: rig variants can enumerate the same skeleton in a different order
    # (e.g. Truebones KingCobra), and the T-pose reference is simply the
    # first NPZ in sorted order. Align by the expected name list when given;
    # a silent positional paste would tag every joint with the wrong label.
    raw_clean = clean_names if clean_names else raw_names
    if clean_names and expected_names is not None:
        if len(clean_names) != len(expected_names):
            raise ValueError(
                f'[{object_type}] clean_joint_names has {len(clean_names)} '
                f'entries but joint_names.json has {len(expected_names)}; '
                f'clean_joint_names.json is stale — re-run the joint-name '
                f'cleaner.')
        if raw_names != expected_names:
            index = {n: i for i, n in enumerate(expected_names)}
            try:
                raw_clean = [clean_names[index[n]] for n in raw_names]
            except KeyError as e:
                raise ValueError(
                    f'[{object_type}] T-pose reference NPZ contains joint '
                    f'{e} that is absent from joint_names.json; the export '
                    f'metadata is out of date.')
            logger.info(f'[{object_type}] T-pose NPZ joint order differs '
                        f'from joint_names.json; clean names realigned by '
                        f'name.')

    corps_idxs = None
    if corps_names is not None:
        try:
            corps_idxs = [raw_names.index(n) for n in corps_names]
        except ValueError as e:
            raise ValueError(
                f'Failed to resolve the requested joint subset (mixamo core '
                f'joints / corps_names) for {object_type} in tpos_data: {e}. '
                f'Raw names: {raw_names}')
        clean_names_list = [raw_clean[i] for i in corps_idxs]
    else:
        clean_names_list = list(raw_clean)

    return corps_idxs, corps_names, clean_names_list, raw_parents, raw_njoints


# ===========================================================================
# NPZ loading
# ===========================================================================

def _names_permutation(ref_names, ref_parents, names, parents):
    """Permutation aligning a clip's joint order to the reference skeleton.

    Returns ``perm`` such that ``array[:, perm]`` reorders per-joint data
    into the reference joint order, or None when the clip is not the same
    skeleton (different name set, duplicate names, or a topology that
    disagrees under the name mapping).
    """
    if sorted(names) != sorted(ref_names) or len(set(names)) != len(names):
        return None
    index = {n: i for i, n in enumerate(names)}
    perm = [index[n] for n in ref_names]
    for i, p in enumerate(ref_parents):
        q = parents[perm[i]]
        if (q == -1) != (p == -1) or (p != -1 and q != perm[p]):
            return None
    return perm


def _load_npz_anim(object_npz_path, raw_parents, raw_njoints,
                   corps_idxs, bfs_order, fps, raw_names=None):
    """Load a motion NPZ, apply joint subset/reorder, and downsample.

    Returns:
        Tuple of (motion_anim, motion_fps, None) on success, or
        (None, None, reason_string) when the motion should be skipped.
    """
    anim_data = np.load(object_npz_path, allow_pickle=True)
    nframes, njoints = anim_data['anim_local_rot'].shape[:2]

    if njoints != raw_njoints:
        reason = f'joint count mismatch: expected {raw_njoints}, got {njoints}'
        logger.warning(f'Skipping clip ({reason}): {object_npz_path}')
        return None, None, reason

    anim_local_rot = anim_data['anim_local_rot']
    anim_local_pos = anim_data['anim_local_pos']
    rest_local_pos = anim_data['rest_local_pos']

    # corps_idxs/bfs_order are positional indices into the T-pose clip's
    # joint order; a source file enumerating the same skeleton in a
    # different order would silently misassign joints, so align by name.
    if raw_names is not None:
        clip_names = anim_data['names'].tolist()
        if clip_names != raw_names:
            perm = _names_permutation(raw_names, raw_parents, clip_names,
                                      anim_data['parents'].tolist())
            if perm is None:
                reason = 'joint names mismatch with T-pose skeleton'
                logger.warning(f'Skipping clip ({reason}): {object_npz_path}')
                return None, None, reason
            logger.info(f'Joint order differs from T-pose skeleton; '
                        f'reordering by name: {object_npz_path}')
            anim_local_rot = anim_local_rot[:, perm]
            anim_local_pos = anim_local_pos[:, perm]
            rest_local_pos = rest_local_pos[perm]

    motion_positions = rest_local_pos.copy()[None, :].repeat(nframes, axis=0)
    motion_positions[:, 0] = anim_local_pos[:, 0]  # root joint's local position is the animated global position

    motion_anim = Animation(
        rotations=Quaternions(anim_local_rot),
        positions=motion_positions,
        orients=Quaternions.id(raw_njoints),
        offsets=rest_local_pos,
        parents=raw_parents,
    )

    if corps_idxs is not None:
        motion_anim = motion_anim[:, corps_idxs]
    motion_anim = motion_anim[:, bfs_order]

    if fps >= 60:
        motion_anim = motion_anim[::2]
        motion_fps = fps // 2
    else:
        motion_fps = fps

    return motion_anim, motion_fps, None


def _load_tpos_npz(tpos_data, corps_idxs=None, corps_names=None):
    """Load T-pose from NPZ data, apply joint subset and BFS reorder.

    Returns:
        Tuple of (tpos_anim, parents, names, fps, bfs_order).
    """
    names, fps = tpos_data['names'].tolist(), tpos_data['fps'].item()
    parents = tpos_data['parents'].tolist()
    njoints = len(parents)

    tpos_anim = Animation(
        rotations=Quaternions(tpos_data['rest_local_rot'][None]),
        positions=tpos_data['rest_local_pos'][None],
        orients=Quaternions.id(njoints),
        offsets=tpos_data['rest_local_pos'],
        parents=tpos_data['parents'],
    )

    if corps_idxs is not None:
        tpos_anim = tpos_anim[:, corps_idxs]
        names = corps_names if corps_names is not None else [names[i] for i in corps_idxs]

    tpos_anim, bfs_order, parents = reorder_anim_bfs(tpos_anim)
    names = [names[i] for i in bfs_order]

    return tpos_anim, parents, names, fps, bfs_order


def load_and_canonicalize_motion(object_npz_path, raw_parents, raw_njoints,
                                 corps_idxs, bfs_order, fps, scale_factor,
                                 ground_height, face_joint_idxs,
                                 body_axis=False, raw_names=None):
    """Load a motion NPZ and canonicalize facing / XZ / ground + scale.

    ``scale_factor`` and ``ground_height`` are reused from T-pose
    calibration so every motion of this object type shares a
    diameter-consistent scale and a consistent floor height (matching the
    npz pipeline); ``root_pose_init_xz`` is recomputed fresh per
    motion so each motion starts centered at the origin. Clips are then
    simple slices of the canonicalized motion.

    Returns:
        Tuple of ((motion_anim, motion_fps), None) on success, or
        (None, reason_string) when the motion should be skipped.
    """
    motion_anim, motion_fps, skip_reason = _load_npz_anim(
        object_npz_path, raw_parents, raw_njoints, corps_idxs, bfs_order, fps,
        raw_names=raw_names)
    if motion_anim is None:
        return None, skip_reason

    motion_anim, _, _, _ = canonicalize_anim(
        motion_anim,
        face_joint_idxs=face_joint_idxs, body_axis=body_axis,
        ground_height=ground_height,
        scale_factor=scale_factor)
    return (motion_anim, motion_fps), None


# ===========================================================================
# T-pose processing and topology conditioning
# ===========================================================================

def process_tpose(tpos_data, corps_idxs=None, corps_names=None,
                  face_joints=None, target_diameter=2.0):
    """Load T-pose from NPZ with facing -> center XZ -> scale -> ground.

    ``scale_factor``, ``ground_height`` (and ``offsets``) are threaded
    downstream so motions share a diameter-consistent scale and a
    consistent floor height across the object type (matching the npz
    pipeline); each motion's ``root_pose_init_xz`` is still
    recomputed fresh so every motion starts centered at origin.

    Raises:
        DegenerateSkeletonError: if the skeleton has no measurable extent
            (all-zero bone offsets), which would make diameter scaling divide
            by zero.

    Returns:
        Tuple of (tpos_anim, offsets, scale_factor, ground_height,
                  parents, names, fps, bfs_order,
                  face_joint_idxs, body_axis).
    """
    tpos_anim, parents, names, fps, bfs_order = _load_tpos_npz(
        tpos_data, corps_idxs, corps_names)

    diameter = float(get_skeleton_diameter(tpos_anim))
    if not np.isfinite(diameter) or diameter <= MIN_SKELETON_DIAMETER:
        raise DegenerateSkeletonError(
            f'degenerate skeleton: leaf-to-leaf diameter {diameter:.3e} '
            f'(zero-length bones?), cannot scale to target diameter '
            f'{target_diameter}')

    face_joint_idxs, body_axis = resolve_face_joint_idxs(face_joints, names)
    tpos_anim, _, ground_height, scale_factor = canonicalize_anim(
        tpos_anim,
        face_joint_idxs=face_joint_idxs, body_axis=body_axis,
        target_scale_metric=target_diameter)

    offsets = np.array(tpos_anim.offsets).copy()  # (J, 3)

    return (tpos_anim, offsets, scale_factor, ground_height,
            parents, names, fps, bfs_order,
            face_joint_idxs, body_axis)


def build_topology_cond(object_type, parents, offsets, names, clean_joint_names,
                        tpos_global_pos, tpos_local_rot, tpos_global_rot,
                        max_path_len=5, max_freqs=8,
                        face_joint_idxs=None, body_axis=False,
                        scale_factor=1.0, ground_height=None):
    """Build the topology conditioning dict from a canonicalized T-pose.

    Produces every field consumed by the training dataset loader plus the
    joint-name and optional face-joint metadata fields.

    ``ground_height`` is the vertical offset actually subtracted from every
    clip of this object type, or None when each motion was grounded on its
    own minimum Y (the default). It is recorded together with
    ``ground_height_mode`` (``'tpos'`` / ``'per_motion'``) so the cond stays
    self-describing — do not pass the T-pose's own ground height here unless
    that same value was reused for the motions.
    """
    parents_arr = np.asarray(parents)
    joint_relations, joint_graph_dists = compute_edge_relations_and_distances(
        parents_arr, max_path_len=max_path_len,
    )
    joint_depths = compute_joint_depths(parents_arr)
    edge_indexs = compute_edge_indexs(parents_arr)
    spectral_feats, _ = compute_laplacian_eigenvectors(parents_arr, max_freqs=max_freqs)
    kinematic_chains = compute_kinematic_chains(parents_arr)

    tpos_offsets = offsets_from_positions(tpos_global_pos, parents_arr)

    cond = {
        'object_type': object_type,
        'parents': parents,
        'offsets': offsets,                           # (J, 3)
        'tpos_offsets': tpos_offsets,                 # (J, 3)
        'joint_names': names,
        'clean_joint_names': clean_joint_names,
        'tpos_first_frame': tpos_global_pos,           # (J, 3)
        'tpos_local_rotations': tpos_local_rot,       # (J, 4) quaternion
        'tpos_global_rotations': tpos_global_rot,     # (J, 4) quaternion
        'joint_relations': joint_relations,           # (J, J)
        'joint_graph_dists': joint_graph_dists,       # (J, J)
        'joint_depths': joint_depths,                 # (J,)
        'edge_indexs': edge_indexs,                   # (2, 2*(J-1))
        'spectral_feats': spectral_feats,             # (J, K)
        'kinematic_chains': kinematic_chains,         # list of lists of joint indices
        'scale_factor': scale_factor,                 # float
        'ground_height': ground_height,               # float, or None per-motion
        'ground_height_mode': 'tpos' if ground_height is not None else 'per_motion',
    }

    if face_joint_idxs is not None:
        cond['face_joint_idxs'] = {
            'r_hip': face_joint_idxs[0],
            'l_hip': face_joint_idxs[1],
            'body_axis': body_axis,
        }

    return cond


# ===========================================================================
# Motion clip processing
# ===========================================================================

def trim_static_ends(motion_anim, static_threshold=1e-5):
    """Strip leading/trailing frames where every joint is nearly stationary.

    Per-frame activity is the max over joints of consecutive global-position
    displacement. Frames at the head/tail with activity below
    ``static_threshold`` are removed; interior static spans are preserved.

    Returns:
        Tuple of (trimmed_anim, start_idx, end_idx). ``end_idx`` is
        exclusive; returns an empty slice when the entire motion is static.
    """
    nframes = len(motion_anim)
    if nframes < 2:
        return motion_anim, 0, nframes

    global_positions = positions_global(motion_anim)
    step = np.linalg.norm(np.diff(global_positions, axis=0), axis=-1)  # (F-1, J)
    max_joint_step = step.max(axis=-1)                                 # (F-1,)

    active = max_joint_step > static_threshold
    if not active.any():
        return motion_anim[:0], 0, 0

    active_idx = np.where(active)[0]
    start = int(active_idx[0])
    end = min(int(active_idx[-1]) + 2, nframes)  # +1 diff offset, +1 exclusive
    return motion_anim[start:end], start, end


def extract_clip_feats(motion_anim, clip_start, clip_end,
                       face_joint_idxs, body_axis=False):
    """Slice a canonicalized motion into a clip and extract per-frame data.

    The motion is assumed to already be canonicalized against T-pose
    calibration (facing / XZ / ground / scale) by
    :func:`load_and_canonicalize_motion`, so clips are simple slices — no
    per-clip re-centering.

    Returns:
        Tuple of (local_rotations (F, J, 4), global_positions (F, J, 3),
        root_facing_quat (F, 4)).
    """
    clip_anim = motion_anim[clip_start:clip_end]
    local_rotations = clip_anim.rotations.copy()  # (F, J, 4)
    global_positions = positions_global(clip_anim)
    root_facing_quat = get_root_facing_quat(
        global_positions, face_joint_idxs, body_axis=body_axis)

    return local_rotations.qs, global_positions, root_facing_quat.qs


def _activity_below_threshold(global_positions, threshold):
    """Return (is_low, metrics) from joint-activity metrics for a sequence."""
    metrics = get_motion_activity_metrics(global_positions)
    is_low = (metrics['joint_activity_topk'] < threshold
              and metrics['root_activity'] < threshold)
    return is_low, metrics


def _format_activity(metrics):
    return (f'joint={metrics["joint_activity"]:.4f}, '
            f'topk={metrics["joint_activity_topk"]:.4f}, '
            f'root={metrics["root_activity"]:.4f}')


# ===========================================================================
# Output I/O
# ===========================================================================

def save_clip(clip_name,
              global_positions, local_rotations, root_facing_quat,
              parents, motion_fps,
              motions_save_dir, animations_save_dir,
              spectral_feats=None, caption=None, save_vis=True,
              vis_ground=False):
    """Save per-clip canonicalized positions/rotations/facing + MP4.

    When ``spectral_feats`` is supplied, the MP4 colors joints by PCA-projected
    spectral features so topologically close joints share colors across
    object types; otherwise falls back to the default root/non-root palette.
    ``caption``, if non-empty, is rendered as the video title. The MP4
    dominates per-clip time; pass ``save_vis=False`` to skip it.

    The NPZ is written atomically: nothing downstream re-validates clip files,
    so an interrupted run must not leave a truncated one for the training
    loader to read.
    """
    with atomic_output_path(pjoin(motions_save_dir, clip_name + '.npz')) as tmp_path:
        # A file object keeps np.savez from appending '.npz' to the temp name.
        with open(tmp_path, 'wb') as f:
            np.savez(
                f,
                global_positions=global_positions,
                local_rotations=local_rotations,
                root_facing_quat=root_facing_quat,
                fps=int(motion_fps),
            )

    if not save_vis:
        return
    clip_vis_path = pjoin(animations_save_dir, clip_name + '.mp4')
    title = caption or None
    if vis_ground:
        # Checkerboard ground + follow camera + trajectory/shadows; keeps
        # the spectral palette when available.
        save_skeleton_motion_ground(
            save_path=clip_vis_path, parents=parents,
            positions=global_positions, spectral_feats=spectral_feats,
            fps=int(motion_fps), title=title,
            figsize=(6.4, 6.4), dpi=100,
        )
        return
    if spectral_feats is not None:
        save_skeleton_motion_spectral(
            save_path=clip_vis_path, parents=parents,
            positions=global_positions, spectral_feats=spectral_feats,
            fps=int(motion_fps), title=title,
            figsize=(6.4, 6.4), dpi=100,
        )
    else:
        save_skeleton_motion(
            save_path=clip_vis_path, parents=parents,
            positions=global_positions, fps=int(motion_fps), title=title,
            figsize=(6.4, 6.4), dpi=100,
        )


def _save_tpose_vis(save_dir, object_type, parents, tpos_anim,
                    spectral_feats=None):
    tpos_save_dir = pjoin(save_dir, 'tpose')
    os.makedirs(tpos_save_dir, exist_ok=True)
    tpos_vis_path = pjoin(tpos_save_dir, f'{object_type}.png')
    tpos_global_pos = positions_global(tpos_anim)[0]
    if spectral_feats is not None:
        save_skeleton_tpos_spectral(
            save_path=tpos_vis_path, parents=parents,
            positions=tpos_global_pos, spectral_feats=spectral_feats,
            figsize=(6.4, 6.4), dpi=100,
        )
    else:
        save_skeleton_tpose(
            save_path=tpos_vis_path, parents=parents,
            positions=tpos_global_pos,
            figsize=(6.4, 6.4), dpi=100,
        )


# ===========================================================================
# Per-object pipeline
# ===========================================================================

def _process_motion_file(object_npz_path, ctx):
    """Process one motion NPZ into saved clips.

    ``ctx`` bundles the per-object state shared across every motion file of
    an object type (skeleton topology, scale, thresholds, save dirs,
    captions). Returns a dict with:
        n_saved        : int             — number of clips written to disk
        frames         : int             — total kept frames across clips
        clip_captions  : Dict[str, str]  — per-clip caption map
        filtered       : List[dict]      — entries dropped (with reason)
    """
    base_name = str(Path(object_npz_path).stem)
    min_frames = ctx['min_frames']

    # --- Load + motion-level canonicalize ----------------------------------
    result, skip_reason = load_and_canonicalize_motion(
        object_npz_path, ctx['raw_parents'], ctx['raw_njoints'],
        ctx['corps_idxs'], ctx['bfs_order'], ctx['fps'], ctx['scale_factor'],
        ctx['ground_height'], ctx['face_joint_idxs'],
        body_axis=ctx['body_axis'], raw_names=ctx.get('raw_names'))
    if result is None:
        return {'n_saved': 0, 'frames': 0, 'clip_captions': {},
                'filtered': [{'name': base_name, 'reason': skip_reason}]}
    motion_anim, motion_fps = result

    # --- Strip static pre/post-roll ----------------------------------------
    orig_nframes = len(motion_anim)
    motion_anim, _, _ = trim_static_ends(
        motion_anim, static_threshold=ctx['static_threshold'])
    if len(motion_anim) < min_frames:
        reason = (f'too few frames after static trim: {len(motion_anim)} '
                  f'< {min_frames} (orig {orig_nframes})')
        return {'n_saved': 0, 'frames': 0, 'clip_captions': {},
                'filtered': [{'name': base_name, 'reason': reason}]}

    # --- Clip windowing ----------------------------------------------------
    nframes = len(motion_anim)
    clips = get_motion_clip_windows(
        nframes, ctx['max_clip_len'], ctx['clip_stride'], apply_clip=ctx['apply_clip'])
    caption = ctx['captions'].get(base_name, '') if ctx['captions'] else ''

    clip_captions, filtered, n_saved, frames = {}, [], 0, 0
    for clip_idx, (clip_start, clip_end) in enumerate(clips):
        (clip_local_rotations, clip_global_positions,
         clip_root_facing_quat) = extract_clip_feats(
            motion_anim, clip_start, clip_end,
            ctx['face_joint_idxs'],
            body_axis=ctx['body_axis'],
        )

        is_low, clip_metrics = _activity_below_threshold(
            clip_global_positions, ctx['activity_threshold'])
        clip_name = f'{base_name}-{clip_idx:03d}'
        if is_low:
            filtered.append({
                'name': clip_name,
                'reason': f'low activity: {_format_activity(clip_metrics)}',
            })
            continue

        save_clip(
            clip_name,
            clip_global_positions, clip_local_rotations,
            clip_root_facing_quat,
            ctx['parents'], motion_fps,
            ctx['motions_save_dir'], ctx['animations_save_dir'],
            spectral_feats=ctx.get('spectral_feats'),
            caption=caption, save_vis=ctx['save_vis'],
            vis_ground=ctx.get('vis_ground', False))
        n_saved += 1
        frames += clip_end - clip_start
        if caption:
            clip_captions[clip_name] = caption

    return {
        'n_saved': n_saved,
        'frames': frames,
        'clip_captions': clip_captions,
        'filtered': filtered,
    }


def process_object(object_type, object_npzs, save_dir,
                   max_path_len=5, max_freqs=8,
                   max_clip_len=100, clip_stride=50, apply_clip=True,
                   captions=None,
                   face_joints=None, corps_names=None, clean_names=None,
                   expected_names=None, require_face=False,
                   target_diameter=2.0, activity_threshold=0.02,
                   static_threshold=1e-05, min_frames=8,
                   min_joints=None, max_joints=None,
                   use_tpos_ground_height=False, save_vis=True,
                   vis_ground=False):
    """Process all motion clips for one object type.

    Emits per-clip NPZs holding raw canonicalized arrays — global
    positions, local rotation quaternions, root facing quaternions —
    from which the training side builds its motion representation.

    When ``apply_clip`` is False, each motion yields a single clip truncated
    to the first ``max_clip_len`` frames; when True, motions are cropped
    into overlapping fixed-length clips with stride ``clip_stride``.

    ``use_tpos_ground_height`` controls whether the T-pose's ground height
    is reused for every motion (True) or recomputed per motion from its own
    minimum Y so each motion's lowest joint sits on the ground (False,
    default). Whichever is used is recorded in the cond as ``ground_height``
    / ``ground_height_mode``.

    Object types with no motion NPZs, a joint count outside
    ``[min_joints, max_joints]`` or a degenerate (zero-extent) skeleton are
    skipped with a reason instead of raising.

    Returns:
        Tuple of (cond, n_clips, n_frames, n_joints, filtered).
    """
    def _skip(reason, njoints=0):
        """Skip the whole object type with a recorded reason."""
        logger.info(f'[{object_type}] skipped: {reason}')
        return None, 0, 0, njoints, [{'name': f'{object_type} (all clips)',
                                      'reason': reason}]

    if not object_npzs:
        return _skip('no motion NPZ files found')

    motions_save_dir = pjoin(save_dir, 'motions')
    animations_save_dir = pjoin(save_dir, 'videos')
    for d in (motions_save_dir, animations_save_dir):
        os.makedirs(d, exist_ok=True)

    # ---- T-pose + topology ----
    tpos_data = np.load(object_npzs[0], allow_pickle=True)
    (corps_idxs, corps_names_list, clean_names_list,
     raw_parents, raw_njoints) = resolve_joint_names(
        object_type, tpos_data, corps_names, clean_names,
        expected_names=expected_names)

    # Joint count is fixed by the corps subset (BFS reordering preserves it),
    # so gate before the T-pose canonicalization rather than after.
    njoints = len(corps_idxs) if corps_idxs is not None else raw_njoints
    if ((min_joints is not None and njoints < min_joints) or
            (max_joints is not None and njoints > max_joints)):
        return _skip(f'{njoints} joints outside [{min_joints}, {max_joints}]',
                     njoints)

    try:
        (tpos_anim, offsets, scale_factor, tpos_ground_height,
         parents, names, fps, bfs_order,
         face_joint_idxs, body_axis) = process_tpose(
            tpos_data, corps_idxs=corps_idxs,
            corps_names=corps_names_list,
            face_joints=face_joints,
            target_diameter=target_diameter)
    except DegenerateSkeletonError as e:
        return _skip(str(e), njoints)

    face_named = bool(face_joints) and bool(
        face_joints.get('r_hip', {}).get('raw')
        or face_joints.get('l_hip', {}).get('raw'))
    if face_named and face_joint_idxs == [-1, -1]:
        # The entry exists but its raw names are not in this (possibly
        # corps-filtered) skeleton — canonicalization silently ran with
        # identity facing above. Never acceptable where facing is required.
        detail = (f'face_joints entry did not resolve: '
                  f'r_hip={face_joints.get("r_hip", {}).get("raw")!r}, '
                  f'l_hip={face_joints.get("l_hip", {}).get("raw")!r} '
                  f'not both present in the processed skeleton')
        if require_face:
            raise ValueError(f'[{object_type}] {detail}; refusing to fall '
                             f'back to identity facing for this dataset.')
        logger.warning(f'[{object_type}] {detail}; using identity facing.')

    # Ground height actually applied to the clips: the T-pose's when it is
    # reused, otherwise None (each motion is grounded on its own minimum Y).
    ground_height = tpos_ground_height if use_tpos_ground_height else None

    clean_names_bfs = [clean_names_list[i] for i in bfs_order]
    tpos_global_pos = positions_global(tpos_anim)[0]
    tpos_local_rot = tpos_anim.rotations.qs[0]
    tpos_global_rot = rotations_global(tpos_anim).qs[0]
    object_cond = build_topology_cond(
        object_type, parents, offsets, names, clean_names_bfs,
        tpos_global_pos, tpos_local_rot, tpos_global_rot,
        max_path_len=max_path_len, max_freqs=max_freqs,
        face_joint_idxs=face_joint_idxs, body_axis=body_axis,
        scale_factor=scale_factor, ground_height=ground_height
    )
    spectral_feats = object_cond.get('spectral_feats')

    _save_tpose_vis(save_dir, object_type, parents, tpos_anim,
                    spectral_feats=spectral_feats)

    logger.info(f'[{object_type}] n_joints={njoints} fps={fps} '
                f'scale={scale_factor:.4f} files={len(object_npzs)}')

    # ---- Per-motion processing ----
    ctx = {
        # skeleton / scale / topology
        'raw_parents': raw_parents, 'raw_njoints': raw_njoints,
        'raw_names': tpos_data['names'].tolist(),
        'corps_idxs': corps_idxs, 'bfs_order': bfs_order,
        'parents': parents,
        'fps': fps, 'scale_factor': scale_factor,
        'ground_height': ground_height,  # None => per-motion grounding
        'face_joint_idxs': face_joint_idxs, 'body_axis': body_axis,
        'spectral_feats': spectral_feats,
        # clip windowing / filtering
        'max_clip_len': max_clip_len, 'clip_stride': clip_stride,
        'apply_clip': apply_clip,
        'activity_threshold': activity_threshold,
        'static_threshold': static_threshold,
        'min_frames': min_frames,
        # I/O
        'motions_save_dir': motions_save_dir,
        'animations_save_dir': animations_save_dir,
        'captions': captions,
        'save_vis': save_vis, 'vis_ground': vis_ground,
    }

    filtered_clips, clip_captions = [], {}
    n_clips, n_frames = 0, 0

    for object_npz_path in object_npzs:
        result = _process_motion_file(object_npz_path, ctx)
        filtered_clips.extend(result.get('filtered', []))
        n_clips += result.get('n_saved', 0)
        n_frames += result.get('frames', 0)
        clip_captions.update(result.get('clip_captions', {}))

    if clip_captions:
        object_cond['captions'] = clip_captions

    logger.info(f'[{object_type}] saved {n_clips} clips ({n_frames} frames), '
                f'filtered {len(filtered_clips)}')

    return object_cond, n_clips, n_frames, njoints, filtered_clips
