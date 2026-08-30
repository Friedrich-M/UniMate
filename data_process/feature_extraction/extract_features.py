"""Process exported NPZ motions into canonicalized training clips.

Supports three dataset layouts via ``--dataset_type``:

- ``truebones``: per-object NPZ files under ``data_dir/motions``, named
  ``{object_type}-{motion}.npz``. Object types are discovered from filename
  prefixes.
- ``objaverse``: same layout as truebones, plus an optional
  ``filtered_objects.txt`` listing object types to skip.
- ``mixamo``: single skeleton / object type. All NPZs live directly under
  ``data_dir/motions`` with no ``{object_type}-`` prefix; caption keys are
  bare motion stems.

Stage-2 (captions, category groups) and stage-3 (clean / face joint names)
metadata JSONs are read from ``data_dir`` — see
:mod:`data_process.feature_extraction.metadata`.

Object types are independent, so they can be processed in parallel
(``--num_workers``) and are resumable: each finished object type caches its
result under ``<save_dir>/cond_parts/`` and is skipped on rerun. The cache
records the clip / threshold / topology settings it was built with and is
re-processed automatically when they change; delete ``cond_parts/`` (or one
entry) to force re-processing for any other reason.

Object types that fail are logged to ``<save_dir>/extract_errors.log``,
recorded in ``filtered_clips.json`` and skipped (no cache entry is written,
so they are retried on the next run) — one bad object never aborts the run.

Usage:
    python -m data_process.feature_extraction.extract_features \\
        --dataset_type truebones \\
        --data_dir dataset/export/truebones \\
        --save_dir dataset/features/truebones
"""

import argparse
import hashlib
import json
import multiprocessing as mp
import os
from os.path import join as pjoin

import numpy as np
from loguru import logger
from tqdm import tqdm

from data_process.feature_extraction.metadata import (
    MIXAMO_CORE_JOINTS,
    atomic_np_save,
    build_stats,
    check_metadata_object_types_consistent,
    get_object_captions,
    get_object_face_joints,
    get_object_metadata,
    load_all_metadata,
    print_summary,
    save_metadata_report,
    save_outputs,
)
from data_process.utils.motion_features import process_object


MIXAMO_OBJECT_TYPE = 'mixamo'


def parse_args():
    parser = argparse.ArgumentParser(
        description="Process exported NPZ motions into canonicalized training clips.")
    # Dataset selection
    parser.add_argument("--dataset_type", type=str, required=True,
                        choices=['truebones', 'objaverse', 'mixamo'],
                        help="Which dataset layout to process")
    # Directories
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Root data directory (motions under data_dir/motions, "
                             "metadata JSONs in data_dir)")
    parser.add_argument("--save_dir", type=str, required=True,
                        help="Output directory for processed clips and metadata")
    # Clip windowing
    parser.add_argument("--max_clip_len", type=int, default=180,
                        help="Max frames per saved clip")
    parser.add_argument("--diffusion_max_len", type=int, default=60,
                        help="Model training max length (dataloader random-crops to this)")
    parser.add_argument("--apply_clip", action="store_true", default=False,
                        help="Crop long motions into overlapping fixed-length clips of "
                             "max_clip_len frames (stride = max_clip_len - "
                             "diffusion_max_len). Otherwise (default), keep only the "
                             "first max_clip_len frames of each motion.")
    # Topology
    parser.add_argument("--max_path_len", type=int, default=5,
                        help="Max path length for topology edge relations")
    parser.add_argument("--max_freqs", type=int, default=8,
                        help="Number of Laplacian eigenvector frequencies for "
                             "spectral joint features")
    # Scaling / grounding
    parser.add_argument("--target_diameter", type=float, default=2.0,
                        help="Target skeleton diameter for leaf-to-leaf scaling")
    parser.add_argument("--use_tpos_ground_height", action="store_true", default=False,
                        help="Ground every motion of an object type on the T-pose's "
                             "floor height. Otherwise (default), each motion is "
                             "grounded on its own minimum Y. The applied value is "
                             "recorded in cond as ground_height / ground_height_mode.")
    # Filtering
    parser.add_argument("--activity_threshold", type=float, default=0.02,
                        help="Min joint activity (temporal spread at canonical "
                             "scale) to keep a clip")
    parser.add_argument("--static_threshold", type=float, default=1e-5,
                        help="Per-frame max-joint displacement below which a frame "
                             "is static; leading/trailing static frames are trimmed")
    parser.add_argument("--min_frames", type=int, default=8,
                        help="Minimum frame count for a motion to be kept "
                             "(checked after downsampling and static trimming)")
    parser.add_argument("--mixamo_core_joints", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="(mixamo only) Restrict the rig to the built-in "
                             "22-joint humanoid core (drops finger chains and "
                             "End bones) — replaces the retired "
                             "corps_joint_names.json. Use --no-mixamo_core_joints "
                             "to keep the full 65-joint skeleton.")
    parser.add_argument("--min_joints", type=int, default=8,
                        help="Skip object types with fewer joints than this "
                             "(counted after corps filtering)")
    parser.add_argument("--max_joints", type=int, default=150,
                        help="Skip object types with more joints than this "
                             "(counted after corps filtering)")
    # Objaverse-only filtering
    parser.add_argument("--filtered_objects", type=str, default="auto",
                        help="(objaverse only) Path to a txt file listing object "
                             "types to skip (one per line; blanks and '#' comments "
                             "ignored). Default 'auto' uses "
                             "<data_dir>/filtered_objects.txt if present. Pass an "
                             "empty string to disable.")
    # Execution
    parser.add_argument("--num_workers", type=int, default=1,
                        help="Parallel worker processes (object types are independent)")
    parser.add_argument("--vis", action=argparse.BooleanOptionalAction, default=True,
                        help="Render the per-clip MP4 preview (use --no-vis for bulk runs)")
    parser.add_argument("--vis_ground", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="Render previews with a checkerboard ground plane, "
                             "follow camera and root trajectory (default). "
                             "--no-vis_ground falls back to the legacy plain "
                             "cubic view.")
    return parser.parse_args()


def resolve_filtered_objects_path(path, data_dir):
    if path == "auto":
        default = pjoin(data_dir, 'filtered_objects.txt')
        return default if os.path.isfile(default) else None
    return path or None


def load_filtered_objects(path):
    if not path:
        return set()
    with open(path, 'r') as f:
        return {
            line.strip() for line in f
            if line.strip() and not line.startswith('#')
        }


def discover_object_types(motion_dir, dataset_type, args):
    """Return (object_types, all_motions) given the dataset layout.

    Only ``.npz`` files are considered, so stray side files in ``motions/``
    never turn into object types with zero motions.
    """
    entries = os.listdir(motion_dir)
    all_motions = [f for f in entries if f.endswith('.npz')]
    if len(all_motions) != len(entries):
        logger.info(f'Ignoring {len(entries) - len(all_motions)} non-NPZ '
                    f'entries in {motion_dir}')

    if dataset_type == 'mixamo':
        # Single object type; no prefix-based grouping.
        return [MIXAMO_OBJECT_TYPE], all_motions

    object_types = sorted(set(m.split('-')[0] for m in all_motions))
    logger.info(f'Found {len(all_motions)} motion files, {len(object_types)} object types')

    if dataset_type == 'objaverse':
        filtered_path = resolve_filtered_objects_path(args.filtered_objects, args.data_dir)
        filtered_object_types = load_filtered_objects(filtered_path)
        if filtered_object_types:
            before = len(object_types)
            object_types = [o for o in object_types if o not in filtered_object_types]
            logger.info(f'Loaded {len(filtered_object_types)} entries from {filtered_path}; '
                        f'{before} -> {len(object_types)} object types after filtering')
        elif args.filtered_objects:
            logger.info(f'No filtered_objects file found '
                        f'(--filtered_objects={args.filtered_objects!r}); '
                        f'processing all {len(object_types)} object types')

    return object_types, all_motions


def collect_object_npzs(object_type, dataset_type, motion_dir, all_motions):
    if dataset_type == 'mixamo':
        return sorted(pjoin(motion_dir, f) for f in all_motions if f.endswith('.npz'))
    return sorted(
        pjoin(motion_dir, f) for f in all_motions
        if f.startswith(object_type + '-') and f.endswith('.npz')
    )


def resolve_captions(object_type, dataset_type, motion_captions):
    # Mixamo caption keys are bare motion stems (no "{object_type}-" prefix),
    # so the full captions dict is passed through untouched.
    if dataset_type == 'mixamo':
        return motion_captions
    return get_object_captions(object_type, motion_captions)


def _object_part_path(save_dir, object_type):
    return pjoin(save_dir, 'cond_parts', f'{object_type}.npy')


def _error_log_path(save_dir):
    return pjoin(save_dir, 'extract_errors.log')


# Settings that determine a cached object result's contents. A cache entry
# built with different values is stale and gets re-processed. ``save_vis``
# is deliberately excluded: it only controls the MP4 previews, not the clips
# or the cond.
CACHE_PARAM_KEYS = (
    'max_clip_len', 'clip_stride', 'apply_clip',
    'max_path_len', 'max_freqs',
    'target_diameter', 'activity_threshold', 'static_threshold',
    'min_frames', 'min_joints', 'max_joints',
    'use_tpos_ground_height',
    'corps_names',
)

# Per-object METADATA inputs baked into the result (captions land in the
# cond, face joints steer the facing canonicalization, clean names are
# stored per joint). Hashed into the cache key so a stage-2/3 rerun —
# new captions, corrected face pairs, recleaned labels — re-processes the
# affected object types instead of silently resuming stale results.
CACHE_DATA_KEYS = ('captions', 'face_joints', 'clean_names')


def _digest(value):
    """Stable short digest of a JSON-serializable metadata value."""
    blob = json.dumps(value, sort_keys=True, default=str)
    return hashlib.sha1(blob.encode('utf-8')).hexdigest()[:16]


def _cache_params(task):
    """Extract the cache-invalidating settings from a task dict."""
    params = {k: task[k] for k in CACHE_PARAM_KEYS}
    for k in CACHE_DATA_KEYS:
        params[f'{k}_digest'] = _digest(task[k])
    return params


def _params_hash(params):
    """Stable hash of the cache-invalidating settings."""
    blob = json.dumps(params, sort_keys=True, default=str)
    return hashlib.sha1(blob.encode('utf-8')).hexdigest()


def _load_cached_result(part_path, object_type, params, params_hash):
    """Return a cached object result, or None when absent / stale / unreadable."""
    if not os.path.isfile(part_path):
        return None
    try:
        payload = np.load(part_path, allow_pickle=True).item()
    except Exception as e:  # noqa: BLE001 — a bad cache entry is just a miss
        logger.warning(f'[{object_type}] unreadable cache {part_path} ({e}); '
                       f're-processing')
        return None

    if 'params_hash' not in payload:
        logger.info(f'[{object_type}] cache invalidated (entry predates settings '
                    f'tracking); re-processing')
        return None
    if payload['params_hash'] == params_hash:
        return payload.get('result')

    cached_params = payload.get('params') or {}
    diff = ', '.join(
        f'{k}: {cached_params.get(k, "<missing>")} -> {v}'
        for k, v in sorted(params.items())
        if cached_params.get(k, '<missing>') != v)
    logger.info(f'[{object_type}] cache invalidated (settings changed: {diff}); '
                f're-processing')
    return None


def process_object_task(task):
    """Process one object type, caching its result for resume.

    ``task`` bundles every :func:`process_object` argument plus the part
    path. The cached result is reused only when it was produced with the same
    clip / threshold / topology settings (see ``CACHE_PARAM_KEYS``).

    Failures are contained here: the object is reported as failed (and its
    error appended to ``extract_errors.log``) instead of propagating, which
    with ``mp.Pool`` would kill the whole run. No cache entry is written for
    a failed object, so it is retried on the next run.

    Returns ``(object_type, result_dict, from_cache)``.
    """
    object_type = task.pop('object_type')
    part_path = task.pop('part_path')
    params = _cache_params(task)
    params_hash = _params_hash(params)

    cached = _load_cached_result(part_path, object_type, params, params_hash)
    if cached is not None:
        return object_type, cached, True

    try:
        obj_cond, n_clips, n_frames, n_joints, filtered = process_object(
            object_type, **task)
    except Exception as e:  # noqa: BLE001 — keep the batch going
        logger.exception(f'[{object_type}] failed to process')
        try:
            os.makedirs(task['save_dir'], exist_ok=True)
            with open(_error_log_path(task['save_dir']), 'a') as log_file:
                log_file.write(f'{object_type}\t{type(e).__name__}: {e}\n')
        except OSError as log_err:
            logger.warning(f'[{object_type}] could not write error log: {log_err}')
        return object_type, {
            'cond': None, 'n_clips': 0, 'n_frames': 0, 'n_joints': 0,
            'filtered': [{'name': f'{object_type} (all clips)',
                          'reason': f'processing failed: {type(e).__name__}: {e}'}],
            'error': f'{type(e).__name__}: {e}',
        }, False

    result = {'cond': obj_cond, 'n_clips': n_clips, 'n_frames': n_frames,
              'n_joints': n_joints, 'filtered': filtered}
    # Atomic: a worker killed mid-write can never leave a half-written cache
    # entry behind (unreadable ones are treated as misses, but this keeps them
    # from happening in the first place).
    atomic_np_save(part_path, {'params_hash': params_hash, 'params': params,
                               'result': result})
    return object_type, result, False


def build_object_tasks(args, clip_stride, motion_dir, metadata):
    face_joint_names = metadata.get('face_joint_names')
    clean_joint_names = metadata.get('clean_joint_names')
    joint_names = metadata.get('joint_names')
    motion_captions = metadata.get('motion_captions')

    object_types, all_motions = discover_object_types(motion_dir, args.dataset_type, args)

    tasks = []
    for object_type in object_types:
        face_joints = get_object_face_joints(object_type, face_joint_names)
        if not face_joints:
            if args.dataset_type == 'mixamo':
                raise ValueError(f'No face_joints entry for {object_type}; '
                                 f'face_joint_names.json is required for mixamo')
            logger.info(f'No face_joints entry for {object_type}; using identity facing')

        tasks.append(dict(
            object_type=object_type,
            part_path=_object_part_path(args.save_dir, object_type),
            object_npzs=collect_object_npzs(object_type, args.dataset_type,
                                            motion_dir, all_motions),
            save_dir=args.save_dir,
            max_clip_len=args.max_clip_len, clip_stride=clip_stride,
            apply_clip=args.apply_clip,
            max_path_len=args.max_path_len, max_freqs=args.max_freqs,
            captions=resolve_captions(object_type, args.dataset_type, motion_captions),
            face_joints=face_joints,
            corps_names=(MIXAMO_CORE_JOINTS
                         if args.dataset_type == 'mixamo'
                         and args.mixamo_core_joints else None),
            clean_names=get_object_metadata(object_type, clean_joint_names),
            expected_names=get_object_metadata(object_type, joint_names),
            require_face=args.dataset_type == 'mixamo',
            target_diameter=args.target_diameter,
            use_tpos_ground_height=args.use_tpos_ground_height,
            activity_threshold=args.activity_threshold,
            static_threshold=args.static_threshold,
            min_frames=args.min_frames,
            min_joints=args.min_joints, max_joints=args.max_joints,
            save_vis=args.vis, vis_ground=args.vis_ground,
        ))
    return tasks


def main(args):
    clip_stride = args.max_clip_len - args.diffusion_max_len
    if clip_stride <= 0:
        message = (f'--diffusion_max_len ({args.diffusion_max_len}) must be smaller '
                   f'than --max_clip_len ({args.max_clip_len}): the clip stride is '
                   f'their difference, and {clip_stride} would '
                   + ('never advance the window'
                      if clip_stride == 0 else 'walk the window backwards'))
        if args.apply_clip:
            raise ValueError(message)
        logger.warning(f'{message}. Harmless here because --apply_clip is off '
                       f'(the stride is unused), but fix it before enabling it.')

    os.makedirs(args.save_dir, exist_ok=True)

    motion_dir = pjoin(args.data_dir, 'motions')

    metadata = load_all_metadata(args.data_dir)
    logger.info(f'Loaded metadata from {args.data_dir}: '
                + ', '.join(f'{k}={len(v)}' for k, v in metadata.items()))
    check_metadata_object_types_consistent(metadata)
    category_groups = metadata.get('category_groups')

    tasks = build_object_tasks(args, clip_stride, motion_dir, metadata)

    cond = {}
    all_filtered_clips = {}
    clips_per_object = {}
    joints_per_object = {}
    total_frames = 0
    max_njoints = 0
    n_cached = 0
    failed_objects = {}

    def accumulate(object_type, result, from_cache):
        nonlocal total_frames, max_njoints, n_cached
        n_cached += int(from_cache)
        if result.get('error'):
            failed_objects[object_type] = result['error']
        if result['n_clips']:
            cond[object_type] = result['cond']
            clips_per_object[object_type] = result['n_clips']
            joints_per_object[object_type] = result['n_joints']
            total_frames += result['n_frames']
            max_njoints = max(max_njoints, result['n_joints'])
        if result['filtered']:
            all_filtered_clips[object_type] = result['filtered']

    pbar = tqdm(total=len(tasks), desc="Processing objects", unit="obj")
    if args.num_workers > 1:
        with mp.Pool(args.num_workers) as pool:
            for object_type, result, from_cache in pool.imap_unordered(
                    process_object_task, tasks):
                accumulate(object_type, result, from_cache)
                pbar.update(1)
    else:
        for task in tasks:
            accumulate(*process_object_task(task))
            pbar.update(1)
    pbar.close()

    if n_cached:
        logger.info(f'Resumed {n_cached}/{len(tasks)} object types from cond_parts/ '
                    f'(delete that directory to force re-processing)')
    if failed_objects:
        logger.error(f'{len(failed_objects)}/{len(tasks)} object types failed and were '
                     f'skipped (retried on the next run); see '
                     f'{_error_log_path(args.save_dir)}: '
                     + ', '.join(sorted(failed_objects)))

    all_captions = save_outputs(args.save_dir, cond, all_filtered_clips,
                                category_groups=category_groups)
    stats = build_stats(clips_per_object, joints_per_object, total_frames, max_njoints)
    print_summary(stats)
    save_metadata_report(args.save_dir, stats, all_filtered_clips,
                         all_captions=all_captions, category_groups=category_groups)
    logger.info('Dataset processing complete.')


if __name__ == "__main__":
    main(parse_args())
