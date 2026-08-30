"""Metadata I/O and clip windowing for the feature-extraction stage.

Loads the per-dataset JSON side files produced by the caption and
joint-annotation stages (``motion_captions.json``, ``category_groups.json``,
``clean_joint_names.json``, ``face_joint_names.json``, ...), provides
case-insensitive per-object lookups, writes the stage outputs (``cond.npy``,
``captions.json``, ``filtered_clips.json``, ``metadata.txt``) and computes
clip windows. Used by :mod:`data_process.feature_extraction.extract_features`.
"""

import json
import os
import tempfile
from contextlib import contextmanager
from os.path import join as pjoin

import numpy as np


# ---------------------------------------------------------------------------
# Atomic output writes
# ---------------------------------------------------------------------------

@contextmanager
def atomic_output_path(path):
    # type: (str) -> Iterator[str]
    """Yield a temp path next to *path*, renamed into place on clean exit.

    Stage outputs land at the end of runs that can take hours, so writing them
    in place risks leaving a truncated ``cond.npy`` / ``captions.json`` behind
    if the run is interrupted. Writers fill the yielded temp path instead; the
    final ``os.replace`` is atomic, so readers see either the previous file or
    the complete new one, and a failed write leaves the previous file intact.
    """
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(
        dir=directory, prefix=f'.{os.path.basename(path)}.', suffix='.tmp')
    os.close(fd)
    try:
        yield tmp_path
        os.replace(tmp_path, path)
    except BaseException:
        try:
            os.remove(tmp_path)
        except OSError:
            pass
        raise


def atomic_np_save(path, data):
    # type: (str, object) -> None
    """``np.save`` *data* to *path* atomically (see :func:`atomic_output_path`)."""
    with atomic_output_path(path) as tmp_path:
        # Passing a file object keeps np.save from appending '.npy' to the
        # temp name (it only does that for string paths).
        with open(tmp_path, 'wb') as f:
            np.save(f, data)


# ---------------------------------------------------------------------------
# JSON loading & metadata helpers
# ---------------------------------------------------------------------------

def load_json(path):
    # type: (str) -> Optional[dict]
    """Load a JSON file if it exists, otherwise return None."""
    if path and os.path.isfile(path):
        with open(path, 'r') as f:
            return json.load(f)
    return None


def load_motion_captions(source_dir):
    # type: (str) -> Optional[Dict[str, str]]
    """Load motion captions from ``motion_captions.json`` in *source_dir*."""
    return load_json(pjoin(source_dir, 'motion_captions.json'))


def load_category_groups(source_dir):
    # type: (str) -> Optional[Dict[str, List[str]]]
    """Load category-to-object-type groupings from ``category_groups.json``."""
    return load_json(pjoin(source_dir, 'category_groups.json'))


def load_clean_joint_names(source_dir):
    # type: (str) -> Optional[Dict[str, List[str]]]
    """Load cleaned joint names from ``clean_joint_names.json``."""
    return load_json(pjoin(source_dir, 'clean_joint_names.json'))


def load_face_joint_names(source_dir):
    # type: (str) -> Optional[Dict[str, dict]]
    """Load face joint info from ``face_joint_names.json``."""
    return load_json(pjoin(source_dir, 'face_joint_names.json'))


# The 22-joint humanoid core the mixamo rig is reduced to at extraction time
# (drops the 40 finger bones plus the End bones). Replaces the retired
# corps_joint_names.json side file; toggled by --mixamo_core_joints.
MIXAMO_CORE_JOINTS = [
    "mixamorig:Hips",
    "mixamorig:Spine", "mixamorig:Spine1", "mixamorig:Spine2",
    "mixamorig:Neck", "mixamorig:Head",
    "mixamorig:LeftShoulder", "mixamorig:LeftArm",
    "mixamorig:LeftForeArm", "mixamorig:LeftHand",
    "mixamorig:RightShoulder", "mixamorig:RightArm",
    "mixamorig:RightForeArm", "mixamorig:RightHand",
    "mixamorig:LeftUpLeg", "mixamorig:LeftLeg",
    "mixamorig:LeftFoot", "mixamorig:LeftToeBase",
    "mixamorig:RightUpLeg", "mixamorig:RightLeg",
    "mixamorig:RightFoot", "mixamorig:RightToeBase",
]


def load_joint_names(source_dir):
    # type: (str) -> Optional[Dict[str, List[str]]]
    """Load raw joint names from ``joint_names.json``."""
    return load_json(pjoin(source_dir, 'joint_names.json'))


def load_all_metadata(source_dir):
    # type: (str) -> dict
    """Load all available JSON metadata files from *source_dir*."""
    loaders = {
        'motion_captions': load_motion_captions,
        'category_groups': load_category_groups,
        'clean_joint_names': load_clean_joint_names,
        'face_joint_names': load_face_joint_names,
        'joint_names': load_joint_names,
    }
    meta = {}
    for key, loader in loaders.items():
        data = loader(source_dir)
        if data is not None:
            meta[key] = data
    return meta


def check_metadata_object_types_consistent(
    metadata,
    keys=('clean_joint_names', 'face_joint_names', 'joint_names'),
):
    # type: (dict, tuple) -> None
    """Verify per-object-type metadata dicts cover the same object types.

    Compares the case-insensitive key sets of each present dict named by
    *keys* in *metadata*; absent or empty dicts are skipped. Raises
    ``ValueError`` on mismatch so drivers fail loudly on misaligned JSON
    sources, and prints a consistency-OK line when all present dicts agree.
    """
    present = {n: {k.lower() for k in metadata[n].keys()}
               for n in keys if metadata.get(n)}
    if len(present) < 2:
        return
    ref_name, ref_set = next(iter(present.items()))
    for other_name, other_set in present.items():
        if other_name == ref_name:
            continue
        only_ref = sorted(ref_set - other_set)
        only_other = sorted(other_set - ref_set)
        if only_ref or only_other:
            raise ValueError(
                f"Object type mismatch between '{ref_name}' and '{other_name}': "
                f"only in {ref_name}: {only_ref}; "
                f"only in {other_name}: {only_other}"
            )
    print(f'Metadata consistency OK across {list(present.keys())}: '
          f'{len(ref_set)} object types')


def _dict_get_ci(d, key):
    # type: (dict, str) -> object
    """Case-insensitive dict lookup. Tries exact match first, then lowercase."""
    if key in d:
        return d[key]
    key_lower = key.lower()
    for k, v in d.items():
        if k.lower() == key_lower:
            return v
    return None


def get_object_metadata(object_type, metadata_dict):
    # type: (str, Optional[dict]) -> object
    """Case-insensitive lookup of *object_type* in a metadata dict, or None."""
    if metadata_dict:
        return _dict_get_ci(metadata_dict, object_type)
    return None


def get_object_face_joints(object_type, face_joint_names=None):
    # type: (str, Optional[Dict[str, dict]]) -> Optional[dict]
    """Return face joint info for *object_type*, or None."""
    return get_object_metadata(object_type, face_joint_names)


def get_object_captions(object_type, motion_captions=None):
    # type: (str, Optional[Dict[str, str]]) -> Dict[str, str]
    """Filter motion captions to entries for *object_type*.

    Caption keys follow the ``{object_type}-{motion_name}`` base-name
    convention used throughout the pipeline; matching is case-insensitive
    on the prefix. Returns an empty dict when no captions are supplied or
    none match.
    """
    if not motion_captions:
        return {}
    prefix = f'{object_type.lower()}-'
    return {k: v for k, v in motion_captions.items()
            if k.lower().startswith(prefix)}


def save_json(path, data):
    """Write *data* as indented JSON to *path*, atomically."""
    with atomic_output_path(path) as tmp_path:
        with open(tmp_path, 'w') as f:
            json.dump(data, f, indent=2)


def collect_captions(cond):
    # type: (Dict[str, dict]) -> Dict[str, str]
    """Merge per-object ``captions`` sub-dicts into a single flat dict."""
    all_captions = {}
    for obj_cond in cond.values():
        if 'captions' in obj_cond:
            all_captions.update(obj_cond['captions'])
    return all_captions


def save_outputs(save_dir, cond, all_filtered_clips, category_groups=None):
    """Save ``cond.npy``, filtered clips, captions, and category groupings.

    Captions and category groups are only written when present. Every file is
    written atomically, so an interrupted run cannot leave a truncated output
    behind for the training loader to trip over. Returns the merged captions
    dict so callers can pass it to ``save_metadata_report``.
    """
    atomic_np_save(pjoin(save_dir, 'cond.npy'), cond)

    if all_filtered_clips:
        save_json(pjoin(save_dir, 'filtered_clips.json'), all_filtered_clips)
        total = sum(len(v) for v in all_filtered_clips.values())
        print(f'Saved {total} filtered clips to filtered_clips.json')

    all_captions = collect_captions(cond)
    if all_captions:
        save_json(pjoin(save_dir, 'captions.json'), all_captions)
        print(f'Saved {len(all_captions)} clip captions')

    if category_groups:
        save_json(pjoin(save_dir, 'category_groups.json'), category_groups)

    return all_captions


def build_stats(clips_per_object, joints_per_object, total_frames, max_njoints, fps=30):
    """Aggregate per-object counts into the dataset summary dict.

    ``duration_min`` is a nominal figure: clips carry their own ``fps`` field
    (30 for every source, 30 after the >=60 fps down-sampling in
    :func:`~data_process.utils.motion_features._load_npz_anim`), but the
    duration here simply divides the frame total by *fps*. The assumption is
    reported alongside the value, so ``fps`` is echoed back in the stats.
    """
    return {
        'total_clips': sum(clips_per_object.values()),
        'total_frames': total_frames,
        'duration_min': total_frames / fps / 60,
        'fps': fps,
        'max_njoints': max_njoints,
        'clips_per_object': clips_per_object,
        'joints_per_object': joints_per_object,
    }


def save_metadata_report(save_dir, stats, all_filtered_clips,
                         all_captions=None, category_groups=None):
    """Write a human-readable metadata summary to ``metadata.txt``.

    ``all_captions`` and ``category_groups`` are optional; their sections are
    omitted when not supplied. The file is written atomically.
    """
    total_filtered = sum(len(v) for v in all_filtered_clips.values())

    with atomic_output_path(pjoin(save_dir, 'metadata.txt')) as tmp_path, \
            open(tmp_path, 'w') as f:
        f.write(f'Total clips: {stats["total_clips"]}\n')
        f.write(f'Total frames: {stats["total_frames"]}\n')
        f.write(f'Total duration (minutes, assuming {stats.get("fps", 30)} fps): '
                f'{stats["duration_min"]:.2f}\n')
        f.write(f'Max joints in dataset: {stats["max_njoints"]}\n')
        if all_captions is not None:
            f.write(f'Total captions: {len(all_captions)}\n')

        f.write('Clips per object type:\n')
        for obj, count in stats['clips_per_object'].items():
            f.write(f'  {obj}: {count}\n')

        f.write('Joints per object type:\n')
        for obj, count in stats['joints_per_object'].items():
            f.write(f'  {obj}: {count}\n')

        f.write(f'Total filtered clips: {total_filtered}\n')
        if all_filtered_clips:
            f.write('Filtered clips per object type:\n')
            for obj, clips in all_filtered_clips.items():
                f.write(f'  {obj}: {len(clips)}\n')

        if category_groups:
            f.write('Categories:\n')
            for cat, members in category_groups.items():
                processed = [m for m in members if m in stats['clips_per_object']]
                f.write(f'  {cat}: {len(processed)} object types\n')


# ---------------------------------------------------------------------------
# Clip windowing
# ---------------------------------------------------------------------------

def compute_clip_windows(nframes, max_clip_len, clip_stride):
    """Compute (start, end) windows for cropping a sequence into clips.

    Windows are ``max_clip_len`` frames long and start every ``clip_stride``
    frames. No returned window is ever longer than ``max_clip_len`` (the whole
    sequence is returned unsliced only when it already fits).

    Leftover frames past the last strided window are handled by tail length:

    - at least half a clip: emitted as their own (shorter) trailing window;
    - shorter than that, but still adding at least as many new frames as a
      normal window's overlap (``max_clip_len - clip_stride``): covered by a
      full-length window anchored at the final frame — replacing the last
      window when the frames it gives up stay covered by the one before it,
      otherwise appended;
    - shorter than that: dropped, rather than glued onto the previous window
      (which is what used to push clips past ``max_clip_len``).
    """
    if not max_clip_len or nframes <= max_clip_len:
        return [(0, nframes)]

    clips = []
    for start in range(0, nframes - max_clip_len + 1, clip_stride):
        clips.append((start, start + max_clip_len))

    last_end = clips[-1][1]
    if last_end < nframes:
        tail_len = nframes - last_end
        if tail_len >= max_clip_len // 2:
            clips.append((last_end, nframes))
        else:
            # Cover the tail with a full-length window anchored at the end.
            slid_start = nframes - max_clip_len
            prev_end = clips[-2][1] if len(clips) >= 2 else 0
            if slid_start <= prev_end:
                # Free: the frames the last window gives up are still covered.
                clips[-1] = (slid_start, nframes)
            elif tail_len >= max(1, max_clip_len - clip_stride):
                # Worth an extra window: it adds at least as many new frames
                # as a normal window's overlap.
                clips.append((slid_start, nframes))
            # Otherwise the tail is too small to pay for a window of its own.

    return clips


def get_motion_clip_windows(nframes, max_clip_len, clip_stride, apply_clip=True):
    """Return clip windows for a motion sequence.

    When ``apply_clip`` is True, crops into overlapping fixed-length clips via
    :func:`compute_clip_windows`. When False, returns a single window covering
    at most the first ``max_clip_len`` frames (i.e., truncation, not cropping).
    """
    if apply_clip:
        return compute_clip_windows(nframes, max_clip_len, clip_stride)
    clip_end = min(nframes, max_clip_len) if max_clip_len else nframes
    return [(0, clip_end)]


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def print_summary(stats):
    print(f'Total clips: {stats["total_clips"]}, Frames: {stats["total_frames"]}, '
          f'Duration: {stats["duration_min"]:.1f}m (@{stats.get("fps", 30)} fps), '
          f'Max joints: {stats["max_njoints"]}')
