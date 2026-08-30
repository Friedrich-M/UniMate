"""Merge per-worker export summary JSONs into the canonical files.

Multi-worker export runs write per-worker summary shards
(``joint_names_worker{i}.json``, ``joint_count_worker{i}.json``,
``clip_frames_worker{i}.json``) so parallel Blender processes never race
on one file. This script merges them — together with any existing canonical
files — into ``joint_names.json`` / ``joint_count.json`` /
``clip_frames.json`` (plus ``summary.json`` with the dataset-level
summary) and removes the shards. Idempotent: rerunning with no shards present
just rewrites the canonical files from themselves.

Plain Python (no bpy), so it runs in the conda env directly:
    python -m data_process.tools.merge_summaries --output_dir <export_dir>
"""

import argparse
import glob
import json
import os
import sys

from loguru import logger

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from data_process.utils.export_markers import collect_joint_names_from_markers  # noqa: E402


def _load(path):
    with open(path) as f:
        return json.load(f)


def _save(path, data):
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, path)


def _scan_missing_frame_counts(motion_dir, known):
    """Frame counts for clips in *motion_dir* that *known* doesn't cover."""
    if not os.path.isdir(motion_dir):
        return {}
    import numpy as np  # local: keeps the module importable without numpy

    found = {}
    for npz_file in sorted(os.listdir(motion_dir)):
        if not npz_file.endswith(".npz"):
            continue
        clip_name = os.path.splitext(npz_file)[0]
        if clip_name in known:
            continue
        try:
            data = np.load(os.path.join(motion_dir, npz_file), allow_pickle=True)
            found[clip_name] = int(data["anim_local_rot"].shape[0])
        except Exception as exc:  # noqa: BLE001 — a bad clip shouldn't stop the merge
            logger.warning(f"Could not read frame count from {npz_file}: {exc}")
    return found


def merge_summaries(output_dir, fps=30, keep_shards=False):
    jn_path = os.path.join(output_dir, "joint_names.json")
    joint_names = _load(jn_path) if os.path.isfile(jn_path) else {}

    # Per-asset completion markers are written as each asset finishes, so they
    # survive a killed worker whose in-memory shard never reached disk. Folding
    # them in here is what makes a truncated joint_names.json self-healing.
    from_markers = collect_joint_names_from_markers(output_dir)
    recovered = len(set(from_markers) - set(joint_names))
    joint_names.update(from_markers)
    if recovered:
        logger.info(f"Recovered {recovered} joint-name entries from completion markers")

    jn_shards = sorted(glob.glob(os.path.join(output_dir, "joint_names_worker*.json")))
    for shard in jn_shards:
        joint_names.update(_load(shard))
    if joint_names:
        _save(jn_path, joint_names)
        _save(os.path.join(output_dir, "joint_count.json"),
              {k: len(v) for k, v in joint_names.items()})
    logger.info(f"joint_names.json: {len(joint_names)} entries "
                f"({len(jn_shards)} shard(s) merged)")

    af_path = os.path.join(output_dir, "clip_frames.json")
    frames = {}
    if os.path.isfile(af_path):
        # Tolerate the legacy in-file summary block from older exports.
        frames = {k: v for k, v in _load(af_path).items() if k != "_summary"}
    af_shards = sorted(glob.glob(os.path.join(output_dir, "clip_frames_worker*.json")))
    for shard in af_shards:
        frames.update({k: v for k, v in _load(shard).items() if k != "_summary"})

    # Clips written after the last summary flush (or by a worker that was
    # killed before writing its shard) are on disk but absent here. Scan for
    # those only — the NPZ read is the expensive part, so already-recorded
    # clips are left alone.
    scanned = _scan_missing_frame_counts(os.path.join(output_dir, "motions"), frames)
    if scanned:
        logger.info(f"Recovered {len(scanned)} clip frame counts by scanning motions/")
        frames.update(scanned)

    if frames:
        _save(af_path, frames)
        total_frames = sum(frames.values())
        duration_sec = total_frames / fps if fps else 0.0
        _save(os.path.join(output_dir, "summary.json"), {
            "total_clips": len(frames),
            "total_frames": total_frames,
            "total_duration_sec": round(duration_sec, 2),
            "total_duration_min": round(duration_sec / 60, 2),
            "fps": fps,
        })
    logger.info(f"clip_frames.json: {len(frames)} clips "
                f"({len(af_shards)} shard(s) merged)")

    if not keep_shards:
        jc_shards = sorted(glob.glob(os.path.join(output_dir, "joint_count_worker*.json")))
        for shard in jn_shards + af_shards + jc_shards:
            os.remove(shard)


def main():
    parser = argparse.ArgumentParser(
        description="Merge per-worker export summary JSONs into the canonical files.")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Export directory holding the *_worker{i}.json shards.")
    parser.add_argument("--fps", type=int, default=30,
                        help="Frame rate for the clip_frames duration summary.")
    parser.add_argument("--keep_shards", action="store_true",
                        help="Keep the per-worker shard files after merging.")
    args = parser.parse_args()
    merge_summaries(args.output_dir, fps=args.fps, keep_shards=args.keep_shards)


if __name__ == "__main__":
    main()
