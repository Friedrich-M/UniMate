"""Export a directory of rigged GLB/GLTF files (e.g. Objaverse) to NPZ motion data.

Pipeline per file: import scene → select armature/mesh → discover actions →
extract animations → Z-up → Y-up → prune skeleton (shared across clips) →
remove T-pose frames → save per-clip NPZ + MP4 + rest-pose PNG.

Files can be sharded across parallel Blender processes with
``--worker_id`` / ``--num_workers``; each worker writes its own summary
JSONs (``*_worker{i}.json``).

Usage (Blender headless):
    blender -b -P data_process/motion_export/export_objaverse.py -- \
        --data_dir dataset/raw/objaverse/glb \
        --output_dir dataset/export/objaverse
"""

import bpy
import sys
import os
import numpy as np
import argparse
from pathlib import Path
from loguru import logger
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from data_process.utils.blender_export import (
    import_gltf, list_gltf_files, sanitize_action_name, unique_clip_names,
    load_scene, prepare_skeleton, extract_all_actions, save_rest_pose_vis,
    prune_skeleton_shared, remove_tpose_frames, discover_pose_actions,
    save_motion, write_export_summary,
    is_asset_complete, mark_asset_complete,
)

# ─────────────────────────────────────────────────────────────────────────────
# Main export pipeline
# ─────────────────────────────────────────────────────────────────────────────

def export_objaverse(gltf_path, save_name, output_dir, fps=30, dtype=np.float64,
                     min_frames=5, consider_parent_rotate=True, save_vis=True):
    """Export every pose action in one GLB/GLTF file. Returns ``{save_name: joint_names}``."""
    assert os.path.exists(gltf_path), f"GLTF file not found: {gltf_path}"

    motion_save_dir = os.path.join(output_dir, "motions")
    vis_save_dir = os.path.join(output_dir, "videos")
    tpos_save_dir = os.path.join(output_dir, "tpose")

    if is_asset_complete(output_dir, save_name):
        logger.info(f"Asset '{save_name}' already complete, skipping: {gltf_path}")
        return {}
    existing = sorted(Path(motion_save_dir).glob(f"{save_name}-*.npz")) if os.path.isdir(motion_save_dir) else []
    if existing:
        # Pruning is joint across all of an asset's clips, so a partial
        # (or pre-marker) export must be redone as a whole.
        logger.warning(f"'{save_name}' has {len(existing)} clip(s) but no completion "
                       f"marker (partial or pre-marker export); re-exporting.")

    logger.info(f"Processing GLTF: {gltf_path}")
    os.makedirs(motion_save_dir, exist_ok=True)
    os.makedirs(vis_save_dir, exist_ok=True)

    armature, mesh = load_scene(import_gltf, gltf_path, fps=fps)
    assert mesh is not None, f"No mesh found in {gltf_path}; cannot build skin matrix."

    action_names, frame_ranges = discover_pose_actions()
    logger.info(f"Total relevant actions: {len(action_names)}")
    if not action_names:
        logger.info(f"No valid actions in {gltf_path}; skipping.")
        mark_asset_complete(output_dir, save_name, status="skipped",
                            reason="no pose actions")
        return {}

    skel = prepare_skeleton(armature, mesh, dtype=dtype, apply_world=True)
    extracted = extract_all_actions(
        armature, frame_ranges,
        skel['rest_local_pos'], skel['parents_array'],
        skel['rest_anim'], skel['nbones'], dtype=dtype,
    )

    anims_list = [anim for _, anim in extracted]
    anims_list, rest_anim_pruned, names_pruned, skin_matrix_pruned = prune_skeleton_shared(
        anims_list, skel['rest_anim_shared'], skel['bone_names'].copy(), skel['skin_matrix'].copy(),
        consider_parent_rotate=consider_parent_rotate,
    )

    # No joint-count gate here: stage 4 (feature_extraction --min_joints /
    # --max_joints) decides which skeletons enter training, so the export
    # stays complete.
    save_rest_pose_vis(tpos_save_dir, save_name, rest_anim_pruned)

    # Distinct actions can sanitize to the same name; disambiguate over the
    # full discovered list so the renderer derives identical clip names.
    clip_suffixes = unique_clip_names(action_names)

    scene_fps = bpy.context.scene.render.fps
    saved_clips = []
    for i, (action_name, _) in enumerate(extracted):
        anim = anims_list[i]

        anim, n_tpose = remove_tpose_frames(anim, rest_anim_pruned)
        if n_tpose > 0:
            logger.info(f"Removed {n_tpose} T-pose frames from '{action_name}', "
                        f"{anim.positions.shape[0]} frames remaining")
        if anim.positions.shape[0] < min_frames:
            logger.info(f"Skipping '{action_name}': only {anim.positions.shape[0]} frames "
                        f"after T-pose removal (min={min_frames})")
            continue

        clean_action_name = clip_suffixes.get(
            action_name, sanitize_action_name(action_name) or "action")
        clip_name = f"{save_name}-{clean_action_name}"
        motion_path = os.path.join(motion_save_dir, f"{clip_name}.npz")
        vis_path = os.path.join(vis_save_dir, f"{clip_name}.mp4")
        save_motion(motion_path, vis_path, anim, rest_anim_pruned,
                    names_pruned, skin_matrix_pruned, scene_fps,
                    action_name=action_name, save_vis=save_vis)
        saved_clips.append(clip_name)

    mark_asset_complete(output_dir, save_name,
                        status="exported" if saved_clips else "skipped",
                        n_clips=len(saved_clips),
                        reason="" if saved_clips else "no clips above min frame count",
                        joint_names=names_pruned if saved_clips else None)

    # Return once per asset (skeleton type), not per clip
    if saved_clips:
        return {save_name: names_pruned}
    return {}


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Export GLB/GLTF files to NPZ motion data.")
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Directory containing GLB/GLTF files.')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for NPZ motion data.')
    parser.add_argument('--fps', type=int, default=30,
                        help='Scene FPS for import and export.')
    parser.add_argument('--consider_parent_rotate', action=argparse.BooleanOptionalAction, default=True,
                        help='Preserve non-rotating leaves whose parent rotates during pruning '
                             '(see prune_skeleton_shared; use --no-consider_parent_rotate to disable).')
    parser.add_argument('--min_frames', type=int, default=5,
                        help='Skip clips shorter than this after T-pose removal.')
    parser.add_argument('--worker_id', type=int, default=0,
                        help="Worker index for sharding files across workers.")
    parser.add_argument('--num_workers', type=int, default=1,
                        help="Total number of workers.")
    parser.add_argument('--vis', action=argparse.BooleanOptionalAction, default=True,
                        help="Render the per-clip MP4 preview (use --no-vis for bulk runs).")

    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
    return parser.parse_args(argv)


def main():
    args = parse_args()

    gltf_paths = list_gltf_files(Path(args.data_dir))
    assert len(gltf_paths) > 0, f"No GLB/GLTF files found in {args.data_dir}"
    logger.info(f"Found {len(gltf_paths)} GLB/GLTF files in {args.data_dir}")

    # Shard files across workers
    if args.num_workers > 1:
        gltf_paths = gltf_paths[args.worker_id::args.num_workers]
        logger.info(f"Worker {args.worker_id}/{args.num_workers}: processing {len(gltf_paths)} files")

    os.makedirs(args.output_dir, exist_ok=True)
    error_log = os.path.join(args.output_dir, f'export_errors_worker{args.worker_id}.log')
    all_joint_names = {}
    n_failed = 0
    for glb_path in tqdm(gltf_paths, desc="Exporting GLB/GLTF files"):
        try:
            saved = export_objaverse(glb_path, glb_path.stem, output_dir=args.output_dir,
                                     fps=args.fps, min_frames=args.min_frames,
                                     consider_parent_rotate=args.consider_parent_rotate,
                                     save_vis=args.vis)
            if saved:
                all_joint_names.update(saved)
        except Exception as e:  # noqa: BLE001 — keep the batch going
            n_failed += 1
            logger.exception(f"Failed to export {glb_path}")
            with open(error_log, 'a') as log_file:
                log_file.write(f"Failed to export {glb_path}: {e}\n")

    worker_suffix = f"_worker{args.worker_id}" if args.num_workers > 1 else ""
    write_export_summary(args.output_dir, all_joint_names, worker_suffix=worker_suffix)

    if n_failed:
        logger.error(f"{n_failed}/{len(gltf_paths)} assets failed; see {error_log}")
        sys.exit(1)


if __name__ == "__main__":
    # Blender exits 0 even on an uncaught exception, which hides failures from
    # `set -e` and from Slurm's afterok dependencies.
    try:
        main()
    except SystemExit:
        raise
    except Exception:
        logger.exception("export_objaverse failed")
        sys.exit(1)
