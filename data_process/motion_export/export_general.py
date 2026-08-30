"""Export rigged assets (GLB/GLTF or FBX) to NPZ motion data.

Generic counterpart to the per-dataset batch exporters: ``--input`` is either
a single file or a directory of ``.glb``/``.gltf``/``.fbx`` assets (mixed
formats welcome). The importer is picked per file by extension, the character
mesh is optional, and every pose action found in a file is exported as its
own clip. Directory runs mirror the Objaverse exporter: per-asset completion
markers for resume, worker sharding, a per-worker error log, and per-worker
summary shards (merge with ``data_process.tools.merge_summaries``).

Pipeline: import scene → select armature (+ mesh if present) → discover
actions → build skeleton arrays → extract animations → optional skeleton
pruning → coordinate conversion → save per-action NPZ + MP4 visualization.

Behavior notes:
    - Mesh optional: with a mesh, skin weights are extracted and skeleton
      pruning is enabled by default; without one, ``skin_matrix`` is saved
      as an empty ``(0, nbones)`` array and pruning is skipped (the pruning
      passes rely on skin weights to decide which bones matter).
    - Actions: all relevant pose actions are exported. Files whose action
      does not pass the pose-action filter (e.g. Mixamo animation-only
      clips) fall back to the armature's single bound action.
    - No joint-count filtering: stage 4 (feature_extraction --min_joints /
      --max_joints) decides which skeletons enter training.

Usage (Blender headless):
    blender -b -P data_process/motion_export/export_general.py -- \
        --input assets_dir --output_dir outputs/export

Usage (pip-installed bpy):
    python data_process/motion_export/export_general.py \
        --input assets/dragon.fbx --output_dir outputs/export
"""

import bpy
import sys
import os
import numpy as np
import argparse
from pathlib import Path
from tqdm import tqdm
from loguru import logger

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from data_process.utils.blender_export import (
    import_fbx, import_gltf, sanitize_action_name, unique_clip_names,
    load_scene, prepare_skeleton, extract_all_actions, save_rest_pose_vis,
    prune_skeleton_shared, remove_tpose_frames, discover_pose_actions,
    action_frame_range, save_motion, write_export_summary,
    is_asset_complete, mark_asset_complete,
)

GLTF_EXTS = {'.glb', '.gltf'}
FBX_EXTS = {'.fbx'}
ASSET_EXTS = GLTF_EXTS | FBX_EXTS


def pick_importer(path):
    """Return the importer matching *path*'s extension, or raise."""
    ext = Path(path).suffix.lower()
    if ext in GLTF_EXTS:
        return import_gltf
    if ext in FBX_EXTS:
        return import_fbx
    raise ValueError(f"Unsupported extension '{ext}' (expected one of "
                     f"{sorted(ASSET_EXTS)}): {path}")


def list_asset_files(input_dir):
    """List .glb/.gltf/.fbx files directly under *input_dir*, sorted."""
    input_dir = Path(input_dir)
    return sorted(p for p in input_dir.iterdir()
                  if p.is_file() and p.suffix.lower() in ASSET_EXTS)


def export_asset(input_path, output_dir, save_name=None, fps=30,
                 dtype=np.float64, prune=None, consider_parent_rotate=True,
                 min_frames=5, remove_tpose=True, save_vis=True):
    """Export every pose action in a single GLB/GLTF/FBX file to NPZ clips.

    Args:
        prune: True/False to force skeleton pruning on/off; None (default)
            enables it iff the file contains a skinned mesh.

    Returns:
        {save_name: joint_names} when at least one clip was saved, else {}.
    """
    assert os.path.exists(input_path), f"Input file not found: {input_path}"
    save_name = save_name or Path(input_path).stem

    motion_save_dir = os.path.join(output_dir, "motions")
    vis_save_dir = os.path.join(output_dir, "videos")
    tpos_save_dir = os.path.join(output_dir, "tpose")

    if is_asset_complete(output_dir, save_name):
        logger.info(f"Asset '{save_name}' already complete, skipping: {input_path}")
        return {}
    existing = sorted(Path(motion_save_dir).glob(f"{save_name}-*.npz")) if os.path.isdir(motion_save_dir) else []
    if existing:
        # Pruning is joint across all of an asset's clips, so a partial
        # (or pre-marker) export must be redone as a whole.
        logger.warning(f"'{save_name}' has {len(existing)} clip(s) but no completion "
                       f"marker (partial or pre-marker export); re-exporting.")

    logger.info(f"Processing asset: {input_path}")
    os.makedirs(motion_save_dir, exist_ok=True)
    os.makedirs(vis_save_dir, exist_ok=True)

    armature, mesh = load_scene(pick_importer(input_path), input_path, fps=fps)
    if mesh is None:
        logger.info(f"No mesh in {input_path}; exporting armature only "
                    f"(empty skin matrix, pruning {'forced on' if prune else 'off'}).")
    if prune is None:
        prune = mesh is not None

    # All relevant pose actions; fall back to the armature's bound action
    # for single-action files that the pose filter rejects.
    action_names, frame_ranges = discover_pose_actions()
    if not action_names and armature.animation_data is not None \
            and armature.animation_data.action is not None:
        action = armature.animation_data.action
        action_names = [action.name]
        frame_ranges = {action.name: action_frame_range(action)}
        logger.info(f"No pose actions discovered; falling back to bound action '{action.name}'.")
    logger.info(f"Total actions to export: {len(action_names)}")
    if not action_names:
        logger.info(f"No actions in {input_path}; skipping.")
        mark_asset_complete(output_dir, save_name, status="skipped",
                            reason="no pose actions")
        return {}

    skel = prepare_skeleton(armature, mesh, dtype=dtype, apply_world=True)
    extracted = extract_all_actions(
        armature, {n: frame_ranges[n] for n in action_names},
        skel['rest_local_pos'], skel['parents_array'],
        skel['rest_anim'], skel['nbones'], dtype=dtype,
    )

    anims_list = [anim for _, anim in extracted]
    rest_anim_shared = skel['rest_anim_shared']
    names = list(skel['bone_names'])
    skin_matrix = skel['skin_matrix']
    if prune:
        anims_list, rest_anim_shared, names, skin_matrix = prune_skeleton_shared(
            anims_list, rest_anim_shared, names, skin_matrix.copy(),
            consider_parent_rotate=consider_parent_rotate,
        )

    save_rest_pose_vis(tpos_save_dir, save_name, rest_anim_shared)

    # Distinct actions can sanitize to the same name and would then overwrite
    # each other's NPZ; disambiguate over the full discovered list.
    clip_suffixes = unique_clip_names(action_names)

    scene_fps = bpy.context.scene.render.fps
    saved_clips = []
    for i, (action_name, _) in enumerate(extracted):
        anim = anims_list[i]

        if remove_tpose:
            anim, n_tpose = remove_tpose_frames(anim, rest_anim_shared)
            if n_tpose > 0:
                logger.info(f"Removed {n_tpose} T-pose frames from '{action_name}', "
                            f"{anim.positions.shape[0]} frames remaining")
        if anim.positions.shape[0] < min_frames:
            logger.info(f"Skipping '{action_name}': only {anim.positions.shape[0]} frames "
                        f"(min={min_frames})")
            continue

        clip_name = f"{save_name}-{clip_suffixes.get(action_name) or sanitize_action_name(action_name) or 'action'}"
        save_motion(
            os.path.join(motion_save_dir, f"{clip_name}.npz"),
            os.path.join(vis_save_dir, f"{clip_name}.mp4"),
            anim, rest_anim_shared, names, skin_matrix, scene_fps,
            action_name=action_name, save_vis=save_vis,
        )
        saved_clips.append(clip_name)

    mark_asset_complete(output_dir, save_name,
                        status="exported" if saved_clips else "skipped",
                        n_clips=len(saved_clips),
                        reason=None if saved_clips else "no clips survived filtering",
                        joint_names=names if saved_clips else None)
    logger.info(f"Saved {len(saved_clips)} clip(s) for '{save_name}'")
    if saved_clips:
        return {save_name: names}
    return {}


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Export rigged GLB/GLTF/FBX assets (a file or a directory) "
                    "to NPZ motion data.")
    parser.add_argument('--input', type=str, required=True,
                        help='A .glb/.gltf/.fbx file, or a directory of them.')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory (motions/, videos/, tpose/ created inside).')
    parser.add_argument('--name', type=str, default=None,
                        help='Save name / clip prefix (single-file input only; '
                             'default: input file stem).')
    parser.add_argument('--fps', type=int, default=30)
    parser.add_argument('--prune', action=argparse.BooleanOptionalAction, default=None,
                        help='Force skeleton pruning on/off '
                             '(default: on iff the file has a skinned mesh).')
    parser.add_argument('--consider_parent_rotate', action=argparse.BooleanOptionalAction, default=True,
                        help='Preserve non-rotating leaves whose parent rotates (see prune_skeleton_shared).')
    parser.add_argument('--min_frames', type=int, default=5,
                        help='Skip clips shorter than this many frames '
                             '(shared minimum across all dataset exporters).')
    parser.add_argument('--keep_tpose_frames', action='store_true',
                        help='Keep frames matching the rest pose instead of removing them.')
    parser.add_argument('--vis', action=argparse.BooleanOptionalAction, default=True,
                        help='Render the per-clip MP4 preview (use --no-vis for bulk runs).')
    parser.add_argument('--worker_id', type=int, default=0,
                        help='Worker index for sharding files across workers (directory input).')
    parser.add_argument('--num_workers', type=int, default=1,
                        help='Total number of workers (directory input).')

    # Blender passes script args after "--"; plain python passes them directly.
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
    return parser.parse_args(argv)


def main():
    args = parse_args()
    export_kwargs = dict(
        fps=args.fps, prune=args.prune,
        consider_parent_rotate=args.consider_parent_rotate,
        min_frames=args.min_frames, remove_tpose=not args.keep_tpose_frames,
        save_vis=args.vis,
    )

    if os.path.isdir(args.input):
        asset_paths = list_asset_files(args.input)
        assert asset_paths, f"No {sorted(ASSET_EXTS)} files found in {args.input}"
        logger.info(f"Found {len(asset_paths)} asset files in {args.input}")

        if args.num_workers > 1:
            asset_paths = asset_paths[args.worker_id::args.num_workers]
            logger.info(f"Worker {args.worker_id}/{args.num_workers}: "
                        f"processing {len(asset_paths)} files")

        os.makedirs(args.output_dir, exist_ok=True)
        error_log = os.path.join(args.output_dir,
                                 f'export_errors_worker{args.worker_id}.log')
        all_joint_names = {}
        n_failed = 0
        for asset_path in tqdm(asset_paths, desc="Exporting assets"):
            try:
                saved = export_asset(str(asset_path), args.output_dir,
                                     **export_kwargs)
                if saved:
                    all_joint_names.update(saved)
            except Exception as e:  # noqa: BLE001 — keep the batch going
                logger.exception(f"Failed to export {asset_path}")
                with open(error_log, 'a') as log_file:
                    log_file.write(f"Failed to export {asset_path}: {e}\n")

        worker_suffix = f"_worker{args.worker_id}" if args.num_workers > 1 else ""
        write_export_summary(args.output_dir, all_joint_names,
                             worker_suffix=worker_suffix)
    else:
        saved = export_asset(args.input, args.output_dir, save_name=args.name,
                             **export_kwargs)
        if saved:
            write_export_summary(args.output_dir, saved)

        if n_failed:
            logger.error(f"{n_failed}/{len(asset_paths)} assets failed; "
                         f"see {error_log}")
            sys.exit(1)


if __name__ == "__main__":
    # Blender exits 0 even on an uncaught exception, which hides failures from
    # `set -e` and from Slurm's afterok dependencies.
    try:
        main()
    except SystemExit:
        raise
    except Exception:
        logger.exception("export_general failed")
        sys.exit(1)
