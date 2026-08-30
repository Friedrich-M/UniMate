"""Export Mixamo animation FBX files to NPZ motion data.

Pipeline: import animation FBX (armature + action, no mesh) → build skeleton
arrays → extract animation → Z-up → Y-up → save per-clip NPZ + MP4.

Skin weights are not extracted (there is no character mesh); ``skin_matrix``
is saved as an empty ``(0, nbones)`` array. All Mixamo animations share the
same skeleton, so ``joint_names.json`` and the rest-pose PNG are written once
(by worker 0) and every freshly-extracted clip's rest pose is checked against
the first one.

Files can be sharded across parallel Blender processes with
``--worker_id`` / ``--num_workers``; each worker writes its own summary
JSONs (``*_worker{i}.json``) — merge them afterwards with
``python -m data_process.tools.merge_summaries`` (the
``run_export.sh`` wrapper does both automatically).

Usage (Blender headless):
    blender -b -P data_process/motion_export/export_mixamo.py -- \
        --anim_dir dataset/raw/mixamo/animation_motion \
        --output_dir dataset/export/mixamo
"""

import bpy
import sys
import os
import json
import numpy as np
import argparse
from tqdm import tqdm
from loguru import logger

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from data_process.utils.blender_export import (
    import_fbx,
    load_scene, prepare_skeleton, extract_all_actions, save_rest_pose_vis,
    action_frame_range, save_motion, write_export_summary,
)


# ─────────────────────────────────────────────────────────────────────────────
# Main export function
# ─────────────────────────────────────────────────────────────────────────────

def export_mixamo(anim_path, output_dir, fps=30, dtype=np.float64,
                  min_frames=5, skip_if_exists=True, save_vis=True):
    """Export a single Mixamo animation clip.

    Args:
        anim_path: Path to animation FBX (provides armature + action).
        output_dir: Output root directory.
        fps: Scene FPS.
        dtype: Numpy dtype for arrays.
        skip_if_exists: Return early when the output NPZ already exists.
        save_vis: Also render the per-clip MP4 preview.

    Returns:
        (bone_names, rest_anim) if extracted, else None.
    """
    assert os.path.exists(anim_path), f"Animation file not found: {anim_path}"

    motion_save_dir = os.path.join(output_dir, "motions")
    vis_save_dir = os.path.join(output_dir, "videos")
    os.makedirs(motion_save_dir, exist_ok=True)
    os.makedirs(vis_save_dir, exist_ok=True)

    anim_base_name = os.path.splitext(os.path.basename(anim_path))[0]
    motion_save_path = os.path.join(motion_save_dir, f"{anim_base_name}.npz")
    vis_save_path = os.path.join(vis_save_dir, f"{anim_base_name}.mp4")

    # Skip if output already exists (unless caller still needs metadata)
    output_exists = os.path.exists(motion_save_path)
    if output_exists and skip_if_exists:
        logger.info(f"Output already exists for '{anim_base_name}', skipping.")
        return None

    armature, _ = load_scene(import_fbx, anim_path, fps=fps)
    assert armature.animation_data is not None and armature.animation_data.action is not None, \
        f"Armature '{armature.name}' has no animation data or action."

    action = armature.animation_data.action
    start_frame, end_frame = action_frame_range(action)
    logger.info(f"Action frame range: {start_frame} to {end_frame}")

    skel = prepare_skeleton(armature, mesh=None, dtype=dtype, apply_world=True)
    extracted = extract_all_actions(
        armature, {action.name: (start_frame, end_frame)},
        skel['rest_local_pos'], skel['parents_array'],
        skel['rest_anim'], skel['nbones'], dtype=dtype,
    )
    _, anim_obj = extracted[0]

    nframes = anim_obj.positions.shape[0]
    if nframes < min_frames:
        logger.info(f"Skipping '{anim_base_name}': only {nframes} frames "
                    f"(min={min_frames})")
        return None

    if not output_exists:
        scene_fps = bpy.context.scene.render.fps
        save_motion(motion_save_path, vis_save_path, anim_obj, skel['rest_anim_shared'],
                    list(skel['bone_names']), skel['skin_matrix'], scene_fps,
                    action_name=anim_base_name, save_vis=save_vis)

    return list(skel['bone_names']), skel['rest_anim_shared']


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Export Mixamo animation FBX files to NPZ motion data.")
    parser.add_argument("--anim_dir", type=str, required=True,
                        help='Directory containing animation FBX files.')
    parser.add_argument("--output_dir", type=str, required=True,
                        help='Output directory (motions/, videos/, tpose/ created inside).')
    parser.add_argument("--fps", type=int, default=30,
                        help="Scene FPS for import and export (dataset is 30 fps).")
    parser.add_argument("--min_frames", type=int, default=5,
                        help="Skip clips shorter than this many frames "
                             "(shared minimum across all dataset exporters).")
    parser.add_argument("--worker_id", type=int, default=0,
                        help="Worker index for sharding files across workers.")
    parser.add_argument("--num_workers", type=int, default=1,
                        help="Total number of workers.")
    parser.add_argument("--vis", action=argparse.BooleanOptionalAction, default=True,
                        help="Render the per-clip MP4 preview (use --no-vis for bulk runs).")
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
    return parser.parse_args(argv)


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    worker_suffix = f"_worker{args.worker_id}" if args.num_workers > 1 else ""
    error_log = os.path.join(args.output_dir, f"export_errors{worker_suffix}.log")

    n_failed = 0
    anim_files = sorted([f for f in os.listdir(args.anim_dir) if f.endswith(".fbx")])
    anim_paths = [os.path.join(args.anim_dir, f) for f in anim_files]
    logger.info(f"Found {len(anim_paths)} animation files")

    if args.num_workers > 1:
        anim_paths = anim_paths[args.worker_id::args.num_workers]
        logger.info(f"Worker {args.worker_id}/{args.num_workers}: "
                    f"processing {len(anim_paths)} files")

    joint_names_path = os.path.join(args.output_dir, "joint_names.json")
    tpos_path = os.path.join(args.output_dir, "tpose", "mixamo.png")

    # Only worker 0 writes the shared skeleton metadata (single skeleton for
    # the whole dataset). While it is missing, worker 0 extracts its first
    # file even if the NPZ already exists, to recover the reference skeleton.
    is_primary = args.worker_id == 0
    need_metadata = is_primary and not (
        os.path.exists(joint_names_path) and os.path.exists(tpos_path))

    ref_bone_names = None
    ref_rest_pos = None
    ref_rest_rot = None
    tpos_mismatches = []

    for anim_path in tqdm(anim_paths, desc="Processing animations"):
        try:
            result = export_mixamo(anim_path, args.output_dir, fps=args.fps,
                                   min_frames=args.min_frames,
                                   skip_if_exists=not need_metadata,
                                   save_vis=args.vis)
            if result is None:
                continue

            bone_names, rest_anim = result
            rest_pos = rest_anim.positions[0]   # (nbones, 3)
            rest_rot = rest_anim.rotations[0].qs  # (nbones, 4)

            if ref_bone_names is None:
                ref_bone_names = bone_names
                ref_rest_pos = rest_pos.copy()
                ref_rest_rot = rest_rot.copy()
                if need_metadata:
                    with open(joint_names_path, 'w') as f:
                        json.dump({"mixamo": bone_names}, f, indent=2)
                    logger.info(f"Saved joint names to {joint_names_path}")
                    if not os.path.exists(tpos_path):
                        save_rest_pose_vis(os.path.dirname(tpos_path), "mixamo", rest_anim)
                    need_metadata = False
            else:
                # Check T-pose consistency against the reference skeleton
                anim_name = os.path.basename(anim_path)
                if bone_names != ref_bone_names:
                    logger.warning(f"Bone names mismatch: {anim_name}")
                    tpos_mismatches.append(anim_name)
                else:
                    pos_diff = np.max(np.abs(rest_pos - ref_rest_pos))
                    rot_diff = np.max(np.abs(rest_rot - ref_rest_rot))
                    if pos_diff > 1e-6 or rot_diff > 1e-6:
                        logger.warning(f"T-pose mismatch: {anim_name} "
                                       f"(pos_diff={pos_diff:.8f}, rot_diff={rot_diff:.8f})")
                        tpos_mismatches.append(anim_name)

        except Exception as e:  # noqa: BLE001 — keep the batch going
            n_failed += 1
            logger.exception(f"Error processing {anim_path}")
            with open(error_log, "a") as f:
                f.write(f"Error processing {anim_path}: {e}\n")

    if tpos_mismatches:
        logger.warning(f"{len(tpos_mismatches)} animations have mismatched T-pose: {tpos_mismatches}")
    else:
        logger.info("All extracted animations share the same T-pose.")

    # On a fully-resumed run nothing is extracted; fall back to the saved
    # joint names instead of overwriting the summary files with {}.
    if ref_bone_names is None and os.path.exists(joint_names_path):
        with open(joint_names_path) as f:
            ref_bone_names = json.load(f).get("mixamo")

    all_joint_names = {"mixamo": ref_bone_names} if ref_bone_names else {}
    write_export_summary(args.output_dir, all_joint_names, worker_suffix=worker_suffix)

    if n_failed:
        logger.error(f"{n_failed}/{len(anim_paths)} clips failed; see {error_log}")
        sys.exit(1)


if __name__ == "__main__":
    # Blender exits 0 even on an uncaught exception, which hides failures from
    # `set -e` and from Slurm's afterok dependencies.
    try:
        main(parse_args())
    except SystemExit:
        raise
    except Exception:
        logger.exception("export_mixamo failed")
        sys.exit(1)
