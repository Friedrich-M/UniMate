"""Render multi-view frames for Truebones FBX animations (EEVEE).

Consumes the curated flat layout of the Truebones ZOO dataset: one FBX per
clip, named ``{Species}-{Action}.fbx``, all in a single directory (the same
input as ``motion_export/export_truebones.py``). Each file is rendered as
``{file_stem}/v00{0..3}/*.png`` — the layout consumed by
``vlm_caption/caption_motion.py``.

Usage (plain python — EEVEE needs the pip ``bpy`` module's GPU context):
    python -m data_process.motion_rendering.render_truebones \
        --data_dir dataset/raw/truebones/animation --output_dir <render_dir>
"""

import os
import sys
import argparse
from pathlib import Path

import bpy
from loguru import logger

from data_process.utils.blender_export import (
    discover_pose_actions,
    truebones_clip_name,
)
from data_process.utils.blender_render import (
    RENDER_FPS,
    init_render_engine,
    is_asset_render_skipped,
    is_render_complete,
    load_and_prep_asset,
    mark_asset_render_skipped,
    render_action_multiview,
)


def render_clip_file(path, output_dir, args):
    """Render multi-view frames for one per-clip FBX file."""
    # Must match export_truebones' clip naming exactly — stage 4 looks the
    # caption up by the NPZ stem with no fuzzy matching, so a raw stem here
    # (spaces, extra dashes) silently produces an uncaptioned clip.
    save_name = truebones_clip_name(Path(path).stem)
    save_root = os.path.join(output_dir, save_name)

    if is_render_complete(save_root, require_video=args.video):
        logger.info(f"Skipping complete: {save_root}")
        return
    if is_asset_render_skipped(output_dir, save_name):
        logger.info(f"Previously marked as no-render, skipping: {path}")
        return

    logger.info(f"Processing: {path}")

    armature = load_and_prep_asset(str(path), fps=args.fps)
    if armature is None:
        logger.warning(f"No armature/mesh in {path}; skipping.")
        mark_asset_render_skipped(output_dir, save_name, reason="no armature/mesh")
        return

    # Each curated file carries exactly one take: the action bound to the
    # armature. Export always uses that one, so falling back to a different
    # action here would render a motion the NPZ never contains.
    action_names, frame_ranges = discover_pose_actions(min_frames=args.min_action_frames)
    if not action_names:
        logger.warning(f"No animation data in {path}; skipping.")
        mark_asset_render_skipped(output_dir, save_name, reason="no pose actions")
        return

    bound = None
    if armature.animation_data and armature.animation_data.action:
        bound = armature.animation_data.action.name
    if bound not in action_names:
        logger.warning(
            f"Bound action {bound!r} is not among the renderable pose actions "
            f"{action_names} for {path} (too short, or none bound); skipping "
            f"rather than rendering a different action than the export used.")
        mark_asset_render_skipped(output_dir, save_name,
                                  reason=f"bound action {bound!r} not renderable")
        return
    action_name = bound

    start, end = frame_ranges[action_name]
    logger.info(f"Rendering '{action_name}' -> '{save_name}' frames=[{start},{end}]")

    render_action_multiview(
        armature=armature,
        action_name=action_name,
        start_frame=start,
        end_frame=end,
        save_root=save_root,
        camera_dist=args.camera_dist,
        scene_scale=args.scale,
        resolution=args.resolution,
        max_render_frames=args.max_render_frames,
        compose_video=args.video,
        fps=args.fps,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Render multi-view frames for Truebones FBX animations (EEVEE).")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Flat directory of per-clip {Species}-{Action}.fbx files.")
    parser.add_argument("--obj_type", type=str, default=None,
                        help="Single species to render (default: all).")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory.")
    parser.add_argument("--scale", type=float, default=1.0,
                        help="Scene normalization scale.")
    parser.add_argument("--camera_dist", type=float, default=1.5,
                        help="Camera distance from origin.")
    parser.add_argument("--resolution", type=int, default=512,
                        help="Render resolution (square).")
    parser.add_argument("--fps", type=int, default=RENDER_FPS,
                        help="Scene FPS for import and preview video "
                             "(must match the export stage's --fps).")
    parser.add_argument("--samples", type=int, default=64,
                        help="EEVEE TAA render samples.")
    parser.add_argument("--max_render_frames", type=int, default=200,
                        help="Cap on rendered frames per clip (per view).")
    parser.add_argument("--min_action_frames", type=int, default=5,
                        help="Skip actions shorter than this many frames "
                             "(default 5 — the shared minimum clip length "
                             "across export and rendering).")
    parser.add_argument("--video", action=argparse.BooleanOptionalAction, default=True,
                        help="Compose each view's frames into v00x.mp4 "
                             "(use --no-video to skip).")
    parser.add_argument("--worker_id", type=int, default=0,
                        help="Worker index for sharding across workers.")
    parser.add_argument("--num_workers", type=int, default=1,
                        help="Total number of workers.")
    parser.add_argument("--gpu_id", type=int, default=None,
                        help="GPU device ID to pin this worker to "
                             "(default: inherit CUDA_VISIBLE_DEVICES).")
    args = parser.parse_args()

    if args.gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

    init_render_engine(render_samples=args.samples)

    data_dir = Path(args.data_dir)
    files = sorted(p for p in data_dir.iterdir()
                   if p.is_file() and p.suffix.lower() == ".fbx")
    if args.obj_type:
        files = [p for p in files if p.stem.split('-', 1)[0] == args.obj_type]
    logger.info(f"Found {len(files)} clip files in {data_dir}")

    if args.num_workers > 1:
        files = files[args.worker_id::args.num_workers]
        logger.info(f"Worker {args.worker_id}/{args.num_workers}: {len(files)} files")

    os.makedirs(args.output_dir, exist_ok=True)
    error_log = os.path.join(args.output_dir,
                             f'render_errors_worker{args.worker_id}.log')

    n_failed = 0
    for path in files:
        try:
            render_clip_file(path, args.output_dir, args)
        except Exception as e:  # noqa: BLE001 — keep the batch going
            n_failed += 1
            logger.exception(f"Failed to render {path}")
            with open(error_log, 'a') as f:
                f.write(f"{path}\t{e}\n")

    if n_failed:
        logger.error(f"{n_failed}/{len(files)} clips failed; see {error_log}")
    return 1 if n_failed else 0


if __name__ == "__main__":
    # bpy/Blender exits 0 even on an uncaught exception, which hides failures
    # from `set -e` and from Slurm's afterok dependencies.
    try:
        sys.exit(main())
    except Exception:
        logger.exception("render_truebones failed")
        sys.exit(1)
