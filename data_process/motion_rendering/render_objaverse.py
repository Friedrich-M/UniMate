"""Render multi-view frames for Objaverse GLB/GLTF animations (EEVEE).

For each GLB, discovers all pose actions and renders each one as
``{glb_stem}-{action}/v00{0..3}/*.png`` — the layout consumed by
``vlm_caption/caption_motion.py``.

Usage (plain python — EEVEE needs the pip ``bpy`` module's GPU context):
    python -m data_process.motion_rendering.render_objaverse \
        --data_dir dataset/raw/objaverse/glb --output_dir <render_dir>
"""

import os
import sys
import argparse
from pathlib import Path

import bpy
from loguru import logger

from data_process.utils.blender_export import (
    discover_pose_actions,
    list_gltf_files,
    unique_clip_names,
)
from data_process.utils.blender_render import (
    RENDER_FPS,
    get_expected_clip_dirs,
    init_render_engine,
    is_asset_render_skipped,
    is_render_complete,
    load_and_prep_asset,
    mark_asset_render_done,
    mark_asset_render_skipped,
    render_action_multiview,
)


def find_pending_stems(output_dir, gltf_stems, require_video=False):
    """Return the subset of GLTF stems that still need rendering.

    A stem is done only when a ``.rendered/`` marker lists the clip
    directories it is expected to produce AND every one of them passes
    :func:`is_render_complete`. Requiring the marker is what distinguishes
    "all actions rendered" from "a worker was killed partway through": the
    directories that do exist are complete in both cases, so judging by
    on-disk directories alone would retire the asset with actions still
    unrendered, permanently. Stems that are done skip the expensive GLB load.

    Action folders are named ``{stem}-{sanitized_action}``; stems may
    themselves contain ``-``, so directory names are matched against the
    known stem set by longest-prefix-at-hyphen rather than splitting on
    the first ``-``.
    """
    output_dir = Path(output_dir)
    stems_set = set(gltf_stems)
    if not output_dir.is_dir():
        return stems_set

    # Assets marked as producing no renders (no armature / actions) are done.
    stems_set = {s for s in stems_set
                 if not is_asset_render_skipped(str(output_dir), s)}

    pending = set()
    for stem in stems_set:
        expected = get_expected_clip_dirs(str(output_dir), stem)
        if not expected:
            # Never finished an end-to-end pass (or predates the marker):
            # re-open it. Completed clip dirs are still skipped per-action.
            pending.add(stem)
            continue
        if not all(is_render_complete(str(output_dir / d), require_video=require_video)
                   for d in expected):
            pending.add(stem)

    return pending


def render_glb(glb_path, output_dir, args):
    """Render multi-view frames for every pose action in one GLB/GLTF file."""
    save_name = Path(glb_path).stem
    if is_asset_render_skipped(output_dir, save_name):
        logger.info(f"Previously marked as no-render, skipping: {glb_path}")
        return

    logger.info(f"Processing: {glb_path}")

    armature = load_and_prep_asset(str(glb_path), fps=args.fps)
    if armature is None:
        logger.warning(f"No armature/mesh in {glb_path}; skipping.")
        mark_asset_render_skipped(output_dir, save_name, reason="no armature/mesh")
        return

    # Names are derived from the UNFILTERED action list so they match the
    # export stage, which applies no frame-count filter — deriving them from
    # a filtered subset would shift the collision suffixes out of step.
    all_action_names, frame_ranges = discover_pose_actions()
    clip_suffixes = unique_clip_names(all_action_names)

    action_names = [
        n for n in all_action_names
        if (frame_ranges[n][1] - frame_ranges[n][0] + 1) >= args.min_action_frames
    ]
    logger.info(f"Found {len(action_names)} pose actions "
                f"({len(all_action_names)} before the frame-count filter)")
    if not action_names:
        mark_asset_render_skipped(output_dir, save_name, reason="no pose actions")
        return

    expected_dirs = []
    for action_name in action_names:
        clean_action_name = clip_suffixes[action_name]
        clip_dir_name = f"{save_name}-{clean_action_name}"
        action_dir = os.path.join(output_dir, clip_dir_name)
        expected_dirs.append(clip_dir_name)

        if is_render_complete(action_dir, require_video=args.video):
            logger.info(f"Skipping complete: {action_dir}")
            continue

        start, end = frame_ranges[action_name]
        logger.info(f"Rendering '{action_name}' -> '{clip_dir_name}' "
                    f"frames=[{start},{end}]")

        render_action_multiview(
            armature=armature,
            action_name=action_name,
            start_frame=start,
            end_frame=end,
            save_root=action_dir,
            camera_dist=args.camera_dist,
            scene_scale=args.scale,
            resolution=args.resolution,
            max_render_frames=args.max_render_frames,
            compose_video=args.video,
            fps=args.fps,
        )

    # Only now is the asset known to have had every action handled.
    mark_asset_render_done(output_dir, save_name, expected_dirs)


def main():
    parser = argparse.ArgumentParser(
        description="Render multi-view frames for Objaverse GLB/GLTF animations (EEVEE).")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Directory containing GLB/GLTF files.")
    parser.add_argument("--obj_name", type=str, default=None,
                        help="Single GLB stem to render (default: all files).")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory.")
    parser.add_argument("--scale", type=float, default=1.0,
                        help="Scene normalization scale.")
    parser.add_argument("--camera_dist", type=float, default=1.5,
                        help="Camera distance from origin.")
    parser.add_argument("--resolution", type=int, default=512,
                        help="Render resolution (square).")
    parser.add_argument("--fps", type=int, default=RENDER_FPS,
                        help="Scene FPS for import and preview video. Must match "
                             "the export stage's --fps: glTF keyframes are stored "
                             "in seconds, so the importer resamples at this rate.")
    parser.add_argument("--samples", type=int, default=64,
                        help="EEVEE TAA render samples.")
    parser.add_argument("--max_render_frames", type=int, default=200,
                        help="Cap on rendered frames per clip (per view).")
    parser.add_argument("--min_action_frames", type=int, default=5,
                        help="Skip actions shorter than this many raw frames "
                             "(default 5 — the shared minimum clip length "
                             "across export and rendering; raw length >= "
                             "post-T-pose-removal length, so no exported clip "
                             "is ever skipped).")
    parser.add_argument("--video", action=argparse.BooleanOptionalAction, default=True,
                        help="Compose each view's frames into v00x.mp4 "
                             "(use --no-video to skip).")
    parser.add_argument("--missing-only", dest="missing_only", action="store_true",
                        help="Scan output_dir first; only load GLBs that have "
                             "at least one missing or incomplete action folder.")
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
    if args.obj_name:
        glb_path = data_dir / f"{args.obj_name}.glb"
        if not glb_path.is_file():
            glb_path = data_dir / f"{args.obj_name}.gltf"
        if not glb_path.is_file():
            parser.error(f"Cannot find {args.obj_name}.glb or .gltf in {data_dir}")
        gltf_paths = [glb_path]
    else:
        gltf_paths = list_gltf_files(data_dir)
        logger.info(f"Found {len(gltf_paths)} GLB/GLTF files in {data_dir}")

    if args.missing_only:
        pending = find_pending_stems(args.output_dir, [p.stem for p in gltf_paths],
                                     require_video=args.video)
        before = len(gltf_paths)
        gltf_paths = [p for p in gltf_paths if p.stem in pending]
        logger.info(f"[missing-only] {len(gltf_paths)}/{before} GLBs have pending renders.")

    if args.num_workers > 1:
        gltf_paths = gltf_paths[args.worker_id::args.num_workers]
        logger.info(f"Worker {args.worker_id}/{args.num_workers}: "
                    f"{len(gltf_paths)} files")

    os.makedirs(args.output_dir, exist_ok=True)
    error_log = os.path.join(args.output_dir,
                             f'render_errors_worker{args.worker_id}.log')

    n_failed = 0
    for glb_path in gltf_paths:
        try:
            render_glb(glb_path, args.output_dir, args)
        except Exception as e:  # noqa: BLE001 — keep the batch going
            n_failed += 1
            logger.exception(f"Failed to render {glb_path}")
            with open(error_log, 'a') as f:
                f.write(f"{glb_path}\t{e}\n")

    if n_failed:
        logger.error(f"{n_failed}/{len(gltf_paths)} assets failed; see {error_log}")
    return 1 if n_failed else 0


if __name__ == "__main__":
    # bpy/Blender exits 0 even on an uncaught exception, which hides failures
    # from `set -e` and from Slurm's afterok dependencies.
    try:
        sys.exit(main())
    except Exception:
        logger.exception("render_objaverse failed")
        sys.exit(1)
