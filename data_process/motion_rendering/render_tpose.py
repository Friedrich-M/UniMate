"""Render T-pose 2x2 grid images for rigged assets (EEVEE).

One PNG per asset, saved flat as ``{output_dir}/{name}.png`` (front/back
top row, left/right bottom row) — the "flat" layout consumed by
``vlm_caption/classify_category.py``.

Two input layouts are auto-detected per entry of ``--data_dir``:
  - files:        ``<data_dir>/*.{glb,gltf,fbx}``  -> one grid per file stem
  - subdirectory: ``<data_dir>/<obj_type>/*.fbx``  -> one grid per object
    type, rendered from its main (all-in-one) FBX — Truebones layout.

Usage (plain python — EEVEE needs the pip ``bpy`` module's GPU context):
    python -m data_process.motion_rendering.render_tpose \
        --data_dir dataset/raw/objaverse/glb --output_dir <tpose_dir>
"""

import os
import sys
import argparse
from pathlib import Path

from loguru import logger

from data_process.utils.blender_export import find_main_fbx, sanitize_object_type
from data_process.utils.blender_render import (
    RENDERABLE_EXTS,
    RENDER_FPS,
    init_render_engine,
    load_and_prep_asset,
    render_tpose_grid,
)


def discover_assets(data_dir, species_grids=False):
    """Return (name, path) pairs to render, auto-detecting the layout.

    Files directly under *data_dir* are rendered by stem; each
    subdirectory is treated as a Truebones-style object type and rendered
    from its main FBX (fallback: first FBX in the subdirectory).

    With *species_grids*, files named ``{Species}-{Action}.*`` are grouped
    by the stem's prefix before the first ``-`` and one grid is rendered
    per species, from its first file (curated Truebones layout).
    """
    data_dir = Path(data_dir)
    assets = []
    seen_species = set()

    for p in sorted(data_dir.iterdir()):
        if p.is_file() and p.suffix.lower() in RENDERABLE_EXTS:
            if species_grids:
                species = p.stem.split('-', 1)[0]
                if species in seen_species:
                    continue
                seen_species.add(species)
                assets.append((sanitize_object_type(species), p))
            else:
                assets.append((p.stem, p))
        elif p.is_dir():
            main_fbx = find_main_fbx(str(p))
            if main_fbx is None:
                fbxs = sorted(f for f in p.iterdir()
                              if f.suffix.lower() == ".fbx")
                main_fbx = str(fbxs[0]) if fbxs else None
            if main_fbx is None:
                logger.warning(f"No FBX found in {p}; skipping.")
                continue
            assets.append((sanitize_object_type(p.name), Path(main_fbx)))

    return assets


def render_asset_tpose(name, path, output_dir, args):
    """Render the T-pose grid for one asset to ``{output_dir}/{name}.png``."""
    tpose_path = os.path.join(output_dir, f"{name}.png")
    if os.path.isfile(tpose_path):
        logger.info(f"Skipping complete: {tpose_path}")
        return

    logger.info(f"Processing: {path} -> {name}")

    armature = load_and_prep_asset(str(path),
                                   keep_only_main_mesh=args.keep_main_mesh_only,
                                   fps=args.fps)
    if armature is None:
        logger.warning(f"No armature/mesh in {path}; skipping.")
        return

    render_tpose_grid(
        save_path=tpose_path,
        camera_dist=args.camera_dist,
        scene_scale=args.scale,
        resolution=args.resolution,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Render T-pose 2x2 grid images for rigged assets (EEVEE).")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Directory of GLB/GLTF/FBX files, or of "
                             "per-object-type FBX subdirectories (Truebones).")
    parser.add_argument("--obj_name", type=str, default=None,
                        help="Single asset name (file stem or object type) "
                             "to render (default: all).")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Flat directory for the <name>.png grids.")
    parser.add_argument("--scale", type=float, default=1.0,
                        help="Scene normalization scale.")
    parser.add_argument("--camera_dist", type=float, default=1.5,
                        help="Camera distance from origin.")
    parser.add_argument("--fps", type=int, default=RENDER_FPS,
                        help="Scene FPS for import (must match the export "
                             "stage's --fps).")
    parser.add_argument("--resolution", type=int, default=512,
                        help="Render resolution per view (square).")
    parser.add_argument("--samples", type=int, default=64,
                        help="EEVEE TAA render samples.")
    parser.add_argument("--keep-all-meshes", dest="keep_main_mesh_only",
                        action="store_false",
                        help="Keep every mesh instead of only the "
                             "highest-vertex one (multi-mesh characters).")
    parser.add_argument("--species_grids", action="store_true",
                        help="Group {Species}-{Action}.fbx files by species "
                             "and render one grid per species (curated "
                             "Truebones layout).")
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

    assets = discover_assets(args.data_dir, species_grids=args.species_grids)
    logger.info(f"Found {len(assets)} assets in {args.data_dir}")

    if args.obj_name:
        assets = [(n, p) for n, p in assets if n == args.obj_name]
        if not assets:
            parser.error(f"Asset '{args.obj_name}' not found in {args.data_dir}")

    if args.num_workers > 1:
        assets = assets[args.worker_id::args.num_workers]
        logger.info(f"Worker {args.worker_id}/{args.num_workers}: "
                    f"{len(assets)} assets")

    os.makedirs(args.output_dir, exist_ok=True)
    error_log = os.path.join(args.output_dir,
                             f'tpose_errors_worker{args.worker_id}.log')

    for name, path in assets:
        try:
            render_asset_tpose(name, path, args.output_dir, args)
        except Exception as e:
            logger.error(f"Failed to render {path}: {e}")
            with open(error_log, 'a') as f:
                f.write(f"{path}\t{e}\n")


if __name__ == "__main__":
    main()
