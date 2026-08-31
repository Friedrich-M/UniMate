"""Batch-animate a Mixamo character mesh with a directory of exported NPZs.

Raw Mixamo data only contains animation (armature + action) without a mesh,
so every NPZ in ``--npz_dir`` is paired with the same ``--char_path`` and
written to ``--output_dir`` via :func:`animate_npz.animate_character`.
Per-file failures are logged to ``animate_errors.log`` inside the output dir.

Usage (Blender headless):
    blender -b -P data_process/mesh_animation/animate_mixamo.py -- \\
        --char_path <character.fbx> \\
        --npz_dir <export_dir>/motions \\
        --output_dir outputs/mixamo_characters
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from loguru import logger  # noqa: E402

from data_process.mesh_animation.animate_npz import animate_character  # noqa: E402
from data_process.mesh_animation.common import (  # noqa: E402
    parse_blender_argv,
    run_clip_batch,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Batch-animate a Mixamo character mesh from exported NPZ motions.")
    parser.add_argument("--char_path", type=str, required=True,
                        help="Character FBX file (provides mesh + armature).")
    parser.add_argument("--npz_dir", type=str, required=True,
                        help="Directory containing export-stage motion NPZ files.")
    parser.add_argument("--output_dir", type=str, default='outputs/mixamo_characters',
                        help="Output directory for animated GLB/FBX files.")
    parser.add_argument("--char_anim_type", type=str, default='glb', choices=['glb', 'fbx'],
                        help="Export format for the animated character.")
    parser.add_argument("--overwrite", action='store_true',
                        help="Re-export clips whose output already exists.")
    return parse_blender_argv(parser)


if __name__ == "__main__":
    args = parse_args()
    assert os.path.exists(args.char_path), f"Character file not found: {args.char_path}"
    assert os.path.isdir(args.npz_dir), f"NPZ directory not found: {args.npz_dir}"

    npz_files = sorted(os.path.join(args.npz_dir, f)
                       for f in os.listdir(args.npz_dir) if f.endswith(".npz"))
    logger.info(f"Found {len(npz_files)} NPZ files in {args.npz_dir}")

    run_clip_batch(
        npz_files,
        lambda npz_path: animate_character(
            char_path=args.char_path,
            anim_path=npz_path,
            output_dir=args.output_dir,
            char_anim_type=args.char_anim_type,
            skip_if_exists=not args.overwrite,
        ),
        output_dir=args.output_dir,
        desc="Animating meshes",
    )
