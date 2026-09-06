"""Shared batch driver for the per-file FBX -> GLB conversion tools.

``tools/truebones_fbx2glb.py`` (per-clip animation) and
``tools/character_fbx2glb.py`` (rigged T-pose character) differ only in what one
conversion does; the CLI around it — sharding across workers, resume, per-worker
error log — is identical and lives here.

bpy-free: only the ``convert`` callback touches Blender, so this module imports
in a plain Python process.
"""

import argparse
import os
import sys

from loguru import logger


def parse_batch_args(description):
    """The fbx2glb CLI, parsed from the argv Blender leaves after ``--``."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Flat directory of input FBX files.')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Destination directory for the GLB files.')
    parser.add_argument('--worker_id', type=int, default=0,
                        help='Worker index for sharding across workers.')
    parser.add_argument('--num_workers', type=int, default=1,
                        help='Total number of workers.')
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
    return parser.parse_args(argv)


def run_batch(args, convert, unit='FBX files'):
    """Run ``convert(fbx_path, glb_path)`` over every FBX under ``args.data_dir``.

    Resumable (an existing GLB is skipped) and fault-tolerant: a file that raises
    is recorded in ``<output_dir>/fbx2glb_errors_worker<id>.log`` and the batch
    goes on. Exits non-zero if anything failed.
    """
    fbx_files = sorted(f for f in os.listdir(args.data_dir) if f.lower().endswith('.fbx'))
    logger.info(f"Found {len(fbx_files)} {unit} in {args.data_dir}")

    if args.num_workers > 1:
        fbx_files = fbx_files[args.worker_id::args.num_workers]
        logger.info(f"Worker {args.worker_id}/{args.num_workers}: {len(fbx_files)} files")

    os.makedirs(args.output_dir, exist_ok=True)
    error_log = os.path.join(args.output_dir, f'fbx2glb_errors_worker{args.worker_id}.log')

    n_ok = n_skip = n_fail = 0
    for fbx_file in fbx_files:
        glb_path = os.path.join(args.output_dir, os.path.splitext(fbx_file)[0] + '.glb')
        if os.path.exists(glb_path):
            n_skip += 1
            continue
        try:
            convert(os.path.join(args.data_dir, fbx_file), glb_path)
            n_ok += 1
        except Exception as e:  # noqa: BLE001 — keep the batch going
            logger.error(f"Failed to convert {fbx_file}: {e}")
            with open(error_log, 'a') as f:
                f.write(f"{fbx_file}\t{e}\n")
            n_fail += 1

    logger.info(f"Done: {n_ok} converted, {n_skip} skipped, {n_fail} failed "
                f"-> {args.output_dir}")
    if n_fail:
        sys.exit(1)
