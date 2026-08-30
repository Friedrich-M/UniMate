"""Convert rigged T-pose character FBX files to GLB, one to one.

Consumes a flat directory of character FBXs (mesh + armature, no animation
required — e.g. ``dataset/raw/mixamo/character_refined``) and writes
``{name}.glb`` per file. Any actions the FBX importer binds (static
object-transform takes) are stripped so the GLB is a clean T-pose rig.
Resumable: existing GLBs are skipped.

Usage (Blender headless):
    blender -b -P data_process/tools/character_fbx2glb.py -- \
        --data_dir dataset/raw/mixamo/character_refined \
        --output_dir dataset/raw/mixamo/character_refined_glb
"""

import argparse
import os
import sys

import bpy
from loguru import logger

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from data_process.utils.blender_export import import_fbx, reset_scene


def convert_character_to_glb(fbx_path, glb_path):
    """Import one character FBX and export it as a T-pose GLB (no animation)."""
    reset_scene()
    import_fbx(fbx_path)

    armatures = [o for o in bpy.data.objects if o.type == 'ARMATURE']
    assert armatures, f"No armature found in {fbx_path}"
    meshes = [o for o in bpy.data.objects if o.type == 'MESH']
    assert meshes, f"No mesh found in {fbx_path}"

    # Strip every action/binding so the export carries only the rig in its
    # rest pose — character files should have no animation, but the FBX
    # importer still binds static object-transform takes.
    for obj in bpy.data.objects:
        if obj.animation_data is not None:
            obj.animation_data_clear()
    for action in list(bpy.data.actions):
        bpy.data.actions.remove(action)

    for obj in bpy.data.objects:
        obj.select_set(obj.type in ('MESH', 'ARMATURE'))

    kwargs = dict(
        filepath=glb_path,
        check_existing=False,
        use_selection=True,
        export_format='GLB',
        export_animations=False,
    )
    bpy.ops.export_scene.gltf(**kwargs)
    logger.info(f"Saved T-pose GLB: {glb_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert rigged T-pose character FBX files to GLB, one to one.")
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Flat directory of character FBX files.')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Destination directory for the per-character GLB files.')
    parser.add_argument('--worker_id', type=int, default=0,
                        help='Worker index for sharding across workers.')
    parser.add_argument('--num_workers', type=int, default=1,
                        help='Total number of workers.')
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()

    fbx_files = sorted(f for f in os.listdir(args.data_dir)
                       if f.lower().endswith('.fbx'))
    logger.info(f"Found {len(fbx_files)} character FBXs in {args.data_dir}")

    if args.num_workers > 1:
        fbx_files = fbx_files[args.worker_id::args.num_workers]
        logger.info(f"Worker {args.worker_id}/{args.num_workers}: {len(fbx_files)} files")

    os.makedirs(args.output_dir, exist_ok=True)
    error_log = os.path.join(args.output_dir,
                             f'fbx2glb_errors_worker{args.worker_id}.log')

    n_ok = n_skip = n_fail = 0
    for fbx_file in fbx_files:
        glb_path = os.path.join(args.output_dir,
                                os.path.splitext(fbx_file)[0] + '.glb')
        if os.path.exists(glb_path):
            n_skip += 1
            continue
        try:
            convert_character_to_glb(os.path.join(args.data_dir, fbx_file), glb_path)
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
