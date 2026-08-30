"""Convert curated Truebones per-clip FBX files to GLB, one to one.

Consumes the flat ``{Species}-{Action}.fbx`` layout (the same input as
``export_truebones.py``) and writes ``{Species}-{Action}.glb`` next to it,
preserving mesh, armature, and the clip's single animation take. Resumable:
existing GLBs are skipped.

Usage (Blender headless):
    blender -b -P data_process/tools/truebones_fbx2glb.py -- \
        --data_dir dataset/raw/truebones/animation \
        --output_dir dataset/raw/truebones/animation_glb
"""

import argparse
import os
import sys

import bpy
from loguru import logger

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from data_process.utils.blender_export import import_fbx, reset_scene


def convert_fbx_to_glb(fbx_path, glb_path):
    """Import one per-clip FBX and export it as GLB with its animation."""
    reset_scene()
    import_fbx(fbx_path)

    armatures = [o for o in bpy.data.objects if o.type == 'ARMATURE']
    assert armatures, f"No armature found in {fbx_path}"
    armature = armatures[0]
    assert armature.animation_data is not None and armature.animation_data.action is not None, \
        f"Armature '{armature.name}' has no bound action in {fbx_path}"

    # Name the GLB animation after the clip's action; drop the static
    # object-transform actions the FBX importer binds to mesh objects so the
    # skeletal take is the only animation in the file.
    stem = os.path.splitext(os.path.basename(fbx_path))[0]
    _, sep, action_part = stem.partition('-')
    if not sep:
        raise ValueError(
            f"Clip filename '{stem}' has no '-' separating species from action "
            f"(expected '{{Species}}-{{Action}}.fbx')")
    take = armature.animation_data.action
    take.name = action_part
    for obj in bpy.data.objects:
        if obj.type != 'ARMATURE' and obj.animation_data is not None:
            obj.animation_data_clear()
    for action in list(bpy.data.actions):
        if action is not take:
            bpy.data.actions.remove(action)

    for obj in bpy.data.objects:
        obj.select_set(obj.type in ('MESH', 'ARMATURE'))

    kwargs = dict(
        filepath=glb_path,
        check_existing=False,
        use_selection=True,
        export_format='GLB',
        export_animations=True,
    )
    # One animation per GLB. On Blender 3.2 `export_nla_strips=False` merges
    # everything into a single clip; the option was replaced by
    # `export_animation_mode` in 3.6 (ACTIVE_ACTIONS emits the armature's
    # bound action as the only animation).
    if bpy.app.version >= (3, 6, 0):
        kwargs['export_animation_mode'] = 'ACTIVE_ACTIONS'
    else:
        kwargs['export_nla_strips'] = False
    bpy.ops.export_scene.gltf(**kwargs)
    logger.info(f"Saved animated GLB: {glb_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert curated Truebones per-clip FBX files to GLB, one to one.")
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Flat directory of per-clip {Species}-{Action}.fbx files.')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Destination directory for the per-clip GLB files.')
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
    logger.info(f"Found {len(fbx_files)} FBX clips in {args.data_dir}")

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
            convert_fbx_to_glb(os.path.join(args.data_dir, fbx_file), glb_path)
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
