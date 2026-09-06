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

import os
import sys

import bpy
from loguru import logger

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from data_process.utils.blender_export import import_fbx, reset_scene
from data_process.utils.fbx2glb import parse_batch_args, run_batch


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


if __name__ == "__main__":
    run_batch(parse_batch_args("Convert rigged T-pose character FBX files to GLB, one to one."),
              convert_character_to_glb, unit='character FBXs')
