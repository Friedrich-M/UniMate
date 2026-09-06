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

import os
import sys

import bpy
from loguru import logger

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from data_process.utils.blender_export import import_fbx, reset_scene
from data_process.utils.fbx2glb import parse_batch_args, run_batch


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


if __name__ == "__main__":
    run_batch(parse_batch_args("Convert curated Truebones per-clip FBX files to GLB, one to one."),
              convert_fbx_to_glb, unit='FBX clips')
