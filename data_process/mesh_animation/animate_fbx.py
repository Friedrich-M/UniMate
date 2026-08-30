"""Bake raw animation FBX/GLB clips onto a chosen rigged character.

Unlike :mod:`animate_npz` (which replays an export-stage NPZ), this works
straight from animation files: each clip's action is moved onto the
character's armature by bone name — the Mixamo convention, where every clip
is authored on the shared ``mixamorig:`` skeleton so any compatible
character can play it without remapping. Bones the character lacks are left
undriven; bones the clip does not touch keep their rest pose.

``--anim_path`` accepts a single clip or a directory of clips, so one
command can bake a whole animation library onto any character. Per-file
failures are logged to ``animate_errors.log`` inside the output dir.

Usage (Blender headless):
    blender -b -P data_process/mesh_animation/animate_fbx.py -- \\
        --char_path dataset/raw/mixamo/character_refined/Amy.fbx \\
        --anim_path dataset/raw/mixamo/animation_motion \\
        --output_dir outputs/animated_Amy \\
        --char_anim_type fbx
"""

import argparse
import os
import re
import sys

import bpy
from loguru import logger
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from data_process.utils.blender_export import reset_scene  # noqa: E402
from data_process.utils.blender_rig import (  # noqa: E402
    RECONSTRUCTED_ACTION_NAME,
    export_animated_character,
    get_armature_obj,
    load_file,
    repair_and_pack_textures,
    update_scene,
)

_BONE_PATH_RE = re.compile(r'^pose\.bones\["(.+?)"\]')


def _action_bone_names(action):
    """Bone names an action's F-curves drive."""
    names = set()
    for fcurve in action.fcurves:
        match = _BONE_PATH_RE.match(fcurve.data_path)
        if match:
            names.add(match.group(1))
    return names


def _adopt_action(char_armature, action):
    """Assign *action* to the character armature and mark it for export.

    ``export_animated_character`` keeps only the action named
    ``RECONSTRUCTED_ACTION_NAME``, so the transferred action takes that name
    (after removing any stale holder of it).
    """
    for stale in list(bpy.data.actions):
        if stale is not action and stale.name == RECONSTRUCTED_ACTION_NAME:
            bpy.data.actions.remove(stale)
    action.name = RECONSTRUCTED_ACTION_NAME

    # The importer keys the animation armature's *object* transform (unit
    # scale, axis correction) into the same action; the character must keep
    # its own object transform, so only bone channels come along.
    for fcurve in [f for f in action.fcurves
                   if not f.data_path.startswith('pose.bones')]:
        action.fcurves.remove(fcurve)

    if char_armature.animation_data is None:
        char_armature.animation_data_create()
    char_armature.animation_data.action = action
    # Blender >= 4.4 slotted actions: bind the (single) slot the importer made.
    anim_data = char_armature.animation_data
    if getattr(anim_data, 'action_slot', True) is None and action.slots:
        anim_data.action_slot = action.slots[0]


def animate_character_fbx(char_path, anim_path, output_dir, char_anim_type='glb',
                          fps=30, skip_if_exists=False):
    """Transfer one clip's action onto the character and export GLB/FBX.

    Returns the fraction of the clip's driven bones that exist on the
    character (1.0 = every animated bone landed).
    """
    anim_base = os.path.splitext(os.path.basename(anim_path))[0]
    output_path = os.path.join(output_dir, f"{anim_base}.{char_anim_type}")
    if skip_if_exists and os.path.exists(output_path):
        logger.info(f"Output already exists, skipping: {output_path}")
        return 1.0
    os.makedirs(output_dir, exist_ok=True)

    reset_scene()
    char_armature = get_armature_obj(load_file(char_path))
    assert char_armature is not None, "No armature found in character file"
    repair_and_pack_textures(char_path)

    anim_objs = load_file(anim_path)
    anim_armature = get_armature_obj(anim_objs)
    assert anim_armature is not None, "No armature found in animation file"
    assert anim_armature.animation_data and anim_armature.animation_data.action, \
        "Animation file carries no action"
    action = anim_armature.animation_data.action

    anim_bones = _action_bone_names(action)
    char_bones = {b.name for b in char_armature.data.bones}
    coverage = len(anim_bones & char_bones) / len(anim_bones) if anim_bones else 0.0
    if coverage < 0.5:
        logger.warning(
            f"{anim_base}: only {coverage:.0%} of the clip's bones exist on the "
            f"character — bone naming likely differs; the result will barely move.")

    _adopt_action(char_armature, action)
    for obj in anim_objs:
        bpy.data.objects.remove(obj, do_unlink=True)

    frame_start, frame_end = action.frame_range
    scene = bpy.context.scene
    scene.render.fps = int(fps)
    scene.frame_start = int(round(frame_start))
    scene.frame_end = int(round(frame_end))
    update_scene()

    export_animated_character(os.path.splitext(output_path)[0],
                              formats=(char_anim_type,))
    return coverage


def parse_args():
    parser = argparse.ArgumentParser(
        description="Bake raw animation FBX/GLB clips onto a rigged character.")
    parser.add_argument("--char_path", type=str, required=True,
                        help="Rigged character FBX/GLB (mesh + armature).")
    parser.add_argument("--anim_path", type=str, required=True,
                        help="Animation clip, or a directory of clips, whose "
                             "bone names match the character's rig.")
    parser.add_argument("--output_dir", type=str, default='outputs/animated',
                        help="Destination directory, one file per clip.")
    parser.add_argument("--char_anim_type", type=str, default='glb',
                        choices=['glb', 'fbx'],
                        help="Export format for the animated character.")
    parser.add_argument("--fps", type=int, default=30,
                        help="Scene frame rate for the export (Mixamo is 30).")
    parser.add_argument("--overwrite", action='store_true',
                        help="Re-export clips whose output already exists.")
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    assert os.path.exists(args.char_path), f"Character file not found: {args.char_path}"
    assert os.path.exists(args.anim_path), f"Animation path not found: {args.anim_path}"

    if os.path.isdir(args.anim_path):
        clips = sorted(os.path.join(args.anim_path, f)
                       for f in os.listdir(args.anim_path)
                       if f.lower().endswith(('.fbx', '.glb')))
    else:
        clips = [args.anim_path]
    logger.info(f"Baking {len(clips)} clip(s) onto {args.char_path}")

    os.makedirs(args.output_dir, exist_ok=True)
    err_log = os.path.join(args.output_dir, "animate_errors.log")

    for clip in tqdm(clips, desc="Animating character"):
        try:
            animate_character_fbx(
                char_path=args.char_path,
                anim_path=clip,
                output_dir=args.output_dir,
                char_anim_type=args.char_anim_type,
                fps=args.fps,
                skip_if_exists=not args.overwrite,
            )
        except Exception as e:  # noqa: BLE001 — keep the batch going
            logger.error(f"Error processing {os.path.basename(clip)}: {e}")
            with open(err_log, "a") as f:
                f.write(f"{os.path.basename(clip)}: {e}\n")
