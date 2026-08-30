"""Drive a rigged character with an *exported* motion NPZ and save GLB/FBX.

Takes the NPZ format written by the export stage
(``rest_local_pos`` / ``rest_local_rot`` / ``anim_local_pos`` /
``anim_local_rot`` / ``names`` / ``parents`` / ``offsets``) and re-applies it
to a character whose armature uses the same bone names. Useful for
round-trip checks of the export and for re-rendering exported clips.

For *generated* motions (feature NPZ + ``cond.npy``) use
:mod:`data_process.mesh_animation.animate_motion` instead.

Usage (Blender headless):
    blender -b -P data_process/mesh_animation/animate_npz.py -- \\
        --char_path my_model.glb \\
        --anim_path <export_dir>/motions/my_model-Walk.npz \\
        --output_dir outputs/animated
"""

import argparse
import os
import sys

import numpy as np
from loguru import logger

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from Animation import Animation, Quaternions, transforms_local  # noqa: E402

from data_process.utils.blender_export import reset_scene  # noqa: E402
from data_process.utils.blender_rig import (  # noqa: E402
    compute_bone_keyframes,
    export_animated_character,
    get_armature_obj,
    load_file,
    rebuild_action_from_data,
    set_scene_timing,
    sync_armature_bones,
    update_scene,
)


def _npz_get_scalar(npz, key, default):
    """Read a scalar from an NpzFile, falling back to *default*."""
    return float(npz[key]) if key in npz else default


def build_anims_from_export_npz(anim_data):
    """Build ``(anim, rest_anim)`` Animation objects from an export NPZ."""
    bone_names = anim_data['names'].tolist()
    nbones = anim_data['anim_local_rot'].shape[1]
    scale_factor = _npz_get_scalar(anim_data, 'scale_factor', 1.0)

    anim_positions = anim_data['anim_local_pos'] / scale_factor
    rest_positions = anim_data['rest_local_pos'][None] / scale_factor
    offsets = anim_data['offsets'] / scale_factor
    parents = anim_data['parents']

    anim = Animation(
        rotations=Quaternions(anim_data['anim_local_rot']),
        positions=anim_positions,
        orients=Quaternions.id(nbones),
        offsets=offsets,
        parents=parents,
    )
    rest_anim = Animation(
        rotations=Quaternions(anim_data['rest_local_rot'][None]),
        positions=rest_positions,
        orients=Quaternions.id(nbones),
        offsets=offsets,
        parents=parents,
    )
    return anim, rest_anim, bone_names


def animate_character(char_path, anim_path, output_dir, char_anim_type='glb',
                      extra_bones_strategy='merge', skip_if_exists=False):
    """Drive a character mesh with an export NPZ and save as GLB/FBX.

    Args:
        char_path: Character FBX/GLB (provides mesh + armature).
        anim_path: Motion NPZ produced by the export stage.
        output_dir: Destination directory (created if missing).
        char_anim_type: Export format, ``'glb'`` or ``'fbx'``.
        extra_bones_strategy: How to treat armature bones absent from the
            animation (see :func:`sync_armature_bones`).
        skip_if_exists: Return early when the output file already exists.
    """
    assert os.path.exists(char_path), f"Character file not found: {char_path}"
    assert os.path.exists(anim_path), f"Animation file not found: {anim_path}"

    anim_base = os.path.splitext(os.path.basename(anim_path))[0]
    output_path = os.path.join(output_dir, f"{anim_base}.{char_anim_type}")
    if skip_if_exists and os.path.exists(output_path):
        logger.info(f"Output already exists, skipping: {output_path}")
        return
    os.makedirs(output_dir, exist_ok=True)

    reset_scene()
    char_armature = get_armature_obj(load_file(char_path))
    assert char_armature is not None, "No armature found in character file"

    anim_data = np.load(anim_path, allow_pickle=True)
    anim, rest_anim, bone_names = build_anims_from_export_npz(anim_data)
    anim_local_mat = transforms_local(anim)          # (nframes, nbones, 4, 4)
    rest_local_mat = transforms_local(rest_anim)[0]  # (nbones, 4, 4)

    sync_armature_bones(char_armature, bone_names,
                        extra_bones_strategy=extra_bones_strategy)
    set_scene_timing(anim_local_mat.shape[0], _npz_get_scalar(anim_data, 'fps', 30))

    keyframes = compute_bone_keyframes(rest_local_mat, anim_local_mat, bone_names)
    rebuild_action_from_data(char_armature, keyframes)
    update_scene()

    export_animated_character(os.path.splitext(output_path)[0], formats=(char_anim_type,))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Drive a rigged character with an exported motion NPZ.")
    parser.add_argument("--char_path", type=str, required=True,
                        help="Character FBX/GLB (mesh + armature).")
    parser.add_argument("--anim_path", type=str, required=True,
                        help="Export-stage motion NPZ whose bone names match the rig.")
    parser.add_argument("--output_dir", type=str, default='outputs/animated')
    parser.add_argument("--char_anim_type", type=str, default='glb', choices=['glb', 'fbx'])
    parser.add_argument("--extra_bones_strategy", type=str, default='merge',
                        choices=['keep', 'merge', 'remove'],
                        help="What to do with armature bones missing from the animation.")
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    animate_character(
        char_path=args.char_path,
        anim_path=args.anim_path,
        output_dir=args.output_dir,
        char_anim_type=args.char_anim_type,
        extra_bones_strategy=args.extra_bones_strategy,
    )
