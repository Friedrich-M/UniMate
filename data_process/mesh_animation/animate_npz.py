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

from Animation import Animation, Quaternions  # noqa: E402

from data_process.mesh_animation.common import (  # noqa: E402
    clip_output_path,
    drive_and_export,
    load_character,
    npz_scalar,
    parse_blender_argv,
)
from data_process.utils.blender_rig import EXTRA_BONES_STRATEGIES  # noqa: E402


def build_anims_from_export_npz(anim_data):
    """Build ``(anim, rest_anim, bone_names)`` from an export-stage NPZ."""
    bone_names = anim_data['names'].tolist()
    nbones = anim_data['anim_local_rot'].shape[1]
    scale_factor = npz_scalar(anim_data, 'scale_factor', 1.0)

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
    assert os.path.exists(anim_path), f"Animation file not found: {anim_path}"

    output_path = clip_output_path(anim_path, output_dir, char_anim_type)
    if skip_if_exists and os.path.exists(output_path):
        logger.info(f"Output already exists, skipping: {output_path}")
        return
    os.makedirs(output_dir, exist_ok=True)

    char_armature = load_character(char_path)

    anim_data = np.load(anim_path, allow_pickle=True)
    anim, rest_anim, bone_names = build_anims_from_export_npz(anim_data)

    drive_and_export(
        char_armature, anim, rest_anim, bone_names,
        fps=npz_scalar(anim_data, 'fps', 30),
        output_path_no_ext=os.path.splitext(output_path)[0],
        formats=(char_anim_type,),
        extra_bones_strategy=extra_bones_strategy,
    )


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
                        choices=list(EXTRA_BONES_STRATEGIES),
                        help="What to do with armature bones missing from the animation.")
    return parse_blender_argv(parser)


if __name__ == "__main__":
    args = parse_args()
    animate_character(
        char_path=args.char_path,
        anim_path=args.anim_path,
        output_dir=args.output_dir,
        char_anim_type=args.char_anim_type,
        extra_bones_strategy=args.extra_bones_strategy,
    )
