"""Shared plumbing for the mesh-animation entry points.

Every entry script in this package runs under Blender headless
(``blender -b -P <script>.py -- <args>``), loads a rigged character, binds a
driving animation onto its armature, and exports GLB/FBX. They differ only in
where the driving animation comes from:

    animate_motion   feature NPZ (or legacy model ``.npy``) + ``cond.npy``
    animate_npz      export-stage NPZ (bone names carried in the file)
    animate_fbx      raw animation FBX/GLB clips (action transfer by name)
    animate_mixamo   batch wrapper over ``animate_npz``

This module holds the pieces they share: Blender-style CLI parsing, character
loading, the keyframe-and-export core, and the error-logged batch loop. Rig
primitives (keyframe math, armature reconciliation, exporters) stay in
:mod:`data_process.utils.blender_rig`.
"""

import os
import sys

from loguru import logger
from tqdm import tqdm

from Animation import transforms_local

from data_process.utils.blender_export import reset_scene
from data_process.utils.blender_rig import (
    compute_bone_keyframes,
    export_animated_character,
    get_armature_obj,
    load_file,
    rebuild_action_from_data,
    repair_and_pack_textures,
    set_scene_timing,
    sync_armature_bones,
    update_scene,
)

ERROR_LOG_NAME = 'animate_errors.log'


# ---------------------------------------------------------------------------
# CLI / small helpers
# ---------------------------------------------------------------------------

def parse_blender_argv(parser):
    """Parse CLI args for a script run via ``blender -b -P script.py -- <args>``.

    Blender consumes everything before the ``--`` separator; only what
    follows belongs to the script. Plain ``python script.py <args>`` (no
    separator) also works, so the entry points run under pip ``bpy`` too.
    """
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
    return parser.parse_args(argv)


def npz_scalar(data, key, default):
    """Read a scalar from an NpzFile / dict, falling back to *default*.

    Mapping-free inputs (e.g. the legacy ``.npy`` ndarray in
    ``animate_motion``) have no keys and always yield *default*.
    """
    if hasattr(data, 'keys') and key in data:
        return float(data[key])
    return default


def clip_output_path(anim_path, output_dir, ext=None):
    """``<output_dir>/<clip stem>[.<ext>]`` for a driving-animation file."""
    base = os.path.splitext(os.path.basename(anim_path))[0]
    return os.path.join(output_dir, base + (f'.{ext}' if ext else ''))


def quat_pos_to_mats(q, pos):
    """Batch (…, 4) wxyz quaternions + (…, 3) translations -> (…, 4, 4)."""
    import numpy as np
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    m = np.zeros(q.shape[:-1] + (4, 4))
    m[..., 0, 0] = 1 - 2 * (y * y + z * z)
    m[..., 0, 1] = 2 * (x * y - w * z)
    m[..., 0, 2] = 2 * (x * z + w * y)
    m[..., 1, 0] = 2 * (x * y + w * z)
    m[..., 1, 1] = 1 - 2 * (x * x + z * z)
    m[..., 1, 2] = 2 * (y * z - w * x)
    m[..., 2, 0] = 2 * (x * z - w * y)
    m[..., 2, 1] = 2 * (y * z + w * x)
    m[..., 2, 2] = 1 - 2 * (x * x + y * y)
    m[..., :3, 3] = pos
    m[..., 3, 3] = 1.0
    return m


# ---------------------------------------------------------------------------
# Character loading
# ---------------------------------------------------------------------------

def load_character(char_path, pack_textures=True):
    """Reset the scene, import the character file and return its armature.

    ``pack_textures`` re-links missing texture files found next to the
    character and packs images into memory so the exporters don't silently
    drop them (see :func:`repair_and_pack_textures`).
    """
    assert os.path.exists(char_path), f"Character file not found: {char_path}"
    reset_scene()
    armature = get_armature_obj(load_file(char_path))
    assert armature is not None, f"No armature found in character file: {char_path}"
    if pack_textures:
        repair_and_pack_textures(char_path)
    return armature


# ---------------------------------------------------------------------------
# Keyframe-and-export core (shared by animate_motion / animate_npz)
# ---------------------------------------------------------------------------

def drive_and_export(char_armature, anim, rest_anim, bone_names, fps,
                     output_path_no_ext, formats=('glb',),
                     extra_bones_strategy='merge', tpos_global_rot=None):
    """Bind an ``(anim, rest_anim)`` pair onto the armature and export.

    Args:
        char_armature: The character's ARMATURE object (see
            :func:`load_character`).
        anim: ``Animation`` with the animated local transforms.
        rest_anim: Single-frame ``Animation`` with the rest pose the
            character's bind pose corresponds to.
        bone_names: Bone names, index-aligned with the animations.
        fps: Scene frame rate for the export.
        output_path_no_ext: Output path without extension.
        formats: Iterable of export formats (``'glb'`` / ``'fbx'``).
        extra_bones_strategy: How to treat armature bones absent from
            *bone_names* (see :func:`sync_armature_bones`).
        tpos_global_rot: Optional ``(nbones, 4)`` T-pose global rotations to
            conjugate keyframes by (legacy ``.npy`` path in animate_motion).
    """
    anim_local_mat = transforms_local(anim)          # (nframes, nbones, 4, 4)
    rest_local_mat = transforms_local(rest_anim)[0]  # (nbones, 4, 4)

    sync_armature_bones(char_armature, bone_names,
                        extra_bones_strategy=extra_bones_strategy)
    set_scene_timing(anim_local_mat.shape[0], fps)

    keyframes = compute_bone_keyframes(
        rest_local_mat, anim_local_mat, bone_names, tpos_global_rot)
    rebuild_action_from_data(char_armature, keyframes)
    update_scene()

    export_animated_character(output_path_no_ext, formats=formats)


# ---------------------------------------------------------------------------
# Batch driver (shared by animate_fbx / animate_mixamo)
# ---------------------------------------------------------------------------

def run_clip_batch(clips, process_one, output_dir, desc):
    """Process clips one by one; failures are logged, not fatal.

    Each failed clip is recorded (with its error) in
    ``<output_dir>/animate_errors.log`` and the batch continues.
    """
    os.makedirs(output_dir, exist_ok=True)
    err_log = os.path.join(output_dir, ERROR_LOG_NAME)
    for clip in tqdm(clips, desc=desc):
        try:
            process_one(clip)
        except Exception as e:  # noqa: BLE001 — keep the batch going
            logger.error(f"Error processing {os.path.basename(clip)}: {e}")
            with open(err_log, "a") as f:
                f.write(f"{os.path.basename(clip)}: {e}\n")
