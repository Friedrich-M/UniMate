"""Preprocess one rigged, animated asset through the real export + feature pipeline.

Runs the actual pipeline stages in-process on a single GLB/FBX:

  1. EXPORT   (:func:`export_general.export_asset`) — discover the asset's
     pose actions, extract rest + animations, motion-driven skeleton pruning,
     Z-up -> Y-up; writes export-stage NPZs.
  2. EXTRACT  (:func:`motion_features.process_object`) — canonical T-pose
     (BFS order, facing -> center -> diameter scale -> ground), per-clip
     motion FEATURES, and the full topology conditioning dict the model
     consumes.
  3. BAKE     — rebuild the asset's armature to the canonical T-pose taken
     straight from the cond (same joint order, rest, scale), transform the
     skinned mesh by the same similarity, and save GLB/FBX. The canonical
     joint order rides inside the file as a custom property.

Outputs under ``--output_dir`` (exactly three by default):
    <name>_canonical.{glb,fbx}  canonical rest-pose asset (drives cond-free)
    cond.npy                    ``{asset_name: topology cond}`` — model-side input
    motions/<clip>.npz          motion features (``global_positions`` /
                                ``local_rotations`` / ``root_facing_quat`` / ``fps``)
Intermediates (export-stage NPZs, T-pose/preview visuals) are cleaned up on
success; pass ``--keep_intermediate`` to keep them (also makes reruns skip
the export stage via its completion markers).

Animate afterwards with ONLY a motion-feature NPZ (generated or extracted):

    python -m data_process.mesh_animation.animate_lbs \\
        --char_path <output_dir>/<name>_canonical.glb \\
        --anim_path <feature_motion.npz>

Facing canonicalization needs the asset's face-joint pair (``--face_r`` /
``--face_l`` raw bone names); omit for identity facing (pipeline sentinel).

Assets WITHOUT any action are supported too: a rest-only fallback builds the
cond from the rest pose (full armature skeleton — motion-driven pruning
needs animations — and no ``motions/`` output).

Usage (Blender headless):
    blender -b -P data_process/mesh_animation/preprocess_char.py -- \\
        --char_path my_asset.glb --output_dir outputs/my_asset \\
        --face_r R_Thigh --face_l L_Thigh --formats glb,fbx
"""

import argparse
import glob
import json
import math
import os
import sys

import bpy
import numpy as np
from loguru import logger
from mathutils import Matrix

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from data_process.joint_annotation.names_clean_rule import (  # noqa: E402
    clean_joint_name,
    post_process,
)
from data_process.mesh_animation.common import (  # noqa: E402
    load_character,
    parse_blender_argv,
    quat_pos_to_mats,
)
from data_process.motion_export.export_general import export_asset  # noqa: E402
from data_process.utils.blender_export import (  # noqa: E402
    apply_zup_to_yup,
    build_rest_animation,
    build_skeleton_arrays,
    clear_animation_state,
)
from data_process.utils.blender_rig import (  # noqa: E402
    export_selected_to_file,
    find_skinned_meshes,
    select_objs,
    sync_armature_bones,
    update_scene,
)
from data_process.utils.motion_features import (  # noqa: E402
    build_topology_cond,
    process_object,
    process_tpose,
)


def _umeyama(X, Y):
    """Similarity (s, R, t) with ``Y ~= s * X @ R + t``; returns (4x4, residual)."""
    muX, muY = X.mean(0), Y.mean(0)
    Xc, Yc = X - muX, Y - muY
    U, S, Vt = np.linalg.svd(Xc.T @ Yc)
    D = np.eye(3)
    D[2, 2] = np.sign(np.linalg.det(U @ Vt))
    R = U @ D @ Vt
    s = (S * D.diagonal()).sum() / (Xc ** 2).sum()
    M = np.eye(4)
    M[:3, :3] = s * R.T
    M[:3, 3] = muY - s * muX @ R
    resid = np.abs(s * X @ R + (muY - s * muX @ R) - Y).max()
    return M, resid


# ---------------------------------------------------------------------------
# Rest-only fallback: cond from the rest pose alone (asset has no actions)
# ---------------------------------------------------------------------------

def cond_from_rest(char_path, name, face_joints=None, target_diameter=2.0):
    """Build the topology cond from the asset's rest pose alone.

    Used when the asset carries no animation: the export stage's
    motion-driven skeleton pruning cannot run, so the FULL armature skeleton
    is kept; everything else (world-applied rest, Z-up -> Y-up, BFS reorder,
    facing/scale/ground canonicalization, cond fields) matches the pipeline.
    """
    from Animation import positions_global, rotations_global

    armature = load_character(char_path)
    names, parents, rest_pos, rest_rot, _ = build_skeleton_arrays(
        armature, None, apply_world=True)
    rest_anim = build_rest_animation(parents, rest_pos, rest_rot, len(names))
    _, rest_anim = apply_zup_to_yup(rest_anim, rest_anim)

    tpos_data = {
        'names': np.array(names), 'parents': np.asarray(parents),
        'rest_local_pos': np.asarray(rest_anim.positions[0]),
        'rest_local_rot': np.asarray(rest_anim.rotations.qs[0]),
        'fps': np.array(30),
    }
    clean = [post_process(clean_joint_name(n, name)) for n in names]

    (canon_anim, offsets, scale_factor, _gh, canon_parents, names_bfs,
     _fps, bfs, face_idxs, baxis) = process_tpose(
        tpos_data, face_joints=face_joints, target_diameter=target_diameter)
    clean_bfs = [clean[i] for i in bfs]

    cond = build_topology_cond(
        name, canon_parents, offsets, names_bfs, clean_bfs,
        positions_global(canon_anim)[0],
        canon_anim.rotations.qs[0],
        rotations_global(canon_anim).qs[0],
        face_joint_idxs=face_idxs, body_axis=baxis,
        scale_factor=scale_factor, ground_height=None)
    logger.warning(
        f"'{name}' has no animation: canonical skeleton = FULL armature "
        f"({len(names_bfs)} joints, no motion-driven pruning) and no motion "
        f"features are produced.")
    return cond


# ---------------------------------------------------------------------------
# Stage 3: bake the canonical rest pose into the asset
# ---------------------------------------------------------------------------

def bake_canonical_asset(char_path, cond, out_base, formats=('glb',)):
    """Rebuild the asset's armature to the cond's canonical T-pose and save.

    The canonical skeleton (joint order, rest positions/orientations) is
    taken verbatim from the topology cond so the baked asset and the motion
    features agree by construction. The skinned mesh gets the same
    similarity transform, so binding is untouched; asset bones outside the
    motion skeleton are weight-merged away first.
    """
    names = list(cond['joint_names'])
    canon_parents = [int(p) for p in cond['parents']]
    canon_pos = np.asarray(cond['tpos_first_frame'], dtype=np.float64)
    canon_mats = quat_pos_to_mats(
        np.asarray(cond['tpos_global_rotations'], dtype=np.float64), canon_pos)

    armature = load_character(char_path)
    meshes = find_skinned_meshes(armature)
    assert meshes, "No mesh with an Armature modifier targeting the asset's armature"

    # The canonical asset holds ONLY the rigged character: drop unskinned
    # props, lights, cameras, helper empties (e.g. the egg Icosphere in
    # Truebones' Chicken-EggLaying) — they can't be driven by the motion and
    # would linger as static junk in the export.
    keep = {armature.name} | {m.name for m in meshes}
    dropped = [o.name for o in bpy.context.scene.objects if o.name not in keep]
    for name_ in dropped:
        bpy.data.objects.remove(bpy.data.objects[name_], do_unlink=True)
    if dropped:
        logger.info(f"Dropped {len(dropped)} non-skinned scene object(s): {dropped[:6]}")

    missing = [n for n in names if n not in armature.data.bones]
    assert not missing, f"cond joints missing from the asset: {missing}"
    sync_armature_bones(armature, names, extra_bones_strategy='merge')

    # Similarity mapping Blender world -> canonical frame: the whole export +
    # extract chain is one similarity of the world frame; fit it exactly on
    # the bone heads and verify the residual.
    arm_world = np.array(armature.matrix_world)
    bones = armature.data.bones
    world_pos = np.stack([
        (arm_world @ np.array(bones[n].matrix_local))[:3, 3] for n in names])
    M, _ = _umeyama(world_pos, canon_pos)
    check = (np.concatenate([world_pos, np.ones((len(names), 1))], 1) @ M.T)[:, :3]
    resid = np.abs(check - canon_pos).max() / max(np.abs(canon_pos).max(), 1e-9)
    assert resid < 1e-4, (
        f"world->canonical is not a clean similarity (residual {resid:.2e}); "
        f"non-uniform armature scale, or the cond belongs to another asset?")
    sim_scale = float(np.cbrt(abs(np.linalg.det(M[:3, :3]))))

    # Transform every skinned mesh into the canonical frame. Un-parent BEFORE
    # zeroing transforms: the meshes hang under the armature, so identity-ing
    # the armature afterwards would shift any still-parented mesh again.
    for mesh_obj in meshes:
        T = M @ np.array(mesh_obj.matrix_world)
        for vert in mesh_obj.data.vertices:
            v = T @ np.array([*vert.co, 1.0])
            vert.co = v[:3]
        mesh_obj.data.update()
        mesh_obj.parent = None
        mesh_obj.matrix_world = Matrix.Identity(4)

    # Rebuild the armature: canonical rest, canonical (BFS) bone order.
    old_lengths = {n: bones[n].length for n in names}
    bpy.context.view_layer.objects.active = armature
    bpy.ops.object.mode_set(mode='EDIT')
    edit_bones = armature.data.edit_bones
    for eb in list(edit_bones):
        edit_bones.remove(eb)
    child_of = {p: j for j, p in enumerate(canon_parents) if p >= 0}
    for j, name in enumerate(names):
        eb = edit_bones.new(name)
        eb.head = (0.0, 0.0, 0.0)
        eb.tail = (0.0, 1.0, 0.0)
        length = (np.linalg.norm(canon_pos[child_of[j]] - canon_pos[j])
                  if j in child_of else old_lengths[name] * sim_scale)
        eb.matrix = Matrix(canon_mats[j].tolist())
        eb.length = max(float(length), 1e-5)
    for j, name in enumerate(names):
        if canon_parents[j] >= 0:
            eb = edit_bones[name]
            eb.use_connect = False
            eb.parent = edit_bones[names[canon_parents[j]]]
    bpy.ops.object.mode_set(mode='OBJECT')
    # ARMATURE-SPACE data stays exactly canonical (Y-up feature frame) — the
    # cond-free animate math lives there. The +90°X OBJECT rotation stands
    # the asset up in Blender's Z-up world; verified with an independent
    # glTF evaluator, the exported file is upright BOTH in standard Y-up
    # viewers and when re-imported into Blender.
    yup_to_zup = Matrix.Rotation(math.radians(90.0), 4, 'X')
    armature.matrix_world = yup_to_zup
    for mesh_obj in meshes:            # re-parent with identity LOCAL transforms
        mesh_obj.parent = armature
        mesh_obj.matrix_world = yup_to_zup.copy()

    # Strip animation; store the canonical joint ORDER as an OBJECT-level
    # custom property (glTF extras / FBX user properties) — importers do not
    # keep bone enumeration order stable.
    clear_animation_state(armature)
    for obj in meshes:
        if obj.animation_data:
            obj.animation_data_clear()
    # Unbinding is not enough: the glTF exporter's default animation mode
    # exports every matchable ACTION datablock, bound or not — delete them
    # all so the canonical asset is truly rest-only.
    for action in list(bpy.data.actions):
        bpy.data.actions.remove(action)
    update_scene()
    armature['canonical_joint_order'] = json.dumps(names)

    select_objs([armature] + meshes, deselect_first=True)
    for fmt in formats:
        assert fmt in ('glb', 'fbx'), f"unsupported output format: {fmt}"
        export_selected_to_file(f'{out_base}.{fmt}', fmt, custom_props=True)


# ---------------------------------------------------------------------------
# Full per-asset preprocessing
# ---------------------------------------------------------------------------

def preprocess_asset(char_path, output_dir, face_r=None, face_l=None,
                     body_axis=False, target_diameter=2.0, formats=('glb',),
                     save_vis=False, apply_clip=False, max_clip_len=10 ** 8,
                     keep_intermediate=False):
    """Export + extract + bake one asset; see the module docstring."""
    assert os.path.exists(char_path), f"Asset not found: {char_path}"
    name = os.path.splitext(os.path.basename(char_path))[0]
    os.makedirs(output_dir, exist_ok=True)

    face_joints = None
    if face_r and face_l:
        face_joints = {'r_hip': {'raw': face_r}, 'l_hip': {'raw': face_l},
                       'body_axis': body_axis}
    else:
        logger.info("No face joints given; identity facing (pipeline sentinel).")

    # --- 1. Export stage (real pipeline; resumes via its own markers) ------
    export_dir = os.path.join(output_dir, 'export')
    export_asset(char_path, export_dir, save_name=name, save_vis=False)
    npzs = sorted(glob.glob(os.path.join(export_dir, 'motions', f'{name}-*.npz')))

    if npzs:
        # --- 2a. Feature stage (real pipeline) ------------------------------
        logger.info(f"Export stage: {len(npzs)} clip NPZ(s)")
        tpos_data = np.load(npzs[0], allow_pickle=True)  # first sorted = T-pose ref
        raw_names = tpos_data['names'].tolist()
        clean_names = [post_process(clean_joint_name(n, name)) for n in raw_names]

        cond, n_clips, n_frames, n_joints, _filtered = process_object(
            name, npzs, save_dir=output_dir,
            face_joints=face_joints, clean_names=clean_names,
            expected_names=raw_names,
            apply_clip=apply_clip, max_clip_len=max_clip_len,
            target_diameter=target_diameter, save_vis=save_vis)
        assert cond is not None, ("feature extraction skipped this asset "
                                  "(degenerate skeleton or no usable clips — see log)")
        logger.info(f"Feature stage: {n_clips} clip(s), {n_frames} frames, "
                    f"{n_joints} joints")
    else:
        # --- 2b. Rest-only fallback: asset carries no animation -------------
        cond = cond_from_rest(char_path, name, face_joints=face_joints,
                              target_diameter=target_diameter)

    cond_path = os.path.join(output_dir, 'cond.npy')
    np.save(cond_path, {name: cond})
    logger.info(f"Saved cond -> {cond_path}")

    # --- 3. Bake the canonical rest pose into the asset ---------------------
    bake_canonical_asset(char_path, cond,
                         os.path.join(output_dir, f'{name}_canonical'),
                         formats=formats)

    # --- 4. Keep only the three deliverables --------------------------------
    if not keep_intermediate:
        import shutil
        for sub in ('export', 'tpose', 'videos', 'animations'):
            shutil.rmtree(os.path.join(output_dir, sub), ignore_errors=True)
        motions_dir = os.path.join(output_dir, 'motions')
        if os.path.isdir(motions_dir) and not os.listdir(motions_dir):
            os.rmdir(motions_dir)          # rest-only run: no motion features
        logger.info("Removed intermediates; kept: "
                    f"{name}_canonical.{{{','.join(formats)}}}, cond.npy, motions/")
    return cond


def parse_args():
    parser = argparse.ArgumentParser(
        description="Preprocess a rigged, animated asset: export + feature "
                    "extraction + canonical rest-pose bake.")
    parser.add_argument("--char_path", type=str, required=True,
                        help="Rigged asset GLB/FBX (mesh + armature + weights). "
                             "Assets WITH actions get the full pipeline incl. "
                             "motion-driven skeleton pruning and motion features; "
                             "action-less assets fall back to a rest-only cond "
                             "(full skeleton, no motions/).")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory (export/, motions/, cond.npy, "
                             "<name>_canonical.{glb,fbx}).")
    parser.add_argument("--face_r", type=str, default=None,
                        help="Raw name of the right face joint (facing canonicalization).")
    parser.add_argument("--face_l", type=str, default=None,
                        help="Raw name of the left face joint.")
    parser.add_argument("--body_axis", action='store_true',
                        help="Face pair is a head/tail body axis (serpentine rigs).")
    parser.add_argument("--target_diameter", type=float, default=2.0,
                        help="Canonical skeleton diameter (pipeline default: 2.0).")
    parser.add_argument("--formats", type=str, default='glb',
                        help="Comma-separated canonical-asset formats (glb,fbx).")
    parser.add_argument("--save_vis", action='store_true',
                        help="Also render per-clip preview MP4s (slow).")
    parser.add_argument("--apply_clip", action='store_true',
                        help="Crop motions into overlapping training-length clips "
                             "instead of one full-length feature NPZ per motion.")
    parser.add_argument("--keep_intermediate", action='store_true',
                        help="Keep export/ NPZs and T-pose/preview visuals (default: "
                             "only canonical asset + cond.npy + motions/ remain).")
    return parse_blender_argv(parser)


if __name__ == "__main__":
    args = parse_args()
    preprocess_asset(
        char_path=args.char_path,
        output_dir=args.output_dir,
        face_r=args.face_r,
        face_l=args.face_l,
        body_axis=args.body_axis,
        target_diameter=args.target_diameter,
        formats=tuple(f.strip().lower() for f in args.formats.split(',') if f.strip()),
        save_vis=args.save_vis,
        apply_clip=args.apply_clip,
        keep_intermediate=args.keep_intermediate,
    )
