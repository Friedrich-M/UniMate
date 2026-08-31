"""Animate a rigged asset by applying Linear Blend Skinning manually in NumPy.

Instead of building a Blender Action and letting Blender's armature modifier
deform the mesh (the :mod:`animate_motion` / :mod:`animate_npz` path), this
extracts the skinning data from the rigged asset once — vertices, weights,
bind pose — computes per-frame bone matrices from the motion, and runs the
LBS equation itself:

    v[f] = sum_b  w[v,b] * (G[f,b] @ inv(bind[b])) @ v_rest

Blender is used ONLY to parse the asset (GLB/FBX mesh + weights + rest bones);
all skinning math is plain NumPy, so the result doubles as a reference
implementation and feeds pipelines that want raw deformed vertices (point
clouds, custom renderers, geometry losses) rather than a re-exported rig.

Accepts both motion formats, auto-detected from the NPZ keys:
  - export-stage NPZ (``anim_local_rot`` / ``names`` / ...), self-contained;
  - feature-format NPZ (``local_rotations`` + ``global_positions``, the
    generated-motion format):
      * with ``--dataset_type`` (and optionally ``--cond_path``) the motion
        skeleton is resolved through ``cond.npy`` exactly like
        :mod:`animate_motion`;
      * WITHOUT them, the asset itself is taken as the canonical motion
        skeleton — the cond-free path for assets baked by
        :mod:`preprocess_char` (joint order = bone order, rest = the
        asset's rest, scale = 1).

Bones are matched to the asset by name. Asset bones missing from the motion
keep their rest local transform (they follow their parent, the ``'keep'``
strategy); motion bones missing from the asset are ignored with a warning.

Outputs (per ``--save``; default ``glb``):
  - ``glb`` / ``fbx``: ``<clip>_lbs.<fmt>`` — the animated RIGGED asset,
    exported through the same keyframe path as :mod:`animate_npz` (the
    manual LBS and Blender's armature modifier are numerically equivalent,
    so this is the playable twin of the vertex outputs).
  - ``npz``: ``<clip>_lbs.npz`` with ``vertices (F, V, 3)`` (world space),
    ``faces``, ``fps``, ``frame_count`` — opt-in, for pipelines that want
    raw deformed vertices.
  - ``obj``: one OBJ per frame under ``<clip>_lbs/`` — opt-in.

Usage (Blender headless or plain python with pip bpy):
    blender -b -P data_process/mesh_animation/animate_lbs.py -- \\
        --char_path my_rigged_asset.glb \\
        --anim_path <export_dir>/motions/<clip>.npz \\
        --output_dir outputs/lbs
"""

import argparse
import os
import sys

import numpy as np
from loguru import logger

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from Animation import transforms_local  # noqa: E402

from data_process.mesh_animation.common import (  # noqa: E402
    load_character,
    npz_scalar,
    parse_blender_argv,
    quat_pos_to_mats,
)
from data_process.utils.blender_rig import (  # noqa: E402
    compute_bone_keyframes,
    export_animated_character,
    find_skinned_meshes,
    rebuild_action_from_data,
    set_scene_timing,
    update_scene,
)


# ---------------------------------------------------------------------------
# Skin extraction (the only bpy-dependent step)
# ---------------------------------------------------------------------------

def extract_skin(char_armature):
    """Extract everything LBS needs from a loaded rigged asset.

    Returns a dict with:
        bone_names   : list of N bone names (armature order)
        parents      : (N,) parent indices, -1 for roots
        rest_global  : (N, 4, 4) armature-space bind matrices
        rest_local   : (N, 4, 4) parent-relative rest transforms
        vertices     : (V, 4) homogeneous rest vertices in ARMATURE space
        weights      : (V, N) normalized skinning weights (rows may be all
                       zero for unskinned vertices — those stay at rest)
        faces        : list of per-polygon vertex-index tuples
        arm_world    : (4, 4) armature object world matrix
    """
    bones = list(char_armature.data.bones)
    bone_names = [b.name for b in bones]
    bone_index = {n: i for i, n in enumerate(bone_names)}
    parents = np.array(
        [bone_index[b.parent.name] if b.parent else -1 for b in bones])
    rest_global = np.stack([np.array(b.matrix_local) for b in bones])
    rest_local = np.stack([
        np.array(b.matrix_local) if b.parent is None
        else np.linalg.inv(np.array(b.parent.matrix_local)) @ np.array(b.matrix_local)
        for b in bones])

    arm_world = np.array(char_armature.matrix_world)
    arm_world_inv = np.linalg.inv(arm_world)

    all_verts, all_weights, all_faces = [], [], []
    offset = 0
    meshes = find_skinned_meshes(char_armature)
    assert meshes, "No mesh with an Armature modifier targeting the asset's armature"
    for mesh_obj in meshes:
        mesh = mesh_obj.data
        to_arm = arm_world_inv @ np.array(mesh_obj.matrix_world)

        v = np.ones((len(mesh.vertices), 4))
        for i, vert in enumerate(mesh.vertices):
            v[i, :3] = vert.co
        all_verts.append(v @ to_arm.T)

        group_to_bone = {g.index: bone_index.get(g.name) for g in mesh_obj.vertex_groups}
        w = np.zeros((len(mesh.vertices), len(bones)))
        for i, vert in enumerate(mesh.vertices):
            for g in vert.groups:
                b = group_to_bone.get(g.group)
                if b is not None and g.weight > 0.0:
                    w[i, b] += g.weight
        all_weights.append(w)

        all_faces.extend(tuple(idx + offset for idx in poly.vertices)
                         for poly in mesh.polygons)
        offset += len(mesh.vertices)

    vertices = np.concatenate(all_verts)
    weights = np.concatenate(all_weights)
    sums = weights.sum(axis=1, keepdims=True)
    skinned = sums[:, 0] > 0
    weights[skinned] /= sums[skinned]
    logger.info(f"Extracted {len(vertices)} vertices ({(~skinned).sum()} unskinned), "
                f"{len(all_faces)} faces, {len(bones)} bones from {len(meshes)} mesh(es)")

    # Canonical joint order baked in by preprocess_char (glTF extras carry
    # object-level custom properties); bone enumeration order itself is not
    # stable across glTF round trips.
    import json
    order = (char_armature.get('canonical_joint_order')
             or char_armature.data.get('canonical_joint_order'))
    if isinstance(order, str):
        order = json.loads(order)

    return {
        'bone_names': bone_names, 'parents': parents,
        'rest_global': rest_global, 'rest_local': rest_local,
        'vertices': vertices, 'weights': weights, 'faces': all_faces,
        'arm_world': arm_world, 'canonical_order': order,
    }


# ---------------------------------------------------------------------------
# Motion loading (both NPZ flavors)
# ---------------------------------------------------------------------------

def load_motion_local_mats(anim_path, dataset_type=None, cond_path=None):
    """Return ``(anim_local (F,J,4,4), rest_local (J,4,4), bone_names, fps)``.

    Auto-detects the NPZ flavor: export-stage (self-contained) vs
    feature-format (generated motion; resolved through ``cond.npy`` exactly
    like :mod:`animate_motion`).
    """
    data = np.load(anim_path, allow_pickle=True)
    if 'anim_local_rot' in data:
        from data_process.mesh_animation.animate_npz import build_anims_from_export_npz
        anim, rest_anim, names = build_anims_from_export_npz(data)
        fps = npz_scalar(data, 'fps', 30)
    elif 'local_rotations' in data:
        from data_process.mesh_animation.animate_motion import (
            COND_PATH_TEMPLATE,
            build_anim_from_npz,
            load_cond_data,
        )
        assert dataset_type, "--dataset_type is required for feature-format motion"
        cond_path = cond_path or COND_PATH_TEMPLATE.format(dataset_type=dataset_type)
        cond_data = load_cond_data(cond_path, anim_path, dataset_type)
        anim, rest_anim, _tpos, raw = build_anim_from_npz(anim_path, cond_data)
        names = list(cond_data['joint_names'])
        fps = npz_scalar(raw, 'fps', 30)
    else:
        raise RuntimeError(
            f"Unrecognized motion NPZ (keys: {sorted(data.keys())}); expected "
            "export-stage ('anim_local_rot', ...) or feature format "
            "('local_rotations', ...)")
    return (transforms_local(anim), transforms_local(rest_anim)[0],
            [str(n) for n in names], fps)


def motion_from_canonical_char(data, skin):
    """Feature NPZ driving a canonical asset (:mod:`preprocess_char` output).

    The asset's own skeleton IS the motion skeleton: joint order = bone
    order, rest pose = the asset's rest, scale = 1 — so no ``cond.npy`` is
    needed. Root positions come from ``global_positions``; every other joint
    keeps its rest offset (rigid bones), rotations from ``local_rotations``.
    """
    q = np.asarray(data['local_rotations'], dtype=np.float64)     # (F, J, 4) wxyz
    gp = np.asarray(data['global_positions'], dtype=np.float64)   # (F, J, 3)

    order = skin.get('canonical_order')
    if order is None:
        logger.warning("Asset carries no canonical_joint_order property (not "
                       "produced by preprocess_char?); assuming its bone "
                       "enumeration order matches the motion's joint order")
        order = list(skin['bone_names'])
    J = len(order)
    assert q.shape[1] == J, (
        f"motion has {q.shape[1]} joints but the asset's canonical skeleton "
        f"has {J} bones; the cond-free path needs an asset baked to the "
        f"motion skeleton (see preprocess_char.py) — or pass "
        f"--dataset_type/--cond_path")
    missing = [n for n in order if n not in skin['bone_names']]
    assert not missing, f"canonical_joint_order names missing from armature: {missing}"

    idx = [skin['bone_names'].index(n) for n in order]
    rest_local = skin['rest_local'][idx]                 # motion joint order
    pos = np.repeat(rest_local[None, :, :3, 3], q.shape[0], axis=0)
    pos[:, 0] = gp[:, 0]
    anim_local = quat_pos_to_mats(q, pos)
    return anim_local, rest_local, list(order), npz_scalar(data, 'fps', 30)


# ---------------------------------------------------------------------------
# Manual LBS
# ---------------------------------------------------------------------------

def lbs_deform(skin, anim_local, motion_rest_local, motion_bone_names):
    """Run manual Linear Blend Skinning; returns world-space ``(F, V, 3)``.

    Per frame and asset bone::

        pose_local[b] = rest_local[b] @ inv(m_rest[b]) @ m_anim[f, b]   (driven)
        pose_local[b] = rest_local[b]                                    (undriven)
        G[f, b]       = G[f, parent] @ pose_local[b]
        skin_mat[b]   = G[f, b] @ inv(rest_global[b])
        v[f]          = sum_b w[v, b] * skin_mat[b] @ v_rest

    which is exactly what the Blender armature modifier evaluates for the
    Action built by the keyframe path, so the two pipelines are
    interchangeable.
    """
    bone_names = skin['bone_names']
    parents = skin['parents']
    nframes = anim_local.shape[0]
    nbones = len(bone_names)

    motion_idx = {n: i for i, n in enumerate(motion_bone_names)}
    driven = [(b, motion_idx[n]) for b, n in enumerate(bone_names) if n in motion_idx]
    missing = [n for n in motion_bone_names if n not in set(bone_names)]
    if missing:
        logger.warning(f"{len(missing)} motion bones not on the asset "
                       f"(ignored): {missing[:8]}{'...' if len(missing) > 8 else ''}")
    undriven = nbones - len(driven)
    if undriven:
        logger.info(f"{undriven} asset bones not in the motion keep their rest "
                    f"local transform (follow parent)")
    assert driven, "No motion bone matches the asset's armature by name"

    # Per-frame local transforms for every asset bone.
    pose_local = np.broadcast_to(skin['rest_local'], (nframes, nbones, 4, 4)).copy()
    for b, j in driven:
        rel = np.linalg.inv(motion_rest_local[j])[None] @ anim_local[:, j]
        pose_local[:, b] = skin['rest_local'][b][None] @ rel

    # FK to armature space (bones are stored parent-before-child in Blender).
    glob = np.empty_like(pose_local)
    for b in range(nbones):
        p = parents[b]
        glob[:, b] = pose_local[:, b] if p < 0 else glob[:, p] @ pose_local[:, b]

    skin_mats = glob @ np.linalg.inv(skin['rest_global'])[None]   # (F, N, 4, 4)

    # LBS: blend matrices per vertex, then transform.
    verts = skin['vertices']                                      # (V, 4)
    weights = skin['weights']                                     # (V, N)
    unskinned = weights.sum(axis=1) == 0
    out = np.empty((nframes, len(verts), 3))
    for f in range(nframes):
        vert_mats = np.einsum('vb,bij->vij', weights, skin_mats[f])
        deformed = np.einsum('vij,vj->vi', vert_mats, verts)
        deformed[unskinned] = verts[unskinned]
        out[f] = (deformed @ skin['arm_world'].T)[:, :3]
    return out


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def save_obj_sequence(dirpath, vertices, faces):
    os.makedirs(dirpath, exist_ok=True)
    face_lines = ["f " + " ".join(str(i + 1) for i in poly) for poly in faces]
    for f in range(vertices.shape[0]):
        with open(os.path.join(dirpath, f"{f:04d}.obj"), 'w') as fh:
            fh.write("\n".join(f"v {x:.6f} {y:.6f} {z:.6f}" for x, y, z in vertices[f]))
            fh.write("\n" + "\n".join(face_lines) + "\n")
    logger.info(f"Saved {vertices.shape[0]} OBJ frames to {dirpath}")


def animate_lbs(char_path, anim_path, output_dir, dataset_type=None,
                cond_path=None, save=('glb',)):
    """Deform a rigged asset with a motion NPZ via manual LBS and save."""
    assert os.path.exists(anim_path), f"Animation file not found: {anim_path}"
    os.makedirs(output_dir, exist_ok=True)

    rig_formats = tuple(f for f in save if f in ('glb', 'fbx'))
    char_armature = load_character(char_path, pack_textures=bool(rig_formats))
    skin = extract_skin(char_armature)

    data = np.load(anim_path, allow_pickle=True)
    if 'local_rotations' in data and dataset_type is None and cond_path is None:
        logger.info("Feature NPZ with no cond: treating the asset as the "
                    "canonical motion skeleton (preprocess_char output)")
        anim_local, rest_local, motion_names, fps = motion_from_canonical_char(data, skin)
    else:
        anim_local, rest_local, motion_names, fps = load_motion_local_mats(
            anim_path, dataset_type=dataset_type, cond_path=cond_path)

    vertices = lbs_deform(skin, anim_local, rest_local, motion_names)

    stem = os.path.splitext(os.path.basename(anim_path))[0] + '_lbs'
    faces_arr = np.array(skin['faces'], dtype=object)
    if 'npz' in save:
        out = os.path.join(output_dir, stem + '.npz')
        np.savez_compressed(out, vertices=vertices.astype(np.float32),
                            faces=faces_arr, fps=fps,
                            frame_count=vertices.shape[0])
        logger.info(f"Saved LBS vertex animation: {out} "
                    f"(F={vertices.shape[0]}, V={vertices.shape[1]})")
    if 'obj' in save:
        save_obj_sequence(os.path.join(output_dir, stem), vertices, skin['faces'])
    if rig_formats:
        # Rigged export via the same keyframe path animate_npz uses: the
        # relative keyframes inv(rest) @ anim land on the asset's own rest,
        # exactly what lbs_deform evaluates. Bones absent from the motion
        # keep their rest local transform (no keyframes = follow parent),
        # matching the LBS 'keep' semantics.
        set_scene_timing(anim_local.shape[0], fps)
        keyframes = compute_bone_keyframes(rest_local, anim_local, motion_names)
        rebuild_action_from_data(char_armature, keyframes)
        update_scene()
        export_animated_character(os.path.join(output_dir, stem),
                                  formats=rig_formats)
    return vertices


def parse_args():
    parser = argparse.ArgumentParser(
        description="Animate a rigged asset via manual NumPy LBS.")
    parser.add_argument("--char_path", type=str, required=True,
                        help="Rigged asset GLB/FBX (mesh + armature + weights).")
    parser.add_argument("--anim_path", type=str, required=True,
                        help="Motion NPZ (export-stage or feature-format, auto-detected), "
                             "or a DIRECTORY of NPZs — one output set per action clip.")
    parser.add_argument("--dataset_type", type=str, default=None,
                        choices=['truebones', 'objaverse', 'mixamo'],
                        help="For feature-format motion: resolves cond.npy. Omit it "
                             "(and --cond_path) to drive a canonical asset baked by "
                             "preprocess_char.py cond-free.")
    parser.add_argument("--cond_path", type=str, default=None,
                        help="cond.npy override for feature-format motion.")
    parser.add_argument("--output_dir", type=str, default='outputs/lbs')
    parser.add_argument("--save", type=str, default='glb',
                        help="Comma-separated outputs: glb / fbx (animated rigged "
                             "asset via the animate_npz keyframe path; default), "
                             "npz (vertex animation), obj (per-frame OBJ files).")
    return parse_blender_argv(parser)


if __name__ == "__main__":
    args = parse_args()
    save = tuple(s.strip() for s in args.save.split(',') if s.strip())

    if os.path.isdir(args.anim_path):
        import glob as _glob
        from data_process.mesh_animation.common import run_clip_batch
        clips = sorted(_glob.glob(os.path.join(args.anim_path, '*.npz')))
        logger.info(f"Animating {len(clips)} clip(s) from {args.anim_path} — "
                    f"one output set per action")
        run_clip_batch(
            clips,
            lambda clip: animate_lbs(
                char_path=args.char_path, anim_path=clip,
                output_dir=args.output_dir, dataset_type=args.dataset_type,
                cond_path=args.cond_path, save=save),
            output_dir=args.output_dir,
            desc="LBS animating",
        )
    else:
        animate_lbs(
            char_path=args.char_path,
            anim_path=args.anim_path,
            output_dir=args.output_dir,
            dataset_type=args.dataset_type,
            cond_path=args.cond_path,
            save=save,
        )
