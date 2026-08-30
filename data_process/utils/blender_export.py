"""Blender utilities for exporting rigged assets to NPZ motion data.

Format-agnostic helpers shared by the Truebones (FBX), Mixamo (FBX),
Objaverse (GLB/GLTF) and single-asset exporters: scene loading, action
discovery, skeleton/skin extraction, skeleton pruning, Z-up → Y-up
conversion and NPZ/summary writing.

The counterpart for the *animate* direction (NPZ → rigged character) lives
in :mod:`data_process.utils.blender_rig`.
"""

import csv
import os
import re
import json
import numpy as np
import bpy
from math import floor, ceil
from pathlib import Path
from loguru import logger
from Animation import Animation, Quaternions, positions_global
from data_process.utils.plotting import save_skeleton_motion, save_skeleton_tpose
from data_process.utils.export_markers import (  # noqa: F401  (re-exported)
    collect_joint_names_from_markers,
    is_asset_complete,
    mark_asset_complete,
    merge_joint_names as _merge_joint_names,
)


# ─────────────────────────────────────────────────────────────────────────────
# GLTF I/O
# ─────────────────────────────────────────────────────────────────────────────

def import_gltf(path: str):
    bpy.ops.import_scene.gltf(filepath=path)


def import_fbx(path: str):
    bpy.ops.import_scene.fbx(filepath=path, use_anim=True)


def action_frame_range(action):
    """Return (start, end) integer frame range for an action."""
    fr = action.frame_range
    start = int(floor(fr[0]))
    end = int(ceil(fr[1]))
    return start, max(start, end)


def find_main_fbx(fbx_dir):
    """Find the single FBX file that contains all actions for the object type.

    Priority:
      1. FBX files without '-': if exactly one, use it; if multiple, pick the one
         with 'all' (case-insensitive) in its name.
      2. If no files without '-', pick the one with '-all' in its name (e.g., Foo-all.fbx).
    """
    fbx_files = [f for f in os.listdir(fbx_dir) if f.lower().endswith('.fbx')]

    # Try files without '-' first
    no_dash = [f for f in fbx_files if '-' not in os.path.splitext(f)[0]]
    if len(no_dash) == 1:
        return os.path.join(fbx_dir, no_dash[0])
    if len(no_dash) > 1:
        with_all = [f for f in no_dash if 'all' in os.path.splitext(f)[0].lower()]
        if len(with_all) >= 1:
            return os.path.join(fbx_dir, with_all[0])

    # Fallback: file with '-all' in name (e.g., Foo-all.fbx)
    dash_all = [f for f in fbx_files if '-all' in os.path.splitext(f)[0].lower()]
    if len(dash_all) >= 1:
        return os.path.join(fbx_dir, dash_all[0])

    return None


EXCLUDED_CSV = "excluded.csv"


def load_excluded_stems(input_dir: Path) -> set:
    """Asset stems listed in ``excluded.csv`` beside (or inside) *input_dir*.

    Assets no importer can read — corrupt rigs, singular transforms, NaN
    axis limits — are recorded there once so every later stage skips them
    instead of failing on them again on each rerun. The CSV carries a
    header with at least ``file`` or ``object_id``; ``stage`` and ``reason``
    are informational.
    """
    stems = set()
    for candidate in (input_dir / EXCLUDED_CSV, input_dir.parent / EXCLUDED_CSV):
        if not candidate.is_file():
            continue
        with open(candidate, newline="") as f:
            for row in csv.DictReader(f):
                stem = (row.get("object_id") or "").strip()
                if not stem:
                    stem = Path((row.get("file") or "").strip()).stem
                if stem:
                    stems.add(stem)
    return stems


def list_gltf_files(input_dir: Path):
    exts = {".glb", ".gltf"}
    paths = sorted(p for p in input_dir.glob("*") if p.is_file() and p.suffix.lower() in exts)
    excluded = load_excluded_stems(input_dir)
    if not excluded:
        return paths
    kept = [p for p in paths if p.stem not in excluded]
    if len(kept) != len(paths):
        logger.info(f"Skipping {len(paths) - len(kept)} asset(s) listed in {EXCLUDED_CSV}")
    return kept


def action_is_relevant_pose(action) -> bool:
    """Return True if the action has fcurves that animate pose bones."""
    if action is None or len(action.fcurves) == 0:
        return False
    return any(fc.data_path.startswith("pose.bones[") for fc in action.fcurves)


def discover_pose_actions(min_frames: int = 0, max_frames: int = float('inf')):
    """Find all pose-type actions in the current Blender scene.

    By default no frame-count filter is applied; callers may pass
    *min_frames* / *max_frames* to reject clips outside a desired range.

    Returns:
        (action_names, frame_ranges) where frame_ranges maps action name to
        (start_frame, end_frame) tuples.
    """
    action_names = []
    frame_ranges = {}

    for action_name, action in bpy.data.actions.items():
        if not action_is_relevant_pose(action):
            logger.info(f"Skipping irrelevant action: {action_name}")
            continue

        start, end = action_frame_range(action)
        n_frames = end - start + 1

        if n_frames < min_frames or n_frames > max_frames:
            logger.info(f"Skipping action '{action_name}': {n_frames} frames "
                        f"(valid range: [{min_frames}, {max_frames}])")
            continue

        logger.info(f"Found action: {action_name}, frames=[{start},{end}]")

        if action_name not in frame_ranges:
            action_names.append(action_name)
            frame_ranges[action_name] = (start, end)
        else:
            prev_start, prev_end = frame_ranges[action_name]
            frame_ranges[action_name] = (min(start, prev_start), max(end, prev_end))

    return action_names, frame_ranges


# ─────────────────────────────────────────────────────────────────────────────
# Naming helpers
# ─────────────────────────────────────────────────────────────────────────────

def sanitize_object_type(obj_type):
    """Normalize a Truebones object-type directory name.

    Replaces '-' and whitespace with '_' so compound type names (e.g.
    'Dog-2') don't collide with the object-action separator dash used in
    clip names (e.g. 'Dog_2-Walk').
    """
    return obj_type.replace('-', '_').replace(' ', '_')


def sanitize_action_name(raw_name):
    """Extract a clean action name from Blender action names.

    Strips Blender prefixes (``Armature|``), colon suffixes (``:Base``),
    known domain prefixes (``mixamo.com_``), then replaces all
    non-alphanumeric characters with ``_``.

    Examples:
        'Armature|Walk'                        -> 'Walk'
        'Armature|Run|Run:Base'                -> 'Run'
        'Idle'                                 -> 'Idle'
        'Armature|Big-Mouth Attack'            -> 'Big_Mouth_Attack'
        'mixamo.com_Walk'                      -> 'Walk'
        'Armature|mixamo.com_Run:Base'         -> 'Run'
    """
    parts = raw_name.split('|')
    name = parts[1] if len(parts) > 1 else parts[0]
    name = name.split(':')[0]
    # Remove known domain prefixes
    name = re.sub(r'^mixamo\.com_', '', name)
    # Remove GLTF importer suffixes (e.g. '_GLTF_created_0')
    name = re.sub(r'_GLTF_created_\d+$', '', name)
    # Replace whitespace with '_'
    name = re.sub(r'\s+', '_', name)
    # Replace all non-alphanumeric characters with '_'
    name = re.sub(r'[^a-zA-Z0-9]+', '_', name)
    # Strip leading/trailing underscores
    name = name.strip('_')
    return name


def truebones_clip_name(fbx_stem):
    """Clip name for one curated Truebones per-clip FBX stem.

    The curated layout is ``{Species}-{Action}.fbx`` with the first dash
    separating the two. Export and rendering must derive the same name from
    it or stage 4's exact-match caption lookup silently yields an empty
    caption, so both go through this one helper.

    Raises:
        ValueError: when the stem has no ``-`` to split on.
    """
    species, sep, action = fbx_stem.partition('-')
    if not sep:
        raise ValueError(
            f"Truebones clip filename '{fbx_stem}' has no '-' separating "
            f"species from action; expected '{{Species}}-{{Action}}.fbx'")
    return f"{sanitize_object_type(species)}-{sanitize_action_name(action)}"


def unique_clip_names(action_names):
    """Map each action name to a collision-free sanitized clip suffix.

    ``sanitize_action_name`` keeps only the second ``|``-separated segment,
    so distinct actions routinely collapse onto one name — e.g.
    ``Armature|Take 001|BaseLayer_Obj`` and
    ``Armature|Take 001|BaseLayer.002_Obj`` both yield ``Take_001``. Export
    then overwrites ``motions/{clip}.npz`` (last wins) while the renderer's
    completeness check skips the later actions (first wins), so the NPZ and
    the render under one clip name end up being different animations and the
    caption describes the wrong motion.

    Colliding names get a ``_00``/``_01``… suffix in list order. Callers on
    both sides must pass the SAME ordered action list (the full set of pose
    actions, before any frame-count filtering) or the suffixes diverge.

    Args:
        action_names: Ordered action names, as returned by
            :func:`discover_pose_actions`.

    Returns:
        ``{action_name: clip_suffix}``.
    """
    bases = [sanitize_action_name(n) or "action" for n in action_names]
    total = {}
    for base in bases:
        total[base] = total.get(base, 0) + 1

    seen = {}
    mapping = {}
    for action_name, base in zip(action_names, bases):
        if total[base] == 1:
            mapping[action_name] = base
        else:
            idx = seen.get(base, 0)
            seen[base] = idx + 1
            mapping[action_name] = f"{base}_{idx:02d}"
    return mapping


# ─────────────────────────────────────────────────────────────────────────────
# Constants (skinning threshold and Blender rigging naming conventions)
# ─────────────────────────────────────────────────────────────────────────────

_SKINNING_WEIGHT_EPS = 1e-5

_CONTROL_PREFIXES = ('MCH-', 'ORG-', 'VIS-')
_CONTROL_SUFFIXES_LOWER = (
    '_ik', '.ik', '_pole', '.pole',
    '_target', '_tip',
    '_ctrl', '.ctrl',
    '_tweak', '_helper',
)


# ─────────────────────────────────────────────────────────────────────────────
# Blender scene utilities
# ─────────────────────────────────────────────────────────────────────────────

def reset_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)


def set_scene_fps(fps: int):
    bpy.context.scene.render.fps = int(fps)


def find_armatures():
    return [o for o in bpy.data.objects if o.type == "ARMATURE"]


def find_meshes():
    return [o for o in bpy.data.objects if o.type == "MESH"]


def choose_main_armature(armatures):
    """Heuristic: armature with most bones."""
    if not armatures:
        return None
    return sorted(armatures, key=lambda a: len(a.data.bones), reverse=True)[0]


def choose_main_mesh(meshes):
    """Heuristic: mesh with most vertices."""
    if not meshes:
        return None
    return sorted(meshes, key=lambda m: len(m.data.vertices), reverse=True)[0]


# ─────────────────────────────────────────────────────────────────────────────
# Animation state management
# ─────────────────────────────────────────────────────────────────────────────

def clear_animation_state(arm_obj: bpy.types.Object):
    """Clear action and NLA tracks to avoid cross-action contamination."""
    ad = arm_obj.animation_data
    if ad is None:
        return
    ad.action = None
    if ad.nla_tracks:
        for tr in list(ad.nla_tracks):
            ad.nla_tracks.remove(tr)


def bind_action(arm_obj: bpy.types.Object, action: bpy.types.Action):
    if arm_obj.animation_data is None:
        arm_obj.animation_data_create()
    arm_obj.animation_data.action = action


# ─────────────────────────────────────────────────────────────────────────────
# Skinning utilities
# ─────────────────────────────────────────────────────────────────────────────

def build_vgroup_to_bone_mapping(mesh_obj, arm_obj):
    """Return dict {vgroup_index -> bone_name} for groups that match a bone."""
    bone_names = set(arm_obj.data.bones.keys())
    vg_to_bone = {}
    for i, vg in enumerate(mesh_obj.vertex_groups):
        if vg.name in bone_names:
            vg_to_bone[i] = vg.name
    return vg_to_bone


def build_vertex_groups_sparse(mesh_obj, vg_to_bone: dict, bone_name_to_index: dict, dtype=np.float64):
    """
    Returns:
      vert_groups:  len=V list[list[(bone_idx, weight)]]
      skin_matrix:  (V, B) dense skinning weight matrix
    """
    verts       = mesh_obj.data.vertices
    V           = len(verts)
    vert_groups = [[] for _ in range(V)]
    skin_matrix = np.zeros((V, len(bone_name_to_index)), dtype=dtype)

    for i, vert in enumerate(verts):
        groups = []
        for g in vert.groups:
            w = float(g.weight)
            if w <= 0.0:
                continue
            bname = vg_to_bone.get(g.group)
            if bname is None:
                continue
            bidx = bone_name_to_index.get(bname)
            if bidx is None:
                continue
            skin_matrix[i, bidx] = w
            groups.append((int(bidx), w))
        vert_groups[i] = groups

    return vert_groups, skin_matrix


# ─────────────────────────────────────────────────────────────────────────────
# Skeleton topology utilities
# ─────────────────────────────────────────────────────────────────────────────

def children_from_parents(parents):
    ch = [[] for _ in range(len(parents))]
    for j, p in enumerate(parents):
        if p >= 0:
            ch[p].append(j)
    return ch


def name_is_control_bone(name: str) -> bool:
    return (
        any(name.startswith(p) for p in _CONTROL_PREFIXES)
        or any(name.lower().endswith(s) for s in _CONTROL_SUFFIXES_LOWER)
    )


def rotation_amplitude(j, anim):
    """
    Maximum rotation angle (radians) of joint j relative to its first-frame
    orientation, across all frames.
    """
    rot_0    = anim.rotations[0:1, j]
    delta    = (-rot_0) * anim.rotations[:, j]
    cos_half = np.abs(delta.qs[:, 0]).clip(0.0, 1.0)
    return float(np.max(2.0 * np.arccos(cos_half)))


def prune_secondary_roots(armature, name_to_weight_sum):
    def get_subtree_weight(bone, weight_dict):
        w = weight_dict.get(bone.name, 0)
        for child in bone.children:
            w += get_subtree_weight(child, weight_dict)
        return w

    all_roots = [bone for bone in armature.data.bones if bone.parent is None]
    if len(all_roots) <= 1:
        logger.info("Only one or no root bone found. Skipping prune_secondary_roots.")
        return

    root_weights = []
    for rb in all_roots:
        total_w = get_subtree_weight(rb, name_to_weight_sum)
        root_weights.append((rb.name, total_w))
        logger.info(f"Root bone '{rb.name}' has total subtree weight: {total_w:.4f}")

    best_root_name = max(root_weights, key=lambda x: x[1])[0]
    logger.info(f"Keeping best root: '{best_root_name}'")

    bones_to_remove = []
    for rb_name, _ in root_weights:
        if rb_name != best_root_name:
            rb_obj = armature.data.bones.get(rb_name)
            if rb_obj:
                bones_to_remove.extend([rb_obj.name] + [b.name for b in rb_obj.children_recursive])

    if bones_to_remove:
        logger.info(f"Removing {len(bones_to_remove)} bones from secondary root subtrees.")
        bpy.context.view_layer.objects.active = armature
        bpy.ops.object.mode_set(mode='EDIT')
        eb = armature.data.edit_bones
        for b_name in bones_to_remove:
            if b_name in eb:
                eb.remove(eb[b_name])
        bpy.ops.object.mode_set(mode='OBJECT')
        bpy.context.view_layer.update()


def collapse_unused_root(anim: Animation):
    """Remove an unskinned root with a single child; the child becomes the new root.

    Uses direct local-space composition instead of global FK round-trip
    to avoid floating-point drift.
    """
    parents = anim.parents.copy()
    F, J    = anim.positions.shape[:2]
    assert J == len(parents)

    root_ids = np.where(np.array(parents) == -1)[0]
    assert len(root_ids) == 1, "Multiple root joints found."
    root_id = root_ids[0]

    all_children = children_from_parents(parents)
    assert len(all_children[root_id]) == 1, "Root must have exactly one child."
    true_root_id = all_children[root_id][0]

    def get_all_descendants(idx):
        res = [idx]
        for c in all_children[idx]:
            res.extend(get_all_descendants(c))
        return res

    keep      = sorted(get_all_descendants(true_root_id))
    new_index = {old: new for new, old in enumerate(keep)}
    newJ      = len(keep)

    # Build new parent array
    new_parents = np.full((newJ,), -1, dtype=np.int32)
    for old_j in keep:
        if old_j != true_root_id:
            new_parents[new_index[old_j]] = new_index[parents[old_j]]

    # Copy local transforms for all kept bones
    new_rot_qs = anim.rotations.qs[:, keep].copy()
    new_pos    = anim.positions[:, keep].copy()
    new_offsets = anim.offsets[keep].copy()

    # Compose root transform into the new root (former child):
    # new_child_local = old_root_local * old_child_local
    new_root_idx = new_index[true_root_id]
    root_rot = anim.rotations[:, root_id]
    new_rot_qs[:, new_root_idx] = (root_rot * anim.rotations[:, true_root_id]).qs
    new_pos[:, new_root_idx] = (
        anim.positions[:, root_id] + root_rot * anim.positions[:, true_root_id]
    )

    # Compose rest-pose offset
    rest_root_rot = Quaternions(anim.rotations.qs[0:1, root_id])
    new_offsets[new_root_idx] = (
        anim.offsets[root_id] + (rest_root_rot * anim.offsets[true_root_id:true_root_id+1])[0]
    )

    new_anim = Animation(
        rotations=Quaternions(new_rot_qs),
        positions=new_pos,
        orients=anim.orients[keep].copy(),
        offsets=new_offsets,
        parents=new_parents,
    )
    return new_anim, keep


# ─────────────────────────────────────────────────────────────────────────────
# Skeleton pruning
# ─────────────────────────────────────────────────────────────────────────────

# --- Prunable bone detection ------------------------------------------------

def _is_prunable(j, skin_matrix, anims_list, rest_anim, names,
                 offset_eps, pos_var_eps, rot_eps,
                 mean_bone_len=1.0, is_leaf=True,
                 consider_parent_rotate=True):
    """
    Return True if bone j should be removed from the skeleton.
    Works with one or more animation clips for aggregated decisions.

    Conditions (checked in order, first match wins):

      (0) Name convention   — Blender control/mechanism prefix or suffix.
      (1) Near-zero offset  — bone head coincident with parent in rest pose.
      (2) Near-zero position— bone stays at parent origin in ALL clips.
      (3) Non-rigid (leaf)  — bone length varies from rest in ANY clip;
                              likely an IK-stretch control.
      (4) Zero rotation DOF — bone never rotates above threshold in ALL clips.
          For leaf bones, when *consider_parent_rotate* is True, only prune
          if the parent also doesn't rotate in ANY clip — a static leaf on a
          rotating parent is a meaningful end-effector (e.g. spider toe tip).
          When False, prune such leaves regardless of parent motion.

    Distance conditions are normalized by mean_bone_len (scale-agnostic).
    """
    # Skinned bones are never prunable
    if skin_matrix[:, j].max() >= _SKINNING_WEIGHT_EPS:
        return False

    if name_is_control_bone(names[j]):                                    # (0)
        return True

    norm = max(mean_bone_len, 1e-9)
    rest_pos_j = rest_anim.positions[0, j]

    if np.linalg.norm(rest_pos_j) / norm < offset_eps:                   # (1)
        return True

    if all(np.max(np.linalg.norm(a.positions[:, j], axis=-1)) / norm     # (2)
           < offset_eps for a in anims_list):
        return True

    if is_leaf:                                                           # (3)
        if any(np.max(np.linalg.norm(a.positions[:, j] - rest_pos_j,
                                     axis=-1)) / norm > pos_var_eps
               for a in anims_list):
            return True

    if rot_eps > 0.0 and all(rotation_amplitude(j, a) < rot_eps          # (4)
                             for a in anims_list):
        if is_leaf and consider_parent_rotate:
            parent_j = anims_list[0].parents[j]
            if parent_j >= 0 and any(rotation_amplitude(parent_j, a) >= rot_eps
                                     for a in anims_list):
                return False  # parent rotates → keep this end-effector
        return True

    return False


# --- Bone merge operation ---------------------------------------------------

def _merge_passthrough_bone(anims_list, rest_anim, names, skin_matrix, j):
    """Collapse pass-through bone j into its single child across all clips.

    The child inherits j's transform so that global positions are preserved.
    """
    parents   = list(anims_list[0].parents)
    childrens = children_from_parents(parents)
    c         = childrens[j][0]

    # Compose rest-pose offset: child_new = j_offset + rot_j * child_offset
    new_offsets = anims_list[0].offsets.copy()
    rest_rot_j0 = Quaternions(rest_anim.rotations.qs[0:1, j])
    new_offsets[c] = new_offsets[j] + (rest_rot_j0 * new_offsets[c:c+1])[0]

    # Re-parent child to j's parent, build index mapping
    new_parents    = list(parents)
    new_parents[c] = parents[j]
    keep_idxs      = [i for i in range(len(parents)) if i != j]
    old_to_new     = {old: new for new, old in enumerate(keep_idxs)}
    final_parents  = np.array(
        [old_to_new[new_parents[i]] if new_parents[i] >= 0 else -1
         for i in keep_idxs],
        dtype=np.int32,
    )

    # Update rest anim
    rest_rot_j    = rest_anim.rotations[:, j]
    new_rest_pos  = rest_anim.positions.copy()
    new_rest_rots = rest_anim.rotations.qs.copy()
    new_rest_pos[:, c]  = rest_anim.positions[:, j] + rest_rot_j * rest_anim.positions[:, c]
    new_rest_rots[:, c] = (rest_rot_j * rest_anim.rotations[:, c]).qs
    new_rest_anim = Animation(
        rotations=Quaternions(new_rest_rots[:, keep_idxs]),
        positions=new_rest_pos[:, keep_idxs],
        orients=rest_anim.orients[keep_idxs],
        offsets=new_offsets[keep_idxs],
        parents=final_parents,
    )

    # Update each animation clip
    new_anims = []
    for anim in anims_list:
        rot_j         = anim.rotations[:, j]
        new_anim_pos  = anim.positions.copy()
        new_anim_rots = anim.rotations.qs.copy()
        new_anim_pos[:, c]  = anim.positions[:, j] + rot_j * anim.positions[:, c]
        new_anim_rots[:, c] = (rot_j * anim.rotations[:, c]).qs
        new_anims.append(Animation(
            rotations=Quaternions(new_anim_rots[:, keep_idxs]),
            positions=new_anim_pos[:, keep_idxs],
            orients=anim.orients[keep_idxs],
            offsets=new_offsets[keep_idxs],
            parents=final_parents,
        ))

    new_names = [names[i] for i in keep_idxs]
    new_skin  = skin_matrix[:, keep_idxs]
    return new_anims, new_rest_anim, new_names, new_skin


# --- Helpers ----------------------------------------------------------------

def _compute_mean_bone_len(rest_anim):
    """Mean bone length from rest-pose offsets, excluding root."""
    bone_lens  = np.linalg.norm(rest_anim.offsets[1:], axis=-1)
    valid_lens = bone_lens[bone_lens > 1e-6]
    return float(valid_lens.mean()) if len(valid_lens) > 0 else 1.0


def _find_deepest_in_chain(j, children_map, cand_set):
    """Walk from candidate j down the single-child chain to the deepest candidate."""
    while children_map[j][0] in cand_set:
        j = children_map[j][0]
    return j


def _find_root(parents):
    """Return the index of the root joint (parent == -1)."""
    return int(np.where(np.array(parents) == -1)[0][0])


# --- Pruning passes --------------------------------------------------------

def _pass_prune_leaves(anims_list, rest_anim, names, skin_matrix,
                       offset_eps, pos_var_eps, rot_eps, min_joints,
                       consider_parent_rotate=True):
    """Pass 1: iteratively remove prunable leaf bones."""
    count = 0
    mbl = _compute_mean_bone_len(rest_anim)
    while True:
        n_joints  = len(anims_list[0].parents)
        childrens = children_from_parents(anims_list[0].parents)
        candidates = [j for j in range(n_joints)
                      if len(childrens[j]) == 0
                      and _is_prunable(j, skin_matrix, anims_list, rest_anim, names,
                                       offset_eps, pos_var_eps, rot_eps,
                                       mean_bone_len=mbl, is_leaf=True,
                                       consider_parent_rotate=consider_parent_rotate)]
        if not candidates:
            break
        keep = [j for j in range(n_joints) if j not in set(candidates)]
        if len(keep) < min_joints:
            break
        logger.info(f"Pruning leaf bones: {[names[j] for j in candidates]}")
        anims_list  = [a[:, keep].copy() for a in anims_list]
        rest_anim   = rest_anim[:, keep].copy()
        names       = [names[j] for j in keep]
        skin_matrix = skin_matrix[:, keep]
        count += len(candidates)
    return anims_list, rest_anim, names, skin_matrix, count


def _pass_promote_root(anims_list, rest_anim, names, skin_matrix):
    """Pass 2: collapse unskinned single-child root into its child."""
    count = 0
    while True:
        root_idx = _find_root(anims_list[0].parents)
        children = children_from_parents(anims_list[0].parents)[root_idx]
        if not (len(children) == 1
                and skin_matrix[:, root_idx].max() < _SKINNING_WEIGHT_EPS):
            break
        logger.info(f"Promoting root child (removing '{names[root_idx]}')")
        new_anims = []
        keep_idxs = None
        for anim in anims_list:
            new_anim, ki = collapse_unused_root(anim)
            if keep_idxs is None:
                keep_idxs = ki
            else:
                assert ki == keep_idxs, "Root-clean keep_idxs mismatch across clips"
            new_anims.append(new_anim)
        rest_anim, rest_ki = collapse_unused_root(rest_anim)
        assert rest_ki == keep_idxs, "Root-clean keep_idxs mismatch for rest_anim"
        anims_list  = new_anims
        names       = [names[j] for j in keep_idxs]
        skin_matrix = skin_matrix[:, keep_idxs]
        count += 1
    return anims_list, rest_anim, names, skin_matrix, count


def _pass_merge_passthrough(anims_list, rest_anim, names, skin_matrix,
                            offset_eps, pos_var_eps, rot_eps, min_joints):
    """Pass 3: merge prunable single-child (non-root) bones, deepest first."""
    count = 0
    mbl = _compute_mean_bone_len(rest_anim)
    while True:
        n_joints  = len(anims_list[0].parents)
        childrens = children_from_parents(anims_list[0].parents)
        root_idx  = _find_root(anims_list[0].parents)
        candidates = [j for j in range(n_joints)
                      if j != root_idx
                      and len(childrens[j]) == 1
                      and _is_prunable(j, skin_matrix, anims_list, rest_anim, names,
                                       offset_eps, pos_var_eps, rot_eps,
                                       mean_bone_len=mbl, is_leaf=False)]
        if not candidates:
            break
        cand_set = set(candidates)
        j = _find_deepest_in_chain(candidates[0], childrens, cand_set)
        logger.info(f"Merging pass-through bone '{names[j]}' into its child")
        anims_list, rest_anim, names, skin_matrix = \
            _merge_passthrough_bone(anims_list, rest_anim, names, skin_matrix, j)
        count += 1
    return anims_list, rest_anim, names, skin_matrix, count


# --- Main entry points ------------------------------------------------------

def prune_skeleton_shared(anims_list, rest_anim, names, skin_matrix,
                          offset_eps=1e-3, pos_var_eps=1e-3,
                          rot_eps=np.deg2rad(2.0), min_joints=4,
                          consider_parent_rotate=True):
    """Iterative skeleton pruning across multiple animation clips.

    Guarantees identical topology across all clips.  Three passes per
    iteration, repeated until no pass makes progress:

      1. Remove prunable leaf bones
      2. Promote child when root is unskinned with single child
      3. Merge prunable pass-through bones into their children

    Args:
        consider_parent_rotate: If True (default), leaf bones that never
            rotate are preserved when their parent rotates — treating them
            as meaningful end-effectors (e.g. spider toe tips). If False,
            such leaves are pruned regardless of parent motion.
    """
    stats = {"leaf": 0, "root_promoted": 0, "passthrough": 0}

    while len(anims_list[0].parents) > min_joints:
        anims_list, rest_anim, names, skin_matrix, n_leaf = \
            _pass_prune_leaves(anims_list, rest_anim, names, skin_matrix,
                               offset_eps, pos_var_eps, rot_eps, min_joints,
                               consider_parent_rotate=consider_parent_rotate)

        anims_list, rest_anim, names, skin_matrix, n_root = \
            _pass_promote_root(anims_list, rest_anim, names, skin_matrix)

        anims_list, rest_anim, names, skin_matrix, n_pass = \
            _pass_merge_passthrough(anims_list, rest_anim, names, skin_matrix,
                                    offset_eps, pos_var_eps, rot_eps, min_joints)

        stats["leaf"]          += n_leaf
        stats["root_promoted"] += n_root
        stats["passthrough"]   += n_pass

        if n_leaf + n_root + n_pass == 0:
            break

    logger.info(f"Pruning summary: {stats}, final joints: {len(anims_list[0].parents)}")
    return anims_list, rest_anim, names, skin_matrix


# ─────────────────────────────────────────────────────────────────────────────
# Motion filtering (T-pose frame removal)
# ─────────────────────────────────────────────────────────────────────────────

def remove_tpose_frames(anim, rest_anim, rot_thresh=np.deg2rad(1.0)):
    """Remove frames whose non-root joint rotations match the rest pose.

    For each frame, computes the maximum rotation angle (over all non-root
    joints) between the frame's local rotation and the rest-pose rotation.
    Frames where this maximum is below *rot_thresh* are considered T-pose
    duplicates and removed.

    Args:
        anim:        Animation with shape (F, J, ...).
        rest_anim:   Single-frame rest-pose Animation (1, J, ...).
        rot_thresh:  Maximum allowed rotation deviation (radians) for a
                     frame to be considered identical to the T-pose.

    Returns:
        (pruned_anim, n_removed)  — the filtered Animation and the count
        of removed frames.  Returns the original if no frames are removed
        or if removing them would leave fewer than 2 frames.
    """
    F, J = anim.rotations.qs.shape[:2]
    if J < 2:
        return anim, 0

    # rest rotation for non-root joints: (1, J-1, 4)
    rest_rot = rest_anim.rotations[:, 1:]            # Quaternions (1, J-1)
    frame_rot = anim.rotations[:, 1:]                # Quaternions (F, J-1)

    # relative rotation: rest^-1 * frame  →  identity when poses match
    delta = (-rest_rot) * frame_rot                  # (F, J-1)
    # angle = 2 * arccos(|w|)
    cos_half = np.abs(delta.qs[..., 0]).clip(0.0, 1.0)   # (F, J-1)
    angles = 2.0 * np.arccos(cos_half)                    # (F, J-1)

    max_angle_per_frame = angles.max(axis=1)               # (F,)
    keep_mask = max_angle_per_frame >= rot_thresh

    n_removed = int((~keep_mask).sum())
    if n_removed == 0:
        return anim, 0

    keep_indices = np.where(keep_mask)[0]
    if len(keep_indices) < 2:
        # Don't strip if it would leave fewer than 2 frames
        return anim, 0

    pruned = Animation(
        rotations=Quaternions(anim.rotations.qs[keep_indices]),
        positions=anim.positions[keep_indices],
        orients=anim.orients.copy(),
        offsets=anim.offsets.copy(),
        parents=anim.parents.copy(),
    )
    return pruned, n_removed


# ─────────────────────────────────────────────────────────────────────────────
# Animation extraction helpers
# ─────────────────────────────────────────────────────────────────────────────

def extract_rest_pose(anim_data_bones, armature, dtype=np.float64, apply_world=True):
    """Extract local rest-pose positions and rotations for all bones."""
    nbones         = len(anim_data_bones)
    rest_local_pos = np.zeros((nbones, 3), dtype=dtype)
    rest_local_rot = np.zeros((nbones, 4), dtype=dtype)

    for idxb, bone in enumerate(anim_data_bones):
        if bone.parent:
            mat = bone.parent.matrix_local.inverted() @ bone.matrix_local
            rest_local_pos[idxb] = np.array(mat.to_translation(), dtype=dtype)
            rest_local_rot[idxb] = np.array(mat.to_quaternion(),  dtype=dtype)
        else:
            if apply_world:
                mat = armature.matrix_world @ bone.matrix_local
                scale = np.array(armature.matrix_world.to_scale(), dtype=dtype)
                rest_local_pos[idxb] = np.array(mat.to_translation(), dtype=dtype) / scale
            else:
                mat = bone.matrix_local
                rest_local_pos[idxb] = np.array(mat.to_translation(), dtype=dtype)
            rest_local_rot[idxb] = np.array(mat.to_quaternion(),  dtype=dtype)

    return rest_local_pos, rest_local_rot


def extract_animation_frames(armature, start_frame, end_frame, nbones, dtype=np.float64, apply_world=True):
    """Scrub through frames and extract per-frame local bone transforms."""
    nframes        = end_frame - start_frame + 1
    anim_local_pos = np.zeros((nframes, nbones, 3), dtype=dtype)
    anim_local_rot = np.zeros((nframes, nbones, 4), dtype=dtype)

    pose_bones = armature.pose.bones
    assert len(pose_bones) > 0, f"No pose bones in armature {armature.name}"

    for idxf, f in enumerate(range(start_frame, end_frame + 1)):
        bpy.context.scene.frame_set(f)  # evaluates the depsgraph

        for idxb, bone in enumerate(pose_bones):
            if bone.parent:
                mat = bone.parent.matrix.inverted() @ bone.matrix
                anim_local_pos[idxf, idxb] = np.array(mat.to_translation(), dtype=dtype)
                anim_local_rot[idxf, idxb] = np.array(mat.to_quaternion(),  dtype=dtype)
            else:
                if apply_world:
                    mat = armature.matrix_world @ bone.matrix
                    scale = np.array(armature.matrix_world.to_scale(), dtype=dtype)
                    anim_local_pos[idxf, idxb] = np.array(mat.to_translation(), dtype=dtype) / scale
                else:
                    mat = bone.matrix
                    anim_local_pos[idxf, idxb] = np.array(mat.to_translation(), dtype=dtype)
                anim_local_rot[idxf, idxb] = np.array(mat.to_quaternion(),  dtype=dtype)

    return anim_local_pos, anim_local_rot


# ─────────────────────────────────────────────────────────────────────────────
# Skeleton setup (Blender scene → numpy arrays)
# ─────────────────────────────────────────────────────────────────────────────

def compute_bone_weight_sums(mesh, armature):
    """Sum skin weights per bone across all vertices."""
    vg_to_bone = build_vgroup_to_bone_mapping(mesh, armature)
    weight_sums = {bname: 0.0 for bname in armature.data.bones.keys()}
    for v in mesh.data.vertices:
        for g in v.groups:
            bname = vg_to_bone.get(g.group)
            if bname:
                weight_sums[bname] += g.weight
    return weight_sums


def build_skeleton_arrays(armature, mesh, dtype=np.float64, apply_world=True):
    """Extract skeleton topology, rest pose, and skin matrix from Blender objects.

    Returns:
        bone_names, parents_array, rest_local_pos, rest_local_rot, skin_matrix
    """
    bones = armature.data.bones
    bone_names = list(bones.keys())
    bone_name_to_index = {b.name: i for i, b in enumerate(bones)}

    if mesh is None:
        skin_matrix = np.zeros((0, len(bone_names)), dtype=dtype)
    else:
        vg_to_bone = build_vgroup_to_bone_mapping(mesh, armature)
        _, skin_matrix = build_vertex_groups_sparse(mesh, vg_to_bone, bone_name_to_index)

    parents_array = np.array([
        -1 if bone.parent is None else bones.find(bone.parent.name)
        for bone in bones
    ], dtype=np.int32)

    rest_local_pos, rest_local_rot = extract_rest_pose(bones, armature, dtype, apply_world=apply_world)

    return bone_names, parents_array, rest_local_pos, rest_local_rot, skin_matrix


def build_rest_animation(parents_array, rest_local_pos, rest_local_rot, nbones):
    """Build a single-frame Animation object from rest pose arrays."""
    return Animation(
        rotations=Quaternions(rest_local_rot[None]),
        positions=rest_local_pos[None],
        orients=Quaternions.id(nbones),
        offsets=rest_local_pos,
        parents=parents_array,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Coordinate conversion (Z-up → Y-up)
# ─────────────────────────────────────────────────────────────────────────────

def _apply_root_rotation(anim, root_quat):
    """Apply *root_quat* to the root joint of an Animation (rotations, positions, offsets)."""
    rots = anim.rotations.copy()
    pos = anim.positions.copy()
    off = anim.offsets.copy()
    rots[:, 0] = root_quat * anim.rotations[:, 0]
    pos[:, 0] = root_quat * anim.positions[:, 0]
    off[0] = (root_quat * anim.offsets[0:1])[0]
    return Animation(
        rotations=rots, positions=pos,
        orients=anim.orients, offsets=off, parents=anim.parents,
    )


def apply_zup_to_yup(anim, rest_anim):
    """Rotate root bone -90 deg around X to convert Z-up to Y-up coordinate system."""
    root_quat = Quaternions.from_euler(np.array([[-np.pi / 2, 0, 0]]))
    return _apply_root_rotation(anim, root_quat), _apply_root_rotation(rest_anim, root_quat)


# ─────────────────────────────────────────────────────────────────────────────
# High-level pipeline helpers (shared by FBX and GLB exporters)
# ─────────────────────────────────────────────────────────────────────────────

def load_scene(import_fn, path, fps=30):
    """Reset scene, set FPS, import via *import_fn*, select main armature/mesh.

    Returns (armature, mesh). Mesh may be None if none found.
    """
    reset_scene()
    set_scene_fps(fps=fps)
    import_fn(str(path))

    armatures = find_armatures()
    assert len(armatures) > 0, f"No armature found in {path}."
    armature = choose_main_armature(armatures)
    logger.info(f"Using armature: {armature.name}, bones={len(armature.data.bones)}")

    meshes = find_meshes()
    mesh = choose_main_mesh(meshes) if meshes else None
    if mesh is not None:
        logger.info(f"Using mesh: {mesh.name}, vertices={len(mesh.data.vertices)}")

    return armature, mesh


def prepare_skeleton(armature, mesh=None, dtype=np.float64, apply_world=True):
    """Prune secondary roots, build skeleton arrays, and Z→Y rest poses.

    When *mesh* is None (e.g. Mixamo clips without a character mesh), the
    secondary-root prune step is skipped and skin_matrix is returned empty.

    Returns a dict with keys: nbones, bone_names, parents_array,
    rest_local_pos, rest_local_rot, skin_matrix, rest_anim, rest_anim_shared.
    """
    if mesh is not None:
        weight_sums = compute_bone_weight_sums(mesh, armature)
        prune_secondary_roots(armature, weight_sums)
        bpy.context.view_layer.update()

    nbones = len(armature.data.bones)
    bone_names, parents_array, rest_local_pos, rest_local_rot, skin_matrix = \
        build_skeleton_arrays(armature, mesh, dtype, apply_world=apply_world)

    rest_anim = build_rest_animation(parents_array, rest_local_pos, rest_local_rot, nbones)
    _, rest_anim_shared = apply_zup_to_yup(rest_anim, rest_anim)

    return {
        'nbones': nbones,
        'bone_names': bone_names,
        'parents_array': parents_array,
        'rest_local_pos': rest_local_pos,
        'rest_local_rot': rest_local_rot,
        'skin_matrix': skin_matrix,
        'rest_anim': rest_anim,
        'rest_anim_shared': rest_anim_shared,
    }


def extract_all_actions(armature, frame_ranges, rest_local_pos, parents_array,
                        rest_anim, nbones, dtype=np.float64, apply_world=True):
    """Extract (name, Animation) for every action in *frame_ranges*.

    Args:
        frame_ranges: dict mapping action_name -> (start_frame, end_frame).

    Applies Z-up → Y-up conversion to each extracted clip.
    """
    extracted = []
    for action_name, (start_frame, end_frame) in frame_ranges.items():
        action = bpy.data.actions.get(action_name)
        if action is None:
            logger.warning(f"Action '{action_name}' not found; skipping.")
            continue

        clear_animation_state(armature)
        bind_action(armature, action)

        anim_local_pos, anim_local_rot = extract_animation_frames(
            armature, start_frame, end_frame, nbones, dtype, apply_world=apply_world
        )

        anim = Animation(
            rotations=Quaternions(anim_local_rot),
            positions=anim_local_pos,
            orients=Quaternions.id(nbones),
            offsets=rest_local_pos,
            parents=parents_array,
        )
        anim, _ = apply_zup_to_yup(anim, rest_anim)
        extracted.append((action_name, anim))

    return extracted


def save_rest_pose_vis(tpos_save_dir, save_name, rest_anim_pruned):
    """Render and save the rest-pose skeleton PNG for one asset."""
    os.makedirs(tpos_save_dir, exist_ok=True)
    tpos_path = os.path.join(tpos_save_dir, f"{save_name}.png")
    rest_global_pos = positions_global(rest_anim_pruned)
    save_skeleton_tpose(
        save_path=tpos_path,
        parents=rest_anim_pruned.parents,
        positions=rest_global_pos[0],
    )
    logger.info(f"Saved rest pose visualization: {tpos_path}")
    return tpos_path


# ─────────────────────────────────────────────────────────────────────────────
# Shared NPZ save and summary helpers
# ─────────────────────────────────────────────────────────────────────────────

def save_motion(motion_path, vis_path, anim, rest_anim, names, skin_matrix, fps,
                action_name="", save_vis=True):
    """Save motion NPZ (float64) and optional MP4.

    The MP4 preview dominates per-clip export time; pass ``save_vis=False``
    to skip it on bulk runs.
    """
    if save_vis:
        global_pos = positions_global(anim)
        save_skeleton_motion(
            save_path=vis_path,
            parents=anim.parents,
            positions=global_pos,
            fps=fps,
        )
        logger.info(f"Saved visualization: {vis_path}")

    np.savez(
        motion_path,
        rest_local_pos=rest_anim.positions[0],
        rest_local_rot=rest_anim.rotations[0].qs,
        anim_local_pos=anim.positions,
        anim_local_rot=anim.rotations.qs,
        offsets=anim.offsets,
        names=np.array(names),
        skin_matrix=skin_matrix,
        fps=fps,
        parents=anim.parents,
        action_name=action_name,
    )
    logger.info(f"Saved motion data: {motion_path}")


def scan_clip_frames(motion_dir):
    """Scan saved NPZ files in motion_dir and return {clip_name: nframes}."""
    frames = {}
    if not os.path.isdir(motion_dir):
        return frames
    for npz_file in sorted(os.listdir(motion_dir)):
        if not npz_file.endswith('.npz'):
            continue
        clip_name = os.path.splitext(npz_file)[0]
        try:
            data = np.load(os.path.join(motion_dir, npz_file), allow_pickle=True)
            frames[clip_name] = int(data['anim_local_rot'].shape[0])
        except Exception:
            pass
    return frames


def write_clip_frames_json(path, frames, fps=30):
    """Write clip_frames.json: a pure {clip_name: nframes} mapping."""
    total_frames = sum(frames.values())
    total_clips = len(frames)
    total_duration_sec = total_frames / fps if fps else 0.0
    with open(path, 'w') as f:
        json.dump(frames, f, indent=2)
    logger.info(f"Saved clip frames ({total_clips} clips, {total_frames} frames, "
                f"{total_duration_sec / 60:.1f} min) to {path}")


def write_summary_json(path, frames, fps=30):
    """Write summary.json with the dataset-level summary of the export."""
    total_frames = sum(frames.values())
    total_duration_sec = total_frames / fps if fps else 0.0
    with open(path, 'w') as f:
        json.dump({
            "total_clips": len(frames),
            "total_frames": total_frames,
            "total_duration_sec": round(total_duration_sec, 2),
            "total_duration_min": round(total_duration_sec / 60, 2),
            "fps": fps,
        }, f, indent=2)
    logger.info(f"Saved export metadata to {path}")


def write_export_summary(output_dir, all_joint_names, fps=30, worker_suffix=""):
    """Write per-run summary JSONs: clip_frames, joint_names, joint_count,
    and (for non-sharded runs) summary.json with the dataset-level summary.

    ``all_joint_names`` only covers assets exported by *this* run — assets
    skipped via their completion marker return nothing. Writing it verbatim
    would therefore truncate ``joint_names.json`` on every resumed run, so
    for the canonical (non-sharded) file it is merged over the durable
    per-asset markers and whatever the file already held. Shard files stay
    run-local; :func:`~data_process.tools.merge_summaries.merge_summaries`
    does the same union for them.

    Args:
        output_dir:      Root directory containing the "motions/" NPZ folder.
        all_joint_names: dict mapping obj_type/save_name -> list[str] bone names.
        fps:             Frame rate used for duration summary.
        worker_suffix:   Optional suffix (e.g. "_worker0") appended to filenames.
                         When set, summary.json is left to merge_summaries so
                         parallel workers never race on it.
    """
    frames = scan_clip_frames(os.path.join(output_dir, "motions"))
    write_clip_frames_json(
        os.path.join(output_dir, f"clip_frames{worker_suffix}.json"),
        frames, fps=fps,
    )
    if not worker_suffix:
        write_summary_json(os.path.join(output_dir, "summary.json"), frames, fps=fps)

    joint_names_path = os.path.join(output_dir, f"joint_names{worker_suffix}.json")
    if worker_suffix:
        merged = dict(all_joint_names)
    else:
        merged = _merge_joint_names(output_dir, all_joint_names, log=logger.info)
    with open(joint_names_path, 'w') as f:
        json.dump(merged, f, indent=2)
    logger.info(f"Saved joint names ({len(merged)} entries) to {joint_names_path}")
    all_joint_names = merged

    joint_count = {k: len(v) for k, v in all_joint_names.items()}
    joint_count_path = os.path.join(output_dir, f"joint_count{worker_suffix}.json")
    with open(joint_count_path, 'w') as f:
        json.dump(joint_count, f, indent=2)
    logger.info(f"Saved joint count to {joint_count_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Per-asset completion markers (resume bookkeeping)
# ─────────────────────────────────────────────────────────────────────────────
#
# Re-exported from data_process.utils.export_markers, which is deliberately
# free of any bpy import so plain-Python tools (merge_summaries) can use it.

__all__ = [n for n in dir() if not n.startswith('_')] + [
    "is_asset_complete", "mark_asset_complete",
    "collect_joint_names_from_markers", "merge_joint_names",
]
