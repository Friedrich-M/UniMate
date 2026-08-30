"""Blender utilities for driving a rigged character with an NPZ animation.

Organized as five sections:

1. Scene / object helpers — import meshes, select objects, pack textures.
2. Keyframe computation — per-bone loc/rot keyframes from 4x4 local
   transforms.
3. Action reconstruction — turn per-bone keyframe arrays into a Blender
   Action bound to the armature.
4. Armature reconciliation — handle armature bones that are absent from
   the driving animation (keep them, merge their weights, or delete them).
5. Character export — export the animated mesh + armature to GLB/FBX.

The counterpart for the *export* direction (Blender scene → NPZ) lives in
:mod:`data_process.utils.blender_export`.
"""

import os
import re

import bmesh
import bpy
import numpy as np
from loguru import logger
from mathutils import Matrix, Quaternion

from data_process.utils.blender_export import (
    bind_action,
    clear_animation_state,
    reset_scene,
)

RECONSTRUCTED_ACTION_NAME = "Reconstructed_Action"


# ─────────────────────────────────────────────────────────────────────────────
# 1. Scene / object helpers
# ─────────────────────────────────────────────────────────────────────────────

def update_scene():
    """Force a full scene dependency-graph update."""
    bpy.context.view_layer.update()
    bpy.context.scene.update_tag()
    for obj in bpy.context.scene.objects:
        obj.update_tag()


def deselect():
    """Deselect every object in the scene."""
    bpy.ops.object.select_all(action="DESELECT")


def select_objs(obj_list=None, deselect_first=False):
    """Select Blender objects, optionally deselecting everything first."""
    if not obj_list:
        obj_list = bpy.context.scene.objects
    if deselect_first:
        deselect()
    for obj in obj_list:
        obj.select_set(True)


def load_file(filepath, *args, **kwargs):
    """Import a 3D file and return the list of newly-added scene objects."""
    old_objs = set(bpy.context.scene.objects)
    if filepath.endswith(".glb"):
        bpy.ops.import_scene.gltf(filepath=filepath, *args, **kwargs)
    elif filepath.endswith(".fbx"):
        bpy.ops.import_scene.fbx(filepath=filepath, *args, **kwargs)
    elif filepath.endswith(".obj"):
        bpy.ops.wm.obj_import(filepath=filepath, *args, **kwargs)
    elif filepath.endswith(".ply"):
        bpy.ops.wm.ply_import(filepath=filepath, *args, **kwargs)
    else:
        raise RuntimeError(f"Invalid input file: {filepath}")
    imported = sorted(set(bpy.context.scene.objects) - old_objs, key=lambda x: x.name)
    logger.info(f"Imported: {[obj.name for obj in imported]}")
    return imported


_IMAGE_FORMAT_EXTENSIONS = {
    'PNG': '.png', 'JPEG': '.jpg', 'JPEG2000': '.jp2', 'TARGA': '.tga',
    'TARGA_RAW': '.tga', 'BMP': '.bmp', 'TIFF': '.tif', 'OPEN_EXR': '.exr',
    'HDR': '.hdr', 'WEBP': '.webp',
}


def repair_and_pack_textures(source_path):
    """Re-link missing texture files and pack every image into memory.

    Imported assets frequently reference textures by paths from the
    original author's machine (absolute Windows paths are common in
    Truebones FBX). When the importer can't locate a file, the image
    datablock has no pixel data, and both the GLB and FBX exporters
    silently drop it. This helper:

      1. searches for missing image files next to *source_path*, and
      2. packs every loadable image into the .blend so exports no
         longer depend on external files.

    Returns the list of image names that still have no usable data
    (their texture files could not be found on disk).
    """
    src_dir = os.path.dirname(os.path.abspath(source_path))
    try:
        bpy.ops.file.find_missing_files(directory=src_dir)
    except RuntimeError as exc:
        logger.warning(f"find_missing_files failed: {exc}")

    broken = []
    for img in bpy.data.images:
        if img.source not in ('FILE', 'SEQUENCE', 'TILED'):
            continue
        if img.packed_file is None:
            try:
                img.pack()
            except RuntimeError as exc:
                logger.warning(f"Could not pack image '{img.name}': {exc}")
        if img.packed_file is None:
            filepath = bpy.path.abspath(img.filepath) if img.filepath else ''
            if not (filepath and os.path.isfile(filepath)):
                broken.append(img.name)

    if broken:
        logger.warning(
            f"{len(broken)} image(s) have no pixel data and will be missing "
            f"from exports (texture files not found near {src_dir}): {broken}"
        )
    return broken


def write_packed_images_to_dir(texture_dir):
    """Write packed images that have no on-disk file out to *texture_dir*.

    The FBX exporter's COPY/embed path reads texture bytes from the
    image's file path on disk; images that exist only as packed data
    (typical after a GLB import) are silently skipped, losing textures.
    Writing them out right before FBX export makes embedding work.

    Returns the list of file paths written.
    """
    written = []
    for idx, img in enumerate(bpy.data.images):
        if img.packed_file is None:
            continue
        filepath = bpy.path.abspath(img.filepath) if img.filepath else ''
        if filepath and os.path.isfile(filepath):
            continue
        ext = _IMAGE_FORMAT_EXTENSIONS.get(img.file_format, '.png')
        safe_name = re.sub(r'[^A-Za-z0-9_.-]+', '_', img.name).strip('_') or f'image_{idx}'
        if not safe_name.lower().endswith(ext):
            safe_name += ext
        os.makedirs(texture_dir, exist_ok=True)
        out_path = os.path.join(texture_dir, safe_name)
        img.filepath_raw = out_path
        try:
            # Dump the packed bytes verbatim (no re-encode) exactly where
            # filepath_raw points, so the exporter's embed path finds them.
            # (img.unpack() writes to the image's *original* path instead.)
            with open(out_path, 'wb') as fh:
                fh.write(img.packed_file.data)
            written.append(out_path)
        except OSError as exc:
            logger.warning(f"Could not write packed image '{img.name}' to disk: {exc}")
    if written:
        logger.info(f"Wrote {len(written)} packed image(s) to {texture_dir} for FBX embedding")
    return written


def _find_upstream_image_node(socket, max_depth=8):
    """BFS upstream from *socket* to the first Image Texture node with an image."""
    queue = [(socket, 0)]
    visited = set()
    while queue:
        sock, depth = queue.pop(0)
        if depth > max_depth:
            continue
        for link in sock.links:
            node = link.from_node
            if node in visited:
                continue
            visited.add(node)
            if node.type == 'TEX_IMAGE' and node.image is not None:
                return node
            queue.extend((inp, depth + 1) for inp in node.inputs)
    return None


def simplify_materials_for_fbx(objs):
    """Rewire each material's base-color texture directly into a Principled BSDF.

    The FBX exporter only recognizes a texture when its Image Texture node
    feeds a Principled BSDF input directly. glTF-imported materials often
    interpose extra nodes — a Mix/Multiply for baseColorFactor or vertex
    colors, or (for KHR_materials_unlit) an Emission chain with no
    Principled node at all — which makes the FBX exporter silently drop
    the texture. This flattens those graphs: find the first image texture
    upstream and connect it straight to Base Color, creating a Principled
    BSDF if the material lacks one.
    """
    mats = {slot.material for obj in objs if obj.type == 'MESH'
            for slot in obj.material_slots if slot.material}
    for mat in mats:
        if not mat.use_nodes:
            continue
        tree = mat.node_tree
        principled = next((n for n in tree.nodes if n.type == 'BSDF_PRINCIPLED'), None)

        if principled is not None:
            base = principled.inputs['Base Color']
            if any(l.from_node.type == 'TEX_IMAGE' and l.from_node.image
                   for l in base.links):
                continue  # already directly wired; FBX export handles it
            img_node = _find_upstream_image_node(base)
            if img_node is None:
                continue
            for l in list(base.links):
                tree.links.remove(l)
            tree.links.new(img_node.outputs['Color'], base)
            logger.info(f"Material '{mat.name}': rewired image "
                        f"'{img_node.image.name}' directly to Base Color for FBX")
            continue

        # No Principled BSDF (e.g. unlit glTF material): build one from the
        # image feeding the output surface, so FBX gets a textured material.
        output = next((n for n in tree.nodes
                       if n.type == 'OUTPUT_MATERIAL' and n.is_active_output),
                      None) or next((n for n in tree.nodes
                                     if n.type == 'OUTPUT_MATERIAL'), None)
        if output is None:
            continue
        img_node = _find_upstream_image_node(output.inputs['Surface'])
        if img_node is None:
            continue
        principled = tree.nodes.new('ShaderNodeBsdfPrincipled')
        tree.links.new(img_node.outputs['Color'], principled.inputs['Base Color'])
        for l in list(output.inputs['Surface'].links):
            tree.links.remove(l)
        tree.links.new(principled.outputs['BSDF'], output.inputs['Surface'])
        logger.info(f"Material '{mat.name}': no Principled BSDF; created one "
                    f"with image '{img_node.image.name}' for FBX")


def get_armature_obj(obj_list=None):
    """Return the first ARMATURE object from a list (or the scene), else None."""
    if not obj_list:
        obj_list = bpy.context.scene.objects
    for obj in obj_list:
        if obj.type == "ARMATURE":
            return obj
    return None


def set_scene_timing(nframes, fps):
    """Set the scene frame rate and frame range ``[0, nframes - 1]``."""
    bpy.context.scene.render.fps = int(fps)
    bpy.context.scene.frame_start = 0
    bpy.context.scene.frame_end = nframes - 1


# ─────────────────────────────────────────────────────────────────────────────
# 2. Keyframe computation
# ─────────────────────────────────────────────────────────────────────────────

def compute_bone_keyframes(rest_local_mat, anim_local_mat, bone_names,
                           tpos_global_rot=None):
    """Compute per-bone location/rotation keyframes from local 4x4 transforms.

    The relative transform ``rest_inv @ anim`` is computed in NumPy for all
    frames at once; per-keyframe loc/rot are then extracted via ``mathutils``
    so they match Blender's conventions exactly.

    Args:
        rest_local_mat:  ``(nbones, 4, 4)`` rest-pose local transforms.
        anim_local_mat:  ``(nframes, nbones, 4, 4)`` animated local transforms.
        bone_names:      Bone names, length ``nbones``.
        tpos_global_rot: Optional ``(nbones, 4)`` T-pose global rotations.
            When given, each keyframe is conjugated by the bone's T-pose
            global rotation.

    Returns:
        ``{bone_name: {'location': [(frame, Vector), ...],
                       'rotation': [(frame, Quaternion), ...]}}``
    """
    nframes, nbones = anim_local_mat.shape[:2]

    rest_inv = np.linalg.inv(rest_local_mat)            # (nbones, 4, 4)
    rel_mats = rest_inv[None] @ anim_local_mat           # (nframes, nbones, 4, 4)

    tpos_quats = tpos_quats_inv = None
    if tpos_global_rot is not None:
        tpos_quats = [Quaternion(tpos_global_rot[b]) for b in range(nbones)]
        tpos_quats_inv = [q.inverted() for q in tpos_quats]

    anim_data_dict = {}
    for idxb in range(nbones):
        locs, rots = [], []
        for idxf in range(nframes):
            m = Matrix(rel_mats[idxf, idxb].tolist())
            loc = m.to_translation()
            rot = m.to_quaternion()
            if tpos_quats is not None:
                rot = tpos_quats_inv[idxb] @ rot @ tpos_quats[idxb]
                loc = tpos_quats_inv[idxb] @ loc
            locs.append((idxf, loc))
            rots.append((idxf, rot))
        anim_data_dict[str(bone_names[idxb])] = {'location': locs, 'rotation': rots}

    return anim_data_dict


# ─────────────────────────────────────────────────────────────────────────────
# 3. Action reconstruction
# ─────────────────────────────────────────────────────────────────────────────

def _add_fcurves(action, bone_name, attr, n_components, keyframes):
    """Create per-channel fcurves for one pose-bone attribute."""
    data_path = f'pose.bones["{bone_name}"].{attr}'
    for i in range(n_components):
        fc = action.fcurves.new(data_path=data_path, index=i)
        fc.keyframe_points.add(len(keyframes))
        for idx, (frame, val) in enumerate(keyframes):
            kp = fc.keyframe_points[idx]
            kp.co = (frame, val[i])
            kp.interpolation = 'LINEAR'


def rebuild_action_from_data(armature, anim_data_dict):
    """Replace the armature's animation with a new Action built from keyframes.

    ``anim_data_dict`` has the format returned by
    :func:`compute_bone_keyframes`. Any existing action / NLA tracks on the
    armature are cleared first; bones missing from the armature are skipped
    with a warning.
    """
    clear_animation_state(armature)
    new_action = bpy.data.actions.new(name=RECONSTRUCTED_ACTION_NAME)
    bind_action(armature, new_action)

    valid_bones = {bone.name for bone in armature.data.bones}
    for bone_name, data in anim_data_dict.items():
        if bone_name not in valid_bones:
            logger.warning(f"Skipping keyframes for '{bone_name}': not in armature")
            continue
        _add_fcurves(new_action, bone_name, 'location', 3, data['location'])
        _add_fcurves(new_action, bone_name, 'rotation_quaternion', 4, data['rotation'])


# ─────────────────────────────────────────────────────────────────────────────
# 4. Armature reconciliation: handle bones not present in the animation
# ─────────────────────────────────────────────────────────────────────────────

EXTRA_BONES_STRATEGIES = ('keep', 'merge', 'remove')


def find_skinned_meshes(armature):
    """Return all mesh objects whose Armature modifier targets *armature*."""
    meshes = []
    for obj in bpy.context.scene.objects:
        if obj.type != 'MESH':
            continue
        for mod in obj.modifiers:
            if mod.type == 'ARMATURE' and mod.object == armature:
                meshes.append(obj)
                break
    return meshes


def build_ancestor_map(armature, kept_bone_set):
    """Map each armature bone NOT in *kept_bone_set* to its nearest kept ancestor.

    Returns ``{dropped_name: ancestor_name_or_None}``. ``None`` indicates
    the chain to the root is entirely dropped (orphan bone).
    """
    mapping = {}
    for bone in armature.data.bones:
        if bone.name in kept_bone_set:
            continue
        parent = bone.parent
        while parent is not None and parent.name not in kept_bone_set:
            parent = parent.parent
        mapping[bone.name] = parent.name if parent is not None else None
    return mapping


def transfer_vertex_group(mesh_obj, src_name, dst_name):
    """Merge weights of vertex group *src_name* into *dst_name* and delete src.

    If *dst_name* is ``None``, the src group is just removed. Existing dst
    weights are preserved and incremented (ADD semantics).
    """
    vgs = mesh_obj.vertex_groups
    src = vgs.get(src_name)
    if src is None:
        return
    if dst_name is None:
        vgs.remove(src)
        return

    dst = vgs.get(dst_name) or vgs.new(name=dst_name)
    src_idx = src.index

    # Snapshot (vert_idx, weight) pairs before mutating the groups.
    transfers = []
    for vert in mesh_obj.data.vertices:
        for grp in vert.groups:
            if grp.group == src_idx:
                if grp.weight > 0.0:
                    transfers.append((vert.index, grp.weight))
                break  # a vertex appears at most once per group

    for v_idx, weight in transfers:
        dst.add([v_idx], weight, 'ADD')
    vgs.remove(src)


def merge_extra_bone_weights(armature, kept_bone_set):
    """Transfer weights of dropped bones to their nearest kept ancestor.

    Returns the ancestor map used (for logging).
    """
    ancestor_map = build_ancestor_map(armature, kept_bone_set)
    if not ancestor_map:
        return ancestor_map
    meshes = find_skinned_meshes(armature)
    if not meshes:
        logger.warning("No skinned meshes found targeting this armature; "
                       "weight transfer is a no-op")
    for src_name, dst_name in ancestor_map.items():
        for mesh_obj in meshes:
            transfer_vertex_group(mesh_obj, src_name, dst_name)
    return ancestor_map


def delete_dominant_weighted_vertices(mesh_obj, dropped_bone_names):
    """Delete vertices whose dominant (max-weight) group is in *dropped_bone_names*.

    Vertices with mixed influences whose primary bone survives are kept,
    so we don't punch holes near boundary regions (e.g. wrist verts that
    have minor finger-bone influences but are mostly weighted to the
    forearm).

    Returns the number of vertices deleted from *mesh_obj*.
    """
    vgs = mesh_obj.vertex_groups
    dropped_idxs = {vgs[name].index for name in dropped_bone_names if name in vgs}
    if not dropped_idxs:
        return 0

    verts_to_delete = []
    for vert in mesh_obj.data.vertices:
        if not vert.groups:
            continue
        max_weight = -1.0
        max_group = -1
        for g in vert.groups:
            if g.weight > max_weight:
                max_weight = g.weight
                max_group = g.group
        if max_group in dropped_idxs:
            verts_to_delete.append(vert.index)

    if not verts_to_delete:
        return 0

    bm = bmesh.new()
    bm.from_mesh(mesh_obj.data)
    bm.verts.ensure_lookup_table()
    bmesh.ops.delete(
        bm,
        geom=[bm.verts[i] for i in verts_to_delete],
        context='VERTS',
    )
    bm.to_mesh(mesh_obj.data)
    bm.free()
    mesh_obj.data.update()
    return len(verts_to_delete)


def delete_skin_vertices_for_bones(armature, dropped_bone_names):
    """Delete dominantly-weighted vertices for *dropped_bone_names* across all meshes.

    Returns ``{mesh_name: n_deleted}`` for logging.
    """
    if not dropped_bone_names:
        return {}
    meshes = find_skinned_meshes(armature)
    if not meshes:
        logger.warning("No skinned meshes found targeting this armature; "
                       "vertex deletion is a no-op")
        return {}
    deleted = {}
    for mesh_obj in meshes:
        n = delete_dominant_weighted_vertices(mesh_obj, dropped_bone_names)
        if n:
            deleted[mesh_obj.name] = n
    return deleted


def remove_armature_bones(armature, bone_names):
    """Delete *bone_names* from *armature* in EDIT mode, keeping the
    hierarchy connected.

    Blender does NOT re-parent children when an edit bone is removed —
    they silently become parentless roots. So before deleting, every
    surviving bone whose parent chain passes through a dropped bone is
    re-parented to its nearest surviving ancestor (the same bone its
    vertex weights are merged into under the 'merge' strategy). Edit-bone
    head/tail are stored in armature space, so re-parenting with
    ``use_connect=False`` preserves each bone's rest transform exactly.
    """
    bpy.context.view_layer.objects.active = armature
    bpy.ops.object.mode_set(mode='EDIT')
    edit_bones = armature.data.edit_bones
    dropped = {name for name in bone_names if name in edit_bones}

    for bone in edit_bones:
        if bone.name in dropped:
            continue
        if bone.parent is None or bone.parent.name not in dropped:
            continue
        ancestor = bone.parent
        while ancestor is not None and ancestor.name in dropped:
            ancestor = ancestor.parent
        bone.use_connect = False
        bone.parent = ancestor
        if ancestor is None:
            logger.warning(
                f"Bone '{bone.name}': entire parent chain was dropped; "
                f"it becomes a root bone"
            )
        else:
            logger.info(f"Re-parented '{bone.name}' to surviving ancestor "
                        f"'{ancestor.name}'")

    for bone_name in dropped:
        edit_bones.remove(edit_bones[bone_name])
    bpy.ops.object.mode_set(mode='OBJECT')


def sync_armature_bones(armature, anim_bone_names, extra_bones_strategy='merge'):
    """Reconcile armature bones with the animation bone list.

    *extra_bones_strategy* controls what happens to armature bones that are
    NOT present in *anim_bone_names*:

    - ``'merge'`` (default): transfer each extra bone's vertex weights to
      its nearest kept ancestor on every skinned mesh, then delete the
      bone. Mesh stays attached; articulation in the merged region is lost.
    - ``'remove'``: delete the bones AND every mesh vertex whose dominant
      (max-weight) bone is one of them. Vertices with mixed influences
      whose primary bone survives are kept, so we don't punch holes near
      boundary regions.
    - ``'keep'``: leave the bones in place with no keyframes. They sit at
      identity local pose and inherit their parent's animated transform
      through Blender's hierarchical pose evaluation.

    Animation bones absent from the armature are always logged and skipped
    by :func:`rebuild_action_from_data`.
    """
    if extra_bones_strategy not in EXTRA_BONES_STRATEGIES:
        raise ValueError(
            f"extra_bones_strategy must be one of {EXTRA_BONES_STRATEGIES}, "
            f"got {extra_bones_strategy!r}"
        )

    armature_bone_set = {bone.name for bone in armature.data.bones}
    anim_bone_set = {str(name) for name in anim_bone_names}

    extra = armature_bone_set - anim_bone_set
    if extra:
        if extra_bones_strategy == 'keep':
            logger.info(
                f"Keeping {len(extra)} armature bones not in animation "
                f"(no keyframes; will follow parent pose): {sorted(extra)}"
            )
        elif extra_bones_strategy == 'merge':
            ancestor_map = merge_extra_bone_weights(armature, anim_bone_set)
            summary = ', '.join(
                f"{src}->{dst or '<orphan>'}"
                for src, dst in sorted(ancestor_map.items())
            )
            logger.warning(
                f"Merged {len(extra)} extra armature bones into nearest kept "
                f"ancestor and removed them: {summary}"
            )
            remove_armature_bones(armature, extra)
        else:  # 'remove'
            deleted = delete_skin_vertices_for_bones(armature, extra)
            if deleted:
                total = sum(deleted.values())
                summary = ', '.join(f"{name}: {n}" for name, n in sorted(deleted.items()))
                logger.warning(
                    f"Deleted {total} mesh vertices dominantly weighted to "
                    f"dropped bones ({summary})"
                )
            logger.warning(f"Removing armature bones not in animation: {sorted(extra)}")
            remove_armature_bones(armature, extra)

    missing = anim_bone_set - armature_bone_set
    if missing:
        logger.warning(f"Animation bones not in armature (will be skipped): {missing}")

    update_scene()


# ─────────────────────────────────────────────────────────────────────────────
# 5. Character export
# ─────────────────────────────────────────────────────────────────────────────

def export_animated_character(output_path_no_ext, formats=('glb',)):
    """Export every MESH + ARMATURE object to ``<output_path_no_ext>.<fmt>``.

    Stale actions other than the reconstructed one are removed first so the
    exporters only bake the driving animation.
    """
    export_objs = [obj for obj in bpy.context.scene.objects
                   if obj.type in ('MESH', 'ARMATURE')]
    select_objs(export_objs, deselect_first=True)

    for action in list(bpy.data.actions):
        if action.name != RECONSTRUCTED_ACTION_NAME:
            bpy.data.actions.remove(action)
    # Imported characters can carry object-level animation data (static
    # transform keys); the glTF exporter samples it into a stray extra
    # animation, so drop everything but the reconstructed armature action.
    for obj in export_objs:
        anim_data = obj.animation_data
        if anim_data is not None and (
                anim_data.action is None
                or anim_data.action.name != RECONSTRUCTED_ACTION_NAME):
            obj.animation_data_clear()

    for fmt in formats:
        export_selected_to_file(f"{output_path_no_ext}.{fmt}", fmt)


def export_selected_to_file(filepath, char_anim_type='glb'):
    """Export currently-selected scene objects to GLB or FBX."""
    if char_anim_type == 'glb':
        bpy.ops.export_scene.gltf(
            filepath=filepath,
            check_existing=False,
            use_selection=True,
            export_format='GLB',
            # Export the action's own F-curves rather than resampling every
            # object: sampling also bakes the armature *object's* static
            # transform into a stray second glTF animation.
            export_force_sampling=False,
        )
    elif char_anim_type == 'fbx':
        # The FBX exporter drops textures that aren't wired directly into a
        # Principled BSDF (Mix nodes, unlit materials); flatten those first.
        simplify_materials_for_fbx(bpy.context.selected_objects)
        # Sidecar copies for FBX consumers that ignore embedded media;
        # embedding itself reads packed data or these files.
        write_packed_images_to_dir(
            os.path.join(os.path.dirname(os.path.abspath(filepath)), 'textures'))
        bpy.ops.export_scene.fbx(
            filepath=filepath,
            check_existing=False,
            use_selection=True,
            use_triangles=True,
            add_leaf_bones=False,
            bake_anim=True,
            path_mode="COPY",
            embed_textures=True,
        )
    else:
        raise RuntimeError(f"Invalid char_anim_type: {char_anim_type}")
    logger.info(f"Saved animated character: {filepath}")


