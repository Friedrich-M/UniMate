"""Export a deform hierarchy and apply decoded motion as a new Action."""
import json
import numpy as np
import bpy
from mathutils import Matrix
from .motion import SCHEMA, semantic_name, signature, canonicalize

def armature_for(obj):
    if obj and obj.type == "ARMATURE":
        return obj
    return obj.find_armature() if obj and obj.type == "MESH" else None

def mesh_capsules(rig, skeleton):
    """Fit conservative bone capsules to dominant skin weights in the rest mesh."""
    indices = {name: j for j, name in enumerate(skeleton["bone_names"]) if name}
    points = {j: [] for j in indices.values()}
    for obj in bpy.context.scene.objects:
        if obj.type != "MESH" or obj.find_armature() != rig:
            continue
        groups = {g.index: indices[g.name] for g in obj.vertex_groups if g.name in indices}
        transform = rig.matrix_world.inverted() @ obj.matrix_world
        for v in obj.data.vertices:
            weights = [(g.weight, groups[g.group]) for g in v.groups if g.group in groups]
            if weights:
                weight, j = max(weights)
                if weight >= .3:
                    points[j].append(tuple(transform @ v.co))
    capsules = []
    for j, values in points.items():
        if len(values) < 4:
            continue
        bone = rig.data.bones[skeleton["bone_names"][j]]
        head = np.array(bone.head_local)
        direction = np.array((bone.tail_local-bone.head_local).normalized())
        offsets = np.asarray(values)-head
        axial = offsets@direction
        radial = np.linalg.norm(offsets-axial[:,None]*direction,axis=1)
        radius = float(np.quantile(radial,.98))
        if radius < 1e-6:
            continue
        low, high = float(axial.min()+radius), float(axial.max()-radius)
        if low > high:
            low = high = float((axial.min()+axial.max())*.5)
        capsules.append(dict(joint=j, a=(head+direction*low).tolist(),
                             b=(head+direction*high).tolist(), radius=radius))
    return capsules


def foot_profiles(rig, skeleton):
    """Find terminal support bones; allow per-bone angle overrides."""
    profiles = []
    heads = np.asarray(skeleton["heads"])
    for j, name in enumerate(skeleton["bone_names"]):
        if name is None or skeleton["parents"][j] < 0:
            continue
        bone = rig.data.bones[name]
        label = semantic_name(bone.get("unimate_label", name)).split()
        if not ({"foot", "paw"} & set(label) or bone.get("unimate_foot", False)):
            continue
        parent = skeleton["parents"][j]
        upper = skeleton["parents"][parent]
        if upper < 0:
            continue
        rest_direction = np.asarray(bone.matrix_local)[:3, 1]
        pitch = float(np.degrees(np.arctan2(rest_direction[2], np.linalg.norm(rest_direction[:2]))))
        stance = float(bone.get("unimate_stance_tilt_deg", np.clip(abs(pitch)+15, 20, 45)))
        swing = float(bone.get("unimate_swing_tilt_deg", max(stance+15, 45)))
        if not 10 <= stance <= 120 or not stance <= swing <= 150:
            raise ValueError(f"{name}: foot tilt limits must be 10–150 degrees, with swing >= stance.")
        leg = float(np.linalg.norm(heads[parent]-heads[upper]) +
                    np.linalg.norm(heads[j]-heads[parent]))
        profiles.append(dict(joint=j, parent=parent, upper=upper, leg_length=leg,
                             stance_tilt=stance, swing_tilt=swing))
    return profiles


def export_ground(rig, skeleton, ground_object=None):
    """Export a static ground mesh or use the rig's rest sole level."""
    world_up = np.array([0., 0., 1.])
    basis = np.array(rig.matrix_world.to_3x3())
    normal = basis.T @ world_up
    normal /= np.linalg.norm(normal)
    feet = {p["joint"] for p in skeleton.get("foot_profiles", [])}
    soles = [min(np.dot(c["a"], normal), np.dot(c["b"], normal))-c["radius"]
             for c in skeleton.get("collision_capsules", []) if c["joint"] in feet]
    fallback = float(min(soles)) if soles else float(min(np.dot(h, normal) for h in skeleton["heads"]))
    data = dict(normal=normal.tolist(), height=fallback, triangles=[])
    if ground_object is None:
        return data
    if ground_object.type != "MESH" or ground_object.find_armature() == rig:
        raise ValueError("Choose an independent, non-deforming ground mesh.")
    depsgraph = bpy.context.evaluated_depsgraph_get()
    evaluated = ground_object.evaluated_get(depsgraph)
    mesh = evaluated.to_mesh()
    try:
        mesh.calc_loop_triangles()
        if len(mesh.loop_triangles) > 20000:
            raise ValueError("Ground mesh exceeds 20,000 triangles. Use a simpler collision surface.")
        transform = rig.matrix_world.inverted() @ evaluated.matrix_world
        points = np.array([tuple(transform @ v.co) for v in mesh.vertices])
        triangles = points[np.array([list(face.vertices) for face in mesh.loop_triangles], dtype=int)]
        data["triangles"] = triangles.tolist()
        data["object"] = ground_object.name
    finally:
        evaluated.to_mesh_clear()
    return data


def export_skeleton(rig, forward="-Y", tips=True):
    if rig is None or rig.type != "ARMATURE":
        raise ValueError("Select an armature or its skinned mesh.")
    if rig.mode == "EDIT":
        raise ValueError("Leave Edit Mode before exporting the rig.")
    selected = {b.name for b in rig.data.bones if b.use_deform}
    if not selected:
        raise ValueError("The armature has no deform bones.")
    for name in list(selected):
        parent = rig.data.bones[name].parent
        while parent:
            selected.add(parent.name)
            parent = parent.parent
    bones = []
    def visit(bone):
        if bone.name in selected:
            bones.append(bone)
            for child in bone.children:
                visit(child)
    roots = [b for b in rig.data.bones if b.name in selected and not b.parent]
    if len(roots) != 1:
        raise ValueError("Use a single root hierarchy. Multiple disconnected roots are not supported yet.")
    visit(roots[0])
    constrained = [b.name for b in bones if any(not c.mute and c.influence != 0 for c in rig.pose.bones[b.name].constraints)]
    if constrained:
        raise ValueError("Bake or disable control constraints first: " + ", ".join(constrained[:4]))
    if rig.animation_data and rig.animation_data.drivers:
        raise ValueError("This prototype requires a deform rig without animation drivers.")
    if rig.animation_data and rig.animation_data.use_nla and any(not t.mute for t in rig.animation_data.nla_tracks):
        raise ValueError("Mute NLA tracks before generating a standalone Action.")
    indices = {b.name: i for i, b in enumerate(bones)}
    result = dict(schema=SCHEMA, name=rig.name, forward=forward, tips=tips,
                  joint_names=[b.name for b in bones],
                  bone_names=[b.name for b in bones],
                  labels=[b.get("unimate_label", semantic_name(b.name)) for b in bones],
                  parents=[indices[b.parent.name] if b.parent else -1 for b in bones],
                  heads=[list(b.head_local) for b in bones],
                  rest_matrices=[[list(row) for row in b.matrix_local] for b in bones])
    if tips:
        for bone in bones:
            if not any(c.name in selected for c in bone.children):
                name = bone.name + " [UniMate tip]"
                if name in result["joint_names"]:
                    raise ValueError("A bone name conflicts with a generated tip name.")
                result["joint_names"].append(name)
                result["bone_names"].append(None)
                result["labels"].append(semantic_name(bone.name) + " end")
                result["parents"].append(indices[bone.name])
                result["heads"].append(list(bone.tail_local))
                result["rest_matrices"].append(None)
    canonicalize(result)
    result["collision_capsules"] = mesh_capsules(rig, result)
    result["foot_profiles"] = foot_profiles(rig, result)
    result["signature"] = signature(result)
    return result

def apply_result(rig, skeleton, path, start_frame, scene):
    current = export_skeleton(rig, skeleton["forward"], skeleton["tips"])
    if current["signature"] != skeleton["signature"]:
        raise ValueError("The rig changed after generation. Generate again for the current rest pose.")
    with np.load(path, allow_pickle=False) as result:
        if int(result["schema"]) != SCHEMA or str(result["signature"]) != skeleton["signature"]:
            raise ValueError("Motion does not belong to this rig export.")
        positions = result["positions"].copy()
        rotations = result["rotations"].copy()
        names = result["joint_names"].tolist()
        fps = float(result["fps"])
        prompt = str(result["prompt"])
        seed = int(result["seed"])
    count = len(skeleton["parents"])
    frames = len(positions)
    if names != skeleton["joint_names"] or positions.shape != (frames, count, 3) or rotations.shape != (frames, count, 3, 3):
        raise ValueError("Motion shape or joint names do not match the rig.")
    if not 1 <= frames <= 10000 or not 0 < fps <= 240:
        raise ValueError("Invalid motion duration or frame rate.")
    if not np.isfinite(positions).all() or not np.isfinite(rotations).all():
        raise ValueError("Motion contains invalid numbers.")
    if not np.allclose(rotations.swapaxes(-1, -2) @ rotations, np.eye(3), atol=1e-4) or not np.allclose(np.linalg.det(rotations), 1, atol=1e-4):
        raise ValueError("Motion contains invalid rotation matrices.")
    if rig.library or rig.data.library:
        raise ValueError("Make the armature local before applying motion.")
    animation = rig.animation_data_create()
    old_action = animation.action
    old_slot = getattr(animation, "action_slot", None)
    old_blend = animation.action_blend_type
    old_influence = animation.action_influence
    old_pose = {p.name: (p.rotation_mode, p.matrix_basis.copy()) for p in rig.pose.bones}
    original_frame = scene.frame_current
    action = bpy.data.actions.new("UniMate | " + prompt[:48])
    action.use_fake_user = True
    action["unimate_prompt"], action["unimate_seed"] = prompt, seed
    action["unimate_source"] = str(path)
    try:
        animation.action = action
        animation.action_blend_type = "REPLACE"
        animation.action_influence = 1.
        ratio = (scene.render.fps / scene.render.fps_base) / fps
        for t in range(frames):
            frame = start_frame + t * ratio
            matrices = {}
            for j, name in enumerate(skeleton["bone_names"]):
                if name is None:
                    continue
                bone = rig.data.bones[name]
                pose = rig.pose.bones[name]
                matrix = Matrix(rotations[t, j].tolist()).to_4x4() @ bone.matrix_local
                matrix.translation = positions[t, j].tolist()
                matrices[name] = matrix
                parent_args = {}
                if bone.parent:
                    parent_args = dict(parent_matrix=matrices[bone.parent.name],
                                       parent_matrix_local=bone.parent.matrix_local)
                basis = bone.convert_local_to_pose(matrix, bone.matrix_local, invert=True, **parent_args)
                location, quaternion, scale = basis.decompose()
                if t and pose.rotation_quaternion.dot(quaternion) < 0:
                    quaternion.negate()
                pose.rotation_mode = "QUATERNION"
                pose.location, pose.rotation_quaternion, pose.scale = location, quaternion, scale
                for prop in ("location", "rotation_quaternion", "scale"):
                    pose.keyframe_insert(prop, frame=frame, group=name)
        # Blender 4.4+ stores curves inside Action channel bags.
        if hasattr(action, "layers") and action.layers:
            curves = [fc for layer in action.layers for strip in layer.strips
                      for bag in strip.channelbags for fc in bag.fcurves]
        else:
            curves = list(action.fcurves)
        for fc in curves:
            for point in fc.keyframe_points:
                point.interpolation = "LINEAR"
        if old_action:
            old_action.use_fake_user = True
        scene.frame_end = max(scene.frame_end, int(np.ceil(start_frame + (frames - 1) * ratio)))
        scene.frame_set(start_frame)
        return action
    except Exception:
        animation.action = old_action
        if old_action and old_slot is not None:
            animation.action_slot = old_slot
        animation.action_blend_type, animation.action_influence = old_blend, old_influence
        for name, (mode, matrix) in old_pose.items():
            rig.pose.bones[name].rotation_mode = mode
            rig.pose.bones[name].matrix_basis = matrix
        bpy.data.actions.remove(action)
        scene.frame_set(original_frame)
        raise
