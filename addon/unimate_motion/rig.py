"""Export a deform hierarchy and apply decoded motion as a new Action.

Skeleton data is exported in a world-aligned frame: armature space with the
armature object's rotation and uniform scale applied, but not its location.
Rigs imported Y-up or scaled (such as Mixamo FBX files) then reach the
backend Z-up like any other rig. Code that reads or writes Blender poses
converts with export_frame().
"""
import json
import re
import numpy as np
import bpy
from mathutils import Matrix
from .motion import SCHEMA, semantic_name, signature, canonicalize

FINGER_WORDS = {"thumb", "index", "middle", "ring", "pinky", "little", "finger", "fingers"}

def armature_for(obj):
    if obj and obj.type == "ARMATURE":
        return obj
    return obj.find_armature() if obj and obj.type == "MESH" else None

def export_frame(rig):
    """Return (linear, rotation): armature space to export frame, and its rotation part."""
    linear = np.array(rig.matrix_world.to_3x3(), dtype=float)
    if np.array_equal(linear, np.eye(3)):
        return linear, linear
    scale = np.linalg.norm(linear, axis=0)
    if np.linalg.det(linear) <= 0 or not np.allclose(scale, scale.mean(), rtol=1e-4):
        raise ValueError("Use a uniform, positive armature object scale (apply non-uniform or mirrored scale first).")
    return linear, linear / scale.mean()

def is_finger(bone):
    words = re.sub(r"\d+", " ", semantic_name(bone.get("unimate_label", bone.name))).split()
    return bool(FINGER_WORDS & set(words))

def mesh_capsules(rig, skeleton):
    """Fit conservative bone capsules to dominant skin weights in the rest mesh."""
    indices = {name: j for j, name in enumerate(skeleton["bone_names"]) if name}
    points = {j: [] for j in indices.values()}
    linear, _ = export_frame(rig)
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
        head = np.asarray(skeleton["heads"][j])
        direction = np.asarray(skeleton["rest_matrices"][j])[:3, 1]
        direction = direction / np.linalg.norm(direction)
        offsets = np.asarray(values) @ linear.T - head
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
        rest_direction = np.asarray(skeleton["rest_matrices"][j])[:3, 1]
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
    # The export frame is world-aligned, so world up is +Z.
    normal = np.array([0., 0., 1.])
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
        linear, _ = export_frame(rig)
        points = np.array([tuple(transform @ v.co) for v in mesh.vertices]) @ linear.T
        triangles = points[np.array([list(face.vertices) for face in mesh.loop_triangles], dtype=int)]
        data["triangles"] = triangles.tolist()
        data["object"] = ground_object.name
    finally:
        evaluated.to_mesh_clear()
    return data


def export_skeleton(rig, forward="-Y", tips=True, fingers=True):
    """Export the deform hierarchy. forward is the facing direction in world
    axes; fingers=False leaves finger bones (and their children) out."""
    if rig is None or rig.type != "ARMATURE":
        raise ValueError("Select an armature or its skinned mesh.")
    if rig.mode == "EDIT":
        raise ValueError("Leave Edit Mode before exporting the rig.")
    linear, rotation = export_frame(rig)
    def excluded(bone):
        while bone:
            if is_finger(bone):
                return True
            bone = bone.parent
        return False
    selected = {b.name for b in rig.data.bones if b.use_deform and (fingers or not excluded(b))}
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
    identity = linear is rotation  # export_frame returns one array for the identity frame
    # With an identity frame, keep the exact armature values so signatures,
    # captured poses and saved jobs from earlier versions stay valid.
    def point(vector):
        return list(vector) if identity else (linear @ np.asarray(vector)).tolist()
    def rest(bone):
        if identity:
            return [list(row) for row in bone.matrix_local]
        matrix = np.asarray(bone.matrix_local, dtype=float)
        matrix[:3, :3] = rotation @ matrix[:3, :3]
        matrix[:3, 3] = linear @ matrix[:3, 3]
        return matrix.tolist()
    result = dict(schema=SCHEMA, name=rig.name, forward=forward, tips=tips, fingers=fingers,
                  joint_names=[b.name for b in bones],
                  bone_names=[b.name for b in bones],
                  labels=[b.get("unimate_label", semantic_name(b.name)) for b in bones],
                  parents=[indices[b.parent.name] if b.parent else -1 for b in bones],
                  heads=[point(b.head_local) for b in bones],
                  rest_matrices=[rest(b) for b in bones])
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
                result["heads"].append(point(bone.tail_local))
                result["rest_matrices"].append(None)
    canonicalize(result)
    result["collision_capsules"] = mesh_capsules(rig, result)
    result["foot_profiles"] = foot_profiles(rig, result)
    result["signature"] = signature(result)
    return result

LINEAR = bpy.types.Keyframe.bl_rna.properties["interpolation"].enum_items["LINEAR"].value

def write_curve(action, rig, path, index, group, frames, values):
    """Write one linear F-Curve in a single bulk call."""
    if hasattr(action, "fcurve_ensure_for_datablock"):  # Blender 4.4+ slotted actions
        curve = action.fcurve_ensure_for_datablock(rig, path, index=index, group_name=group)
    else:
        curve = action.fcurves.new(path, index=index, action_group=group)
    points = curve.keyframe_points
    points.add(len(frames))
    points.foreach_set("co", np.column_stack([frames, values]).astype(np.float32).ravel())
    points.foreach_set("interpolation", np.full(len(frames), LINEAR, dtype=np.int32))
    curve.update()

def apply_result(rig, skeleton, path, start_frame, scene):
    current = export_skeleton(rig, skeleton["forward"], skeleton["tips"], skeleton.get("fingers", True))
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
    linear, rotation = export_frame(rig)
    if linear is not rotation:  # export frame -> armature space
        positions = positions @ np.linalg.inv(linear).T
        rotations = rotation.T @ rotations @ rotation
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
        times = start_frame + np.arange(frames) * ratio
        # Only the root moves; child heads follow fixed rest offsets, so key
        # rotations everywhere and location on root bones only. Scale stays 1.
        keyed = [(j, name) for j, name in enumerate(skeleton["bone_names"]) if name]
        quaternions = {name: np.empty((frames, 4)) for _, name in keyed}
        locations = {name: np.empty((frames, 3)) for _, name in keyed if not rig.data.bones[name].parent}
        for t in range(frames):
            matrices = {}
            for j, name in keyed:
                bone = rig.data.bones[name]
                matrix = Matrix(rotations[t, j].tolist()).to_4x4() @ bone.matrix_local
                matrix.translation = positions[t, j].tolist()
                matrices[name] = matrix
                parent_args = {}
                if bone.parent:
                    parent_args = dict(parent_matrix=matrices[bone.parent.name],
                                       parent_matrix_local=bone.parent.matrix_local)
                basis = bone.convert_local_to_pose(matrix, bone.matrix_local, invert=True, **parent_args)
                quaternions[name][t] = basis.to_quaternion()
                if name in locations:
                    locations[name][t] = basis.translation
        for _, name in keyed:
            values = quaternions[name]
            # Keep consecutive quaternions in one hemisphere for clean interpolation.
            flips = np.cumsum(np.einsum("ij,ij->i", values[1:], values[:-1]) < 0) % 2
            values[1:][flips == 1] *= -1
            pose = rig.pose.bones[name]
            pose.rotation_mode = "QUATERNION"
            if name not in locations:
                pose.location = (0, 0, 0)
            pose.scale = (1, 1, 1)
            channels = [("rotation_quaternion", values)]
            if name in locations:
                channels.append(("location", locations[name]))
            for prop, data in channels:
                path = pose.path_from_id(prop)
                for index in range(data.shape[1]):
                    write_curve(action, rig, path, index, name, times, data[:, index])
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
