"""Editable human landmark mapping and pose capture for UniMate conditioning."""
import json
import re
import numpy as np
from mathutils import Matrix, Vector
from .motion import canonicalize, encode_pose, semantic_name

ROLES = ("hips", "spine", "chest", "neck", "head",
         "left_upper_arm", "left_forearm", "left_hand", "right_upper_arm", "right_forearm", "right_hand",
         "left_thigh", "left_shin", "left_foot", "right_thigh", "right_shin", "right_foot")

def auto_mapping(rig):
    aliases = {
        "hips": ("hips", "pelvis"), "spine": ("spine",), "chest": ("chest", "spine2", "upperchest"),
        "neck": ("neck",), "head": ("head",),
        "upper_arm": ("upperarm", "arm"), "forearm": ("forearm", "lowerarm"),
        "hand": ("hand",), "thigh": ("thigh", "upleg", "upperleg"),
        "shin": ("shin", "calf", "leg", "lowerleg"), "foot": ("foot",),
    }
    cleaned = {b.name: re.sub(r"[^a-z0-9]", "", semantic_name(b.get("unimate_label", b.name))) for b in rig.data.bones}
    mapping = {}
    for role in ROLES:
        if role.startswith(("left_", "right_")):
            side, part = role.split("_", 1)
            accepted = {side + a for a in aliases[part]} | {a + side for a in aliases[part]}
        else:
            accepted = set(aliases[role])
        matches = [n for n, label in cleaned.items() if label in accepted]
        mapping[role] = matches[0] if len(matches) == 1 else ""
    return mapping

def capture_pose(rig, skeleton):
    """Read the current pose in armature space and encode it with encode_pose."""
    parents = skeleton["parents"]
    count = len(parents)
    positions = np.zeros((count, 3))
    rotations = np.tile(np.eye(3), (count, 1, 1))
    for j, name in enumerate(skeleton["bone_names"]):
        if name:
            bone, pose = rig.data.bones[name], rig.pose.bones[name]
            if not np.allclose(pose.scale, (1,1,1), atol=1e-4):
                raise ValueError("Pose references must use rotations, not bone scale.")
            if bone.parent and not np.allclose(pose.location, (0,0,0), atol=1e-4):
                raise ValueError("Non-root bone translations cannot be represented by this motion model.")
            positions[j] = np.asarray(pose.head)
            rotations[j] = np.array(pose.matrix.to_3x3()) @ np.array(bone.matrix_local.to_3x3()).T
        else:
            parent = parents[j]
            positions[j] = np.asarray(rig.pose.bones[skeleton["bone_names"][parent]].tail)
            rotations[j] = rotations[parent]
    return encode_pose(positions, rotations, skeleton)

def preview_estimate(rig, skeleton, estimate, mapping, threshold=.5):
    canon = canonicalize(skeleton)
    world = np.asarray(estimate["world"], dtype=float)
    if world.shape != (33, 5) or not np.isfinite(world).all():
        raise ValueError("Invalid human pose estimation result.")
    # MediaPipe camera axes -> canonical Y-up, toward-camera +Z -> armature local.
    points = world[:, :3] * np.array([1.,-1.,-1.])
    points = points @ canon["basis"]
    confidence = np.minimum(world[:, 3], world[:, 4])
    core = [11,12,23,24]
    if np.min(confidence[core]) < threshold:
        raise ValueError("Torso landmarks are uncertain. Use a clearer full-body image.")
    hip = (points[23]+points[24])*.5
    shoulders = (points[11]+points[12])*.5
    ears = (points[7]+points[8])*.5
    directions = {
        "hips": (shoulders-hip, core), "spine": (shoulders-hip, core),
        "chest": (shoulders-hip, core), "neck": (ears-shoulders, [7,8,11,12]),
        "head": (ears-shoulders, [7,8,11,12]),
    }
    for side, indices in (("left", (11,13,15,19,23,25,27,31)), ("right", (12,14,16,20,24,26,28,32))):
        a,b,c,d,e,f,g,h = indices
        for role, p, q in (("upper_arm",a,b),("forearm",b,c),("hand",c,d),("thigh",e,f),("shin",f,g),("foot",g,h)):
            directions[side+"_"+role] = (points[q]-points[p], [p,q])
    required = ("hips","left_upper_arm","left_forearm","right_upper_arm","right_forearm","left_thigh","left_shin","right_thigh","right_shin")
    missing = [role for role in required if not mapping.get(role) or mapping[role] not in rig.data.bones]
    if missing:
        raise ValueError("Map these human bones first: " + ", ".join(missing))
    filled = [n for n in mapping.values() if n]
    if len(filled) != len(set(filled)):
        raise ValueError("Each human mapping role needs a distinct bone.")
    def body_frame(across, vertical):
        if np.linalg.norm(across) < 1e-6:
            raise ValueError("Hip positions do not define a reliable body width.")
        x = across / np.linalg.norm(across)
        y = vertical - x * np.dot(x, vertical)
        if np.linalg.norm(y) < 1e-6:
            raise ValueError("Body landmarks do not define a reliable torso orientation.")
        y /= np.linalg.norm(y)
        return np.stack([x, y, np.cross(x, y)], axis=-1)
    rest_across = np.array(rig.data.bones[mapping["left_thigh"]].head_local) - np.array(rig.data.bones[mapping["right_thigh"]].head_local)
    upper = mapping.get("chest") or mapping.get("spine") or mapping.get("head")
    if not upper:
        raise ValueError("Map a chest, spine, or head bone for torso orientation.")
    rest_up = np.array(rig.data.bones[upper].tail_local) - np.array(rig.data.bones[mapping["hips"]].head_local)
    body_delta = Matrix((body_frame(points[23]-points[24], shoulders-hip) @ body_frame(rest_across, rest_up).T).tolist())
    by_bone = {name: role for role, name in mapping.items() if name}
    matrices, deltas = {}, {}
    skipped = []
    for name in skeleton["bone_names"]:
        if name is None:
            continue
        bone = rig.data.bones[name]
        parent_delta = deltas[bone.parent.name] if bone.parent else body_delta
        delta = parent_delta.copy()
        role = by_bone.get(name)
        if role in directions:
            target, landmarks = directions[role]
            if np.min(confidence[landmarks]) >= threshold and np.linalg.norm(target) > 1e-6:
                current = parent_delta @ (bone.tail_local-bone.head_local)
                swing = current.normalized().rotation_difference(Vector(target).normalized()).to_matrix()
                delta = swing @ parent_delta
            else:
                skipped.append(role)
        deltas[name] = delta
        head = bone.head_local.copy() if not bone.parent else (
            matrices[bone.parent.name].translation + parent_delta @ (bone.head_local-bone.parent.head_local))
        matrix = delta.to_4x4() @ bone.matrix_local
        matrix.translation = head
        matrices[name] = matrix
        kwargs = {}
        if bone.parent:
            kwargs = dict(parent_matrix=matrices[bone.parent.name], parent_matrix_local=bone.parent.matrix_local)
        pose = rig.pose.bones[name]
        pose.rotation_mode = "QUATERNION"
        pose.matrix_basis = bone.convert_local_to_pose(matrix, bone.matrix_local, invert=True, **kwargs)
    # Single-view landmarks are hip-relative. Place the estimated body on the rig's
    # rest ground plane using its reconstructed extremities, rather than leaving a
    # sitting pose floating at the standing pelvis height.
    low = min(min(matrix.translation.z,
                  (matrix.translation + deltas[name] @ (rig.data.bones[name].tail_local-rig.data.bones[name].head_local)).z)
              for name, matrix in matrices.items())
    ground = min(head[2] for head in skeleton["heads"])
    for matrix in matrices.values():
        matrix.translation.z += ground-low
    for name, matrix in matrices.items():
        bone = rig.data.bones[name]
        kwargs = dict(parent_matrix=matrices[bone.parent.name], parent_matrix_local=bone.parent.matrix_local) if bone.parent else {}
        rig.pose.bones[name].matrix_basis = bone.convert_local_to_pose(matrix, bone.matrix_local, invert=True, **kwargs)
    return skipped
