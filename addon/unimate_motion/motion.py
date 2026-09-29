"""Rig geometry and UniMate feature decoding. No Blender or ML dependencies."""
import hashlib
import json
import re
import numpy as np

SCHEMA = 1
AXES = {"Y": (0, 1, 0), "-Y": (0, -1, 0), "X": (1, 0, 0), "-X": (-1, 0, 0)}

def semantic_name(name):
    name = name.split(":")[-1]
    name = re.sub(r"^(DEF|ORG|MCH)[-_]", "", name, flags=re.I)
    name = re.sub(r"([a-z])([A-Z])", r"\1 \2", name)
    name = re.sub(r"[._-]L$", " left", name, flags=re.I)
    name = re.sub(r"[._-]R$", " right", name, flags=re.I)
    return re.sub(r"[_.-]+", " ", name).strip().lower()

def signature(skeleton):
    data = {k: skeleton[k] for k in ("joint_names", "bone_names", "parents", "heads", "rest_matrices", "forward")}
    return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()

def canonicalize(skeleton):
    parents = np.asarray(skeleton["parents"], dtype=np.int64)
    heads = np.asarray(skeleton["heads"], dtype=np.float64)
    n = len(parents)
    if heads.shape != (n, 3) or n < 2 or not np.isfinite(heads).all():
        raise ValueError("Skeleton must contain finite joint positions.")
    if parents[0] != -1 or any(p < 0 or p >= j for j, p in enumerate(parents[1:], 1)):
        raise ValueError("Skeleton must be one tree ordered parent before child.")
    if len(set(skeleton["joint_names"])) != n:
        raise ValueError("Joint names must be unique.")
    forward = np.array(AXES[skeleton["forward"]], dtype=float)
    up = np.array([0., 0., 1.])
    basis = np.stack([np.cross(up, forward), up, forward])
    pos = heads @ basis.T
    origin = np.array([pos[0, 0], pos[:, 1].min(), pos[0, 2]])
    adjacency = [[] for _ in range(n)]
    for j, p in enumerate(parents[1:], 1):
        length = float(np.linalg.norm(pos[j] - pos[p]))
        adjacency[j].append((p, length))
        adjacency[p].append((j, length))
    def farthest(start):
        distances = {start: 0.}
        queue = [start]
        for i in queue:
            for j, length in adjacency[i]:
                if j not in distances:
                    distances[j] = distances[i] + length
                    queue.append(j)
        end = max(distances, key=distances.get)
        return end, distances[end]
    diameter = farthest(farthest(0)[0])[1]
    if diameter < 1e-8:
        raise ValueError("Skeleton has no measurable length.")
    scale = 2. / diameter
    pos = (pos - origin) * scale
    offsets = pos.copy()
    offsets[1:] -= pos[parents[1:]]
    return dict(positions=pos, offsets=offsets, parents=parents,
                basis=basis, origin=origin, scale=scale)

def rotation_6d(values):
    x, raw_y = values[..., :3], values[..., 3:]
    x_len = np.linalg.norm(x, axis=-1, keepdims=True)
    z = np.cross(x, raw_y)
    z_len = np.linalg.norm(z, axis=-1, keepdims=True)
    if np.any(x_len < 1e-8) or np.any(z_len < 1e-8):
        raise ValueError("Generated motion contains degenerate rotations.")
    x, z = x / x_len, z / z_len
    return np.stack([x, np.cross(z, x), z], axis=-1)

def decode_features(features, canonical):
    """Match upstream FK: child slots store their parent's local rotation."""
    data = np.asarray(features, dtype=float)
    parents, offsets = canonical["parents"], canonical["offsets"]
    if data.ndim != 3 or data.shape[1:] != (len(parents), 12) or not np.isfinite(data).all():
        raise ValueError("Expected finite UniMate features with shape (frames, joints, 12).")
    frames, joints, _ = data.shape
    hml = rotation_6d(data[:, :, 3:9])
    local = np.tile(np.eye(3), (frames, joints, 1, 1))
    for j, parent in enumerate(parents[1:], 1):
        local[:, parent] = hml[:, j]
    velocity = np.zeros((frames, 3))
    velocity[1:, 0] = data[:-1, 0, 9]
    velocity[1:, 2] = data[:-1, 0, 11]
    root = np.cumsum(np.einsum("tji,tj->ti", hml[:, 0], velocity), axis=0)
    root[:, 1] = data[:, 0, 1]
    positions = np.zeros((frames, joints, 3))
    rotations = np.zeros((frames, joints, 3, 3))
    for j, parent in enumerate(parents):
        if parent < 0:
            rotations[:, j], positions[:, j] = local[:, j], root
        else:
            rotations[:, j] = rotations[:, parent] @ local[:, j]
            positions[:, j] = positions[:, parent] + np.einsum("tij,j->ti", rotations[:, parent], offsets[j])
    basis = canonical["basis"]
    positions = (positions / canonical["scale"] + canonical["origin"]) @ basis
    rotations = basis.T @ rotations @ basis
    return positions, rotations


def encode_pose(positions, rotations, skeleton):
    """Capture a static pose as UniMate features (root travel is not constrained)."""
    canon = canonicalize(skeleton)
    basis, scale, origin, parents = (canon[k] for k in ("basis", "scale", "origin", "parents"))
    positions = (np.asarray(positions) @ basis.T - origin) * scale
    global_delta = basis @ np.asarray(rotations) @ basis.T
    local = global_delta.copy()
    for j, parent in enumerate(parents[1:], 1):
        local[j] = global_delta[parent].T @ global_delta[j]
    direction = global_delta[0] @ np.array([0., 0., 1.])
    yaw = np.arctan2(direction[0], direction[2])
    c, s = np.cos(yaw), np.sin(yaw)
    facing = np.array([[c,0,-s],[0,1,0],[s,0,c]])
    positions[:, [0,2]] -= positions[0, [0,2]]
    features = np.zeros((len(parents), 12))
    features[:, :3] = positions @ facing.T
    features[0, 3:9] = np.concatenate([facing[:,0], facing[:,1]])
    for j, parent in enumerate(parents[1:], 1):
        features[j, 3:9] = np.concatenate([local[parent,:,0], local[parent,:,1]])
    return dict(signature=skeleton["signature"], features=features.tolist())
