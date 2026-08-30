"""Visualize an exported motion with a root-anchored orientation arrow.

Sanity-check tool for facing-direction annotations: loads an export-stage
motion NPZ, reconstructs per-frame global joint positions via forward
kinematics, derives a per-frame orientation vector, and saves an MP4 where
an arrow at the root joint points along that orientation. With
``--direction face --face-joints i j`` the arrow shows the heading implied
by a candidate face-joint pair.

Expected NPZ keys:

    rest_local_pos : (J, 3)       rest bone offsets
    anim_local_pos : (T, J, 3)    per-frame local joint positions
    anim_local_rot : (T, J, 4)    per-frame local joint rotations (quaternions)
    parents        : (J,)         parent joint indices (-1 for root)

The orientation vector is chosen by ``--direction``:

    velocity  (default) — root horizontal velocity, held forward on stalls
    x / y / z           — fixed unit axis (world frame, pre-rotate)
    face                — vector from root to the mean of two face joints
                          specified via ``--face-joints i j``

Usage:
    python -m data_process.tools.vis_motion \\
        <path/to/motion.npz> [--output out.mp4] [--fps 30] \\
        [--direction velocity|x|y|z|face] [--face-joints i j]
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from Animation import Animation, Quaternions, positions_global  # noqa: E402
from data_process.utils.plotting import render_skeleton_motion_directed  # noqa: E402
import imageio  # noqa: E402


def load_motion_from_npz(npz_path):
    """Load a motion NPZ and compute per-frame global joint positions.

    Returns:
        (parents, global_positions) where ``global_positions`` is a
        float32 (T, J, 3) array.
    """
    data = np.load(npz_path, allow_pickle=True)
    required = ('rest_local_pos', 'anim_local_pos', 'anim_local_rot', 'parents')
    missing = [k for k in required if k not in data.files]
    if missing:
        raise KeyError(
            f"{npz_path} is missing required keys {missing}; "
            f"available keys: {list(data.files)}"
        )

    rest_local_pos = data['rest_local_pos']      # (J, 3)
    anim_local_pos = data['anim_local_pos']      # (T, J, 3)
    anim_local_rot = data['anim_local_rot']      # (T, J, 4)
    parents = np.asarray(data['parents'])        # (J,)
    njoints = len(parents)

    motion_anim = Animation(
        rotations=Quaternions(anim_local_rot),
        positions=anim_local_pos,
        orients=Quaternions.id(njoints),
        offsets=rest_local_pos,
        parents=parents,
    )
    global_pos = positions_global(motion_anim).astype(np.float32)  # (T, J, 3)
    return parents, global_pos


def compute_directions(global_pos, mode, face_joints=None):
    """Build a per-frame orientation vector from global joint positions.

    Args:
        global_pos: (T, J, 3) float array.
        mode: One of ``'velocity'``, ``'x'``, ``'y'``, ``'z'``, ``'face'``.
        face_joints: Tuple ``(i, j)`` of joint indices used when
            ``mode == 'face'``; the direction points from the root joint
            toward the midpoint of joints ``i`` and ``j``.

    Returns:
        (T, 3) float32 array. Not normalized — the renderer normalizes
        internally and skips zero-length frames.
    """
    T = global_pos.shape[0]

    if mode == 'velocity':
        # Horizontal velocity of the root joint, held forward on stalls so
        # the arrow doesn't flicker off on stationary frames.
        roots = global_pos[:, 0]                       # (T, 3)
        vel = np.zeros_like(roots)
        vel[1:] = roots[1:] - roots[:-1]
        if T >= 2:
            vel[0] = vel[1]
        vel[:, 1] = 0.0                                # project to ground plane (y=up)
        # Forward-fill through stalls so the arrow stays visible.
        last = np.zeros(3, dtype=np.float32)
        for t in range(T):
            if np.linalg.norm(vel[t]) < 1e-6:
                vel[t] = last
            else:
                last = vel[t]
        return vel.astype(np.float32)

    if mode in ('x', 'y', 'z'):
        axis = {'x': [1, 0, 0], 'y': [0, 1, 0], 'z': [0, 0, 1]}[mode]
        return np.tile(np.array(axis, dtype=np.float32)[None, :], (T, 1))

    if mode == 'face':
        if face_joints is None or len(face_joints) != 2:
            raise ValueError("--face-joints i j required when --direction=face")
        i, j = face_joints
        mid = 0.5 * (global_pos[:, i] + global_pos[:, j])  # (T, 3)
        return (mid - global_pos[:, 0]).astype(np.float32)

    raise ValueError(f"unknown direction mode: {mode}")


DEFAULT_OUTPUT_DIR = "outputs/motion_vis"


def default_output_path(npz_path):
    """Default output: ``<DEFAULT_OUTPUT_DIR>/<stem>_motion_directed.mp4``."""
    return os.path.join(DEFAULT_OUTPUT_DIR, f"{Path(npz_path).stem}_motion_directed.mp4")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        'npz_path', type=str,
        help='Path to a motion NPZ containing anim_local_pos / anim_local_rot / rest_local_pos / parents',
    )
    parser.add_argument(
        '--output', type=str, default=None,
        help=f'Output MP4 path (default: {DEFAULT_OUTPUT_DIR}/<input_stem>_motion_directed.mp4)',
    )
    parser.add_argument('--fps', type=int, default=30, help='Output video fps (default: 30)')
    parser.add_argument(
        '--direction', type=str, default='velocity',
        choices=['velocity', 'x', 'y', 'z', 'face'],
        help='Source of the orientation vector (default: velocity)',
    )
    parser.add_argument(
        '--face-joints', type=int, nargs=2, default=None, metavar=('I', 'J'),
        help='Joint index pair for --direction=face (midpoint target)',
    )
    parser.add_argument('--elev', type=float, default=30.0, help='Camera elevation')
    parser.add_argument('--azim', type=float, default=-60.0, help='Camera azimuth')
    parser.add_argument(
        '--no-rotate-root', action='store_true',
        help='Disable Z-up -> Y-up swap (default: enabled, matches save_skeleton_motion)',
    )
    parser.add_argument(
        '--arrow-scale', type=float, default=0.25,
        help='Arrow length as a fraction of the scene side (default: 0.25)',
    )
    parser.add_argument('--arrow-color', type=str, default='green', help='Arrow color')
    args = parser.parse_args()

    if not os.path.exists(args.npz_path):
        parser.error(f"Input NPZ not found: {args.npz_path}")

    parents, global_pos = load_motion_from_npz(args.npz_path)
    directions = compute_directions(global_pos, args.direction, args.face_joints)

    frames = render_skeleton_motion_directed(
        parents=parents,
        positions=global_pos,
        directions=directions,
        elev=args.elev,
        azim=args.azim,
        rotate_root=not args.no_rotate_root,
        arrow_scale=args.arrow_scale,
        arrow_color=args.arrow_color,
    )

    output_path = args.output or default_output_path(args.npz_path)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    imageio.mimwrite(output_path, frames, fps=args.fps)
    print(
        f"Saved directed motion ({frames.shape[0]} frames, "
        f"{global_pos.shape[1]} joints, direction={args.direction}) "
        f"to {output_path}"
    )


if __name__ == '__main__':
    main()
