"""3D skeleton visualization utilities (matplotlib, headless Agg backend).

All public functions operate on a parent-array skeleton representation:

    parents : array_like of shape (J,)
        parents[j] is the index of joint j's parent, or -1 for the root.
    positions : array_like
        Joint positions. Single pose: (J, 3) or (J, D) with D >= 3 (extra
        feature dims are stripped). Motion: (T, J, 3) or (T, J, D).

Two families of functions:

    render_skeleton_*  -> RGB numpy array(s) in memory
    save_skeleton_*    -> write PNG (pose) / MP4 (motion) to disk
"""

import textwrap
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import matplotlib
import imageio
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.ticker import LinearLocator
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from mpl_toolkits.mplot3d import proj3d

matplotlib.use('Agg')


# ============================================================================
# Geometry / array helpers
# ============================================================================

def _as_xyz(positions) -> np.ndarray:
    """Cast to float32 and truncate the last axis to xyz."""
    pts = np.asarray(positions, dtype=np.float32)
    if pts.shape[-1] > 3:
        pts = pts[..., :3]
    return pts


def _maybe_rotate_y_up(pts: np.ndarray, rotate: bool) -> np.ndarray:
    """Swap Y/Z so a Z-up skeleton renders upright in matplotlib's Y-up view."""
    if not rotate:
        return pts
    out = pts.copy()
    out[..., 1] = -pts[..., 2]
    out[..., 2] = pts[..., 1]
    return out


def _edges_from_parents(parents: Sequence[int]) -> np.ndarray:
    """(E, 2) edge list of (child, parent) pairs, skipping the root."""
    return np.array(
        [(i, p) for i, p in enumerate(parents) if p != -1],
        dtype=np.int32,
    )


def _cubic_bounds(points: np.ndarray):
    """Axis-aligned cubic bounding box centered on the points (preserves aspect)."""
    x_min, x_max = points[..., 0].min(), points[..., 0].max()
    y_min, y_max = points[..., 1].min(), points[..., 1].max()
    z_min, z_max = points[..., 2].min(), points[..., 2].max()
    side = max(x_max - x_min, y_max - y_min, z_max - z_min)
    cx, cy, cz = (x_min + x_max) / 2, (y_min + y_max) / 2, (z_min + z_max) / 2
    half = side / 2
    return cx - half, cx + half, cy - half, cy + half, cz - half, cz + half


def _tight_bounds(points: np.ndarray, pad_ratio: float = 0.08,
                  abs_pad_ratio: float = 0.0):
    """Per-axis bounding box with proportional padding; pairs with non-cubic box_aspect.

    ``pad_ratio`` adds *proportional* padding (``pad_ratio * axis_length``) —
    tiny on a thin axis.
    ``abs_pad_ratio`` adds *absolute* padding tied to the overall scale
    (``abs_pad_ratio * max_axis_length``) — the right knob when labels or
    annotations are placed at offsets proportional to the full skeleton size.
    """
    x_min, x_max = points[..., 0].min(), points[..., 0].max()
    y_min, y_max = points[..., 1].min(), points[..., 1].max()
    z_min, z_max = points[..., 2].min(), points[..., 2].max()
    # Guard against degenerate flat axes (e.g. planar skeleton).
    dx = max(x_max - x_min, 1e-6)
    dy = max(y_max - y_min, 1e-6)
    dz = max(z_max - z_min, 1e-6)
    scale = max(dx, dy, dz)
    abs_pad = abs_pad_ratio * scale
    px = dx * pad_ratio + abs_pad
    py = dy * pad_ratio + abs_pad
    pz = dz * pad_ratio + abs_pad
    return (x_min - px, x_max + px,
            y_min - py, y_max + py,
            z_min - pz, z_max + pz)


def _box_aspect_from_bounds(bounds) -> Tuple[float, float, float]:
    x_min, x_max, y_min, y_max, z_min, z_max = bounds
    return (max(x_max - x_min, 1e-6),
            max(y_max - y_min, 1e-6),
            max(z_max - z_min, 1e-6))


# ============================================================================
# Matplotlib figure helpers
# ============================================================================

def _new_figure(bounds, elev: Optional[float], azim: Optional[float],
                figsize=(6.4, 4.8), dpi: int = 120,
                box_aspect=(1, 1, 1)):
    """Create a 3D figure/axes configured for skeleton rendering.

    Use ``box_aspect=(1,1,1)`` with :func:`_cubic_bounds`, or per-axis extents
    with :func:`_tight_bounds` to fill the figure with a long-thin subject.
    """
    fig = Figure(figsize=figsize, dpi=dpi)
    canvas = FigureCanvasAgg(fig)
    ax = fig.add_subplot(111, projection='3d', computed_zorder=False)

    x_min, x_max, y_min, y_max, z_min, z_max = bounds
    ax.set_xlim([x_min, x_max])
    ax.set_ylim([y_min, y_max])
    ax.set_zlim([z_min, z_max])

    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    ax.set_xticklabels([]); ax.set_yticklabels([]); ax.set_zticklabels([])
    ax.xaxis.set_major_locator(LinearLocator(8))
    ax.yaxis.set_major_locator(LinearLocator(8))
    ax.zaxis.set_major_locator(LinearLocator(8))
    ax.grid(True, color='gray', linestyle='--', linewidth=0.5)
    ax.set_box_aspect(box_aspect)

    if elev is not None and azim is not None:
        ax.view_init(elev=elev, azim=azim)

    fig.subplots_adjust(left=-0.05, right=1.05, top=1.05, bottom=-0.05)
    return fig, canvas, ax


def _capture_rgb(canvas: FigureCanvasAgg) -> np.ndarray:
    """Draw the canvas and return its pixels as an (H, W, 3) uint8 array."""
    canvas.draw()
    w, h = canvas.get_width_height()
    rgba = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)
    return rgba[..., :3].copy()


# ============================================================================
# Skeleton drawing primitives
# ============================================================================

def _draw_static_skeleton(ax, pts: np.ndarray, parents: Sequence[int],
                          edges: np.ndarray) -> None:
    """Draw bones (gray) and joints (root red, others blue) for a single pose."""
    segs = np.zeros((len(edges), 2, 3), dtype=np.float32)
    segs[:, 0] = pts[edges[:, 0]]
    segs[:, 1] = pts[edges[:, 1]]
    ax.add_collection3d(Line3DCollection(segs, colors='gray', linewidths=1.0))

    joint_colors = np.where(np.asarray(parents) == -1, 'r', 'b')
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=10, c=joint_colors)
    ax.scatter(pts[0:1, 0], pts[0:1, 1], pts[0:1, 2], s=15, c='r')


def _init_motion_artists(ax, edges: np.ndarray, first_frame: np.ndarray,
                         joint_colors, joint_size: float,
                         bone_color: str = 'gray',
                         bone_linewidth: float = 1.0,
                         joint_edgecolors=None,
                         joint_edgewidth: float = 0.0):
    """Create reusable bone/joint artists for a motion render.

    Returns ``(segs, lc, sc)`` where ``segs`` is the mutable (E, 2, 3) buffer
    backing the Line3DCollection ``lc``, and ``sc`` is the joint scatter.
    Callers mutate ``segs`` and ``sc._offsets3d`` per frame.
    """
    segs = np.zeros((len(edges), 2, 3), dtype=np.float32)
    lc = Line3DCollection(segs, colors=bone_color, linewidths=bone_linewidth)
    ax.add_collection3d(lc)

    kwargs = dict(s=joint_size, c=joint_colors, depthshade=False, zorder=2)
    if joint_edgecolors is not None:
        kwargs['edgecolors'] = joint_edgecolors
        kwargs['linewidths'] = joint_edgewidth
    sc = ax.scatter(first_frame[:, 0], first_frame[:, 1], first_frame[:, 2],
                    **kwargs)
    return segs, lc, sc


def _fibonacci_sphere_dirs(n: int) -> np.ndarray:
    """``n`` approximately uniform unit vectors on the sphere (Fibonacci spiral)."""
    phi = np.pi * (3.0 - np.sqrt(5.0))  # golden angle
    idx = np.arange(n, dtype=np.float32)
    y = 1.0 - 2.0 * idx / max(n - 1, 1)
    radius = np.sqrt(np.maximum(1.0 - y * y, 0.0))
    theta = phi * idx
    x = np.cos(theta) * radius
    z = np.sin(theta) * radius
    return np.stack([x, y, z], axis=1).astype(np.float32)


def _anchor_for_dir(d: np.ndarray):
    """Pick matching (ha, va) text anchors so text grows *away* from the joint."""
    ha = 'left' if d[0] > 0.15 else 'right' if d[0] < -0.15 else 'center'
    va = 'bottom' if d[1] > 0.15 else 'top' if d[1] < -0.15 else 'center'
    return ha, va


def _sample_bone_obstacles(pts: np.ndarray, parents: Sequence[int],
                           samples_per_bone: int = 3) -> np.ndarray:
    """Sample interior points along every bone segment; used as label obstacles."""
    pts = np.asarray(pts, dtype=np.float32)
    samples: List[np.ndarray] = []
    ts = np.linspace(0.0, 1.0, samples_per_bone + 2)[1:-1]  # interior only
    for c, p in enumerate(parents):
        if p == -1:
            continue
        a, b = pts[c], pts[p]
        for t in ts:
            samples.append(a * (1.0 - t) + b * t)
    if not samples:
        return np.zeros((0, 3), dtype=np.float32)
    return np.stack(samples, axis=0)


def _annotate_joints(ax, pts: np.ndarray, joint_names: Sequence[str],
                     bounds, font_size: int,
                     parents: Optional[Sequence[int]] = None) -> None:
    """Place each joint's name next to its joint without overlapping bones/labels.

    For each joint, scores 24 × 4 candidate offset directions (Fibonacci sphere
    × multiple radii) in screen space by distance from other joints, bone
    samples, and already-placed labels. The chosen anchor gets a colored
    leader line back to the joint so the association stays visible even when
    labels are pushed far out into empty pockets.
    """
    x_min, x_max, y_min, y_max, z_min, z_max = bounds
    scale = max(x_max - x_min, y_max - y_min, z_max - z_min)
    base = 0.08 * scale  # leader lines let us push labels well clear of bones

    unit_dirs_single = _fibonacci_sphere_dirs(36)
    radii = np.array([1.0, 1.5, 2.2, 3.0], dtype=np.float32)
    unit_dirs = np.repeat(unit_dirs_single, len(radii), axis=0)
    radius_scales = np.tile(radii, len(unit_dirs_single)).astype(np.float32)

    # Force a draw so box_aspect/view_init propagate into get_proj().
    ax.figure.canvas.draw()
    proj_matrix = ax.get_proj()

    def project2d(xyz: np.ndarray) -> np.ndarray:
        xs, ys, _ = proj3d.proj_transform(
            xyz[..., 0], xyz[..., 1], xyz[..., 2], proj_matrix,
        )
        return np.stack([xs, ys], axis=-1)

    J = len(pts)
    parents_arr = (np.asarray(parents) if parents is not None
                   else np.full(J, -1, dtype=int))
    joint_colors = [
        ('red' if parents_arr[i] == -1 else 'steelblue') for i in range(J)
    ]

    joint_obstacles = pts
    bone_obstacles = (_sample_bone_obstacles(pts, parents)
                      if parents is not None
                      else np.zeros((0, 3), dtype=np.float32))
    placed_2d: List[np.ndarray] = []

    joint_obstacles_2d = project2d(joint_obstacles)
    bone_obstacles_2d = (project2d(bone_obstacles)
                         if len(bone_obstacles) > 0
                         else np.zeros((0, 2), dtype=np.float32))

    for i, (p, name) in enumerate(zip(pts, joint_names)):
        offsets = unit_dirs * (radius_scales[:, None] * base)
        candidates = p[None, :] + offsets                         # (N, 3)
        candidates_2d = project2d(candidates)                     # (N, 2)

        # Avoidance set in 2D: all joints (except self) + bone samples + placed labels.
        mask = np.ones(len(joint_obstacles_2d), dtype=bool)
        mask[i] = False
        avoid_parts = [joint_obstacles_2d[mask]]
        if len(bone_obstacles_2d) > 0:
            avoid_parts.append(bone_obstacles_2d)
        if placed_2d:
            avoid_parts.append(np.stack(placed_2d, axis=0))
        avoid_2d = np.concatenate(avoid_parts, axis=0)

        diffs = candidates_2d[:, None, :] - avoid_2d[None, :, :]
        min_dists = np.linalg.norm(diffs, axis=-1).min(axis=1)

        # Small penalty on longer radii — prefer shorter leaders when free.
        screen_span = float(
            max(np.ptp(candidates_2d[:, 0]), np.ptp(candidates_2d[:, 1]), 1e-6)
        )
        score = min_dists - 0.03 * radius_scales * screen_span
        slot = int(np.argmax(score))

        lp = candidates[slot]
        ha, va = _anchor_for_dir(unit_dirs[slot])
        color = joint_colors[i]

        leader, = ax.plot([p[0], lp[0]], [p[1], lp[1]], [p[2], lp[2]],
                          color=color, linewidth=0.9, linestyle='-',
                          alpha=0.9, zorder=9)
        leader.set_clip_on(False)
        txt = ax.text(lp[0], lp[1], lp[2],
                      f"[{i}] {name}", fontsize=font_size, color='black',
                      ha=ha, va=va, zorder=10,
                      bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                                edgecolor=color, linewidth=0.8, alpha=0.85))
        txt.set_clip_on(False)
        placed_2d.append(candidates_2d[slot])


def _spectral_to_rgb(spectral_feats: np.ndarray) -> np.ndarray:
    """Map (J, K) spectral features to (J, 3) RGB via PCA + per-channel min-max.

    Topologically close joints get similar colors; distant joints contrast.
    """
    J, K = spectral_feats.shape
    if K >= 3:
        centered = spectral_feats - spectral_feats.mean(axis=0, keepdims=True)
        _, _, Vt = np.linalg.svd(centered, full_matrices=False)
        rgb = centered @ Vt[:3].T
    else:
        rgb = np.zeros((J, 3), dtype=np.float32)
        rgb[:, :K] = spectral_feats

    for c in range(3):
        lo, hi = rgb[:, c].min(), rgb[:, c].max()
        if hi - lo > 1e-8:
            rgb[:, c] = (rgb[:, c] - lo) / (hi - lo)
        else:
            rgb[:, c] = 0.5
    return rgb.astype(np.float32)


# ============================================================================
# Renderers: in-memory RGB arrays
# ============================================================================

def render_skeleton_tpose(parents: Sequence[int], positions,
                          elev: float = 30,
                          azim: float = -60,
                          rotate_root: bool = True,
                          title: Optional[str] = None,
                          figsize: Tuple[float, float] = (6, 6),
                          dpi: int = 120) -> np.ndarray:
    """Render a single skeleton pose as an (H, W, 3) uint8 image."""
    pts = _as_xyz(positions).reshape(-1, 3)
    pts = _maybe_rotate_y_up(pts, rotate_root)
    edges = _edges_from_parents(parents)
    bounds = _cubic_bounds(pts)

    fig, canvas, ax = _new_figure(bounds, elev, azim, figsize=figsize, dpi=dpi)
    _draw_static_skeleton(ax, pts, parents, edges)
    if title:
        fig.suptitle(title, fontsize=12, fontweight='bold', wrap=True, y=0.98)
        fig.subplots_adjust(top=0.90)
    return _capture_rgb(canvas)


def render_skeleton_tpose_facing(parents: Sequence[int], positions,
                                 face_joint_idxs: Optional[Sequence[int]] = None,
                                 forward=None,
                                 target_forward=None,
                                 elev: float = 30,
                                 azim: float = -60,
                                 rotate_root: bool = True,
                                 arrow_scale: float = 0.35,
                                 title: Optional[str] = None,
                                 extra_markers=None,
                                 figsize: Tuple[float, float] = (6, 6),
                                 dpi: int = 120) -> np.ndarray:
    """Rest pose with its facing pair highlighted, as an (H, W, 3) uint8 image.

    ``face_joint_idxs`` is ``[r_hip, l_hip]`` (drawn red / blue, joined by a
    dashed magenta line); ``forward`` is the facing direction those joints
    imply, drawn as a green arrow from the root, and ``target_forward`` (e.g.
    +Z after canonicalization) as a grey arrow. Arrows are ``arrow_scale``
    times the cubic-bound side. ``extra_markers`` is an optional list of
    ``(joint_indices, color, marker, label)`` tuples drawn on top (with a
    legend) — e.g. the Head / Tail End joints, to read the facing at a glance.
    Same camera as :func:`render_skeleton_tpose`, so the image lines up with
    the export-stage ``tpose/<rig>.png``.
    """
    pts = _as_xyz(positions).reshape(-1, 3)
    pts = _maybe_rotate_y_up(pts, rotate_root)
    edges = _edges_from_parents(parents)
    bounds = _cubic_bounds(pts)
    x_min, x_max, y_min, y_max, z_min, z_max = bounds
    arrow_len = max(x_max - x_min, y_max - y_min, z_max - z_min) * arrow_scale

    fig, canvas, ax = _new_figure(bounds, elev, azim, figsize=figsize, dpi=dpi)
    _draw_static_skeleton(ax, pts, parents, edges)

    if face_joint_idxs is not None and len(face_joint_idxs) >= 2 \
            and min(face_joint_idxs[:2]) >= 0:
        r, l = int(face_joint_idxs[0]), int(face_joint_idxs[1])
        for j, color in ((r, 'red'), (l, 'blue')):
            ax.scatter(pts[j:j + 1, 0], pts[j:j + 1, 1], pts[j:j + 1, 2],
                       s=90, c=color, edgecolors='black', linewidths=1.0,
                       depthshade=False, zorder=12)
        ax.plot([pts[r, 0], pts[l, 0]], [pts[r, 1], pts[l, 1]],
                [pts[r, 2], pts[l, 2]], color='magenta', linestyle='--',
                linewidth=1.5, zorder=11)

    root = pts[0]
    for vec, color, width in ((target_forward, '0.55', 3.0), (forward, 'green', 2.5)):
        if vec is None:
            continue
        d = np.asarray(vec, dtype=np.float32).reshape(3)
        n = float(np.linalg.norm(d))
        if n < 1e-8:
            continue
        d = _maybe_rotate_y_up(d / n, rotate_root)
        ax.quiver(root[0], root[1], root[2], d[0], d[1], d[2],
                  length=arrow_len, normalize=False, color=color,
                  linewidth=width, arrow_length_ratio=0.25, zorder=13)

    if extra_markers:
        drawn = False
        for idxs, color, marker, label in extra_markers:
            idxs = [int(i) for i in idxs if 0 <= int(i) < len(pts)]
            if not idxs:
                continue
            ax.scatter(pts[idxs, 0], pts[idxs, 1], pts[idxs, 2], s=110, c=color,
                       marker=marker, edgecolors='black', linewidths=0.8,
                       depthshade=False, zorder=14, label=label)
            drawn = True
        if drawn:
            handles, labels = ax.get_legend_handles_labels()
            fig.legend(handles, labels, loc='lower center', ncol=max(1, len(labels)),
                       fontsize=7, framealpha=0.8, bbox_to_anchor=(0.5, 0.012))

    if title:
        fig.suptitle(title, fontsize=9, wrap=True, y=0.99)
        fig.subplots_adjust(top=0.90)
    return _capture_rgb(canvas)


def render_skeleton_tpose_annotated(parents: Sequence[int], positions,
                                    joint_names: Sequence[str],
                                    elev: float = 30,
                                    azim: float = -60,
                                    rotate_root: bool = True,
                                    font_size: int = 6,
                                    joint_marker_size: float = 80.0,
                                    bone_linewidth: float = 2.0,
                                    figsize: Tuple[float, float] = (12, 12),
                                    dpi: int = 120) -> np.ndarray:
    """Render a T-pose with each joint labeled by name for manual inspection.

    Uses tight per-axis bounds + matching box_aspect so long-thin subjects
    (snakes, centipedes) fill the figure without distortion, leaving more
    room for labels.
    """
    pts = _as_xyz(positions).reshape(-1, 3)
    pts = _maybe_rotate_y_up(pts, rotate_root)
    assert len(joint_names) == pts.shape[0], (
        f"joint_names has {len(joint_names)} entries but positions has "
        f"{pts.shape[0]} joints"
    )
    edges = _edges_from_parents(parents)
    # Labels are placed at offsets up to ~0.24*scale from each joint (see
    # ``_annotate_joints``: base=0.08*scale × radii up to 3.0). Add 0.32*scale
    # of absolute padding so the label bboxes and leader endpoints stay inside
    # the plotted volume even on thin/long subjects (snakes, tentacles).
    bounds = _tight_bounds(pts, abs_pad_ratio=0.32)
    box_aspect = _box_aspect_from_bounds(bounds)

    _, canvas, ax = _new_figure(
        bounds, elev, azim,
        figsize=figsize, dpi=dpi, box_aspect=box_aspect,
    )

    # Drop the dashed grid + axis panes so labels stand out against a clean
    # background. Scoped to this renderer only — other plots keep their grids.
    ax.grid(False)
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.set_visible(False)
    ax.set_axis_off()

    segs = np.zeros((len(edges), 2, 3), dtype=np.float32)
    segs[:, 0] = pts[edges[:, 0]]
    segs[:, 1] = pts[edges[:, 1]]
    ax.add_collection3d(Line3DCollection(
        segs, colors='dimgray', linewidths=bone_linewidth, zorder=5,
    ))

    # Root drawn separately (red, larger) so skeleton origin is obvious.
    parents_arr = np.asarray(parents)
    non_root = parents_arr != -1
    root = ~non_root
    ax.scatter(pts[non_root, 0], pts[non_root, 1], pts[non_root, 2],
               s=joint_marker_size, c='steelblue',
               edgecolors='black', linewidths=0.8, depthshade=False, zorder=8)
    ax.scatter(pts[root, 0], pts[root, 1], pts[root, 2],
               s=joint_marker_size * 1.8, c='red',
               edgecolors='black', linewidths=1.2, depthshade=False, zorder=9)

    _annotate_joints(ax, pts, joint_names, bounds, font_size, parents=parents)
    return _capture_rgb(canvas)


def render_skeleton_tpos_spectral(parents: Sequence[int], positions,
                                  spectral_feats,
                                  elev: float = 30,
                                  azim: float = -60,
                                  rotate_root: bool = True,
                                  title: Optional[str] = None,
                                  figsize: Tuple[float, float] = (8, 8),
                                  dpi: int = 120) -> np.ndarray:
    """Render T-pose with joints PCA-colored from (J, K) spectral features."""
    pts = _as_xyz(positions).reshape(-1, 3)
    pts = _maybe_rotate_y_up(pts, rotate_root)
    spectral_feats = np.asarray(spectral_feats, dtype=np.float32)
    edges = _edges_from_parents(parents)
    bounds = _cubic_bounds(pts)

    fig, canvas, ax = _new_figure(bounds, elev, azim, figsize=figsize, dpi=dpi)
    colors = _spectral_to_rgb(spectral_feats)

    for child, parent in edges:
        ax.plot([pts[child, 0], pts[parent, 0]],
                [pts[child, 1], pts[parent, 1]],
                [pts[child, 2], pts[parent, 2]],
                color='black', linewidth=2.0, alpha=0.8, zorder=1)

    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2],
               c=colors, s=80, edgecolors='k', linewidths=0.5,
               zorder=2, depthshade=False)

    if title:
        fig.suptitle(title, fontsize=12, fontweight='bold', y=0.95)
    return _capture_rgb(canvas)


def render_skeleton_spectral_grid(parents: Sequence[int], positions,
                                  spectral_feats,
                                  elev: float = 30,
                                  azim: float = -60,
                                  rotate_root: bool = True,
                                  title: Optional[str] = None,
                                  figsize: Optional[Tuple[float, float]] = None,
                                  cell_size: float = 5.0,
                                  dpi: int = 120) -> np.ndarray:
    """Grid of skeleton subplots, one per eigenvector frequency (coolwarm).

    ``figsize`` overrides the full figure size; when ``None``, the figure is
    sized as ``(cell_size * ncols, cell_size * nrows)``.
    """
    pts = _as_xyz(positions).reshape(-1, 3)
    pts = _maybe_rotate_y_up(pts, rotate_root)
    spectral_feats = np.asarray(spectral_feats, dtype=np.float32)
    K = spectral_feats.shape[1]
    edges = _edges_from_parents(parents)
    bounds = _tight_bounds(pts, pad_ratio=0.10)
    x_min, x_max, y_min, y_max, z_min, z_max = bounds
    box_aspect = _box_aspect_from_bounds(bounds)

    ncols = min(K, 4)
    nrows = (K + ncols - 1) // ncols

    if figsize is None:
        figsize = (cell_size * ncols, cell_size * nrows)
    fig = Figure(figsize=figsize, dpi=dpi)
    canvas = FigureCanvasAgg(fig)

    top_margin = 0.04 if title else 0.01
    usable_h = 1.0 - top_margin

    for k in range(K):
        row, col = divmod(k, ncols)
        cell_w = 1.0 / ncols
        cell_h = usable_h / nrows
        x0 = col * cell_w
        y0 = 1.0 - top_margin - (row + 1) * cell_h
        ax = fig.add_axes([x0, y0, cell_w, cell_h], projection='3d',
                          computed_zorder=False)
        ax.set_xlim([x_min, x_max])
        ax.set_ylim([y_min, y_max])
        ax.set_zlim([z_min, z_max])
        ax.set_box_aspect(box_aspect)
        ax.set_xticklabels([]); ax.set_yticklabels([]); ax.set_zticklabels([])
        ax.xaxis.set_major_locator(LinearLocator(6))
        ax.yaxis.set_major_locator(LinearLocator(6))
        ax.zaxis.set_major_locator(LinearLocator(6))
        ax.tick_params(axis='both', length=0)
        ax.grid(True, color='gray', linestyle='--', linewidth=0.4)
        if elev is not None and azim is not None:
            ax.view_init(elev=elev, azim=azim)

        for child, parent in edges:
            ax.plot([pts[child, 0], pts[parent, 0]],
                    [pts[child, 1], pts[parent, 1]],
                    [pts[child, 2], pts[parent, 2]],
                    color='black', linewidth=1.5, alpha=0.9, zorder=1)

        vals = spectral_feats[:, k]
        vmax = max(np.abs(vals).max(), 1e-8)
        sc = ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2],
                        c=vals, cmap='coolwarm', vmin=-vmax, vmax=vmax,
                        s=50, edgecolors='k', linewidths=0.4,
                        zorder=2, depthshade=False)
        fig.colorbar(sc, ax=ax, shrink=0.5, pad=0.0, aspect=15)
        ax.set_title(f'Frequency {k + 1}', fontsize=11, fontweight='bold', pad=0)

    if title:
        fig.suptitle(title, fontsize=14, fontweight='bold',
                     y=1.0 - top_margin * 0.3)
    return _capture_rgb(canvas)


def render_skeleton_motion(parents: Sequence[int], positions,
                           elev: float = 30,
                           azim: float = -60,
                           rotate_root: bool = True,
                           title: Optional[str] = None,
                           figsize: Tuple[float, float] = (6, 6),
                           dpi: int = 120) -> np.ndarray:
    """Render a motion sequence as (T, H, W, 3). Reuses artists across frames."""
    pts = _as_xyz(positions)
    assert pts.ndim == 3, f"motion positions must be (T, J, D), got {pts.shape}"
    pts = _maybe_rotate_y_up(pts, rotate_root)
    T = pts.shape[0]

    edges = _edges_from_parents(parents)
    bounds = _cubic_bounds(pts)
    fig, canvas, ax = _new_figure(bounds, elev, azim, figsize=figsize, dpi=dpi)
    if title:
        _draw_caption(fig, title, figsize)

    joint_colors = np.where(np.asarray(parents) == -1, 'r', 'b')
    segs, lc, sc = _init_motion_artists(ax, edges, pts[0], joint_colors,
                                        joint_size=10.0)
    sc_root = ax.scatter(pts[0, 0:1, 0], pts[0, 0:1, 1], pts[0, 0:1, 2],
                         s=15, c='r')

    frames: List[np.ndarray] = []
    for t in range(T):
        P = pts[t]
        segs[:, 0] = P[edges[:, 0]]
        segs[:, 1] = P[edges[:, 1]]
        lc.set_segments(segs)
        sc._offsets3d = (P[:, 0], P[:, 1], P[:, 2])
        sc_root._offsets3d = (P[0:1, 0], P[0:1, 1], P[0:1, 2])
        frames.append(_capture_rgb(canvas))

    return np.stack(frames, axis=0)


def render_skeleton_motion_spectral(parents: Sequence[int], positions,
                                    spectral_feats,
                                    elev: float = 30,
                                    azim: float = -60,
                                    rotate_root: bool = True,
                                    title: Optional[str] = None,
                                    joint_marker_size: float = 30.0,
                                    bone_linewidth: float = 1.5,
                                    figsize: Tuple[float, float] = (6, 6),
                                    dpi: int = 120) -> np.ndarray:
    """Motion sequence with joints PCA-colored from spectral features (fixed across frames)."""
    pts = _as_xyz(positions)
    assert pts.ndim == 3, f"motion positions must be (T, J, D), got {pts.shape}"
    pts = _maybe_rotate_y_up(pts, rotate_root)
    T, J, _ = pts.shape
    spectral_feats = np.asarray(spectral_feats, dtype=np.float32)
    assert spectral_feats.shape[0] == J, (
        f"spectral_feats has {spectral_feats.shape[0]} joints but positions "
        f"has {J} joints"
    )

    edges = _edges_from_parents(parents)
    bounds = _cubic_bounds(pts)
    fig, canvas, ax = _new_figure(bounds, elev, azim, figsize=figsize, dpi=dpi)
    if title:
        _draw_caption(fig, title, figsize)

    colors = _spectral_to_rgb(spectral_feats)
    segs, lc, sc = _init_motion_artists(
        ax, edges, pts[0], colors,
        joint_size=joint_marker_size, bone_linewidth=bone_linewidth,
        joint_edgecolors='k', joint_edgewidth=0.4,
    )

    frames: List[np.ndarray] = []
    for t in range(T):
        P = pts[t]
        segs[:, 0] = P[edges[:, 0]]
        segs[:, 1] = P[edges[:, 1]]
        lc.set_segments(segs)
        sc._offsets3d = (P[:, 0], P[:, 1], P[:, 2])
        frames.append(_capture_rgb(canvas))

    return np.stack(frames, axis=0)


def render_skeleton_motion_directed(parents: Sequence[int], positions,
                                    directions,
                                    elev: float = 30,
                                    azim: float = -60,
                                    rotate_root: bool = True,
                                    arrow_scale: float = 0.25,
                                    arrow_color: str = 'green',
                                    figsize: Tuple[float, float] = (6, 6),
                                    dpi: int = 120) -> np.ndarray:
    """Motion sequence with a per-frame orientation arrow anchored at the root.

    ``directions`` is (3,) (constant) or (T, 3). Zero-length vectors render
    no arrow for that frame. ``arrow_scale`` is a fraction of the scene's
    cubic-bound side length.
    """
    pts = _as_xyz(positions)
    assert pts.ndim == 3, f"motion positions must be (T, J, D), got {pts.shape}"
    pts = _maybe_rotate_y_up(pts, rotate_root)
    T = pts.shape[0]

    dirs = np.asarray(directions, dtype=np.float32)
    if dirs.ndim == 1:
        dirs = np.broadcast_to(dirs, (T, 3)).copy()
    assert dirs.shape == (T, 3), (
        f"directions must be (3,) or (T, 3) with T={T}, got {dirs.shape}"
    )
    dirs = _maybe_rotate_y_up(dirs, rotate_root)
    norms = np.linalg.norm(dirs, axis=-1, keepdims=True)
    unit_dirs = np.where(norms > 1e-8, dirs / np.maximum(norms, 1e-8), 0.0)

    edges = _edges_from_parents(parents)
    bounds = _cubic_bounds(pts)
    x_min, x_max, y_min, y_max, z_min, z_max = bounds
    side = max(x_max - x_min, y_max - y_min, z_max - z_min)
    arrow_len = side * arrow_scale

    _, canvas, ax = _new_figure(bounds, elev, azim, figsize=figsize, dpi=dpi)

    joint_colors = np.where(np.asarray(parents) == -1, 'r', 'b')
    segs, lc, sc = _init_motion_artists(ax, edges, pts[0], joint_colors,
                                        joint_size=10.0)
    sc_root = ax.scatter(pts[0, 0:1, 0], pts[0, 0:1, 1], pts[0, 0:1, 2],
                         s=15, c='r')

    # 3D quiver can't be mutated in place; remove+re-add each frame.
    quiver = None
    frames: List[np.ndarray] = []
    for t in range(T):
        P = pts[t]
        segs[:, 0] = P[edges[:, 0]]
        segs[:, 1] = P[edges[:, 1]]
        lc.set_segments(segs)
        sc._offsets3d = (P[:, 0], P[:, 1], P[:, 2])
        sc_root._offsets3d = (P[0:1, 0], P[0:1, 1], P[0:1, 2])

        if quiver is not None:
            quiver.remove()
            quiver = None
        d = unit_dirs[t]
        if np.any(d != 0.0):
            root = P[0]
            quiver = ax.quiver(
                root[0], root[1], root[2],
                d[0], d[1], d[2],
                length=arrow_len, normalize=False,
                color=arrow_color, linewidth=2.0,
                arrow_length_ratio=0.25, zorder=11,
            )
        frames.append(_capture_rgb(canvas))

    return np.stack(frames, axis=0)


# ============================================================================
# Ground-plane motion render
# ============================================================================

# Two checker greys plus the background the floor dissolves into toward the
# horizon. Cool-tinted and more separated than a hairline pair: the tiles
# must read as a floor at 640 px without competing with the skeleton.
_CHECKER_COLORS = np.array([(0.910, 0.917, 0.930),
                            (0.822, 0.834, 0.860)], dtype=np.float32)
_GROUND_BG = np.array([1.0, 1.0, 1.0], dtype=np.float32)
_SHADOW_COLOR = (0.55, 0.57, 0.61)
_TRAIL_COLOR = (0.95, 0.49, 0.11)
_BONE_COLOR = (0.27, 0.29, 0.33)


def _ground_tiles(ax, cx: float, cy: float, azim: float, view_half: float,
                  tile: float, z: float,
                  fade_start: float = 1.0, fade_end: float = 4.5,
                  fade_subdiv: int = 2):
    """World-anchored whole checker tiles for the ground under the camera.

    Every tile is drawn complete, so the drawn set only changes by whole
    tiles entering/leaving as the camera glides — a rigid floor scrolling
    underfoot, with no edge boiling and no clipped slivers.

    Two things keep the floor from reading as a floating slab. Candidate
    tiles are culled by projecting their corners to pixels and keeping only
    those the frame actually shows, so the near floor stays solid into the
    bottom corners while the drawn set stays small. And the fade runs along
    the camera's view direction alone, from ``fade_start`` to ``fade_end``
    (multiples of ``view_half``): the floor dissolves into the background
    toward the horizon and nowhere else.

    A flat-shaded quad can only carry one colour, so a fade evaluated per
    tile quantises into tile-sized steps — and against the alternating
    checker those steps read as hard banding at the horizon. Each tile is
    therefore split into ``fade_subdiv`` x ``fade_subdiv`` sub-quads that
    share the tile's checker colour but fade individually, which divides the
    step size by ``fade_subdiv`` without touching the pattern — the fade then
    reads as a gradient across each tile rather than as a flat step between
    tiles. Only tiles inside the ramp are split (see below); 2 buys most of
    the smoothing at ~1.6x the frame cost, 3 is smoother again at ~2.5x.

    Returns ``(quads (N, 4, 3), colors (N, 3))``.
    """
    theta = np.deg2rad(azim)
    # Horizontal direction the camera looks along (it sits at ``azim``), and
    # its normal: "deeper into the scene" and "across it".
    u_hat = np.array([-np.cos(theta), -np.sin(theta)], dtype=np.float32)
    v_hat = np.array([-u_hat[1], u_hat[0]], dtype=np.float32)

    # Candidate window: the axis-aligned bounds of a ground wedge that
    # comfortably contains everything the camera can see out to fade_end.
    near, width, spread = 3.0, 3.8, 1.2
    far_width = width + spread * fade_end
    corners = np.array([(-near, -width), (-near, width),
                        (fade_end, -far_width), (fade_end, far_width)],
                       dtype=np.float32) * view_half
    world = (corners[:, :1] * u_hat + corners[:, 1:] * v_hat
             + np.array([cx, cy], dtype=np.float32))
    i0 = int(np.floor(world[:, 0].min() / tile))
    i1 = int(np.floor(world[:, 0].max() / tile))
    j0 = int(np.floor(world[:, 1].min() / tile))
    j1 = int(np.floor(world[:, 1].max() / tile))
    ii, jj = np.meshgrid(np.arange(i0, i1 + 1), np.arange(j0, j1 + 1),
                         indexing='ij')
    ii = ii.ravel()
    jj = jj.ravel()
    xa = (ii * tile).astype(np.float32)
    ya = (jj * tile).astype(np.float32)

    # Coarse cull first, in the wedge's own frame: it drops the ~half of the
    # index rectangle that lies outside the wedge for the price of two dot
    # products, so the exact (but pricier) pixel test below runs on far
    # fewer tiles.
    dx = xa + tile / 2 - cx
    dy = ya + tile / 2 - cy
    du = dx * u_hat[0] + dy * u_hat[1]
    dv = dx * v_hat[0] + dy * v_hat[1]
    inside = ((du > -near * view_half - tile) &
              (du < fade_end * view_half + tile) &
              (np.abs(dv) < width * view_half
               + spread * np.maximum(du, 0.0) + tile))
    ii, jj = ii[inside], jj[inside]
    xa, ya, du = xa[inside], ya[inside], du[inside]

    zc = np.full_like(xa, z)
    quads = np.stack([
        np.stack([xa, ya, zc], axis=1),
        np.stack([xa + tile, ya, zc], axis=1),
        np.stack([xa + tile, ya + tile, zc], axis=1),
        np.stack([xa, ya + tile, zc], axis=1),
    ], axis=1)
    # The pixel cull below runs on whole tiles — cheaper, and a sub-quad can
    # never fall outside its parent.

    # Keep a tile when any corner lands on (or just off) the canvas.
    flat = quads.reshape(-1, 3)
    sx, sy, _ = proj3d.proj_transform(flat[:, 0], flat[:, 1], flat[:, 2],
                                      ax.get_proj())
    scr = ax.transData.transform(np.column_stack([sx, sy]))
    scr = scr.reshape(-1, 4, 2)
    w, h = ax.figure.bbox.size
    pad = 0.05 * max(w, h)
    on_screen = ((scr[..., 0] > -pad) & (scr[..., 0] < w + pad) &
                 (scr[..., 1] > -pad) & (scr[..., 1] < h + pad)).any(axis=1)
    ii, jj, du = ii[on_screen], jj[on_screen], du[on_screen]
    quads = quads[on_screen]

    # Sub-divide, but only across the ramp: nearer than fade_start every
    # sub-quad would come out at f = 0 and beyond fade_end at f = 1, so
    # splitting there buys nothing and the extra polygons are drawn on every
    # frame, not just on a floor rebuild. One tile of margin covers the tiles
    # that straddle either end.
    n = max(1, int(fade_subdiv))
    if n > 1:
        margin = tile / max(view_half, 1e-9)
        dun = du / max(view_half, 1e-9)
        in_ramp = (dun > fade_start - margin) & (dun < fade_end + margin)
        if in_ramp.any():
            sub = tile / n
            off = np.arange(n, dtype=np.float32) * sub
            ox, oy = np.meshgrid(off, off, indexing='ij')
            ox, oy = ox.ravel(), oy.ravel()             # (n*n,)
            qr = quads[in_ramp]
            xs = (qr[:, 0, 0][:, None] + ox[None, :]).ravel()
            ys = (qr[:, 0, 1][:, None] + oy[None, :]).ravel()
            zs = np.full_like(xs, z)
            sub_quads = np.stack([
                np.stack([xs, ys, zs], axis=1),
                np.stack([xs + sub, ys, zs], axis=1),
                np.stack([xs + sub, ys + sub, zs], axis=1),
                np.stack([xs, ys + sub, zs], axis=1),
            ], axis=1)
            dx = xs + sub / 2 - cx
            dy = ys + sub / 2 - cy
            quads = np.concatenate([quads[~in_ramp], sub_quads], axis=0)
            du = np.concatenate([du[~in_ramp],
                                 dx * u_hat[0] + dy * u_hat[1]])
            ii = np.concatenate([ii[~in_ramp], np.repeat(ii[in_ramp], n * n)])
            jj = np.concatenate([jj[~in_ramp], np.repeat(jj[in_ramp], n * n)])

    base = _CHECKER_COLORS[(ii + jj) % 2]
    du = du / max(view_half, 1e-9)
    f = np.clip((du - fade_start) / max(fade_end - fade_start, 1e-9), 0.0, 1.0)
    f = (f * f * (3.0 - 2.0 * f))[:, None]              # smoothstep
    colors = base * (1.0 - f) + _GROUND_BG * f
    return quads, colors


def _smooth_path(path: np.ndarray, window: int) -> np.ndarray:
    """Moving-average smoothing with edge padding; kills gait sway so the
    follow camera glides instead of bobbing with each step."""
    if window <= 1 or len(path) < 3:
        return path
    kernel = np.ones(window, dtype=np.float32) / window
    pad = window // 2
    padded = np.pad(path, ((pad, window - 1 - pad), (0, 0)), mode='edge')
    return np.stack([np.convolve(padded[:, d], kernel, mode='valid')
                     for d in range(path.shape[1])], axis=1)


# The house sans stack, shared with the dataset-statistics figures: those
# import SANS_STACK from here for their ``font.sans-serif``, so a caption in a
# clip preview and a label in clip_frames_distribution.png are the same face by
# construction and cannot drift apart. Order and contents are the figures'
# original list, kept verbatim.
SANS_STACK = ['Helvetica', 'Arial', 'DejaVu Sans', 'Liberation Sans']


def _installed_family(preferred):
    """Keep only the faces this machine actually has, + the bundled fallback.

    Naming a missing family makes matplotlib emit one ``findfont: Font family
    ... not found`` warning per text draw — tens of thousands of lines over a
    dataset render — so the stack is resolved once, at import. DejaVu Sans
    ships with matplotlib itself, so the result is never empty.
    """
    try:
        from matplotlib import font_manager as fm
        have = {f.name for f in fm.fontManager.ttflist}
    except Exception:  # noqa: BLE001 — never fail an import over fonts
        have = set()
    stack = [f for f in preferred if f in have]
    if 'DejaVu Sans' not in stack:
        stack.append('DejaVu Sans')
    return stack


# Caption typography for the rendered clip previews and T-pose cards.
CAPTION_FONT_FAMILY = _installed_family(SANS_STACK)
CAPTION_FONT_SIZE = 13.0
# dist_plot.finish_figure titles at this weight. DejaVu Sans has no semibold
# face, so matplotlib resolves it to DejaVuSans-Bold.ttf — matching the family
# is not enough, the weight is what picks the actual font file.
CAPTION_FONT_WEIGHT = 'semibold'
CAPTION_COLOR = '#1F2937'          # dist_plot.INK
# Figure-fraction height one caption line costs the plot area, per point of
# type. Multiplied by the font size and the line count by _draw_caption.
_CAPTION_LINE_DROP = 0.0042


def _mean_advance_pt(font_pt: float, family=None) -> float:
    """Mean glyph advance of *family* at *font_pt*, in points.

    Measured from the real face rather than assumed, so the wrap width stays
    right when the caption font or size changes (DejaVu averages ~0.545 em,
    Nimbus Sans ~0.50 — hard-coding either mis-wraps the other).
    """
    probe = ('abcdefghijklmnopqrstuvwxyz '
             'ABCDEFGHIJKLMNOPQRSTUVWXYZ etaoinshrdlu')
    try:
        from matplotlib.font_manager import FontProperties
        from matplotlib.textpath import TextPath
        fp = FontProperties(family=family or CAPTION_FONT_FAMILY, size=font_pt,
                            weight=CAPTION_FONT_WEIGHT)
        width = TextPath((0, 0), probe, prop=fp).get_extents().width
        if width > 0:
            return float(width) / len(probe)
    except Exception:  # noqa: BLE001 — fall back rather than fail a render
        pass
    return 0.545 * font_pt


def _wrap_title(text: str, fig_width_in: float, font_pt: float,
                family=None, fill: float = 0.92) -> str:
    """Hard-wrap a caption to the figure width (suptitle's own wrapping is
    unreliable and long captions ran off both edges).

    ``fill`` is the share of the figure width a full line may occupy.
    """
    chars = max(24, int(fig_width_in * 72.0 * fill / _mean_advance_pt(font_pt, family)))
    return '\n'.join(textwrap.wrap(text, chars)) or text


def _draw_caption(fig, text: str, figsize, font_pt: float = CAPTION_FONT_SIZE,
                  top_base: float = 0.94) -> None:
    """Draw a wrapped clip caption across the top of *fig* and make room for it.

    Plain label type, not a heading: a caption is there to be read off, not to
    be the loudest thing in the frame. The plot area gives up height in
    proportion to the type size and the number of wrapped lines, so a long
    caption never lands on top of the subject.
    """
    wrapped = _wrap_title(text, figsize[0], font_pt, CAPTION_FONT_FAMILY)
    n_lines = wrapped.count('\n') + 1
    fig.text(0.5, 0.988, wrapped, ha='center', va='top',
             color=CAPTION_COLOR, fontsize=font_pt,
             fontfamily=CAPTION_FONT_FAMILY, fontweight=CAPTION_FONT_WEIGHT)
    fig.subplots_adjust(top=top_base - _CAPTION_LINE_DROP * font_pt * n_lines)


def render_skeleton_motion_ground(parents: Sequence[int], positions,
                                  spectral_feats=None,
                                  elev: float = 20,
                                  azim: float = -60,
                                  rotate_root: bool = True,
                                  follow: bool = True,
                                  title: Optional[str] = None,
                                  title_font_size: float = CAPTION_FONT_SIZE,
                                  joint_marker_size: float = 18.0,
                                  bone_linewidth: float = 1.9,
                                  shadow: bool = True,
                                  trail: bool = True,
                                  view_margin: float = 1.0,
                                  figsize: Tuple[float, float] = (6, 6),
                                  dpi: int = 120) -> np.ndarray:
    """Motion render on a checkerboard floor, with a camera that follows the
    subject, a contact shadow and a fading root trail.

    The floor sits at the motion's lowest point and is world-anchored and
    unbounded (tiles are rebuilt around the camera each frame and fade out
    toward the horizon), so travel reads off the tiles scrolling underfoot
    rather than off a slab edge sweeping past. The view window is sized to
    the subject, not to the trajectory, so a long walk does not shrink the
    character to a speck.

    ``shadow`` drops the skeleton straight down onto the floor, which is what
    makes ground contact (and its absence, in a jump) legible. ``trail``
    draws the path the root has travelled so far, fading with age. ``title``
    is drawn as a plain black caption (hard-wrapped to the figure width) in
    the same label type the annotated T-pose renderer uses. Joints use
    the spectral PCA palette when ``spectral_feats`` is given, else the
    default root-red / joint-blue scheme.
    """
    pts = _as_xyz(positions)
    assert pts.ndim == 3, f"motion positions must be (T, J, D), got {pts.shape}"
    pts = _maybe_rotate_y_up(pts, rotate_root)
    T, J, _ = pts.shape

    edges = _edges_from_parents(parents)
    ground_z = float(pts[..., 2].min())
    root_xy = pts[:, 0, :2].astype(np.float32)                # (T, 2)

    # Window half-extent from the subject's own size, not the trajectory.
    radius_xy = float(np.abs(pts[..., :2] - root_xy[:, None, :]).max())
    height = float(pts[..., 2].max() - ground_z)
    # ``view_margin`` > 1 pulls the camera back. The default frames a moving
    # subject tightly; a still pose is read for its shape, and a rest pose
    # that spreads its limbs along the ground (or rears a tail overhead)
    # wants the extra room so nothing touches the frame edge.
    half = max(radius_xy * 1.05, height * 0.50, 1e-3) * max(view_margin, 1e-3)

    # Coordinated motion: the camera GLIDES after the subject (smoothed root
    # path) while the floor stays fixed in the WORLD, so subject, floor and
    # camera all move together and world travel reads off the tiles sliding
    # through the frame plus the trail.
    traj_center = 0.5 * (root_xy.max(axis=0) + root_xy.min(axis=0))
    traj_half = float((root_xy.max(axis=0) - root_xy.min(axis=0)).max()) / 2.0
    view_half = half if follow else max(traj_half + half, half)
    centers = _smooth_path(root_xy, window=max(5, T // 12)) if follow \
        else np.repeat(traj_center[None, :], T, axis=0)
    center0 = centers[0]

    # Vertical extent hugs the subject (tall bipeds and flat crawlers both
    # fill the frame); the box aspect mirrors the ranges to avoid distortion.
    z_top = ground_z + max(height * 1.15, view_half * 0.3)
    bounds = (center0[0] - view_half, center0[0] + view_half,
              center0[1] - view_half, center0[1] + view_half,
              ground_z, z_top)
    fig, canvas, ax = _new_figure(
        bounds, elev, azim, figsize=figsize, dpi=dpi,
        box_aspect=(1.0, 1.0, (z_top - ground_z) / (2 * view_half)))
    ax.set_axis_off()
    # Fill the canvas: with the axes cage hidden there is nothing to clip,
    # so spill the 3D box past the figure edges. The bottom stays the most
    # conservative side — the subject's lowest joint sits exactly on the
    # lower z limit, so cropping there would cut its feet.
    fig.subplots_adjust(left=-0.26, right=1.26, top=1.24, bottom=-0.18)
    if title:
        _draw_caption(fig, title, figsize, title_font_size, top_base=1.15)

    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    # Tile size from BOTH body dimensions — the geometric mean of height and
    # horizontal span: an upright biped and a sprawling low body each get
    # tiles proportioned to their bulk, and neither dimension alone can blow
    # the scale up. The view-based clamp keeps 6-10 tiles across the window,
    # which is what makes travel legible without turning the floor into a
    # busy grid.
    body_span = 2.0 * radius_xy
    tile = float(np.clip(0.35 * np.sqrt(max(height * body_span, 1e-12)),
                         view_half / 5.0, view_half / 3.0))
    quads, cols = _ground_tiles(ax, *center0, azim, view_half, tile,
                                ground_z)
    # Stroking each tile in its own fill colour closes the hairline seams
    # matplotlib leaves between adjacent quads (they show up as white pin
    # dots where four tiles meet).
    platform = Poly3DCollection(quads, facecolors=cols, edgecolors=cols,
                                linewidths=0.6, zorder=0)
    ax.add_collection3d(platform)
    # Rebuilding the floor is the single most expensive thing per frame, and
    # the tiles are world-anchored: nothing about them changes until the
    # camera has moved. Rebuild only after a fraction of a tile of travel
    # (an in-place motion then builds the floor exactly once); the cull pad
    # covers the tiles that enter during the interval.
    rebuild_at = np.array(center0, dtype=np.float32)
    rebuild_step = 0.2 * tile

    if spectral_feats is not None:
        spectral_feats = np.asarray(spectral_feats, dtype=np.float32)
        assert spectral_feats.shape[0] == J
        joint_colors = _spectral_to_rgb(spectral_feats)
        edge_kwargs = dict(joint_edgecolors='0.15', joint_edgewidth=0.4)
    else:
        joint_colors = np.where(np.asarray(parents) == -1, 'r', 'b')
        edge_kwargs = {}

    # Ground-level artists sit just above the tiles to avoid z-fighting.
    eps = max(view_half * 4e-3, 1e-6)
    shadow_lc = None
    if shadow:
        # A straight-down projection of the bones. Cheap, but it is what
        # makes foot contact — and its absence during a jump — legible.
        shadow_lc = Line3DCollection(np.zeros((len(edges), 2, 3), np.float32),
                                     colors=[_SHADOW_COLOR],
                                     linewidths=bone_linewidth * 1.5,
                                     alpha=0.28, zorder=1)
        ax.add_collection3d(shadow_lc)

    trail_lc = None
    if trail:
        # Age-faded so the head of the path (where the subject is now) is the
        # strongest mark and the tail recedes instead of ringing the floor.
        trail_lc = Line3DCollection(np.zeros((0, 2, 3), np.float32),
                                    linewidths=1.6, zorder=2)
        ax.add_collection3d(trail_lc)

    segs, lc, sc = _init_motion_artists(
        ax, edges, pts[0], joint_colors,
        joint_size=joint_marker_size, bone_color=_BONE_COLOR,
        bone_linewidth=bone_linewidth, **edge_kwargs)
    lc.set_zorder(3)
    sc.set_zorder(4)
    # The root must never be occluded: inside a single scatter, points paint
    # in array order, so joints drawn after index 0 cover it. Re-draw the
    # root as its own topmost scatter (same size; red in the default
    # palette, its own spectral color otherwise).
    root_color = ([joint_colors[0]] if spectral_feats is not None else 'r')
    sc_root = ax.scatter(pts[0, 0:1, 0], pts[0, 0:1, 1], pts[0, 0:1, 2],
                         s=joint_marker_size, c=root_color,
                         depthshade=False, zorder=5, **(
                             dict(edgecolors='0.15', linewidths=0.4)
                             if spectral_feats is not None else {}))

    shadow_segs = np.zeros((len(edges), 2, 3), np.float32)
    shadow_segs[..., 2] = ground_z + eps
    trail_rgba = np.zeros((max(T - 1, 1), 4), np.float32)
    trail_rgba[:, :3] = _TRAIL_COLOR

    frames: List[np.ndarray] = []
    for t in range(T):
        P = pts[t]
        segs[:, 0] = P[edges[:, 0]]
        segs[:, 1] = P[edges[:, 1]]
        lc.set_segments(segs)
        sc._offsets3d = (P[:, 0], P[:, 1], P[:, 2])
        sc_root._offsets3d = (P[0:1, 0], P[0:1, 1], P[0:1, 2])
        cx, cy = centers[t]

        if shadow_lc is not None:
            shadow_segs[..., :2] = segs[..., :2]
            shadow_lc.set_segments(shadow_segs)

        if trail_lc is not None and t >= 1:
            path = root_xy[:t + 1]
            # Drop the part of the path that has left the faded floor, so the
            # trail never hangs in empty white space.
            keep = np.max(np.abs(path - np.array([cx, cy], np.float32)),
                          axis=1) < 1.6 * view_half
            keep = keep[:-1] & keep[1:]
            seg = np.empty((t, 2, 3), np.float32)
            seg[:, 0, :2] = path[:-1]
            seg[:, 1, :2] = path[1:]
            seg[..., 2] = ground_z + eps
            rgba = trail_rgba[:t].copy()
            rgba[:, 3] = np.linspace(0.04, 0.7, t) * keep
            trail_lc.set_segments(seg)
            trail_lc.set_color(rgba)

        # The limits move the camera, and the tile cull projects through
        # them, so pan first and rebuild the floor after.
        if follow:
            ax.set_xlim(cx - view_half, cx + view_half)
            ax.set_ylim(cy - view_half, cy + view_half)
        if t == 0 or max(abs(cx - rebuild_at[0]),
                         abs(cy - rebuild_at[1])) > rebuild_step:
            quads, cols = _ground_tiles(ax, cx, cy, azim, view_half, tile,
                                        ground_z)
            platform.set_verts(quads)
            platform.set_facecolor(cols)
            platform.set_edgecolor(cols)
            rebuild_at = np.array([cx, cy], dtype=np.float32)
        frames.append(_capture_rgb(canvas))

    return np.stack(frames, axis=0)


# ============================================================================
# Savers: write PNG / MP4 to disk
# ============================================================================

def _write_image(save_path: str, frame: np.ndarray) -> None:
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    imageio.imwrite(save_path, frame)


def _write_video(save_path: str, frames: np.ndarray, fps: int) -> None:
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    imageio.mimwrite(save_path, frames, fps=fps)


# Defaults (elev=30, azim=-60, rotate_root=True) match the pipeline convention.

def save_skeleton_tpose(save_path: str, parents: Sequence[int], positions,
                        elev: float = 30, azim: float = -60,
                        rotate_root: bool = True,
                        figsize: Tuple[float, float] = (6, 6),
                        dpi: int = 120) -> None:
    """Render a T-pose and save as PNG."""
    _write_image(save_path, render_skeleton_tpose(
        parents, positions, elev=elev, azim=azim, rotate_root=rotate_root,
        figsize=figsize, dpi=dpi,
    ))


def save_skeleton_tpose_annotated(save_path: str, parents: Sequence[int],
                                  positions, joint_names: Sequence[str],
                                  elev: float = 30, azim: float = -60,
                                  rotate_root: bool = True,
                                  font_size: int = 6,
                                  joint_marker_size: float = 80.0,
                                  bone_linewidth: float = 2.0,
                                  figsize: Tuple[float, float] = (12, 12),
                                  dpi: int = 120) -> None:
    """Render an annotated T-pose and save as PNG."""
    _write_image(save_path, render_skeleton_tpose_annotated(
        parents, positions, joint_names,
        elev=elev, azim=azim, rotate_root=rotate_root, font_size=font_size,
        joint_marker_size=joint_marker_size, bone_linewidth=bone_linewidth,
        figsize=figsize, dpi=dpi,
    ))


def save_skeleton_tpos_spectral(save_path: str, parents: Sequence[int], positions,
                                spectral_feats,
                                elev: float = 30, azim: float = -60,
                                rotate_root: bool = True,
                                title: Optional[str] = None,
                                figsize: Tuple[float, float] = (8, 8),
                                dpi: int = 120) -> None:
    """Render spectral-colored T-pose and save as PNG."""
    _write_image(save_path, render_skeleton_tpos_spectral(
        parents, positions, spectral_feats,
        elev=elev, azim=azim, rotate_root=rotate_root, title=title,
        figsize=figsize, dpi=dpi,
    ))


def save_skeleton_spectral_grid(save_path: str, parents: Sequence[int],
                                positions, spectral_feats,
                                elev: float = 30, azim: float = -60,
                                rotate_root: bool = True,
                                title: Optional[str] = None,
                                figsize: Optional[Tuple[float, float]] = None,
                                cell_size: float = 5.0,
                                dpi: int = 120) -> None:
    """Render per-frequency spectral grid and save as PNG."""
    _write_image(save_path, render_skeleton_spectral_grid(
        parents, positions, spectral_feats,
        elev=elev, azim=azim, rotate_root=rotate_root, title=title,
        figsize=figsize, cell_size=cell_size, dpi=dpi,
    ))


def save_skeleton_motion(save_path: str, parents: Sequence[int], positions,
                         fps: int = 30, elev: float = 30, azim: float = -60,
                         rotate_root: bool = True,
                         title: Optional[str] = None,
                         figsize: Tuple[float, float] = (6, 6),
                         dpi: int = 120) -> None:
    """Render a motion sequence and save as MP4."""
    frames = render_skeleton_motion(
        parents, positions, elev=elev, azim=azim, rotate_root=rotate_root,
        title=title, figsize=figsize, dpi=dpi,
    )
    _write_video(save_path, frames, fps)


def save_skeleton_motion_spectral(save_path: str, parents: Sequence[int],
                                  positions, spectral_feats,
                                  fps: int = 30, elev: float = 30,
                                  azim: float = -60,
                                  rotate_root: bool = True,
                                  title: Optional[str] = None,
                                  figsize: Tuple[float, float] = (6, 6),
                                  dpi: int = 120) -> None:
    """Render a spectral-colored motion sequence and save as MP4."""
    frames = render_skeleton_motion_spectral(
        parents, positions, spectral_feats,
        elev=elev, azim=azim, rotate_root=rotate_root, title=title,
        figsize=figsize, dpi=dpi,
    )
    _write_video(save_path, frames, fps)


def render_skeleton_tpose_ground(parents: Sequence[int], positions,
                                 spectral_feats=None,
                                 elev: float = 20,
                                 azim: float = -60,
                                 rotate_root: bool = True,
                                 title: Optional[str] = None,
                                 title_font_size: float = CAPTION_FONT_SIZE,
                                 joint_marker_size: float = 18.0,
                                 bone_linewidth: float = 1.9,
                                 shadow: bool = True,
                                 view_margin: float = 1.25,
                                 figsize: Tuple[float, float] = (6, 6),
                                 dpi: int = 120) -> np.ndarray:
    """A single rest pose on the clip previews' checkerboard floor.

    Same framing, floor, contact shadow, spectral palette and caption type as
    :func:`render_skeleton_motion_ground`, so an object's T-pose card and its
    clip previews read as one set. Rendered as a one-frame motion with the
    camera parked (``follow=False``) and no root trail — there is nothing
    moving to follow or to trace.

    ``positions`` is one pose, ``(J, 3)``; a ``(1, J, 3)`` sequence is
    accepted too. ``title`` is normally the object type.
    """
    pts = _as_xyz(positions)
    if pts.ndim == 2:
        pts = pts[None]
    assert pts.ndim == 3 and pts.shape[0] == 1, (
        f"a T-pose is one frame, got {pts.shape}")
    return render_skeleton_motion_ground(
        parents, pts, spectral_feats=spectral_feats, elev=elev, azim=azim,
        rotate_root=rotate_root, follow=False, title=title,
        title_font_size=title_font_size,
        joint_marker_size=joint_marker_size, bone_linewidth=bone_linewidth,
        shadow=shadow, trail=False, view_margin=view_margin,
        figsize=figsize, dpi=dpi)[0]


def save_skeleton_tpose_ground(save_path: str, parents: Sequence[int],
                               positions, spectral_feats=None,
                               elev: float = 20, azim: float = -60,
                               rotate_root: bool = True,
                               title: Optional[str] = None,
                               title_font_size: float = CAPTION_FONT_SIZE,
                               shadow: bool = True,
                               view_margin: float = 1.25,
                               figsize: Tuple[float, float] = (6, 6),
                               dpi: int = 120) -> None:
    """Render and save a ground-plane T-pose PNG (see
    :func:`render_skeleton_tpose_ground`)."""
    _write_image(save_path, render_skeleton_tpose_ground(
        parents, positions, spectral_feats=spectral_feats, elev=elev,
        azim=azim, rotate_root=rotate_root, title=title,
        title_font_size=title_font_size, shadow=shadow,
        view_margin=view_margin, figsize=figsize, dpi=dpi))


def save_skeleton_motion_ground(save_path: str, parents: Sequence[int],
                                positions, spectral_feats=None,
                                fps: int = 20, elev: float = 20,
                                azim: float = -60, rotate_root: bool = True,
                                follow: bool = True,
                                title: Optional[str] = None,
                                title_font_size: float = CAPTION_FONT_SIZE,
                                shadow: bool = True, trail: bool = True,
                                view_margin: float = 1.0,
                                figsize: Tuple[float, float] = (6, 6),
                                dpi: int = 120) -> None:
    """Render and save a ground-plane motion MP4 (see
    :func:`render_skeleton_motion_ground`)."""
    frames = render_skeleton_motion_ground(
        parents, positions, spectral_feats=spectral_feats, elev=elev,
        azim=azim, rotate_root=rotate_root, follow=follow, title=title,
        title_font_size=title_font_size, shadow=shadow, trail=trail,
        view_margin=view_margin, figsize=figsize, dpi=dpi)
    _write_video(save_path, frames, fps)
