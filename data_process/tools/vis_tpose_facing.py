"""Compare an exported rest pose with its stage-4 facing canonicalization.

The export stage saves every skeleton's rest pose as ``export/<ds>/tpose/<rig>.png``
in the asset's own frame. Stage 4 (``feature_extraction/extract_features.py`` via
``utils/motion_features.process_tpose``) rotates that rest pose so the facing
direction implied by the face-joint pair points to +Z — ``forward = Y x (r_hip -
l_hip)``, or the body axis for ``body_axis`` rigs — then centres, scales and
grounds it. This tool runs exactly that canonicalization on the rest pose and
saves one comparison image per skeleton, same camera as ``tpose/<rig>.png``:

    left  — the original rest pose (export frame) with the face joints marked
    right — the same pose after the face-pair correction (facing +Z)

``r_hip`` is red, ``l_hip`` blue (joined by a dashed line), the pair-derived
forward is a green arrow and the grey arrow on the right panel is +Z, where the
green one must end up. The title carries the pair's raw names, source and the
pair's lateral (XZ) separation in the rest pose and at the first animation
frame. Flags: ``COINCIDENT`` — the two joints sit on the midline in both poses,
so the pair yields a meaningless facing (pick another pair);
``REST-DEGENERATE`` — lateral at frame 0 but not in the rest pose, i.e. the
exported rest pose lies on its side and the stage-4 T-pose facing is unreliable
even though the pair is fine; ``UNRESOLVED`` — a named joint is missing from the
skeleton. A ``facing_summary.tsv`` with the same numbers is written next to the
PNGs.

The pair comes from ``<export_dir>/face_joint_names.json`` (the sidecar
``patch_annotations.py`` maintains); pass ``--face_json`` to preview another
sidecar or a ``dataset/UniML3D/patches/<ds>_face_pairs.json`` override file
before applying it.

Usage:
    python -m data_process.tools.vis_tpose_facing --export_dir dataset/export/objaverse
    python -m data_process.tools.vis_tpose_facing --export_dir dataset/export/objaverse --rigs RIG1 RIG2
    python -m data_process.tools.vis_tpose_facing --export_dir dataset/export/truebones \\
        --face_json dataset/UniML3D/patches/truebones_face_pairs.json --output_dir outputs/tmp

See ``data_process/scripts/run_joints_vis_facing.sh`` for the wrapper.
"""

import argparse
import csv
import json
import multiprocessing as mp
import os
import sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from Animation import Animation, Quaternions, positions_global  # noqa: E402
from data_process.utils.motion_features import (  # noqa: E402
    DegenerateSkeletonError, process_tpose, resolve_face_joint_idxs,
)
from data_process.utils.plotting import render_skeleton_tpose_facing  # noqa: E402


DEFAULT_OUTPUT_DIR = 'outputs/tpose_facing_vis'
COINCIDENT_SEP = 0.03   # pair separation (fraction of the skeleton extent) below which facing is noise
PANEL_FIGSIZE = (6, 6)  # matches export/<ds>/tpose/<rig>.png (save_skeleton_tpose defaults)
PANEL_DPI = 120


# ---------------------------------------------------------------------------
# Face-pair loading
# ---------------------------------------------------------------------------

def load_face_pairs(path):
    """Load ``face_joint_names.json`` (sidecar) or a ``*_face_pairs.json`` patch.

    Both are normalised to the sidecar layout
    ``{rig: {"r_hip": {"raw", "clean"}, "l_hip": {...}, "source", "body_axis"}}``;
    keys starting with ``_`` (``_comment``) are skipped.
    """
    with open(path) as f:
        data = json.load(f)
    out = {}
    for rig, entry in data.items():
        if rig.startswith('_') or not isinstance(entry, dict):
            continue
        norm = {}
        for key in ('r_hip', 'l_hip'):
            val = entry.get(key, '')
            if isinstance(val, dict):
                norm[key] = {'raw': val.get('raw', ''), 'clean': val.get('clean', '')}
            else:
                norm[key] = {'raw': val or '', 'clean': ''}
        norm['source'] = entry.get('source', '')
        norm['body_axis'] = bool(entry.get('body_axis', False))
        out[rig] = norm
    return out


def index_motion_npzs(motions_dir):
    """First NPZ per skeleton key (the prefix before the first ``-``), from one
    directory scan — any clip carries the rest pose. Mixamo shares one skeleton
    across un-prefixed clips, so its ``mixamo`` key maps to the first NPZ.
    """
    index = {}
    for p in sorted(Path(motions_dir).glob('*.npz')):
        key = p.stem.split('-', 1)[0] if '-' in p.stem else p.stem
        index.setdefault(key, p)
        index.setdefault('mixamo', p)
    return index


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

def rest_positions(data):
    """Global rest-pose joint positions (J, 3) of an export NPZ, in export order."""
    parents = np.asarray(data['parents']).astype(int)
    anim = Animation(
        rotations=Quaternions(data['rest_local_rot'][None]),
        positions=data['rest_local_pos'][None],
        orients=Quaternions.id(len(parents)),
        offsets=data['rest_local_pos'],
        parents=parents,
    )
    return positions_global(anim)[0].astype(np.float32), parents


def first_frame_positions(data):
    """Global joint positions (J, 3) at the first animation frame of an export NPZ."""
    parents = np.asarray(data['parents']).astype(int)
    anim = Animation(
        rotations=Quaternions(data['anim_local_rot'][:1]),
        positions=data['anim_local_pos'][:1],
        orients=Quaternions.id(len(parents)),
        offsets=data['rest_local_pos'],
        parents=parents,
    )
    return positions_global(anim)[0].astype(np.float32)


def pair_forward(positions, face_joint_idxs, body_axis):
    """Facing direction the pair implies (same math as ``skeleton.get_root_facing_quat``).

    Returns ``(forward (3,), lateral_separation)`` where the separation is the
    horizontal distance between the two joints as a fraction of the skeleton's
    largest extent; ``forward`` is ``None`` for the ``[-1, -1]`` sentinel.
    """
    if face_joint_idxs is None or face_joint_idxs[0] < 0 or face_joint_idxs[1] < 0:
        return None, 0.0
    r, l = face_joint_idxs[:2]
    across = positions[r] - positions[l]
    extent = float(np.ptp(positions, axis=0).max()) or 1.0
    sep = float(np.linalg.norm(across * np.array([1.0, 0.0, 1.0]))) / extent
    n = float(np.linalg.norm(across))
    if n < 1e-9:
        return np.zeros(3, dtype=np.float32), sep
    across = across / n
    forward = np.cross(np.array([0.0, 1.0, 0.0]), across)
    n = float(np.linalg.norm(forward))
    forward = forward / n if n > 1e-9 else np.zeros(3)
    if body_axis:
        rot = Quaternions.from_euler(np.array([0.0, -np.pi / 2, 0.0]), 'xyz')
        forward = np.asarray(rot * forward[None])[0]
    return forward.astype(np.float32), sep


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def _fit_height(img, height):
    """Nearest-neighbour resample *img* to *height* rows (keeps aspect)."""
    if img.shape[0] == height:
        return img
    width = max(1, int(round(img.shape[1] * height / img.shape[0])))
    rows = (np.arange(height) * img.shape[0] / height).astype(int)
    cols = (np.arange(width) * img.shape[1] / width).astype(int)
    return img[rows][:, cols]


def _blank_panel(shape, text):
    """Light-grey panel with a caption, for a missing export PNG."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    fig = Figure(figsize=PANEL_FIGSIZE, dpi=PANEL_DPI)
    canvas = FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)
    ax.set_axis_off()
    ax.text(0.5, 0.5, text, ha='center', va='center', fontsize=10, wrap=True)
    canvas.draw()
    w, h = canvas.get_width_height()
    rgb = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)[..., :3]
    return _fit_height(rgb.copy(), shape[0])


MARKER_STYLES = (('orange', '*'), ('purple', 'v'), ('cyan', 's'), ('yellow', 'D'), ('lime', 'P'))


def label_markers(labels, mark_labels):
    """``extra_markers`` for :func:`render_skeleton_tpose_facing`: one style per
    requested clean label, joints found by exact label match (``labels`` is
    aligned with the rig's joint order)."""
    if not labels or not mark_labels:
        return None
    out = []
    for k, lab in enumerate(mark_labels):
        idxs = [i for i, l in enumerate(labels) if l == lab]
        if idxs:
            color, marker = MARKER_STYLES[k % len(MARKER_STYLES)]
            out.append((idxs, color, marker, lab))
    return out or None


def render_rig(rig, npz_path, entry, output_path, labels=None, mark_labels=()):
    """Render the original | corrected comparison for one skeleton; returns the summary row.

    ``labels`` are the rig's clean joint labels aligned with the NPZ joint order
    (optional); joints whose label is in ``mark_labels`` get an extra marker.
    """
    data = np.load(npz_path, allow_pickle=True)
    names = [str(n) for n in data['names']]
    if labels is not None and len(labels) != len(names):
        labels = None
    raw_pos, raw_parents = rest_positions(data)
    raw_idxs, body_axis = resolve_face_joint_idxs(entry, names)
    fwd_raw, sep = pair_forward(raw_pos, raw_idxs, body_axis)
    # the same pair at the first animation frame: a pair that is lateral there
    # but not in the rest pose means the rest pose lies on its side, not that
    # the joints coincide
    f0_pos = first_frame_positions(data)
    _, sep_f0 = pair_forward(f0_pos, raw_idxs, body_axis)

    r_raw, l_raw = entry['r_hip']['raw'], entry['l_hip']['raw']
    flag = ''
    if raw_idxs == [-1, -1]:
        flag = 'UNRESOLVED' if (r_raw or l_raw) else 'EMPTY'
    elif sep < COINCIDENT_SEP and sep_f0 < COINCIDENT_SEP:
        flag = 'COINCIDENT'
    elif sep < COINCIDENT_SEP:
        flag = 'REST-DEGENERATE'
    fx, fz = (float(fwd_raw[0]), float(fwd_raw[2])) if fwd_raw is not None else (float('nan'),) * 2
    pair_txt = 'r_hip={} ({})   l_hip={} ({})   src={}{}'.format(
        r_raw or '-', entry['r_hip']['clean'] or '?', l_raw or '-',
        entry['l_hip']['clean'] or '?', entry['source'] or '-',
        '  body_axis' if body_axis else '')
    info = 'forward=({:+.2f}, {:+.2f})  pair sep rest={:.3f} frame0={:.3f} {}'.format(
        fx, fz, sep, sep_f0, flag).rstrip()

    panel_raw = render_skeleton_tpose_facing(
        raw_parents, raw_pos, raw_idxs, fwd_raw,
        title='{}\nORIGINAL rest pose (export frame)\n{}\n{}'.format(rig, pair_txt, info),
        extra_markers=label_markers(labels, mark_labels),
        figsize=PANEL_FIGSIZE, dpi=PANEL_DPI)

    try:
        (tpos_anim, _, _, _, parents_c, names_c, _, _,
         idxs_c, body_axis_c) = process_tpose(data, face_joints=entry)
        pos_c = positions_global(tpos_anim)[0].astype(np.float32)
        fwd_c, _ = pair_forward(pos_c, idxs_c, body_axis_c)
        cz = float(fwd_c[2]) if fwd_c is not None else float('nan')
        # process_tpose reorders joints (BFS): carry the labels over by name
        labels_c = ([labels[names.index(n)] for n in names_c]
                    if labels is not None and all(n in names for n in names_c) else None)
        panel_can = render_skeleton_tpose_facing(
            parents_c, pos_c, idxs_c, fwd_c, target_forward=np.array([0.0, 0.0, 1.0]),
            title='{}\nCORRECTED by face pair (stage-4: facing -> +Z, centred, scaled, grounded)\n'
                  '{}\nforward.z after rotation = {:+.2f} (grey arrow = +Z)'.format(rig, pair_txt, cz),
            extra_markers=label_markers(labels_c, mark_labels),
            figsize=PANEL_FIGSIZE, dpi=PANEL_DPI)
        canon_z = cz
    except DegenerateSkeletonError as e:
        panel_can = _blank_panel(panel_raw.shape, 'degenerate skeleton:\n{}'.format(e))
        canon_z = float('nan')
        flag = (flag + ' DEGENERATE').strip()

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    imageio.imwrite(output_path, np.concatenate([panel_raw, panel_can], axis=1))
    return {'rig': rig, 'r_hip': r_raw, 'l_hip': l_raw, 'source': entry['source'],
            'body_axis': int(body_axis), 'forward_x': '{:.3f}'.format(fx),
            'forward_z': '{:.3f}'.format(fz), 'pair_sep': '{:.3f}'.format(sep),
            'pair_sep_frame0': '{:.3f}'.format(sep_f0),
            'canon_forward_z': '{:.3f}'.format(canon_z), 'flag': flag,
            'png': output_path}


def _worker(task):
    rig, npz_path, entry, output_path, labels, mark_labels = task
    try:
        return render_rig(rig, npz_path, entry, output_path, labels, mark_labels)
    except Exception as e:  # noqa: BLE001 — keep the batch going
        return {'rig': rig, 'r_hip': entry['r_hip']['raw'], 'l_hip': entry['l_hip']['raw'],
                'source': entry['source'], 'body_axis': int(entry.get('body_axis', False)),
                'forward_x': '', 'forward_z': '', 'pair_sep': '', 'pair_sep_frame0': '',
                'canon_forward_z': '', 'flag': 'ERROR {}: {}'.format(type(e).__name__, e), 'png': ''}


SUMMARY_FIELDS = ('rig', 'r_hip', 'l_hip', 'source', 'body_axis', 'forward_x', 'forward_z',
                  'pair_sep', 'pair_sep_frame0', 'canon_forward_z', 'flag', 'png')


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--export_dir', required=True,
                        help='Stage-1 export directory (dataset/export/<ds>): motions/ and '
                             'face_joint_names.json')
    parser.add_argument('--face_json', default=None,
                        help='Face pairs to visualize: a face_joint_names.json sidecar or a '
                             'patches/<ds>_face_pairs.json override file '
                             '(default: <export_dir>/face_joint_names.json)')
    parser.add_argument('--output_dir', default=None,
                        help='Output directory (default: {}/<export_dir basename>)'.format(DEFAULT_OUTPUT_DIR))
    parser.add_argument('--rigs', nargs='*', default=None, help='Only these skeleton keys')
    parser.add_argument('--rigs_file', default=None,
                        help='Text file with one skeleton key per line (# comments allowed)')
    parser.add_argument('--source', default=None,
                        help='Only pairs whose "source" equals this (e.g. thigh, body_axis)')
    parser.add_argument('--include_empty', action='store_true',
                        help='Also render skeletons whose pair is empty (identity facing)')
    parser.add_argument('--limit', type=int, default=None, help='Stop after N skeletons')
    parser.add_argument('--mark_labels', default='',
                        help='Comma-separated clean labels to mark with extra symbols on both '
                             'panels, e.g. "Head,Tail End,Toe" (looked up in '
                             '<export_dir>/clean_joint_names.json)')
    parser.add_argument('--overwrite', action='store_true', help='Re-render existing PNGs')
    parser.add_argument('--workers', type=int, default=max(1, min(8, (os.cpu_count() or 2) // 2)),
                        help='Parallel renderers (default: min(8, cpus/2))')
    return parser.parse_args()


def main():
    args = parse_args()
    export_dir = args.export_dir.rstrip('/')
    motions_dir = os.path.join(export_dir, 'motions')
    face_json = args.face_json or os.path.join(export_dir, 'face_joint_names.json')
    output_dir = args.output_dir or os.path.join(DEFAULT_OUTPUT_DIR, os.path.basename(export_dir))
    if not os.path.isdir(motions_dir):
        sys.exit('motions dir not found: {}'.format(motions_dir))
    if not os.path.isfile(face_json):
        sys.exit('face json not found: {}'.format(face_json))

    pairs = load_face_pairs(face_json)
    keys = list(pairs)
    if args.rigs_file:
        with open(args.rigs_file) as f:
            wanted = [ln.split('#', 1)[0].strip() for ln in f]
        wanted = [w for w in wanted if w]
        keys = [k for k in wanted if k in pairs] + [k for k in wanted if k not in pairs]
    if args.rigs:
        keys = list(args.rigs)
    missing = [k for k in keys if k not in pairs]
    if missing:
        print('warning: {} requested skeleton(s) not in {}: {}'.format(
            len(missing), face_json, ', '.join(missing[:5])), file=sys.stderr)
        keys = [k for k in keys if k in pairs]
    if args.source:
        keys = [k for k in keys if pairs[k]['source'] == args.source]
    if not args.include_empty:
        keys = [k for k in keys if pairs[k]['r_hip']['raw'] and pairs[k]['l_hip']['raw']]
    if args.limit:
        keys = keys[:args.limit]

    mark_labels = tuple(s.strip() for s in args.mark_labels.split(',') if s.strip())
    clean = {}
    if mark_labels:
        clean_path = os.path.join(export_dir, 'clean_joint_names.json')
        if os.path.isfile(clean_path):
            with open(clean_path) as f:
                clean = json.load(f)
        else:
            print('warning: --mark_labels needs {}; markers disabled'.format(clean_path), file=sys.stderr)
            mark_labels = ()

    npz_index = index_motion_npzs(motions_dir)
    done = set(os.listdir(output_dir)) if os.path.isdir(output_dir) else set()
    tasks, skipped = [], 0
    for rig in keys:
        out = os.path.join(output_dir, '{}.png'.format(rig))
        if '{}.png'.format(rig) in done and not args.overwrite:
            skipped += 1
            continue
        npz = npz_index.get(rig)
        if npz is None:
            print('warning: no motion NPZ for {}'.format(rig), file=sys.stderr)
            continue
        tasks.append((rig, str(npz), pairs[rig], out, clean.get(rig), mark_labels))

    print('{} skeleton(s) to render ({} already done) -> {}'.format(len(tasks), skipped, output_dir))
    os.makedirs(output_dir, exist_ok=True)
    rows = []
    if args.workers > 1 and len(tasks) > 1:
        with mp.Pool(args.workers) as pool:
            for row in tqdm(pool.imap_unordered(_worker, tasks), total=len(tasks), unit='rig'):
                rows.append(row)
    else:
        for task in tqdm(tasks, unit='rig'):
            rows.append(_worker(task))

    rows.sort(key=lambda r: r['rig'])
    summary_path = os.path.join(output_dir, 'facing_summary.tsv')
    existing = {}
    # --overwrite only discards the old summary for a full run; a subset re-render
    # (--rigs/--rigs_file/--source/--limit) merges its rows into the existing file.
    subset = bool(args.rigs or args.rigs_file or args.source or args.limit)
    if os.path.isfile(summary_path) and (subset or not args.overwrite):
        with open(summary_path) as f:
            for r in csv.DictReader(f, delimiter='\t'):
                existing[r['rig']] = r
    for r in rows:
        existing[r['rig']] = r
    with open(summary_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS, delimiter='\t')
        w.writeheader()
        for rig in sorted(existing):
            w.writerow({k: existing[rig].get(k, '') for k in SUMMARY_FIELDS})

    flagged = [r for r in rows if r['flag']]
    print('rendered {} skeleton(s); {} flagged -> {}'.format(len(rows), len(flagged), summary_path))
    for r in flagged[:20]:
        print('  {}\t{}\t{} / {}'.format(r['rig'], r['flag'], r['r_hip'], r['l_hip']))
    if len(flagged) > 20:
        print('  ... {} more (see summary)'.format(len(flagged) - 20))


if __name__ == '__main__':
    main()
