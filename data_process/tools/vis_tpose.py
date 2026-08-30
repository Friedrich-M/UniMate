"""Visualize skeleton T-poses with joint names annotated, for manual review.

Workflow companion for ``face_select*``: when automatic face-joint
resolution produces ``"source": "empty"`` (or picks the wrong pair), render
the T-pose with every joint labeled so the left/right symmetric pair can be
identified by eye. Each motion NPZ produces one PNG.

Accepts the export-stage NPZ layout (see
:func:`data_process.utils.blender_export.save_motion`) whose relevant fields are:

    rest_local_pos : (J, 3)   local bone offsets at rest
    rest_local_rot : (J, 4)   local bone rotations at rest (quaternions)
    names          : (J,)     joint name strings
    parents        : (J,)     parent joint indices (-1 for root)
    offsets        : (J, 3)   bind-pose offsets (optional — falls back to
                              rest_local_pos when absent)

The T-pose global positions are reconstructed via forward kinematics on the
rest pose and passed to :func:`save_skeleton_tpose_annotated`.

Modes:
    - Single: ``--npz_path X`` → one PNG.
    - Batch (default): ``--motions_dir M [--output_dir O]`` → one PNG per
      unique key (prefix before the first ``-``) discovered in ``M``.
    - Batch (face_json): ``--face_json F --motions_dir M [--source empty] [--output_dir O]``
      → one PNG per key in F whose ``source`` field matches ``--source``.

See ``data_process/scripts/run_joints_vis_tpose.sh`` for the typical batch call.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from Animation import Animation, Quaternions, positions_global  # noqa: E402
from data_process.utils.plotting import save_skeleton_tpose_annotated  # noqa: E402


REQUIRED_KEYS = ("rest_local_pos", "rest_local_rot", "names", "parents")


def load_tpose_from_npz(npz_path):
    # type: (str) -> Tuple[np.ndarray, List[str], np.ndarray]
    """Load rest-pose data from a motion NPZ and compute global T-pose positions.

    Returns:
        ``(parents, joint_names, tpose_positions)`` where ``tpose_positions``
        is a float32 ``(J, 3)`` array of global joint coordinates at rest.
    """
    data = np.load(npz_path, allow_pickle=True)
    missing = [k for k in REQUIRED_KEYS if k not in data.files]
    if missing:
        raise KeyError(
            "{} is missing required keys {}; available: {}".format(
                npz_path, missing, list(data.files)
            )
        )

    rest_local_pos = data['rest_local_pos']   # (J, 3)
    rest_local_rot = data['rest_local_rot']   # (J, 4) quaternions
    parents = np.asarray(data['parents'])     # (J,)
    names = [str(n) for n in data['names']]   # length J
    # Prefer the explicit bind-pose offsets when the export recorded them;
    # fall back to the rest-pose local translation when absent.
    offsets = (np.asarray(data['offsets']) if 'offsets' in data.files
               else rest_local_pos)

    njoints = len(parents)

    rest_anim = Animation(
        rotations=Quaternions(rest_local_rot[None]),  # (1, J, 4)
        positions=rest_local_pos[None],               # (1, J, 3)
        orients=Quaternions.id(njoints),
        offsets=offsets,
        parents=parents,
    )
    tpose_positions = positions_global(rest_anim)[0].astype(np.float32)  # (J, 3)

    return parents, names, tpose_positions


def render_one(npz_path, output_path, *, font_size, joint_marker_size,
               bone_linewidth):
    # type: (str, str, int, float, float) -> None
    """Render one NPZ as an annotated T-pose PNG."""
    parents, names, tpose_positions = load_tpose_from_npz(npz_path)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    save_skeleton_tpose_annotated(
        save_path=output_path,
        parents=parents,
        positions=tpose_positions,
        joint_names=names,
        font_size=font_size,
        joint_marker_size=joint_marker_size,
        bone_linewidth=bone_linewidth,
    )


def find_motion_npz(motions_dir, key):
    # type: (str, str) -> Optional[Path]
    """Resolve the first motion NPZ whose prefix-before-'-' matches ``key``.

    Motion files are named ``<key>-<clip_name>.npz`` (e.g.
    ``669a728e57bbceb71e6b8ce6_fbx-mixamo_com.npz``). Any single clip is
    enough to render the rest-pose skeleton, so we take the first sorted
    match. Returns ``None`` if no file matches.
    """
    hits = sorted(Path(motions_dir).glob("{}-*.npz".format(key)))
    return hits[0] if hits else None


def collect_keys_from_motions_dir(motions_dir):
    # type: (str) -> List[str]
    """Collect unique keys (prefix before the first ``-``) from NPZs in ``motions_dir``.

    NPZs without a ``-`` in their stem are treated as their own key.
    """
    keys = set()
    for p in Path(motions_dir).glob("*.npz"):
        stem = p.stem
        keys.add(stem.split("-", 1)[0] if "-" in stem else stem)
    return sorted(keys)


DEFAULT_OUTPUT_DIR = "outputs/face_joints_vis"


def default_single_output_path(npz_path):
    # type: (str) -> str
    """Default single-mode output: ``<DEFAULT_OUTPUT_DIR>/<stem>_tpose_annotated.png``."""
    return os.path.join(DEFAULT_OUTPUT_DIR,
                        "{}_tpose_annotated.png".format(Path(npz_path).stem))


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    mode = parser.add_mutually_exclusive_group(required=False)
    mode.add_argument(
        '--npz_path', type=str, default=None,
        help='Single mode: path to a motion NPZ (skips batch logic).',
    )
    mode.add_argument(
        '--face_json', type=str, default=None,
        help='Batch mode (alt): path to face_joint_names.json. Motion files '
             'are matched as <motions_dir>/<key>-*.npz (prefix before the '
             'first "-"). Without this flag, batch mode scans --motions_dir '
             'directly for unique keys.',
    )

    # Batch-only
    parser.add_argument(
        '--motions_dir', type=str, default=None,
        help='[batch] Directory of motion NPZs (default: sibling "motions/" '
             'of --face_json).',
    )
    parser.add_argument(
        '--source', type=str, default='empty',
        help='[batch] Only render uuids whose "source" field equals this '
             '(default: "empty"). Use "ALL" to render every key.',
    )
    parser.add_argument(
        '--output_dir', type=str, default=DEFAULT_OUTPUT_DIR,
        help='[batch] Output directory (default: {}).'.format(DEFAULT_OUTPUT_DIR),
    )
    parser.add_argument(
        '--limit', type=int, default=None,
        help='[batch] Stop after N uuids (debug aid).',
    )
    parser.add_argument(
        '--overwrite', action='store_true',
        help='[batch] Re-render even if the output PNG already exists.',
    )

    # Single-only
    parser.add_argument(
        '--output', type=str, default=None,
        help='[single] Output PNG path (default: '
             '<output_dir>/<input_stem>_tpose_annotated.png).',
    )

    # Plot knobs
    parser.add_argument('--font-size', type=int, default=7,
                        help='Font size for joint name annotations.')
    parser.add_argument('--joint-marker-size', type=float, default=80.0,
                        help='Joint dot marker size in points^2.')
    parser.add_argument('--bone-linewidth', type=float, default=2.0,
                        help='Bone line width.')
    args = parser.parse_args()

    plot_kwargs = dict(
        font_size=args.font_size,
        joint_marker_size=args.joint_marker_size,
        bone_linewidth=args.bone_linewidth,
    )

    # ── Single mode ─────────────────────────────────────────────────────────
    if args.npz_path:
        if not os.path.exists(args.npz_path):
            parser.error("Input NPZ not found: {}".format(args.npz_path))
        out = args.output or default_single_output_path(args.npz_path)
        render_one(args.npz_path, out, **plot_kwargs)
        print("Saved: {}".format(out))
        return

    # ── Batch mode: resolve the list of keys ────────────────────────────────
    # face_json takes precedence when explicitly provided; otherwise we scan
    # --motions_dir directly for unique keys.
    if args.face_json:
        if not os.path.exists(args.face_json):
            parser.error("face_json not found: {}".format(args.face_json))
        motions_dir = args.motions_dir or str(Path(args.face_json).parent / "motions")
        if not os.path.isdir(motions_dir):
            parser.error("motions_dir not found: {}".format(motions_dir))
        with open(args.face_json) as f:
            face = json.load(f)
        if args.source == "ALL":
            keys = sorted(face.keys())
        else:
            keys = sorted(
                k for k, v in face.items()
                if isinstance(v, dict) and v.get("source") == args.source
            )
        source_desc = "face_json={} source={!r}".format(args.face_json, args.source)
    else:
        if args.motions_dir is None:
            parser.error("--motions_dir is required in batch mode")
        motions_dir = args.motions_dir
        if not os.path.isdir(motions_dir):
            parser.error("motions_dir not found: {}".format(motions_dir))
        keys = collect_keys_from_motions_dir(motions_dir)
        source_desc = "motions_dir={}".format(motions_dir)

    if args.limit is not None:
        keys = keys[: args.limit]

    if not keys:
        print("No entries selected ({})".format(source_desc))
        return

    os.makedirs(args.output_dir, exist_ok=True)
    print("motions_dir : {}".format(motions_dir))
    print("output_dir  : {}".format(args.output_dir))
    print("Rendering {} T-poses ({})".format(len(keys), source_desc))

    rendered = skipped_existing = missing_npz = failed = 0
    missing_examples = []  # type: List[str]
    failed_examples = []   # type: List[str]

    for key in tqdm(keys, unit="seq", dynamic_ncols=True):
        out_path = os.path.join(args.output_dir, "{}.png".format(key))
        if (not args.overwrite) and os.path.exists(out_path):
            skipped_existing += 1
            continue
        npz = find_motion_npz(motions_dir, key)
        if npz is None:
            missing_npz += 1
            if len(missing_examples) < 5:
                missing_examples.append(key)
            continue
        try:
            render_one(str(npz), out_path, **plot_kwargs)
            rendered += 1
        except Exception as e:  # noqa: BLE001
            failed += 1
            if len(failed_examples) < 5:
                failed_examples.append("{}: {}".format(key, e))

    print()
    print("rendered        : {}".format(rendered))
    print("skipped (exists): {}".format(skipped_existing))
    print("missing NPZ     : {}".format(missing_npz))
    if missing_examples:
        for k in missing_examples:
            print("    - {}".format(k))
        if missing_npz > len(missing_examples):
            print("    ... ({} total)".format(missing_npz))
    print("failed          : {}".format(failed))
    if failed_examples:
        for line in failed_examples:
            print("    - {}".format(line))
        if failed > len(failed_examples):
            print("    ... ({} total)".format(failed))


if __name__ == '__main__':
    main()
