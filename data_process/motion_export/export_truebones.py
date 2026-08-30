"""Export Truebones FBX clips to NPZ motion data.

Consumes the curated flat layout of the Truebones ZOO dataset: one FBX per
clip, named ``{Species}-{Action}.fbx`` (species never contains ``-``), all in
a single directory. Files are grouped by species so skeleton pruning is
shared across every clip of that species.

Pipeline: import → select armature/mesh → extract animation → Z-up → Y-up →
prune skeleton (shared across the species' clips) → save per-clip NPZ + MP4
+ rest-pose PNG.

Usage (Blender headless):
    blender -b -P data_process/motion_export/export_truebones.py -- \
        --data_dir dataset/raw/truebones/animation \
        --output_dir dataset/export/truebones
"""

import bpy
import sys
import os
import numpy as np
from tqdm import tqdm
import argparse
from pathlib import Path
from loguru import logger

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from data_process.utils.blender_export import (
    import_fbx,
    load_scene, prepare_skeleton, extract_all_actions, save_rest_pose_vis,
    prune_skeleton_shared,
    action_frame_range, sanitize_action_name, sanitize_object_type,
    save_motion, write_export_summary,
    is_asset_complete, mark_asset_complete,
)


# ─────────────────────────────────────────────────────────────────────────────
# Per-FBX extraction (one action per file)
# ─────────────────────────────────────────────────────────────────────────────

def extract_single_fbx(fbx_path, fps=30, dtype=np.float64):
    """Import one FBX file and return its (anim, rest_anim, names, skin_matrix, nbones)."""
    assert os.path.exists(fbx_path), f"FBX file not found: {fbx_path}"

    armature, mesh = load_scene(import_fbx, fbx_path, fps=fps)
    assert armature.animation_data is not None and armature.animation_data.action is not None, \
        f"Armature '{armature.name}' has no animation data or action."

    action = armature.animation_data.action
    start_frame, end_frame = action_frame_range(action)
    logger.info(f"Action frame range: {start_frame} to {end_frame}")

    skel = prepare_skeleton(armature, mesh, dtype=dtype, apply_world=True)
    extracted = extract_all_actions(
        armature, {action.name: (start_frame, end_frame)},
        skel['rest_local_pos'], skel['parents_array'],
        skel['rest_anim'], skel['nbones'], dtype=dtype,
    )
    _, anim = extracted[0]
    return {
        'anim': anim,
        'rest_anim': skel['rest_anim_shared'],
        'names': skel['bone_names'],
        'skin_matrix': skel['skin_matrix'],
        'nbones': skel['nbones'],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Shared post-processing: pruning, visualization, saving
# ─────────────────────────────────────────────────────────────────────────────

def postprocess_and_save(obj_type, extracted, rest_anim_shared, bone_names,
                         skin_matrix, output_dir, fps,
                         consider_parent_rotate=True, save_vis=True,
                         save_tpose=True):
    """Shared skeleton pruning, visualization, and per-clip NPZ saving."""
    motion_save_dir = os.path.join(output_dir, "motions")
    vis_save_dir    = os.path.join(output_dir, "videos")
    tpos_save_dir   = os.path.join(output_dir, "tpose")
    os.makedirs(motion_save_dir, exist_ok=True)
    os.makedirs(vis_save_dir,    exist_ok=True)

    anims_list = [anim for _, anim in extracted]
    anims_list, rest_anim_pruned, names_pruned, skin_matrix_pruned = prune_skeleton_shared(
        anims_list, rest_anim_shared, list(bone_names), skin_matrix.copy(),
        consider_parent_rotate=consider_parent_rotate,
    )

    # No joint-count gate here: stage 4 (feature_extraction --min_joints /
    # --max_joints) decides which skeletons enter training, so the export
    # stays complete.
    if save_tpose:
        save_rest_pose_vis(tpos_save_dir, obj_type, rest_anim_pruned)

    for i, (action_name, _) in enumerate(extracted):
        clean_action_name = sanitize_action_name(action_name)
        clip_name = f"{obj_type}-{clean_action_name}"
        save_motion(
            os.path.join(motion_save_dir, f"{clip_name}.npz"),
            os.path.join(vis_save_dir,    f"{clip_name}.mp4"),
            anims_list[i], rest_anim_pruned,
            names_pruned, skin_matrix_pruned, fps,
            action_name=clean_action_name, save_vis=save_vis,
        )

    logger.info(f"Saved {len(extracted)} clips for '{obj_type}'")
    return {obj_type: names_pruned}


# ─────────────────────────────────────────────────────────────────────────────
# Per-species export
# ─────────────────────────────────────────────────────────────────────────────

def export_species(species, fbx_paths, output_dir, fps=30, dtype=np.float64,
                   min_frames=5, consider_parent_rotate=True, save_vis=True):
    """Export one species: extract each of its per-clip FBX files, then prune
    the skeleton jointly across all clips and save them."""

    obj_type = sanitize_object_type(species)

    if is_asset_complete(output_dir, obj_type):
        logger.info(f"Species '{obj_type}' already complete, skipping.")
        return {}
    motion_save_dir = os.path.join(output_dir, "motions")
    existing = sorted(Path(motion_save_dir).glob(f"{obj_type}-*.npz")) if os.path.isdir(motion_save_dir) else []
    if existing:
        # Pruning is joint across all of a species' clips, so a partial
        # (or pre-marker) export must be redone as a whole.
        logger.warning(f"'{obj_type}' has {len(existing)} clip(s) but no completion "
                       f"marker (partial or pre-marker export); re-exporting.")

    # A few species ship rig variants (different node counts or container
    # names — e.g. KingCobra's extra Null, Monkey's A01/B01/B02 containers).
    # prune_skeleton_shared requires identical topology across its inputs, so
    # clips are grouped by joint-name signature and each group is pruned
    # separately. The dominant group defines the species' joint_names entry
    # and rest-pose PNG.
    groups = {}  # names signature -> {'rest_anim', 'skin_matrix', 'extracted'}
    for fbx_path in fbx_paths:
        stem = Path(fbx_path).stem
        species_part, sep, action_name = stem.partition('-')
        if not sep:
            # Without the separator there is no action name; skipping one file
            # beats aborting the whole species on an IndexError.
            logger.warning(f"Skipping {fbx_path}: filename has no '-' separating "
                           f"species from action (expected "
                           f"'{{Species}}-{{Action}}.fbx').")
            continue

        try:
            result = extract_single_fbx(fbx_path, fps=fps, dtype=dtype)
        except Exception as e:
            logger.warning(f"Failed to extract {fbx_path}: {e}")
            continue

        nframes = result['anim'].positions.shape[0]
        if nframes < min_frames:
            logger.info(f"Skipping {fbx_path}: only {nframes} frames "
                        f"(min={min_frames})")
            continue

        sig = tuple(result['names'])
        group = groups.setdefault(sig, {
            'rest_anim':   result['rest_anim'],
            'names':       list(result['names']),
            'skin_matrix': result['skin_matrix'],
            'extracted':   [],
        })
        group['extracted'].append((action_name, result['anim']))

    saved = {}
    if groups:
        by_size = sorted(groups.values(), key=lambda g: len(g['extracted']),
                         reverse=True)
        if len(by_size) > 1:
            logger.warning(f"'{obj_type}' has {len(by_size)} rig variants "
                           f"(clip counts: {[len(g['extracted']) for g in by_size]}); "
                           f"pruning each separately.")
        for i, group in enumerate(by_size):
            logger.info(f"Extracted {len(group['extracted'])} animations for "
                        f"'{obj_type}' (variant {i})")
            result = postprocess_and_save(
                obj_type, group['extracted'],
                group['rest_anim'], group['names'], group['skin_matrix'],
                output_dir, fps=fps,
                consider_parent_rotate=consider_parent_rotate, save_vis=save_vis,
                save_tpose=(i == 0),
            )
            if i == 0:
                saved = result
    else:
        logger.info(f"No valid FBX files extracted for '{obj_type}'; skipping.")

    mark_asset_complete(output_dir, obj_type,
                        status="exported" if saved else "skipped",
                        joint_names=saved.get(obj_type))
    return saved


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Export Truebones FBX clips to NPZ motion data.")
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Flat directory of per-clip {Species}-{Action}.fbx files.')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory (motions/, videos/, tpose/ created inside).')
    parser.add_argument('--fps', type=int, default=30,
                        help='Scene FPS for import and export (dataset is 30 fps).')
    parser.add_argument('--min_frames', type=int, default=5,
                        help='Skip clips shorter than this many frames '
                             '(shared minimum across all dataset exporters).')
    parser.add_argument('--consider_parent_rotate', action=argparse.BooleanOptionalAction, default=True,
                        help="Preserve non-rotating leaves whose parent rotates during pruning "
                             "(see prune_skeleton_shared; use --no-consider_parent_rotate to disable).")
    parser.add_argument('--vis', action=argparse.BooleanOptionalAction, default=True,
                        help="Render the per-clip MP4 preview (use --no-vis for bulk runs).")
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
    return parser.parse_args(argv)


def main():
    args = parse_args()

    fbx_files = sorted(f for f in os.listdir(args.data_dir) if f.lower().endswith('.fbx'))
    by_species = {}
    for f in fbx_files:
        species = os.path.splitext(f)[0].split('-', 1)[0]
        by_species.setdefault(species, []).append(os.path.join(args.data_dir, f))
    logger.info(f"Found {len(fbx_files)} clips across {len(by_species)} species "
                f"in {args.data_dir}")

    os.makedirs(args.output_dir, exist_ok=True)
    error_log = os.path.join(args.output_dir, 'export_errors.log')
    all_joint_names = {}
    n_failed = 0
    for species, fbx_paths in tqdm(sorted(by_species.items()), desc="Exporting species"):
        try:
            saved = export_species(species, fbx_paths, output_dir=args.output_dir,
                                   fps=args.fps, min_frames=args.min_frames,
                                   consider_parent_rotate=args.consider_parent_rotate,
                                   save_vis=args.vis)
            if saved:
                all_joint_names.update(saved)
        except Exception as e:  # noqa: BLE001 — keep the batch going
            n_failed += 1
            logger.exception(f"Failed to export species {species}")
            with open(error_log, 'a') as log_file:
                log_file.write(f"Failed to export species {species}: {e}\n")

    write_export_summary(args.output_dir, all_joint_names)
    if n_failed:
        logger.error(f"{n_failed}/{len(by_species)} species failed; see {error_log}")
        sys.exit(1)


if __name__ == "__main__":
    # Blender exits 0 even on an uncaught exception, which hides failures from
    # `set -e` and from Slurm's afterok dependencies.
    try:
        main()
    except SystemExit:
        raise
    except Exception:
        logger.exception("export_truebones failed")
        sys.exit(1)
