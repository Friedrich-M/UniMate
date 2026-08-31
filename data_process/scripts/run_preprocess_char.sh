#!/bin/bash
# Stage 5 — preprocess one rigged, animated asset through the real export +
# feature pipeline. Leaves exactly three deliverables in OUTPUT_DIR:
#   <name>_canonical.{glb,fbx}  canonical rest-pose asset
#   cond.npy                    model-side topology conditioning
#   motions/<clip>.npz          motion features
# Feature NPZs then drive the canonical asset cond-free via run_animate_lbs.sh.
# (KEEP_INTERMEDIATE=1 keeps the export-stage NPZs and visuals.)
#
# Usage:
#   CHAR_PATH=asset.glb OUTPUT_DIR=outputs/asset \
#       FACE_R=R_Thigh FACE_L=L_Thigh FORMATS=glb,fbx \
#       bash data_process/scripts/run_preprocess_char.sh
#
# Env overrides:
#   CHAR_PATH     (required — rigged, ANIMATED asset GLB/FBX)
#   OUTPUT_DIR    (required — export/, motions/, cond.npy, <name>_canonical.*)
#   FACE_R/FACE_L raw face-joint names for facing canonicalization
#   BODY_AXIS=1   face pair is a head/tail body axis (serpentine rigs)
#   FORMATS       comma-separated canonical formats (default: glb)
#   SAVE_VIS=1    also render per-clip preview MP4s
#   APPLY_CLIP=1  crop motions into training-length clips
#   KEEP_INTERMEDIATE=1  keep export-stage NPZs and T-pose/preview visuals

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

handle_help "$@"

CHAR_PATH=${CHAR_PATH:?Set CHAR_PATH to a rigged, animated asset GLB/FBX}
OUTPUT_DIR=${OUTPUT_DIR:?Set OUTPUT_DIR for the preprocessed outputs}

EXTRA_ARGS=()
[[ -n "${FACE_R:-}" ]] && EXTRA_ARGS+=(--face_r="$FACE_R")
[[ -n "${FACE_L:-}" ]] && EXTRA_ARGS+=(--face_l="$FACE_L")
[[ -n "${BODY_AXIS:-}" ]] && EXTRA_ARGS+=(--body_axis)
[[ -n "${FORMATS:-}" ]] && EXTRA_ARGS+=(--formats="$FORMATS")
[[ -n "${SAVE_VIS:-}" ]] && EXTRA_ARGS+=(--save_vis)
[[ -n "${APPLY_CLIP:-}" ]] && EXTRA_ARGS+=(--apply_clip)
[[ -n "${KEEP_INTERMEDIATE:-}" ]] && EXTRA_ARGS+=(--keep_intermediate)

blender -b -P data_process/mesh_animation/preprocess_char.py -- \
    --char_path="$CHAR_PATH" \
    --output_dir="$OUTPUT_DIR" \
    "${EXTRA_ARGS[@]}" "$@"
