#!/bin/bash
# Stage 5 — drive a rigged character with a canonicalized motion clip (stage-4
# format, or a generated motion in that format) and export GLB + FBX. The
# character is auto-resolved from the dataset + clip name for truebones and
# objaverse; mixamo needs CHAR_PATH.
#
# Usage:
#   ANIM_PATH=clip.npz bash data_process/scripts/run_animate_motion.sh objaverse
#   ANIM_PATH=clip.npz bash data_process/scripts/run_animate_motion.sh truebones
#   ANIM_PATH=clip.npz CHAR_PATH=char.fbx bash data_process/scripts/run_animate_motion.sh mixamo
#
# Env overrides:
#   ANIM_PATH   (required) motion .npz (or legacy .npy)
#   CHAR_PATH   character mesh (auto-resolved for truebones/objaverse; required for mixamo)
#   COND_PATH   cond.npy (default: dataset/features/<dataset>/cond.npy)
#   OUTPUT_DIR  (default: outputs/animated)
#   ANIM_MODE   fk|ik, legacy .npy only (default: fk)

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

handle_help "$@"

DATASET=${1:?Usage: run_animate_motion.sh <truebones|mixamo|objaverse> [extra args...]}
shift
require_dataset "$DATASET"

ANIM_PATH=${ANIM_PATH:?Set ANIM_PATH to a motion .npz/.npy for dataset=$DATASET}
OUTPUT_DIR=${OUTPUT_DIR:-outputs/animated}
ANIM_MODE=${ANIM_MODE:-fk}

EXTRA_ARGS=()
[[ -n "${CHAR_PATH:-}" ]] && EXTRA_ARGS+=(--char_path="$CHAR_PATH")
[[ -n "${COND_PATH:-}" ]] && EXTRA_ARGS+=(--cond_path="$COND_PATH")
if [[ "$DATASET" == "mixamo" && -z "${CHAR_PATH:-}" ]]; then
    echo "dataset=mixamo requires CHAR_PATH (no auto-resolved character mesh)" >&2
    exit 1
fi

blender -b -P data_process/mesh_animation/animate_motion.py -- \
    --dataset_type="$DATASET" \
    --anim_path="$ANIM_PATH" \
    --output_dir="$OUTPUT_DIR" \
    --anim_mode="$ANIM_MODE" \
    "${EXTRA_ARGS[@]}" "$@"
