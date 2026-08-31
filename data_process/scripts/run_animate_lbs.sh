#!/bin/bash
# Stage 5 — deform a rigged asset with a motion NPZ via manual NumPy LBS and
# save the vertex animation (NPZ, optionally per-frame OBJ).
#
# Accepts both motion flavors (auto-detected): export-stage NPZ, or
# feature-format (generated) NPZ — the latter needs DATASET_TYPE.
#
# Usage:
#   CHAR_PATH=asset.glb ANIM_PATH=clip.npz bash data_process/scripts/run_animate_lbs.sh
#   CHAR_PATH=asset.glb ANIM_PATH=gen.npz DATASET_TYPE=objaverse bash data_process/scripts/run_animate_lbs.sh
#
# Env overrides:
#   CHAR_PATH     (required — rigged asset GLB/FBX with skinning weights)
#   ANIM_PATH     (required — motion NPZ, or a directory of NPZs: one output
#                  set per action clip)
#   DATASET_TYPE  truebones|objaverse|mixamo (feature-format motion only)
#   COND_PATH     cond.npy override (feature-format motion only)
#   OUTPUT_DIR    (default: outputs/lbs)
#   SAVE          comma-separated: glb,fbx,npz,obj (default: glb)
#                 glb/fbx = animated rigged asset via the animate_npz keyframe path

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

handle_help "$@"

CHAR_PATH=${CHAR_PATH:?Set CHAR_PATH to a rigged asset GLB/FBX}
ANIM_PATH=${ANIM_PATH:?Set ANIM_PATH to a motion NPZ}
OUTPUT_DIR=${OUTPUT_DIR:-outputs/lbs}
SAVE=${SAVE:-glb}

EXTRA_ARGS=()
[[ -n "${DATASET_TYPE:-}" ]] && EXTRA_ARGS+=(--dataset_type="$DATASET_TYPE")
[[ -n "${COND_PATH:-}" ]] && EXTRA_ARGS+=(--cond_path="$COND_PATH")

blender -b -P data_process/mesh_animation/animate_lbs.py -- \
    --char_path="$CHAR_PATH" \
    --anim_path="$ANIM_PATH" \
    --output_dir="$OUTPUT_DIR" \
    --save="$SAVE" \
    "${EXTRA_ARGS[@]}" "$@"
