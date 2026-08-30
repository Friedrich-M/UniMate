#!/bin/bash
# Stage 5 — drive a rigged character with an *exported* motion NPZ (stage-1
# format) and save the animated character as GLB/FBX.
#
# Usage:
#   CHAR_PATH=char.glb ANIM_PATH=clip.npz bash data_process/scripts/run_animate_npz.sh
#
# Env overrides:
#   CHAR_PATH      (required — rigged character GLB/FBX matching the NPZ skeleton)
#   ANIM_PATH      (required — NPZ whose skeleton matches the rig)
#   OUTPUT_DIR     (default: outputs/animated)
#   CHAR_ANIM_TYPE glb|fbx (default: glb)

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

handle_help "$@"

CHAR_PATH=${CHAR_PATH:?Set CHAR_PATH to a rigged character GLB/FBX}
ANIM_PATH=${ANIM_PATH:?Set ANIM_PATH to a motion NPZ matching the rig}
OUTPUT_DIR=${OUTPUT_DIR:-outputs/animated}
CHAR_ANIM_TYPE=${CHAR_ANIM_TYPE:-glb}

blender -b -P data_process/mesh_animation/animate_npz.py -- \
    --char_path="$CHAR_PATH" \
    --anim_path="$ANIM_PATH" \
    --output_dir="$OUTPUT_DIR" \
    --char_anim_type="$CHAR_ANIM_TYPE" "$@"
