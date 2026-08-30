#!/bin/bash
# Stage 5 — bake raw animation FBX/GLB clips onto a chosen rigged character
# (action transfer by bone name — made for Mixamo-style shared skeletons,
# where any clip can drive any compatible character without remapping).
#
# Usage:
#   bash data_process/scripts/run_animate_fbx.sh                    # Y Bot x animation_motion/
#   CHAR_PATH=dataset/raw/mixamo/character_refined/Amy.fbx \
#       bash data_process/scripts/run_animate_fbx.sh                # whole library on Amy
#   ANIM_PATH=dataset/raw/mixamo/animation_motion/Jab_Cross.fbx \
#   CHAR_PATH=dataset/raw/mixamo/character_refined/Amy.fbx \
#       bash data_process/scripts/run_animate_fbx.sh                # a single clip
#
# Env overrides:
#   CHAR_PATH       rigged character FBX/GLB (default: character_refined/Y_Bot.fbx)
#   ANIM_PATH       animation clip or directory of clips
#                   (default: dataset/raw/mixamo/animation_motion)
#   OUTPUT_DIR      destination directory (default: outputs/animated_<character stem>)
#   CHAR_ANIM_TYPE  glb|fbx (default: fbx)
#
# Extra arguments are passed through to animate_fbx.py (e.g. --overwrite).

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

handle_help "$@"

CHAR_PATH=${CHAR_PATH:-dataset/raw/mixamo/character_refined/Y_Bot.fbx}
if [[ ! -f "$CHAR_PATH" ]]; then
    echo "Character file not found: $CHAR_PATH" >&2
    echo "Download the mixamo raw data (run_download.sh mixamo) or set CHAR_PATH." >&2
    exit 1
fi
ANIM_PATH=${ANIM_PATH:-dataset/raw/mixamo/animation_motion}
CHAR_STEM=$(basename "${CHAR_PATH%.*}")
OUTPUT_DIR=${OUTPUT_DIR:-outputs/animated_${CHAR_STEM}}
CHAR_ANIM_TYPE=${CHAR_ANIM_TYPE:-fbx}

blender -b -P data_process/mesh_animation/animate_fbx.py -- \
    --char_path="$CHAR_PATH" \
    --anim_path="$ANIM_PATH" \
    --output_dir="$OUTPUT_DIR" \
    --char_anim_type="$CHAR_ANIM_TYPE" "$@"
