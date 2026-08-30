#!/bin/bash
# Stage 5 — batch-animate a Mixamo character mesh with a directory of exported
# motion NPZs (raw Mixamo animations carry no mesh, so a character FBX is required).
#
# Usage:
#   bash data_process/scripts/run_animate_mixamo.sh                 # default character: Y Bot
#   CHAR_PATH=char.fbx bash data_process/scripts/run_animate_mixamo.sh
#
# Env overrides:
#   CHAR_PATH      rigged Mixamo character FBX (default: Y_Bot — the character
#                  whose skeleton the animation FBXs are authored on)
#   NPZ_DIR        (default: dataset/export/mixamo/motions)
#   OUTPUT_DIR     (default: outputs/mixamo_characters)
#   CHAR_ANIM_TYPE glb|fbx (default: fbx)

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

handle_help "$@"

DEFAULT_CHAR=dataset/raw/mixamo/character_refined/Y_Bot.fbx
CHAR_PATH=${CHAR_PATH:-$DEFAULT_CHAR}
if [[ ! -f "$CHAR_PATH" ]]; then
    echo "Character FBX not found: $CHAR_PATH" >&2
    echo "Download the mixamo raw data (run_download.sh mixamo) or set CHAR_PATH." >&2
    exit 1
fi
NPZ_DIR=${NPZ_DIR:-$(export_dir mixamo)/motions}
OUTPUT_DIR=${OUTPUT_DIR:-outputs/mixamo_characters}
CHAR_ANIM_TYPE=${CHAR_ANIM_TYPE:-fbx}

blender -b -P data_process/mesh_animation/animate_mixamo.py -- \
    --char_path="$CHAR_PATH" \
    --npz_dir="$NPZ_DIR" \
    --output_dir="$OUTPUT_DIR" \
    --char_anim_type="$CHAR_ANIM_TYPE" "$@"
