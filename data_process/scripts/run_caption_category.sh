#!/bin/bash
# Stage 2 — classify assets into body-plan categories with a local Qwen3.5 / Qwen3-VL model.
# Single-GPU only — one classification per object, no sharding.
#
# Usage:
#   bash data_process/scripts/run_caption_category.sh objaverse
#   MODEL=Qwen/Qwen3-VL-2B-Instruct bash data_process/scripts/run_caption_category.sh truebones
#
# Env overrides:
#   MODEL, RENDER_ROOT (default: dataset/render/<dataset>_tpose),
#   EXPORT_DIR (default: dataset/export/<dataset>; skeleton facts + captions shown to the model),
#   MOTION_RENDER_ROOT (default: dataset/render/<dataset>; first frame of a clip, 4 cameras, as stance reference),
#   CATEGORY_GROUPS_JSON (default: dataset/export/<dataset>/category_groups.json),
#   VOTES (default 3 sampled answers, majority wins), MAX_TOKENS (default 96)
# Extra arguments are passed through to classify_category.py.
# On offline compute nodes with a populated HuggingFace cache, export
# HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1.

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

handle_help "$@"
DATASET=${1:?Usage: run_caption_category.sh <truebones|mixamo|objaverse> [extra args...]}
shift
require_dataset "$DATASET"

RENDER_ROOT=${RENDER_ROOT:-$(tpose_dir "$DATASET")}
CATEGORY_GROUPS_JSON=${CATEGORY_GROUPS_JSON:-$(export_dir "$DATASET")/category_groups.json}
EXPORT_DIR=${EXPORT_DIR:-$(export_dir "$DATASET")}
MOTION_RENDER_ROOT=${MOTION_RENDER_ROOT:-$(render_dir "$DATASET")}
VOTES=${VOTES:-3}
if [[ ! -d "$RENDER_ROOT" ]]; then
    echo "RENDER_ROOT does not exist: $RENDER_ROOT" >&2
    exit 2
fi
mkdir -p "$(dirname "$CATEGORY_GROUPS_JSON")"

# Local Qwen only: Qwen/Qwen3.5-9B (default, same as the captioner), Qwen/Qwen3.8-27B, or Qwen/Qwen3-VL-{2B,8B,32B,72B}-Instruct.
MODEL=${MODEL:-Qwen/Qwen3.5-9B}
MAX_TOKENS=${MAX_TOKENS:-96}

echo "Dataset: $DATASET | Model: $MODEL"
echo "Render root: $RENDER_ROOT"
echo "Output JSON: $CATEGORY_GROUPS_JSON"
echo "Export dir (skeleton facts + captions): $EXPORT_DIR | motion renders: $MOTION_RENDER_ROOT | votes: $VOTES"

# Mixamo is one shared humanoid rig ("mixamo" in the export sidecars); its
# T-pose folder holds the 114 character renders, which are not what stage 4
# indexes, so the file is written directly instead of classified.
if [[ "$DATASET" == "mixamo" ]]; then
    printf '{\n  "bipedal": [\n    "mixamo"\n  ]\n}\n' > "$CATEGORY_GROUPS_JSON"
    echo "mixamo: single humanoid rig -> wrote $CATEGORY_GROUPS_JSON"
    exit 0
fi

python -m data_process.vlm_caption.classify_category \
    --render_root="$RENDER_ROOT" \
    --model="$MODEL" \
    --max_tokens="$MAX_TOKENS" \
    --votes="$VOTES" \
    --export_dir="$EXPORT_DIR" \
    --motion_render_root="$MOTION_RENDER_ROOT" \
    --category_groups_json="$CATEGORY_GROUPS_JSON" "$@"
