#!/bin/bash
# Stage 2 — classify assets into body-plan categories with a local Qwen3-VL model.
# Single-GPU only — one classification per object, no sharding.
#
# Usage:
#   bash data_process/scripts/run_caption_category.sh objaverse
#   MODEL=Qwen/Qwen3-VL-2B-Instruct bash data_process/scripts/run_caption_category.sh truebones
#
# Env overrides:
#   MODEL, RENDER_ROOT (default: dataset/render/<dataset>_tpose),
#   CATEGORY_GROUPS_JSON (default: dataset/export/<dataset>/category_groups.json),
#   MAX_TOKENS
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
if [[ ! -d "$RENDER_ROOT" ]]; then
    echo "RENDER_ROOT does not exist: $RENDER_ROOT" >&2
    exit 2
fi
mkdir -p "$(dirname "$CATEGORY_GROUPS_JSON")"

# Local Qwen3-VL only. Sizes: Qwen/Qwen3-VL-{2B,8B,32B,72B}-Instruct.
MODEL=${MODEL:-Qwen/Qwen3-VL-8B-Instruct}
MAX_TOKENS=${MAX_TOKENS:-10}

echo "Dataset: $DATASET | Model: $MODEL"
echo "Render root: $RENDER_ROOT"
echo "Output JSON: $CATEGORY_GROUPS_JSON"

python -m data_process.vlm_caption.classify_category \
    --render_root="$RENDER_ROOT" \
    --model="$MODEL" \
    --max_tokens="$MAX_TOKENS" \
    --category_groups_json="$CATEGORY_GROUPS_JSON" "$@"
