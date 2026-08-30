#!/bin/bash
# Stage 3 — LLM joint-name cleaning: joint_names.json -> clean_joint_names.json.
#
# Usage:
#   bash data_process/scripts/run_joints_names_clean_llm.sh objaverse                     # DeepSeek v4-flash (default)
#   MODEL=deepseek-v4-flash bash data_process/scripts/run_joints_names_clean_llm.sh objaverse # DeepSeek API
#   MODEL=Qwen/Qwen3-8B bash data_process/scripts/run_joints_names_clean_llm.sh objaverse # local HF model
#
# Env overrides: INPUT, OUTPUT, MODEL, MAX_TOKENS. Extra args are passed through
# (e.g. --no_fallback). API backends read OPENAI_API_KEY / DEEPSEEK_API_KEY.

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

handle_help "$@"

DATASET=${1:?Usage: run_joints_names_clean_llm.sh <truebones|mixamo|objaverse> [extra args...]}
shift
require_dataset "$DATASET"

INPUT=${INPUT:-$(export_dir "$DATASET")/joint_names.json}
OUTPUT=${OUTPUT:-$(export_dir "$DATASET")/clean_joint_names.json}
MODEL=${MODEL:-deepseek-v4-flash}
MAX_TOKENS=${MAX_TOKENS:-2048}

python -m data_process.joint_annotation.names_clean_llm \
    --input="$INPUT" --output="$OUTPUT" \
    --model="$MODEL" --max_tokens="$MAX_TOKENS" \
    --report "$@"
