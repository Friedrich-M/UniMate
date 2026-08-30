#!/bin/bash
# Stage 3 — LLM facing-direction joint pair per rig: -> face_joint_names.json.
#
# Usage:
#   bash data_process/scripts/run_joints_face_select_llm.sh objaverse                    # DeepSeek v4-flash (default)
#   MODEL=Qwen/Qwen3-8B bash data_process/scripts/run_joints_face_select_llm.sh objaverse # local HF model
#
# Env overrides: INPUT_DIR, OUTPUT_NAME, MODEL, MAX_TOKENS. Extra args are passed
# through (e.g. --no_fallback). API backends read OPENAI_API_KEY / DEEPSEEK_API_KEY.

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

handle_help "$@"

DATASET=${1:?Usage: run_joints_face_select_llm.sh <truebones|mixamo|objaverse> [extra args...]}
shift
require_dataset "$DATASET"

INPUT_DIR=${INPUT_DIR:-$(export_dir "$DATASET")}
OUTPUT_NAME=${OUTPUT_NAME:-face_joint_names.json}
MODEL=${MODEL:-deepseek-v4-flash}
# Thinking backends hold a call for tens of seconds (DeepSeek low-effort
# ~19s, plus server keep-alive under load), so the per-rig SIGALRM budget
# scales with the backend; override with RIG_TIMEOUT.
if [[ "$MODEL" == deepseek* ]]; then RIG_TIMEOUT=${RIG_TIMEOUT:-300}; else RIG_TIMEOUT=${RIG_TIMEOUT:-15}; fi
MAX_TOKENS=${MAX_TOKENS:-512}

python -m data_process.joint_annotation.face_select_llm \
    --input_dir="$INPUT_DIR" --output_name="$OUTPUT_NAME" \
    --model="$MODEL" --max_tokens="$MAX_TOKENS" --rig_timeout="$RIG_TIMEOUT" \
"$@"
