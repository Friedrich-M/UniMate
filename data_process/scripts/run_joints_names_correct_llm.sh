#!/bin/bash
# Stage 3 — LLM check-and-correct sweep over clean_joint_names.json (in place).
#
# Every rig in joint_names.json is sent to the LLM together with its current
# cleaned label; the LLM keeps or corrects each label. The cleaned file is
# updated IN PLACE (a *.bak copy is made on first run). Set FAILED_LIST to
# restrict the sweep to the rig ids listed in a text file.
#
# Usage:
#   bash data_process/scripts/run_joints_names_correct_llm.sh objaverse
#   FAILED_LIST=.../failed_clean_names.txt bash data_process/scripts/run_joints_names_correct_llm.sh objaverse
#
# Env overrides: RAW, CLEANED, FAILED_LIST, MODEL, MAX_TOKENS. Extra args are
# passed through. API backends read OPENAI_API_KEY / DEEPSEEK_API_KEY.

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

handle_help "$@"

DATASET=${1:?Usage: run_joints_names_correct_llm.sh <truebones|mixamo|objaverse> [extra args...]}
shift
require_dataset "$DATASET"

RAW=${RAW:-$(export_dir "$DATASET")/joint_names.json}
CLEANED=${CLEANED:-$(export_dir "$DATASET")/clean_joint_names.json}
MODEL=${MODEL:-deepseek-v4-flash}
# Thinking backends hold a call for tens of seconds (DeepSeek low-effort
# ~19s, plus server keep-alive under load), so the per-rig SIGALRM budget
# scales with the backend; override with RIG_TIMEOUT.
if [[ "$MODEL" == deepseek* ]]; then RIG_TIMEOUT=${RIG_TIMEOUT:-300}; else RIG_TIMEOUT=${RIG_TIMEOUT:-15}; fi
MAX_TOKENS=${MAX_TOKENS:-4096}

EXTRA_ARGS=()
[[ -n "${FAILED_LIST:-}" ]] && EXTRA_ARGS+=(--failed_list="$FAILED_LIST")

python -m data_process.joint_annotation.names_correct_llm \
    --raw="$RAW" --cleaned="$CLEANED" \
    --model="$MODEL" --max_tokens="$MAX_TOKENS" --rig_timeout="$RIG_TIMEOUT" \
    "${EXTRA_ARGS[@]}" "$@"
