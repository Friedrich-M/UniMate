#!/bin/bash
# Stage 3 — re-run LLM face-joint selection on rigs whose source == "empty".
#
# For each empty entry in face_joint_names.json the LLM is asked again to
# pick a bilateral pair (or body-axis endpoints). Successful picks overwrite
# the entry in place (a *.bak copy is made on first run); rigs that stay
# empty are listed in still_empty_face_joints.txt.
#
# Usage:
#   bash data_process/scripts/run_joints_face_correct_llm.sh objaverse
#   MODEL=gpt-5 bash data_process/scripts/run_joints_face_correct_llm.sh objaverse
#
# Env overrides: RAW, CLEANED, FACE, FAILED_LIST (optional id subset), MODEL,
# MAX_TOKENS, RIG_TIMEOUT. Extra args are passed through.

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

handle_help "$@"

DATASET=${1:?Usage: run_joints_face_correct_llm.sh <truebones|mixamo|objaverse> [extra args...]}
shift
require_dataset "$DATASET"

RAW=${RAW:-$(export_dir "$DATASET")/joint_names.json}
CLEANED=${CLEANED:-$(export_dir "$DATASET")/clean_joint_names.json}
FACE=${FACE:-$(export_dir "$DATASET")/face_joint_names.json}
MODEL=${MODEL:-deepseek-v4-flash}
MAX_TOKENS=${MAX_TOKENS:-512}
# Thinking backends need a much larger per-rig budget (see names_correct wrapper).
if [[ "$MODEL" == deepseek* ]]; then RIG_TIMEOUT=${RIG_TIMEOUT:-300}; else RIG_TIMEOUT=${RIG_TIMEOUT:-15}; fi

for f in "$RAW" "$CLEANED" "$FACE"; do
    [[ -f "$f" ]] || { echo "MISSING FILE: $f" >&2; exit 1; }
done

EXTRA_ARGS=()
[[ -n "${FAILED_LIST:-}" ]] && EXTRA_ARGS+=(--failed_list="$FAILED_LIST")

python -m data_process.joint_annotation.face_correct_llm \
    --raw="$RAW" --cleaned="$CLEANED" --face="$FACE" \
    --model="$MODEL" --max_tokens="$MAX_TOKENS" --rig_timeout="$RIG_TIMEOUT" \
    "${EXTRA_ARGS[@]}" "$@"
