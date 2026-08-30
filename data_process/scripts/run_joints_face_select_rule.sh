#!/bin/bash
# Stage 3 — rule-based facing-direction joint pair: -> face_joint_names.json.
# Reads joint_names.json + clean_joint_names.json from the export directory.
#
# Usage:
#   bash data_process/scripts/run_joints_face_select_rule.sh <truebones|mixamo|objaverse>
#
# Env overrides: INPUT_DIR, OUTPUT

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

handle_help "$@"

DATASET=${1:?Usage: run_joints_face_select_rule.sh <truebones|mixamo|objaverse> [extra args...]}
shift
require_dataset "$DATASET"

INPUT_DIR=${INPUT_DIR:-$(export_dir "$DATASET")}
OUTPUT=${OUTPUT:-$INPUT_DIR/face_joint_names.json}

python -m data_process.joint_annotation.face_select_rule \
    --input_dir "$INPUT_DIR" --output "$OUTPUT" "$@"
