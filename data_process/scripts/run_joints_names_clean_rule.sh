#!/bin/bash
# Stage 3 — rule-based joint-name cleaning: joint_names.json -> clean_joint_names.json.
#
# Usage:
#   bash data_process/scripts/run_joints_names_clean_rule.sh <truebones|mixamo|objaverse> [--report]
#
# Env overrides: INPUT, OUTPUT (default: inside dataset/export/<dataset>)

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

handle_help "$@"

DATASET=${1:?Usage: run_joints_names_clean_rule.sh <truebones|mixamo|objaverse> [extra args...]}
shift
require_dataset "$DATASET"

INPUT=${INPUT:-$(export_dir "$DATASET")/joint_names.json}
OUTPUT=${OUTPUT:-$(export_dir "$DATASET")/clean_joint_names.json}

python -m data_process.joint_annotation.names_clean_rule \
    --input="$INPUT" --output="$OUTPUT" "$@"
