#!/bin/bash
# Stage 4 — process exported NPZ motions into canonicalized training clips.
#
# Usage:
#   bash data_process/scripts/run_extract_features.sh truebones
#   bash data_process/scripts/run_extract_features.sh objaverse
#   bash data_process/scripts/run_extract_features.sh mixamo
#   APPLY_CLIP=1 bash data_process/scripts/run_extract_features.sh truebones
#   NUM_WORKERS=8 NO_VIS=1 bash data_process/scripts/run_extract_features.sh objaverse
#
# Env overrides:
#   DATA_DIR     Stage-1 export directory (default: dataset/export/<dataset>)
#   SAVE_DIR     Output directory (default: dataset/features/<dataset>)
#   APPLY_CLIP   If 1/true, crop long motions into overlapping fixed-length
#                clips; otherwise (default) keep only the first max_clip_len frames.
#   NUM_WORKERS  Parallel worker processes over object types (default: 1)
#   NO_VIS       If 1/true, skip the per-clip MP4 previews
# Extra arguments are passed through to extract_features.py.

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

handle_help "$@"

DATASET=${1:?Usage: run_extract_features.sh <truebones|mixamo|objaverse> [extra args...]}
shift
require_dataset "$DATASET"

DATA_DIR=${DATA_DIR:-$(export_dir "$DATASET")}
SAVE_DIR=${SAVE_DIR:-$(features_dir "$DATASET")}

EXTRA_ARGS=()
# Per-dataset joint-count range (skeletons outside it are skipped here, not
# at export — the exports stay complete). Defaults in extract_features.py
# are 8..150; objaverse keeps its wider historical range.
case "$DATASET" in
    objaverse) EXTRA_ARGS+=(--min_joints 4 --max_joints 180) ;;
esac
case "${APPLY_CLIP:-0}" in
    1|true|TRUE) EXTRA_ARGS+=(--apply_clip) ;;
esac
case "${NO_VIS:-0}" in
    1|true|TRUE) EXTRA_ARGS+=(--no-vis) ;;
esac
[[ "${NUM_WORKERS:-1}" -gt 1 ]] && EXTRA_ARGS+=(--num_workers="$NUM_WORKERS")

python -m data_process.feature_extraction.extract_features \
    --dataset_type="$DATASET" \
    --data_dir="$DATA_DIR" \
    --save_dir="$SAVE_DIR" \
    "${EXTRA_ARGS[@]}" "$@"
