#!/bin/bash
# Stage 3 QA — compare each exported rest pose with its stage-4 facing
# canonicalization: [original rest pose with the face pair marked | the same pose
# corrected by the pair, facing +Z], one PNG per skeleton plus facing_summary.tsv.
#
# Usage:
#   bash data_process/scripts/run_joints_vis_facing.sh objaverse
#   RIGS="RIG1 RIG2" bash data_process/scripts/run_joints_vis_facing.sh objaverse
#   FACE_JSON=dataset/UniML3D/patches/objaverse_face_pairs.json OUTPUT_DIR=outputs/tmp \
#       bash data_process/scripts/run_joints_vis_facing.sh objaverse      # preview a patch
#   LIMIT=50 bash data_process/scripts/run_joints_vis_facing.sh truebones
#
# Env overrides:
#   FACE_JSON    face pairs to draw (default: <export_dir>/face_joint_names.json)
#   OUTPUT_DIR   PNG output directory (default: outputs/tpose_facing_vis/<dataset>)
#   RIGS         space-separated skeleton keys (default: every non-empty pair)
#   RIGS_FILE    text file with one skeleton key per line
#   SOURCE       only pairs with this "source" (thigh, shoulder, body_axis, ...)
#   LIMIT        stop after N skeletons
#   MARK_LABELS  comma-separated clean labels to mark with extra symbols, e.g. "Head,Tail End,Toe"
#   WORKERS      parallel renderers (default: min(8, cpus/2))
#   OVERWRITE=1  re-render existing PNGs

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

handle_help "$@"

DATASET=${1:?Usage: run_joints_vis_facing.sh <truebones|mixamo|objaverse>}
require_dataset "$DATASET"
EXPORT_DIR=$(export_dir "$DATASET")
[[ -d "$EXPORT_DIR/motions" ]] || { echo "export motions not found: $EXPORT_DIR/motions" >&2; exit 1; }

cmd=(python -m data_process.tools.vis_tpose_facing --export_dir "$EXPORT_DIR")
[[ -n "${FACE_JSON:-}" ]]  && cmd+=(--face_json "$FACE_JSON")
[[ -n "${OUTPUT_DIR:-}" ]] && cmd+=(--output_dir "$OUTPUT_DIR")
[[ -n "${RIGS:-}" ]]       && cmd+=(--rigs $RIGS)
[[ -n "${RIGS_FILE:-}" ]]  && cmd+=(--rigs_file "$RIGS_FILE")
[[ -n "${SOURCE:-}" ]]     && cmd+=(--source "$SOURCE")
[[ -n "${LIMIT:-}" ]]      && cmd+=(--limit "$LIMIT")
[[ -n "${MARK_LABELS:-}" ]] && cmd+=(--mark_labels "$MARK_LABELS")
[[ -n "${WORKERS:-}" ]]    && cmd+=(--workers "$WORKERS")
[[ "${OVERWRITE:-0}" == 1 ]] && cmd+=(--overwrite)

"${cmd[@]}"
