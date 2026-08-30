#!/bin/bash
# Stage 3 — render annotated T-pose PNGs from exported motion NPZs.
# Use the PNGs to identify left/right symmetric face joints by eye.
#
# Usage:
#   # Batch mode — one PNG per unique skeleton in the export directory:
#   bash data_process/scripts/run_joints_vis_tpose.sh truebones
#   LIMIT=50 bash data_process/scripts/run_joints_vis_tpose.sh truebones
#
#   # Single mode — one NPZ:
#   NPZ_PATH=/path/to/motion.npz bash data_process/scripts/run_joints_vis_tpose.sh
#   NPZ_PATH=/path/to/motion.npz OUTPUT_PATH=out.png bash data_process/scripts/run_joints_vis_tpose.sh
#
# Env overrides:
#   NPZ_PATH     single mode: path to one motion NPZ (takes precedence over batch mode)
#   OUTPUT_PATH  [single] output PNG path
#   MOTIONS_DIR  [batch] motion NPZ directory (default: <export_dir>/motions)
#   OUTPUT_DIR   [batch] PNG output directory (default: <export_dir>/face_joints_vis)
#   LIMIT        [batch] stop after N skeletons
#   FONT_SIZE    label font size (default: 7)

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

handle_help "$@"

cmd=(python -m data_process.tools.vis_tpose)

if [[ -n "${NPZ_PATH:-}" ]]; then
    [[ -f "$NPZ_PATH" ]] || { echo "NPZ_PATH not found: $NPZ_PATH" >&2; exit 1; }
    cmd+=(--npz_path "$NPZ_PATH")
    [[ -n "${OUTPUT_PATH:-}" ]] && cmd+=(--output "$OUTPUT_PATH")
else
    DATASET=${1:?Usage: run_joints_vis_tpose.sh <truebones|mixamo|objaverse>  (or set NPZ_PATH)}
    require_dataset "$DATASET"
    MOTIONS_DIR=${MOTIONS_DIR:-$(export_dir "$DATASET")/motions}
    OUTPUT_DIR=${OUTPUT_DIR:-$(export_dir "$DATASET")/face_joints_vis}
    [[ -d "$MOTIONS_DIR" ]] || { echo "MOTIONS_DIR not found: $MOTIONS_DIR" >&2; exit 1; }
    cmd+=(--motions_dir "$MOTIONS_DIR" --output_dir "$OUTPUT_DIR")
    [[ -n "${LIMIT:-}" ]] && cmd+=(--limit "$LIMIT")
fi
[[ -n "${FONT_SIZE:-}" ]] && cmd+=(--font-size "$FONT_SIZE")

"${cmd[@]}"
