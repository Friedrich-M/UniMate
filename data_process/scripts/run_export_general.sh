#!/bin/bash
# Stage 1 — export rigged GLB/GLTF/FBX assets to NPZ motion data.
# Input is a single file or a directory of assets (mixed formats welcome).
# Mesh optional (armature-only FBX works); every pose action becomes a clip.
#
# Usage:
#   bash data_process/scripts/run_export_general.sh <asset.glb|.fbx|dir> [extra args...]
#   bash data_process/scripts/run_export_general.sh my_assets/ --multi-worker 8 --no-vis
#   bash data_process/scripts/run_export_general.sh model.fbx --no-prune
#
# Env overrides: OUTPUT_DIR (default: dataset/export/custom)

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

handle_help "$@"
INPUT="${1:?Usage: run_export_general.sh <asset.glb|.fbx|dir> [extra args...]}"
shift
OUTPUT_DIR=${OUTPUT_DIR:-$(export_dir custom)}

NUM_WORKERS=1
EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --multi-worker) NUM_WORKERS="${2:?--multi-worker requires a value}"; shift 2 ;;
        *) EXTRA_ARGS+=("$1"); shift ;;
    esac
done

if [[ "$NUM_WORKERS" -gt 1 && -d "$INPUT" ]]; then
    LOG_DIR=$(worker_log_dir "export-general")
    echo "Launching $NUM_WORKERS export workers (per-worker logs: $LOG_DIR)..."
    pids=()
    for worker_id in $(seq 0 $((NUM_WORKERS - 1))); do
        blender -b -P data_process/motion_export/export_general.py -- \
            --input="$INPUT" --output_dir="$OUTPUT_DIR" \
            --worker_id="$worker_id" --num_workers="$NUM_WORKERS" \
            "${EXTRA_ARGS[@]}" > >(tee "$LOG_DIR/worker${worker_id}.log") 2>&1 &
        pids+=($!)
        echo "  Started worker $worker_id (PID ${pids[-1]})"
    done

    # A bare `wait` always returns 0 — check every worker individually, and
    # never merge a truncated set of shards (merging deletes them).
    failed=0
    for worker_id in "${!pids[@]}"; do
        wait "${pids[worker_id]}" \
            || { echo "Worker $worker_id failed (log: $LOG_DIR/worker${worker_id}.log)" >&2; failed=1; }
    done
    if [[ $failed -ne 0 ]]; then
        echo "Some export workers failed — NOT merging summary JSONs." >&2
        echo "Per-worker shards are preserved in $OUTPUT_DIR; rerun to finish." >&2
        exit 1
    fi
    echo "All workers finished. Merging summary JSONs..."
    python -m data_process.tools.merge_summaries --output_dir="$OUTPUT_DIR"
else
    if [[ "$NUM_WORKERS" -gt 1 ]]; then
        echo "WARNING: --multi-worker is ignored for a single-file input" \
             "($INPUT); running one exporter process." >&2
    fi
    blender -b -P data_process/motion_export/export_general.py -- \
        --input="$INPUT" --output_dir="$OUTPUT_DIR" "${EXTRA_ARGS[@]}"
fi
