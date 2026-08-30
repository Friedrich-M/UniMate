#!/bin/bash
# Stage 2a — render a T-pose 2x2 grid per asset (EEVEE), the input of
# body-plan category classification (run_caption_category.sh).
#
# Usage:
#   bash data_process/scripts/run_render_tpose.sh truebones
#   bash data_process/scripts/run_render_tpose.sh objaverse --multi-worker 8
#   DATA_DIR=outputs/mixamo_characters bash data_process/scripts/run_render_tpose.sh mixamo
#
# Grids are written flat as <tpose_dir>/<name>.png (one per GLB stem, or
# per Truebones object type). Runs with plain python + the pip `bpy`
# module: EEVEE needs the module's GPU context.
#
# Env overrides: DATA_DIR, OUTPUT_DIR, RESOLUTION, SAMPLES, CAMERA_DIST
# Extra arguments are passed through to the renderer.

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

handle_help "$@"
DATASET=${1:-}
[[ -n "$DATASET" ]] || { print_usage; exit 0; }
require_dataset "$DATASET"
shift

NUM_WORKERS=1
EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --multi-worker) NUM_WORKERS="${2:?--multi-worker requires a value}"; shift 2 ;;
        *) EXTRA_ARGS+=("$1"); shift ;;
    esac
done

case "$DATASET" in
    truebones) DATA_DIR=${DATA_DIR:-dataset/raw/truebones/animation}
               EXTRA_ARGS+=(--species_grids) ;;     # one grid per species
    mixamo)    DATA_DIR=${DATA_DIR:-dataset/raw/mixamo/character_refined} ;;
    objaverse) DATA_DIR=${DATA_DIR:-dataset/raw/objaverse/glb} ;;
esac
OUTPUT_DIR=${OUTPUT_DIR:-$(tpose_dir "$DATASET")}

COMMON_ARGS=(
    --data_dir="$DATA_DIR"
    --output_dir="$OUTPUT_DIR"
    --resolution="${RESOLUTION:-512}"
    --samples="${SAMPLES:-64}"
    --camera_dist="${CAMERA_DIST:-1.5}"
)

echo "Dataset: $DATASET | Input: $DATA_DIR | Output: $OUTPUT_DIR"
if [[ "$NUM_WORKERS" -gt 1 ]]; then
    LOG_DIR=$(worker_log_dir "render-tpose-$DATASET")
    echo "Launching $NUM_WORKERS render workers (per-worker logs: $LOG_DIR)..."
    pids=()
    for worker_id in $(seq 0 $((NUM_WORKERS - 1))); do
        python -m data_process.motion_rendering.render_tpose "${COMMON_ARGS[@]}" \
            --worker_id="$worker_id" --num_workers="$NUM_WORKERS" \
            "${EXTRA_ARGS[@]}" > >(tee "$LOG_DIR/worker${worker_id}.log") 2>&1 &
        pids+=($!)
        echo "  Started worker $worker_id (PID ${pids[-1]})"
    done

    # A bare `wait` always returns 0 — check every worker individually.
    failed=0
    for worker_id in "${!pids[@]}"; do
        wait "${pids[worker_id]}" \
            || { echo "Worker $worker_id failed (log: $LOG_DIR/worker${worker_id}.log)" >&2; failed=1; }
    done
    if [[ $failed -ne 0 ]]; then
        echo "Some render workers failed. Rerun to retry the missing grids." >&2
        exit 1
    fi
    echo "All workers finished."
else
    python -m data_process.motion_rendering.render_tpose "${COMMON_ARGS[@]}" "${EXTRA_ARGS[@]}"
fi
