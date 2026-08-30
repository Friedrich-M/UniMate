#!/bin/bash
# Stage 2a — render multi-view frames of every motion (EEVEE), the input of
# VLM motion captioning (run_caption_motion.sh).
#
# Usage:
#   bash data_process/scripts/run_render_motion.sh truebones
#   bash data_process/scripts/run_render_motion.sh objaverse --multi-worker 8
#   bash data_process/scripts/run_render_motion.sh objaverse --missing-only
#   DATA_DIR=outputs/mixamo_characters bash data_process/scripts/run_render_motion.sh mixamo
#
# Inputs per dataset: truebones/objaverse render the raw FBX/GLB assets;
# mixamo renders the animated character files produced by
# run_animate_mixamo.sh (raw Mixamo FBXs carry no mesh).
#
# Runs with plain python + the pip `bpy` module: EEVEE needs the module's
# GPU context (`blender -b` has no display surface). --multi-worker N
# shards the assets over N processes; with NUM_GPUS>1 the workers are
# spread round-robin over the visible GPUs (worker i -> GPU i % NUM_GPUS),
# so N is the *total* worker count, not the per-GPU count.
#
# KNOWN LIMITATION: the CUDA_VISIBLE_DEVICES pinning does NOT bind EEVEE's
# EGL context — on a multi-GPU machine every worker still renders on the
# first GPU. Give each render process (or Slurm job) a machine/cgroup that
# exposes exactly one GPU instead of relying on NUM_GPUS>1.
#
# Env overrides: DATA_DIR, OUTPUT_DIR, RESOLUTION, SAMPLES, CAMERA_DIST,
# MAX_RENDER_FRAMES (per-clip frame cap, default 200), MIN_ACTION_FRAMES,
# NUM_GPUS (default 1). Extra arguments are passed through to the renderer.

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

handle_help "$@"
DATASET=${1:-}
[[ -n "$DATASET" ]] || { print_usage; exit 0; }
require_dataset "$DATASET"
shift

NUM_WORKERS=1
NUM_GPUS=${NUM_GPUS:-1}
EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --multi-worker) NUM_WORKERS="${2:?--multi-worker requires a value}"; shift 2 ;;
        *) EXTRA_ARGS+=("$1"); shift ;;
    esac
done

case "$DATASET" in
    truebones) DATA_DIR=${DATA_DIR:-dataset/raw/truebones/animation} ;;
    mixamo)    DATA_DIR=${DATA_DIR:-outputs/mixamo_characters} ;;
    objaverse) DATA_DIR=${DATA_DIR:-dataset/raw/objaverse/glb} ;;
esac
OUTPUT_DIR=${OUTPUT_DIR:-$(render_dir "$DATASET")}

RENDERER="data_process.motion_rendering.render_${DATASET}"
COMMON_ARGS=(
    --data_dir="$DATA_DIR"
    --output_dir="$OUTPUT_DIR"
    --resolution="${RESOLUTION:-512}"
    --samples="${SAMPLES:-64}"
    --camera_dist="${CAMERA_DIST:-1.5}"
)
[[ -n "${MAX_RENDER_FRAMES:-}" ]] && COMMON_ARGS+=(--max_render_frames="$MAX_RENDER_FRAMES")
[[ -n "${MIN_ACTION_FRAMES:-}" ]] && COMMON_ARGS+=(--min_action_frames="$MIN_ACTION_FRAMES")

echo "Dataset: $DATASET | Input: $DATA_DIR | Output: $OUTPUT_DIR"
if [[ "$NUM_WORKERS" -gt 1 ]]; then
    LOG_DIR=$(worker_log_dir "render-motion-$DATASET")
    echo "Launching $NUM_WORKERS render workers (per-worker logs: $LOG_DIR)..."
    pids=()
    for worker_id in $(seq 0 $((NUM_WORKERS - 1))); do
        # Pin each worker to one GPU before the process starts: EEVEE picks
        # its device when the bpy module builds its GL context, so the
        # restriction has to be in the environment, not set from Python.
        gpu=$((worker_id % NUM_GPUS))
        CUDA_VISIBLE_DEVICES="$gpu" \
        python -m "$RENDERER" "${COMMON_ARGS[@]}" \
            --worker_id="$worker_id" --num_workers="$NUM_WORKERS" \
            "${EXTRA_ARGS[@]}" > >(tee "$LOG_DIR/worker${worker_id}.log") 2>&1 &
        pids+=($!)
        echo "  Started worker $worker_id on GPU $gpu (PID ${pids[-1]})"
    done

    # A bare `wait` always returns 0 — check every worker individually.
    failed=0
    for worker_id in "${!pids[@]}"; do
        wait "${pids[worker_id]}" \
            || { echo "Worker $worker_id failed (log: $LOG_DIR/worker${worker_id}.log)" >&2; failed=1; }
    done
    if [[ $failed -ne 0 ]]; then
        echo "Some render workers failed. Rerun to retry the missing frames." >&2
        exit 1
    fi
    echo "All workers finished."
else
    python -m "$RENDERER" "${COMMON_ARGS[@]}" "${EXTRA_ARGS[@]}"
fi
