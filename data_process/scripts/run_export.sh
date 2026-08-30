#!/bin/bash
# Stage 1 — export raw rigged assets (GLB/GLTF and FBX) to NPZ motion data.
#
# Usage:
#   bash data_process/scripts/run_export.sh truebones           # dataset preset (flat {Species}-{Action}.fbx clips)
#   bash data_process/scripts/run_export.sh mixamo              # dataset preset (animation-only FBX)
#   bash data_process/scripts/run_export.sh objaverse           # dataset preset (GLB/GLTF)
#   bash data_process/scripts/run_export.sh objaverse --multi-worker 8   # parallel (mixamo too)
#   bash data_process/scripts/run_export.sh mixamo --multi-worker 8 --no-vis   # skip MP4 previews
#   DATA_DIR=my_assets bash data_process/scripts/run_export.sh  # auto: .glb/.gltf and .fbx side by side
#
# --multi-worker only applies to a *directory* input; truebones is exported by
# a single-process exporter and ignores it (with a warning).
#
# Auto mode runs export_general.py on the directory: GLB/GLTF and FBX side
# by side, importer picked per file by extension, mesh optional (armature-only
# FBX works too).
#
# Env overrides: DATA_DIR, OUTPUT_DIR
# Extra arguments are passed through to the exporter.

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

EXPORTERS=data_process/motion_export

# ── Parse mode + flags ───────────────────────────────────────────────────────
handle_help "$@"

MODE=auto
case "${1:-}" in
    truebones|mixamo|objaverse|auto) MODE="$1"; shift ;;
esac

NUM_WORKERS=1
EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --multi-worker) NUM_WORKERS="${2:?--multi-worker requires a value}"; shift 2 ;;
        *) EXTRA_ARGS+=("$1"); shift ;;
    esac
done

# ── Per-mode defaults ────────────────────────────────────────────────────────
case "$MODE" in
    truebones)
        DATA_DIR=${DATA_DIR:-dataset/raw/truebones/animation}
        OUTPUT_DIR=${OUTPUT_DIR:-$(export_dir truebones)} ;;
    mixamo)
        DATA_DIR=${DATA_DIR:-dataset/raw/mixamo/animation_motion}
        OUTPUT_DIR=${OUTPUT_DIR:-$(export_dir mixamo)} ;;
    objaverse)
        DATA_DIR=${DATA_DIR:-dataset/raw/objaverse/glb}
        OUTPUT_DIR=${OUTPUT_DIR:-$(export_dir objaverse)} ;;
    auto)
        DATA_DIR=${DATA_DIR:?auto mode requires DATA_DIR (directory of .glb/.gltf/.fbx assets)}
        OUTPUT_DIR=${OUTPUT_DIR:-$(export_dir custom)} ;;
esac

# ── Exporter invocations ─────────────────────────────────────────────────────
# launch_workers <script> <input flag>: run one exporter, sharding over
# NUM_WORKERS parallel Blender processes and merging the per-worker summary
# JSONs afterwards. Only a directory input is sharded — the exporters write
# unsuffixed canonical summaries for a single file, which N workers would
# concurrently overwrite.
launch_workers() {
    local script="$1" input_flag="$2"
    if [[ "$NUM_WORKERS" -gt 1 && -d "$DATA_DIR" ]]; then
        local log_dir
        log_dir=$(worker_log_dir "export-$MODE")
        echo "Launching $NUM_WORKERS export workers (per-worker logs: $log_dir)..."
        local pids=() worker_id failed=0
        for worker_id in $(seq 0 $((NUM_WORKERS - 1))); do
            blender -b -P "$EXPORTERS/$script" -- \
                "$input_flag=$DATA_DIR" --output_dir="$OUTPUT_DIR" \
                --worker_id="$worker_id" --num_workers="$NUM_WORKERS" \
                "${EXTRA_ARGS[@]}" > >(tee "$log_dir/worker${worker_id}.log") 2>&1 &
            pids+=($!)
            echo "  Started worker $worker_id (PID ${pids[-1]})"
        done

        # A bare `wait` always returns 0 — check every worker individually.
        # Merging a truncated set of shards would silently shrink the
        # canonical JSONs and then delete the shards, so never merge on failure.
        for worker_id in "${!pids[@]}"; do
            wait "${pids[worker_id]}" \
                || { echo "Worker $worker_id failed (log: $log_dir/worker${worker_id}.log)" >&2; failed=1; }
        done
        if [[ $failed -ne 0 ]]; then
            echo "Some export workers failed — NOT merging summary JSONs." >&2
            echo "Per-worker shards are preserved in $OUTPUT_DIR; rerun to finish," >&2
            echo "then merge with: python -m data_process.tools.merge_summaries --output_dir=$OUTPUT_DIR" >&2
            exit 1
        fi
        echo "All workers finished. Merging summary JSONs..."
        python -m data_process.tools.merge_summaries --output_dir="$OUTPUT_DIR"
    else
        if [[ "$NUM_WORKERS" -gt 1 ]]; then
            echo "WARNING: --multi-worker is ignored for a single-file input" \
                 "($DATA_DIR); running one exporter process." >&2
        fi
        blender -b -P "$EXPORTERS/$script" -- \
            "$input_flag=$DATA_DIR" --output_dir="$OUTPUT_DIR" "${EXTRA_ARGS[@]}"
    fi
}

export_glb()    { launch_workers export_objaverse.py --data_dir; }
export_mixamo() { launch_workers export_mixamo.py --anim_dir; }

export_fbx() {
    if [[ "$NUM_WORKERS" -gt 1 ]]; then
        echo "WARNING: --multi-worker is ignored for truebones — export_truebones.py" \
             "is single-process; running one exporter process." >&2
    fi
    blender -b -P "$EXPORTERS/export_truebones.py" -- \
        --data_dir="$DATA_DIR" --output_dir="$OUTPUT_DIR" "$@" "${EXTRA_ARGS[@]}"
}

# ── Dispatch ─────────────────────────────────────────────────────────────────
echo "Mode: $MODE | Input: $DATA_DIR | Output: $OUTPUT_DIR"
case "$MODE" in
    objaverse) export_glb ;;
    mixamo)    export_mixamo ;;
    truebones) export_fbx ;;
    auto)      launch_workers export_general.py --input ;;
esac
