#!/bin/bash
# Stage 2 — caption multi-view motion renders with a VLM.
# The backend is picked from MODEL, with the same rules as detect_backend() in
# vlm_caption/backends.py:
#   *qwen* (default, any case) -> local Qwen3-VL (GPU-bound; supports --multi-gpu)
#   gemini*                    -> Gemini API    (needs GOOGLE_API_KEY)
#   anything else              -> OpenAI-compatible API (needs OPENAI_API_KEY)
#
# Usage:
#   bash data_process/scripts/run_caption_motion.sh objaverse                # local Qwen3-VL
#   bash data_process/scripts/run_caption_motion.sh truebones --multi-gpu    # all visible GPUs (local backend only)
#   MODEL=gemini-3-flash-preview bash data_process/scripts/run_caption_motion.sh objaverse
#   MODEL=gpt-5-mini NUM_WORKERS=16 bash data_process/scripts/run_caption_motion.sh mixamo
#
# Env overrides:
#   MODEL, RENDER_ROOT (default: dataset/render/<dataset>),
#   OUTPUT_JSON (default: dataset/export/<dataset>/motion_captions.json),
#   DOWNSAMPLE_RATE, MAX_TOKENS, NUM_WORKERS, MAX_RETRIES, BASE_URL (OpenAI-compatible proxy),
#   THINKING_LEVEL / MEDIA_RESOLUTION (gemini-3), THINKING_BUDGET (gemini-2.5)
# Unrecognized arguments are passed through to caption_motion.py.
# On offline compute nodes with a populated HuggingFace cache, export
# HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1.

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

# ── Arguments ────────────────────────────────────────────────────────────────
handle_help "$@"
DATASET=${1:?Usage: run_caption_motion.sh <truebones|mixamo|objaverse> [--multi-gpu] [extra args...]}
shift
require_dataset "$DATASET"

MODE=single
PASSTHRU=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --multi-gpu) MODE=multi-gpu; shift ;;
        -h|--help) print_usage; exit 0 ;;
        # Everything else (e.g. --views v000 v002, --downsample_rate 4) goes to
        # the Python entry point, which owns the full flag set.
        *) PASSTHRU+=("$1"); shift ;;
    esac
done

# ── Backend from MODEL (mirrors vlm_caption/backends.py detect_backend) ─
MODEL=${MODEL:-Qwen/Qwen3-VL-8B-Instruct}
if [[ "${MODEL,,}" == *qwen* ]]; then
    backend=local                       # backends.py: BACKEND_QWEN
elif [[ "$MODEL" == gemini* ]]; then
    backend=gemini
else
    backend=openai
fi

if [[ "$MODE" == "multi-gpu" && "$backend" != "local" ]]; then
    echo "ERROR: --multi-gpu only applies to the local Qwen backend "\
"(API backends scale via NUM_WORKERS instead)." >&2
    exit 2
fi
[[ "$backend" == "gemini" ]] && require_env GOOGLE_API_KEY
[[ "$backend" == "openai" ]] && require_env OPENAI_API_KEY

# ── Paths ────────────────────────────────────────────────────────────────────
RENDER_ROOT=${RENDER_ROOT:-$(render_dir "$DATASET")}
OUTPUT_JSON=${OUTPUT_JSON:-$(export_dir "$DATASET")/motion_captions.json}
if [[ ! -d "$RENDER_ROOT" ]]; then
    echo "RENDER_ROOT does not exist: $RENDER_ROOT" >&2
    exit 2
fi
mkdir -p "$(dirname "$OUTPUT_JSON")"

# ── Backend-specific settings ────────────────────────────────────────────────
# Full 30 fps for every dataset: temporal continuity is what lets the model
# track heading turns and posture on appearance-poor subjects (measured on
# mixamo: 180-turn and sit/stand errors gone at 1x vs 2x downsampling).
DOWNSAMPLE_RATE=${DOWNSAMPLE_RATE:-1}
MAX_RETRIES=${MAX_RETRIES:-4}
EXTRA=()
case "$backend" in
    local)
        MAX_TOKENS=${MAX_TOKENS:-800} ;;
    gemini)
        # Reasoning/answer tokens share one budget -> generous default.
        # Default 2 workers — token/minute quotas are easy to exhaust with
        # many concurrent image-heavy requests.
        MAX_TOKENS=${MAX_TOKENS:-1024}
        NUM_WORKERS=${NUM_WORKERS:-2}
        EXTRA+=(--thinking_level="${THINKING_LEVEL:-low}"
                --media_resolution="${MEDIA_RESOLUTION:-medium}"
                --thinking_budget="${THINKING_BUDGET:-0}"
                --max_retries="$MAX_RETRIES"
                --num_workers="$NUM_WORKERS") ;;
    openai)
        # GPT-5 reasoning tokens count against max_completion_tokens.
        MAX_TOKENS=${MAX_TOKENS:-1024}
        NUM_WORKERS=${NUM_WORKERS:-4}
        EXTRA+=(--max_retries="$MAX_RETRIES" --num_workers="$NUM_WORKERS")
        [[ -n "${BASE_URL:-}" ]] && EXTRA+=(--base_url="$BASE_URL") ;;
esac

# Mixamo ships official catalogue labels and truebones the T2M4LVO
# annotations; pass them as verify-against-frames hints
# (HINTS_JSON overrides, HINTS_JSON="" disables).
case "$DATASET" in
    mixamo)    HINTS_JSON=${HINTS_JSON-dataset/raw/mixamo/animation_motion_prompts.json} ;;
    truebones) HINTS_JSON=${HINTS_JSON-dataset/raw/truebones/animation_prompts.json} ;;
    *)         HINTS_JSON=${HINTS_JSON:-} ;;
esac

COMMON=(--task="$DATASET" --render_root="$RENDER_ROOT" --model="$MODEL"
        --downsample_rate="$DOWNSAMPLE_RATE" --max_tokens="$MAX_TOKENS"
        --output_json="$OUTPUT_JSON")
[[ -n "$HINTS_JSON" ]] && COMMON+=(--hints_json="$HINTS_JSON")

echo "Dataset: $DATASET | Backend: $backend | Model: $MODEL | Mode: $MODE"
[[ -n "$HINTS_JSON" ]] && echo "Caption hints: $HINTS_JSON"
echo "Render root: $RENDER_ROOT"
echo "Output JSON: $OUTPUT_JSON"

# ── Run ──────────────────────────────────────────────────────────────────────
if [[ "$MODE" == "multi-gpu" ]]; then
    num_gpus=$(python -c "import torch; print(torch.cuda.device_count())")
    if ! [[ "$num_gpus" =~ ^[0-9]+$ ]] || [[ "$num_gpus" -lt 1 ]]; then
        echo "No visible CUDA devices (num_gpus=$num_gpus). Aborting --multi-gpu." >&2
        exit 1
    fi
    LOG_DIR=$(worker_log_dir "caption-motion-$DATASET")
    echo "Launching $num_gpus GPU processes (per-GPU logs: $LOG_DIR)..."

    pids=()
    for gpu_id in $(seq 0 $((num_gpus - 1))); do
        python -m data_process.vlm_caption.caption_motion "${COMMON[@]}" \
            --num_gpus="$num_gpus" --gpu_id="$gpu_id" \
            "${PASSTHRU[@]}" > >(tee "$LOG_DIR/gpu${gpu_id}.log") 2>&1 &
        pids+=($!)
        echo "  Started GPU $gpu_id (PID ${pids[-1]})"
    done

    failed=0
    for gpu_id in "${!pids[@]}"; do
        wait "${pids[gpu_id]}" \
            || { echo "GPU $gpu_id failed (log: $LOG_DIR/gpu${gpu_id}.log)" >&2; failed=1; }
    done
    if [[ $failed -ne 0 ]]; then
        echo "Some processes failed. Shard files preserved for inspection." >&2
        exit 1
    fi

    echo "Merging shards..."
    python -m data_process.vlm_caption.caption_motion \
        --task="$DATASET" --merge_shards --output_json="$OUTPUT_JSON" --num_gpus="$num_gpus"
    echo "Done! Output: $OUTPUT_JSON"
else
    python -m data_process.vlm_caption.caption_motion \
        "${COMMON[@]}" "${EXTRA[@]}" "${PASSTHRU[@]}"
fi
