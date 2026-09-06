#!/bin/bash
# Train a UniMate model on a single GPU.
#
# Usage:
#   bash scripts/run_train.sh [config] [-- extra args...]
#
# Arguments:
#   config     training config JSON (default: configs/uniml3d_60frames_graph_adaln.json)
#   extra      anything after `--` is forwarded to unimate.training.train,
#              e.g. --batch_size 8 --resume outputs/<run>/checkpoints/<ckpt>.pt
#
# Env overrides:
#   OUTPUT_DIR            (default: the config's experiment.output_dir)
#   CONDA_ENV             conda environment to activate (default: unimate)
#   CUDA_VISIBLE_DEVICES  GPU to use (default: the one with the most free memory)
#
# For multi-GPU runs use scripts/slurm/slurm_train.sh instead.

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
handle_help "$@"

CONFIG=${1:-configs/uniml3d_60frames_graph_adaln.json}
shift || true
[[ "${1:-}" == "--" ]] && shift

[[ -f "$CONFIG" ]] || { echo "No such config: $CONFIG" >&2; exit 1; }

select_gpu
echo "Config:     $CONFIG"
echo "Output dir: ${OUTPUT_DIR:-<config default>}"

CMD=(accelerate launch -m unimate.training.train --config "$CONFIG")
[[ -n "${OUTPUT_DIR:-}" ]] && CMD+=(--output_dir "$OUTPUT_DIR")

"${CMD[@]}" "$@"
