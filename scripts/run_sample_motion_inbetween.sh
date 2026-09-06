#!/bin/bash
# Motion in-betweening: hold the listed frames of a reference clip at their
# ground truth and let the flow ODE generate the rest.
#
# Usage:
#   bash scripts/run_sample_motion_inbetween.sh <exp_dir> [test_cases_json] [cfg_scale]
#   KEEP_FRAMES="0,15,-1" bash scripts/run_sample_motion_inbetween.sh <exp_dir> cases.json
#
# Arguments:
#   exp_dir          training output directory (config.json, dataset_stats.npy, checkpoints/)
#   test_cases_json  {"<object_type>-<clip_id>": "prompt"} — keys must name dataset
#                    clips (their motion is the clamped ground truth); omit to use
#                    every clip of the eval split with its original caption
#   cfg_scale        classifier-free guidance scale (default: value saved in the run's config)
#
# Env overrides:
#   KEEP_FRAMES           comma-separated signed frame indices to hold; negatives
#                         count from the end (default: "0,-1" — first and last frame)
#   GT_START_FRAME        absolute frame at which the reference window starts
#                         (default: random window)
#   REPLICATE             samples per clip (default: 3); each draws a new window and noise
#   SEED                  integer seed for deterministic cropping and noise (default: unset)
#   OUTPUT_DIR            (default: <exp_dir>/samples[_<stem>]_inbetween_<keep tag>)
#   CONDA_ENV             conda environment to activate (default: unimate)
#   CUDA_VISIBLE_DEVICES  GPU to use (default: the one with the most free memory)
#
# Results land in <OUTPUT_DIR>/inbetween/, next to the clamped ground truth (gt_rep_<r>-*).

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
handle_help "$@"

EXP_DIR=${1:?Usage: run_sample_motion_inbetween.sh <exp_dir> [test_cases_json] [cfg_scale]}
TEST_CASES=${2:-}
CFG_SCALE=${3:-}
KEEP_FRAMES=${KEEP_FRAMES:-0,-1}
GT_START_FRAME=${GT_START_FRAME:-}
REPLICATE=${REPLICATE:-3}
SEED=${SEED:-}

TEST_NAME=${TEST_CASES:+$(basename "${TEST_CASES%.json}")}
KEEP_TAG=$(echo "$KEEP_FRAMES" | sed 's/,/_/g; s/-/n/g')   # "0,-1" -> "0_n1"
OUTPUT_DIR=${OUTPUT_DIR:-$EXP_DIR/samples${TEST_NAME:+_$TEST_NAME}_inbetween_${KEEP_TAG}}

select_gpu
echo "Exp dir:     $EXP_DIR"
echo "Test cases:  ${TEST_CASES:-<eval split>}"
echo "Cfg scale:   ${CFG_SCALE:-<config default>}"
echo "Keep frames: $KEEP_FRAMES"
echo "GT start:    ${GT_START_FRAME:-<random window>}"
echo "Replicate:   $REPLICATE"
echo "Output dir:  $OUTPUT_DIR/inbetween"

CMD=(python -m unimate.inference.sample
     --exp_dir "$EXP_DIR" --output_dir "$OUTPUT_DIR" --num_repetitions "$REPLICATE"
     --inbetween --keep_frames "$KEEP_FRAMES")
[[ -n "$TEST_CASES" ]]     && CMD+=(--test_cases_json "$TEST_CASES")
[[ -n "$CFG_SCALE" ]]      && CMD+=(--cfg_scale "$CFG_SCALE")
[[ -n "$GT_START_FRAME" ]] && CMD+=(--gt_start_frame "$GT_START_FRAME")
[[ -n "$SEED" ]]           && CMD+=(--seed "$SEED")

"${CMD[@]}"
