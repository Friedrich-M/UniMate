#!/bin/bash
# Text-conditioned sampling from a trained UniMate run.
#
# Usage:
#   bash scripts/run_sample_motion_text.sh <exp_dir> [test_cases_json] [cfg_scale]
#
# Arguments:
#   exp_dir          training output directory (config.json, dataset_stats.npy, checkpoints/)
#   test_cases_json  {"<object_type>-<case_id>": "prompt"}; omit to enumerate the
#                    dataset's eval split (or its train prompts when there is none)
#   cfg_scale        classifier-free guidance scale (default: value saved in the run's config)
#
# Env overrides:
#   REPLICATE             samples per test case (default: 3)
#   SEED                  integer seed for deterministic noise (default: unset)
#   OUTPUT_DIR            (default: <exp_dir>/samples[_<test_cases stem>])
#   CONDA_ENV             conda environment to activate (default: unimate)
#   CUDA_VISIBLE_DEVICES  GPU to use (default: the one with the most free memory)

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
handle_help "$@"

EXP_DIR=${1:?Usage: run_sample_motion_text.sh <exp_dir> [test_cases_json] [cfg_scale]}
TEST_CASES=${2:-}
CFG_SCALE=${3:-}
REPLICATE=${REPLICATE:-3}
SEED=${SEED:-}

TEST_NAME=${TEST_CASES:+$(basename "${TEST_CASES%.json}")}
OUTPUT_DIR=${OUTPUT_DIR:-$EXP_DIR/samples${TEST_NAME:+_$TEST_NAME}}

select_gpu
echo "Exp dir:    $EXP_DIR"
echo "Test cases: ${TEST_CASES:-<dataset split>}"
echo "Cfg scale:  ${CFG_SCALE:-<config default>}"
echo "Replicate:  $REPLICATE"
echo "Output dir: $OUTPUT_DIR"

CMD=(python -m unimate.inference.sample
     --exp_dir "$EXP_DIR" --output_dir "$OUTPUT_DIR" --num_repetitions "$REPLICATE")
[[ -n "$TEST_CASES" ]] && CMD+=(--test_cases_json "$TEST_CASES")
[[ -n "$CFG_SCALE" ]]  && CMD+=(--cfg_scale "$CFG_SCALE")
[[ -n "$SEED" ]]       && CMD+=(--seed "$SEED")

"${CMD[@]}"
