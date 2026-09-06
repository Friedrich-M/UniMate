#!/bin/bash
# Text-guided motion editing: hold the listed joints of a reference clip at
# their ground-truth motion and regenerate the remaining joints under a new
# prompt (e.g. keep the lower body of a walk and add an upper-body wave).
#
# Usage:
#   KEEP_JOINTS="Hips,LeftUpLeg,LeftLeg,RightUpLeg,RightLeg" \
#       bash scripts/run_sample_motion_edit.sh <exp_dir> [test_cases_json] [cfg_scale]
#
# Arguments:
#   exp_dir          training output directory (config.json, dataset_stats.npy, checkpoints/)
#   test_cases_json  {"<object_type>-<clip_id>": "edited prompt"} — keys must name
#                    dataset clips (their motion is the clamped ground truth); omit
#                    to use every clip of the eval split with its original caption
#   cfg_scale        classifier-free guidance scale (default: value saved in the run's config)
#
# Env overrides:
#   KEEP_JOINTS           (required) comma-separated joint names to hold, matched
#                         case-insensitively against the rig's joint names
#   GT_START_FRAME        absolute frame at which the reference window starts
#                         (default: random window)
#   REPLICATE             edits per clip (default: 3); each draws a new window and noise
#   SEED                  integer seed for deterministic cropping and noise (default: unset)
#   OUTPUT_DIR            (default: <exp_dir>/samples[_<stem>]_edit_<first joint>_n<count>)
#   CONDA_ENV             conda environment to activate (default: unimate)
#   CUDA_VISIBLE_DEVICES  GPU to use (default: the one with the most free memory)
#
# Results land in <OUTPUT_DIR>/motion_edit/, next to the clamped ground truth (gt_rep_<r>-*).

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
handle_help "$@"

EXP_DIR=${1:?Usage: run_sample_motion_edit.sh <exp_dir> [test_cases_json] [cfg_scale]}
TEST_CASES=${2:-}
CFG_SCALE=${3:-}
KEEP_JOINTS=${KEEP_JOINTS:?Set KEEP_JOINTS to a comma-separated list of joint names to hold}
GT_START_FRAME=${GT_START_FRAME:-}
REPLICATE=${REPLICATE:-3}
SEED=${SEED:-}

TEST_NAME=${TEST_CASES:+$(basename "${TEST_CASES%.json}")}
# Filesystem-safe tag for the keep list: first joint name + joint count.
FIRST_JOINT=$(echo "$KEEP_JOINTS" | cut -d',' -f1 | tr -cd '[:alnum:]_')
N_JOINTS=$(echo "$KEEP_JOINTS" | awk -F',' '{print NF}')
OUTPUT_DIR=${OUTPUT_DIR:-$EXP_DIR/samples${TEST_NAME:+_$TEST_NAME}_edit_${FIRST_JOINT}_n${N_JOINTS}}

select_gpu
echo "Exp dir:     $EXP_DIR"
echo "Test cases:  ${TEST_CASES:-<eval split>}"
echo "Cfg scale:   ${CFG_SCALE:-<config default>}"
echo "Keep joints: $KEEP_JOINTS"
echo "GT start:    ${GT_START_FRAME:-<random window>}"
echo "Replicate:   $REPLICATE"
echo "Output dir:  $OUTPUT_DIR/motion_edit"

CMD=(python -m unimate.inference.sample
     --exp_dir "$EXP_DIR" --output_dir "$OUTPUT_DIR" --num_repetitions "$REPLICATE"
     --motion_edit --keep_joints "$KEEP_JOINTS")
[[ -n "$TEST_CASES" ]]     && CMD+=(--test_cases_json "$TEST_CASES")
[[ -n "$CFG_SCALE" ]]      && CMD+=(--cfg_scale "$CFG_SCALE")
[[ -n "$GT_START_FRAME" ]] && CMD+=(--gt_start_frame "$GT_START_FRAME")
[[ -n "$SEED" ]]           && CMD+=(--seed "$SEED")

"${CMD[@]}"
