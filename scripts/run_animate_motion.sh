#!/bin/bash
# Drive a rigged character with motion clips — generated .npy or extracted .npz —
# and export GLB + FBX (batch front-end for stage 5 of data_process/).
#
# Usage:
#   bash scripts/run_animate_motion.sh <DATASET_TYPE> <INPUT...> [OUTPUT_DIR]
#   bash scripts/run_animate_motion.sh -h | --help
#
# DATASET_TYPE:  truebones | objaverse | mixamo
# INPUT...:      one or more .npy / .npz files, or a single directory containing them
# OUTPUT_DIR:    directory to save .glb / .fbx outputs (default: outputs/animated)
#                Detected as the last arg when it does not end in .npy / .npz.
#
# Env overrides:
#   ANIM_MODE (default: fk), CHAR_PATH (default: auto),
#   COND_PATH (default: dataset/features/<DATASET_TYPE>/cond.npy)
#
# Output: $OUTPUT_DIR/<anim_stem>.{glb,fbx}

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

STAGE1_WRAPPER="$PROJECT_ROOT/data_process/scripts/run_animate_motion.sh"
# COND_PATH is an optional override; when unset, the data_process wrapper
# defaults to the feature-extraction layout: dataset/features/<dataset>/cond.npy.

if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
    sed -n '2,/^$/p' "$0" | sed 's/^# \{0,1\}//'
    exit 0
fi

if [ "$#" -lt 2 ]; then
    echo "ERROR: expected at least 2 positional args: <DATASET_TYPE> <INPUT...> [OUTPUT_DIR]" >&2
    echo "Usage: bash $(basename "$0") <DATASET_TYPE> <INPUT...> [OUTPUT_DIR]" >&2
    exit 1
fi

DATASET_TYPE="$1"
shift

case "$DATASET_TYPE" in
    truebones|objaverse|mixamo) ;;
    *)
        echo "ERROR: unknown DATASET_TYPE='$DATASET_TYPE' (expected: truebones | objaverse | mixamo)" >&2
        exit 1
        ;;
esac

# The last arg is OUTPUT_DIR iff it doesn't look like a motion file
# (and there are at least 2 remaining args, so an input is still present).
# Use a portable last-arg lookup so this still works when bash is invoked as `sh`
# (where `${!#}` and `${@:1:N}` can misbehave under POSIX mode).
ALL_ARGS=("$@")
NARGS=$#
LAST_ARG="${ALL_ARGS[$((NARGS - 1))]}"
case "$LAST_ARG" in
    *.npy|*.npz)
        OUTPUT_DIR="outputs/animated"
        INPUT_ARGS=("${ALL_ARGS[@]}")
        ;;
    *)
        if [ "$NARGS" -eq 1 ]; then
            OUTPUT_DIR="outputs/animated"
            INPUT_ARGS=("${ALL_ARGS[@]}")
        else
            OUTPUT_DIR="$LAST_ARG"
            unset 'ALL_ARGS[NARGS - 1]'
            INPUT_ARGS=("${ALL_ARGS[@]}")
        fi
        ;;
esac

ANIM_PATHS=()
if [ "${#INPUT_ARGS[@]}" -eq 1 ] && [ -d "${INPUT_ARGS[0]}" ]; then
    INPUT_DIR="${INPUT_ARGS[0]}"
    shopt -s nullglob
    ANIM_PATHS=("$INPUT_DIR"/*.npy "$INPUT_DIR"/*.npz)
    shopt -u nullglob
    if [ "${#ANIM_PATHS[@]}" -eq 0 ]; then
        echo "ERROR: no .npy / .npz motion files found in: '$INPUT_DIR'" >&2
        exit 1
    fi
    INPUT_DESC="$INPUT_DIR"
else
    for f in "${INPUT_ARGS[@]}"; do
        if [ ! -f "$f" ]; then
            echo "ERROR: input not found: '$f'" >&2
            exit 1
        fi
        ANIM_PATHS+=("$f")
    done
    INPUT_DESC="${#INPUT_ARGS[@]} file(s)"
fi

ANIM_MODE=${ANIM_MODE:-fk}

mkdir -p "$OUTPUT_DIR"

process_one_motion() {
    local anim_path="$1"
    local idx="$2"
    local total="$3"

    local anim_stem
    anim_stem=$(basename "$anim_path")
    anim_stem=${anim_stem%.*}

    local exported_fbx="$OUTPUT_DIR/$anim_stem.fbx"
    local exported_glb="$OUTPUT_DIR/$anim_stem.glb"

    echo "############################################################"
    echo "# [$idx/$total] $anim_stem"
    echo "#   ANIM_PATH    = $anim_path"
    echo "#   DATASET_TYPE = $DATASET_TYPE"
    echo "#   OUTPUT_DIR   = $OUTPUT_DIR"
    echo "#   ANIM_MODE    = $ANIM_MODE"
    echo "#   CHAR_PATH    = ${CHAR_PATH:-<auto>}"
    echo "############################################################"
    (
        export OUTPUT_DIR ANIM_MODE
        export ANIM_PATH="$anim_path"
        if [ -n "${COND_PATH:-}" ]; then
            export COND_PATH
        fi
        if [ -n "${CHAR_PATH:-}" ]; then
            export CHAR_PATH
        fi
        bash "$STAGE1_WRAPPER" "$DATASET_TYPE"
    )

    if [ ! -f "$exported_fbx" ]; then
        echo "ERROR: did not produce expected FBX at '$exported_fbx'" >&2
        return 1
    fi

    echo "[$idx/$total] DONE  glb=$exported_glb  fbx=$exported_fbx"
}

TOTAL=${#ANIM_PATHS[@]}
echo "============================================================"
echo "DATASET_TYPE = $DATASET_TYPE   (${TOTAL} motion(s))"
echo "INPUT        = $INPUT_DESC"
echo "OUTPUT_DIR   = $OUTPUT_DIR"
for ((i = 0; i < TOTAL; i++)); do
    echo "  [$((i + 1))/$TOTAL] ${ANIM_PATHS[$i]}"
done
echo "============================================================"

for ((i = 0; i < TOTAL; i++)); do
    process_one_motion "${ANIM_PATHS[$i]}" "$((i + 1))" "$TOTAL"
done

echo "============================================================"
echo "BATCH DONE — ${TOTAL} motion(s) processed."
echo "============================================================"
