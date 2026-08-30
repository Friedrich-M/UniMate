#!/bin/bash
# Stage 0 — download raw source assets from the Hugging Face Hub into
# dataset/raw/<dataset>/, the layout every downstream wrapper expects.
#
# Usage:
#   bash data_process/scripts/run_download.sh mixamo      # animations + rigged characters
#   bash data_process/scripts/run_download.sh objaverse   # rigged + animated GLBs
#   bash data_process/scripts/run_download.sh objaverse --include 'glb/*'   # skip metadata
#
# Sources:
#   mixamo     https://huggingface.co/datasets/Linzhan/Mixamo-Animations-Characters
#   objaverse  https://huggingface.co/datasets/Linzhan/Objaverse-XL-Rigged-Animated
#   truebones  https://huggingface.co/datasets/Linzhan/Truebones-ZOO-Annotations
#
# The truebones repo carries annotations only (prompts, per-clip metadata,
# T-pose renders, build scripts) — the Truebones ZOO animal pack itself is a
# commercial product that we cannot redistribute. Purchase it from Truebones
# (https://truebones.com), unpack it to dataset/raw/truebones/Truebone_Z-OO/
# (one folder per animal), and rebuild the per-clip animation/ layout with the
# downloaded scripts/pipeline/ (see the repo's README).
#
# Needs the `hf` CLI (https://hf.co/cli); gated/private repos also need
# `hf auth login` or HF_TOKEN.
#
# Env overrides: RAW_DIR (target directory), TRUEBONES_REPO
# Extra arguments are passed through to `hf download`.

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

handle_help "$@"
DATASET=${1:-}
[[ -n "$DATASET" ]] || { print_usage; exit 0; }
require_dataset "$DATASET"
shift

if ! command -v hf >/dev/null 2>&1; then
    echo "ERROR: the 'hf' CLI is not installed. Install it with:" >&2
    echo "  curl -LsSf https://hf.co/cli/install.sh | bash -s" >&2
    exit 1
fi

case "$DATASET" in
    mixamo)    REPO=Linzhan/Mixamo-Animations-Characters ;;
    objaverse) REPO=Linzhan/Objaverse-XL-Rigged-Animated ;;
    truebones)
        REPO=${TRUEBONES_REPO:-Linzhan/Truebones-ZOO-Annotations}
        cat >&2 <<'EOF'
NOTE: Truebones ZOO is a commercial asset pack and is not redistributed; this
      downloads annotations only (prompts, metadata, renders, build scripts).
      To obtain the motion data, purchase it from https://truebones.com, unpack
      it to dataset/raw/truebones/Truebone_Z-OO/ (one folder per animal), and
      rebuild animation/ with the downloaded scripts/pipeline/.
EOF
        ;;
esac
RAW_DIR=${RAW_DIR:-dataset/raw/$DATASET}

echo "Downloading $REPO -> $RAW_DIR"
if ! hf download "$REPO" --type dataset --local-dir "$RAW_DIR" "$@"; then
    if [[ "$DATASET" == "truebones" ]]; then
        echo "Download failed. Note that only annotations are hosted; the motion files" >&2
        echo "are commercial: purchase the Truebones ZOO pack from https://truebones.com" >&2
        echo "and unpack it to dataset/raw/truebones/Truebone_Z-OO/<Animal>/..." >&2
    fi
    exit 1
fi
