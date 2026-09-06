# shellcheck shell=bash
# Shared prologue for every scripts/run_*.sh wrapper.
#
#   source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
#
# Resolves PROJECT_ROOT (the repository root), cds into it, activates the
# conda environment named by $CONDA_ENV (default: unimate) when conda is
# available, and defines the helpers below.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
# Resolve the sourcing wrapper's own path before the cd below so that
# print_usage still finds it when the wrapper was invoked by a relative path.
WRAPPER_PATH="${BASH_SOURCE[1]:-$0}"
[[ "$WRAPPER_PATH" == /* ]] || WRAPPER_PATH="$PWD/$WRAPPER_PATH"
cd "$PROJECT_ROOT"

CONDA_ENV=${CONDA_ENV:-unimate}
if command -v conda >/dev/null 2>&1; then
    # conda activation hooks reference unset variables; relax `set -u` around them.
    case $- in *u*) _had_nounset=1 ;; *) _had_nounset=0 ;; esac
    set +u
    eval "$(conda shell.bash hook 2>/dev/null)"
    conda activate "$CONDA_ENV" 2>/dev/null \
        || echo "warning: could not activate conda env '$CONDA_ENV'; using current python" >&2
    [[ "$_had_nounset" == 1 ]] && set -u
fi

# print_usage : print the leading comment block of the calling wrapper.
print_usage() {
    sed -n '2,/^$/p' "$WRAPPER_PATH" | sed 's/^# \{0,1\}//'
}

# handle_help "$@" : print the wrapper usage and exit 0 on -h / --help.
# Call it right after sourcing this file, before any other argument checks.
handle_help() {
    case "${1:-}" in
        -h|--help) print_usage; exit 0 ;;
    esac
}

# select_gpu : unless CUDA_VISIBLE_DEVICES is already set, pin the run to the
# GPU with the most free memory. No-op when nvidia-smi is unavailable.
select_gpu() {
    if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
        echo "GPU:        CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
        return 0
    fi
    command -v nvidia-smi >/dev/null 2>&1 || return 0
    local gpu
    gpu=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits \
          | awk '{print NR-1, $1}' | sort -k2 -nr | head -n 1 | awk '{print $1}')
    [[ -n "$gpu" ]] || return 0
    export CUDA_VISIBLE_DEVICES=$gpu
    echo "GPU:        $gpu (most free memory)"
}
