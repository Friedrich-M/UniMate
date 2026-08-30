# shellcheck shell=bash
# Shared prologue for every data_process/scripts/run_*.sh wrapper.
#
#   source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
#
# Resolves PROJECT_ROOT (the repository root), cds into it, activates the
# conda environment named by $CONDA_ENV (default: unimate), and defines the
# default dataset layout used by every stage. Override any DEFAULT_* path by
# exporting the corresponding variable before calling a wrapper.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
# Resolve the sourcing wrapper's own path *before* the cd below, so that
# print_usage still finds it when the wrapper was invoked by a relative path
# (e.g. `cd data_process/scripts && bash run_export.sh -h`).
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

# Avoid user startup hooks leaking into Blender / worker processes.
unset PYTHONSTARTUP PYTHONBREAKPOINT

# ── Default dataset layout (relative to PROJECT_ROOT) ───────────────────────
#   dataset/raw/<dataset>/                    raw FBX / GLB assets
#   dataset/export/<dataset>/                 stage-1 export + stage-2/3 metadata
#   dataset/render/<dataset>/                 multi-view renders (captioning input)
#   dataset/render/<dataset>_tpose/           T-pose grids (category input)
#   dataset/features/<dataset>/               stage-4 training clips + cond.npy
export_dir()    { echo "dataset/export/$1"; }
render_dir()    { echo "dataset/render/$1"; }
tpose_dir()     { echo "dataset/render/${1}_tpose"; }
features_dir()  { echo "dataset/features/$1"; }

# ── Helpers ──────────────────────────────────────────────────────────────────
# require_dataset <name> : validate a dataset name.
require_dataset() {
    case "$1" in
        truebones|mixamo|objaverse) ;;
        *) echo "Unknown dataset '$1' (expected: truebones | mixamo | objaverse)" >&2; exit 2 ;;
    esac
}

# require_env VAR : abort if the environment variable is unset/empty.
require_env() {
    if [[ -z "${!1:-}" ]]; then
        echo "ERROR: $1 is not set. Export it in your shell." >&2
        exit 1
    fi
}

# print_usage [script] : print the leading comment block of a wrapper
# (defaults to the wrapper that sourced this file).
print_usage() {
    sed -n '2,/^$/p' "${1:-$WRAPPER_PATH}" | sed 's/^# \{0,1\}//'
}

# handle_help "$@" : print the wrapper usage and exit 0 when the first
# argument is -h/--help. Call it right after sourcing this file, before any
# other argument validation, so `-h` works on every wrapper.
handle_help() {
    case "${1:-}" in
        -h|--help) print_usage; exit 0 ;;
    esac
}

# worker_log_dir <name> : create and echo a directory for per-worker logs, so
# that a failing shard of a parallel run can be attributed. Override with
# WORKER_LOG_DIR.
worker_log_dir() {
    local dir="${WORKER_LOG_DIR:-outputs/logs/$1}"
    mkdir -p "$dir"
    echo "$dir"
}
