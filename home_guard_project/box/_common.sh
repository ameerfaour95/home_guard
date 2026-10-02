#!/usr/bin/env bash
# Shared setup for the box runner scripts. Source this, do not execute it.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

LOG_DIR="$PROJECT_ROOT/logs"
mkdir -p "$LOG_DIR"

# uv installs to ~/.local/bin, which a scheduled task's PATH may not include.
export PATH="$HOME/.local/bin:$PATH"
export PYTHONUNBUFFERED=1

log() { echo "$(date '+%Y-%m-%d %H:%M:%S')  $*"; }

# Create or refresh the virtualenv. Offline is fine once it exists.
ensure_venv() {
    if command -v uv &>/dev/null; then
        uv sync --python 3.12 --quiet || log "uv sync failed (offline?) - using the existing environment"
    else
        log "uv not found on PATH - using the existing environment"
    fi
    PY="$PROJECT_ROOT/.venv/Scripts/python.exe"
    [[ -x "$PY" ]] || PY="$PROJECT_ROOT/.venv/bin/python"
    if [[ ! -x "$PY" ]]; then
        log "No Python found in .venv - run setup_box.ps1 first"
        return 1
    fi
}
