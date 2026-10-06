#!/usr/bin/env bash
# Shared setup for the box runner scripts. Source this, do not execute it.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

# Where this box keeps its config, data and logs (paths.py has the same rule):
#   $HOMEGUARD_HOME (a folder, or "legacy"), else C:\ProgramData\HomeGuard once it has been
#   migrated (its layout.json marker), else the old places inside the code folder.
# HOMEGUARD_HOME is exported, so the Python programs started from here agree.
_hg_posix() { if command -v cygpath &>/dev/null; then cygpath -u "$1"; else printf '%s\n' "$1"; fi; }
_hg_windows() { if command -v cygpath &>/dev/null; then cygpath -w "$1"; else printf '%s\n' "$1"; fi; }
HG_HOME=""
if [[ -n "${HOMEGUARD_HOME:-}" && "${HOMEGUARD_HOME,,}" != "legacy" ]]; then
    HG_HOME="$(_hg_posix "$HOMEGUARD_HOME")"
elif [[ -z "${HOMEGUARD_HOME:-}" ]]; then
    _hg_default="$(_hg_posix "${ProgramData:-${PROGRAMDATA:-C:\\ProgramData}}")/HomeGuard"
    [[ -f "$_hg_default/layout.json" ]] && HG_HOME="$_hg_default"
fi
if [[ -n "$HG_HOME" ]]; then
    HG_LAYOUT="home"
    export HOMEGUARD_HOME="$(_hg_windows "$HG_HOME")"
    LOG_DIR="$HG_HOME/logs"
    CONFIG_DIR="$HG_HOME/config"
else
    HG_LAYOUT="legacy"
    export HOMEGUARD_HOME="legacy"
    LOG_DIR="$PROJECT_ROOT/logs"
    CONFIG_DIR="$PROJECT_ROOT/home_guard_project/data_collection"
fi
CAMERAS_YAML="$CONFIG_DIR/cameras.yaml"
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
