#!/usr/bin/env bash
# ============================================================================
#  screen.sh - what the box shows on its own screen
#
#  Lives in:  home_guard_project/box/screen.sh
#  Started by "Home Guard" on the Desktop and at sign-in (live_view.cmd).
#
#  The log window always opens. The camera windows open only when the installer
#  chose so in the setup program (box.yaml: show_cameras: true):
#    show_cameras: true   -> watch_live.sh  (a window per camera + the log)
#    show_cameras: false  -> watch_logs.sh  (the log only; background collector untouched)
# ============================================================================
set -uo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

ensure_venv || exit 1

# A clean GUI close succeeds and never reaches the legacy collector controls.
if [[ "${1:-}" != "--legacy" ]] && "$PY" -m home_guard_project.box.app; then
    exit 0
fi
printf '%s\n' 'Home Guard app could not start; opening the legacy screen.'
if [[ "$("$PY" -m home_guard_project.box get-option show_cameras 2>/dev/null)" == "true" ]]; then
    exec "$SCRIPT_DIR/watch_live.sh"
else
    exec "$SCRIPT_DIR/watch_logs.sh"
fi
