#!/usr/bin/env bash
# ============================================================================
#  watch_live.sh - see the box work: every camera in a window, with detections
#
#  Lives in:  home_guard_project/box/watch_live.sh
#  Run on the box itself, signed in, with a screen (or double-click
#  "Home Guard Live View" on its Desktop).
#
#  What it does:
#    1. Stops the background collector (two collectors must not run at once)
#    2. Runs the same collector in this window with one live window per camera
#       and the detection boxes drawn; the log scrolls here; clips are still saved
#    3. When you press q in a camera window (or Ctrl+C here), starts the
#       background collector again
# ============================================================================
set -uo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

ensure_venv || exit 1

export HOME_GUARD_CONFIG_OVERLAY="$SCRIPT_DIR/config.live.yaml"
cores="$(nproc 2>/dev/null || echo 4)"
export OMP_NUM_THREADS="$(( cores > 1 ? cores - 1 : 1 ))"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"

log "Stopping the background collector..."
schtasks //End //TN HomeGuard-Collector &>/dev/null || true
"$SCRIPT_DIR/stop_collector.sh"

# Keep collector.alive fresh while the live view runs, so the hourly heartbeat
# knows something is collecting and does not start a second collector.
( while true; do touch "$LOG_DIR/collector.alive"; sleep 30; done ) &
toucher=$!

restart_background() {
    kill "$toucher" 2>/dev/null
    log "Starting the background collector again..."
    schtasks //Run //TN HomeGuard-Collector &>/dev/null || log "Could not start it - run: schtasks /Run /TN HomeGuard-Collector"
}
trap restart_background EXIT

log "Live view: one window per camera. Press q in a camera window, or Ctrl+C here, to stop."
log "Clips are still being saved while you watch."
echo

"$PY" -u home_guard_project/data_collection/data_collection.py 2>&1 | grep --line-buffered -a -v -E '^\[[A-Za-z0-9_]+ @ [0-9a-fA-Fx]+\]'
