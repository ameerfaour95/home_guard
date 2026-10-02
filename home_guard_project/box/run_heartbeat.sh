#!/usr/bin/env bash
# ============================================================================
#  run_heartbeat.sh - write the box status JSON to S3
#
#  Lives in:  home_guard_project/box/run_heartbeat.sh
#  Started by the "HomeGuard-Heartbeat" scheduled task (hourly).
# ============================================================================
set -uo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

HEARTBEAT_LOG="$LOG_DIR/heartbeat.log"
ensure_venv >> "$HEARTBEAT_LOG" 2>&1 || exit 1

# Self-heal: the runner and the live view both touch collector.alive every 30 s.
# If it is stale, nothing is collecting (the runner died, or a live view window
# was closed without restarting the background collector): start the task again.
# Starting a task that is already running does nothing.
if [[ -z "$(find "$LOG_DIR/collector.alive" -mmin -5 2>/dev/null)" ]]; then
    log "Collector not alive - starting the HomeGuard-Collector task" >> "$HEARTBEAT_LOG"
    schtasks //Run //TN HomeGuard-Collector >> "$HEARTBEAT_LOG" 2>&1 || true
fi

"$PY" -m home_guard_project.box heartbeat >> "$HEARTBEAT_LOG" 2>&1
