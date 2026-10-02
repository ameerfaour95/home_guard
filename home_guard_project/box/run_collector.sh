#!/usr/bin/env bash
# ============================================================================
#  run_collector.sh — unattended data collection for a collector box
#
#  Lives in:  home_guard_project/box/run_collector.sh
#  Started by the "HomeGuard-Collector" scheduled task at boot.
#
#  What it does:
#    1. Waits until cameras.yaml exists (camera discovery is a one-time manual step)
#    2. Runs data_collection.py headless (config.box.yaml overlay), logging to logs/
#    3. Touches logs/collector.alive while the collector runs (read by the heartbeat)
#    4. Restarts the collector 15 s after it exits, forever
# ============================================================================
set -uo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

export HOME_GUARD_CONFIG_OVERLAY="${HOME_GUARD_CONFIG_OVERLAY:-$SCRIPT_DIR/config.box.yaml}"

CAMERAS_YAML="$PROJECT_ROOT/home_guard_project/data_collection/cameras.yaml"
ALIVE_FILE="$LOG_DIR/collector.alive"
PID_FILE="$LOG_DIR/collector.winpid"
RUNNER_LOG="$LOG_DIR/runner.log"

child=""

# A scheduled-task stop kills this script without running the trap, which
# would leave the collector running. Kill a leftover one before starting.
kill_leftover() {
    [[ -f "$PID_FILE" ]] || return 0
    local winpid
    winpid="$(cat "$PID_FILE")"
    if command -v tasklist &>/dev/null && tasklist //FI "PID eq $winpid" 2>/dev/null | grep -qi python; then
        log "Stopping leftover collector (PID $winpid)" >> "$RUNNER_LOG"
        taskkill //PID "$winpid" //F &>/dev/null || true
    fi
    rm -f "$PID_FILE"
}

stop() {
    log "Stopping." >> "$RUNNER_LOG"
    [[ -n "$child" ]] && kill "$child" 2>/dev/null
    rm -f "$PID_FILE"
    exit 0
}
trap stop INT TERM

ensure_venv >> "$RUNNER_LOG" 2>&1 || exit 1
kill_leftover

waiting_logged=false
while true; do
    if [[ ! -f "$CAMERAS_YAML" ]]; then
        if [[ "$waiting_logged" == "false" ]]; then
            log "cameras.yaml not found — run camera discovery (see box/README.md). Checking every 60 s." >> "$RUNNER_LOG"
            waiting_logged=true
        fi
        sleep 60
        continue
    fi
    waiting_logged=false

    collector_log="$LOG_DIR/collector-$(date +%F).log"
    log "Starting collector (overlay: $HOME_GUARD_CONFIG_OVERLAY)" >> "$RUNNER_LOG"
    "$PY" -u home_guard_project/data_collection/data_collection.py >> "$collector_log" 2>&1 &
    child=$!
    cat "/proc/$child/winpid" > "$PID_FILE" 2>/dev/null || echo "$child" > "$PID_FILE"

    while kill -0 "$child" 2>/dev/null; do
        touch "$ALIVE_FILE"
        sleep 30
    done
    wait "$child"
    log "Collector exited with code $? — restarting in 15 s" >> "$RUNNER_LOG"
    child=""
    rm -f "$PID_FILE"
    sleep 15
done
