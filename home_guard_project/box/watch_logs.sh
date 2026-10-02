#!/usr/bin/env bash
# ============================================================================
#  watch_logs.sh - show what the box is doing, as scrolling log lines
#
#  Lives in:  home_guard_project/box/watch_logs.sh
#  Opened by screen.sh when the camera windows are switched off.
#
#  It only reads the log files. The collector keeps running in the background,
#  and closing this window does not stop it.
# ============================================================================
set -uo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

echo "Home Guard is working in the background. This window shows what it is doing."
echo "Closing this window does not stop it."
echo

# A new log file starts each day: follow today's, and switch when the date changes.
while true; do
    today="$(date +%F)"
    touch "$LOG_DIR/collector-$today.log"
    tail -n 25 -F "$LOG_DIR/runner.log" "$LOG_DIR/collector-$today.log" 2>/dev/null &
    tail_pid=$!
    while [[ "$(date +%F)" == "$today" ]] && kill -0 "$tail_pid" 2>/dev/null; do
        sleep 60
    done
    kill "$tail_pid" 2>/dev/null
done
