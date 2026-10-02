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
echo "Clips are saved on this box first. Every few minutes they are sent to the online"
echo "folder and removed from this box; the lines marked UPLOAD show that."
echo "Closing this window does not stop it."
echo

# The upload log is mostly progress bars; these are the lines that say what happened.
UPLOAD_LINES='Moved [0-9]+ |Found [0-9]+ files to sync|Done\. Uploaded|Local cleanup|Heartbeat written|ERROR|WARNING'

# A new log file starts each day: follow today's, and switch when the date changes.
while true; do
    today="$(date +%F)"
    touch "$LOG_DIR/collector-$today.log" "$LOG_DIR/upload-$today.log"
    tail -n 25 -F "$LOG_DIR/runner.log" "$LOG_DIR/collector-$today.log" 2>/dev/null &
    tail_pid=$!
    grep -E "$UPLOAD_LINES" "$LOG_DIR/upload-$today.log" | tail -n 5 | sed 's/^/UPLOAD  /'
    # $! is the tail; when it is killed the filter behind it ends too.
    tail -n 0 -F "$LOG_DIR/upload-$today.log" 2>/dev/null \
        > >(grep --line-buffered -E "$UPLOAD_LINES" | sed -u 's/^/UPLOAD  /') &
    upload_pid=$!
    while [[ "$(date +%F)" == "$today" ]] && kill -0 "$tail_pid" 2>/dev/null; do
        sleep 60
    done
    kill "$tail_pid" "$upload_pid" 2>/dev/null
done
