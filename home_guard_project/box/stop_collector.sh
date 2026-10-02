#!/usr/bin/env bash
# ============================================================================
#  stop_collector.sh - stop the runner and the collector on a collector box
#
#  Lives in:  home_guard_project/box/stop_collector.sh
#
#  "schtasks /End /TN HomeGuard-Collector" alone leaves the real bash runner
#  alive. Run this afterwards for a full stop. Start again with
#  "schtasks /Run /TN HomeGuard-Collector".
# ============================================================================
set -uo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

for pid_file in "$LOG_DIR/runner.winpid" "$LOG_DIR/collector.winpid"; do
    [[ -f "$pid_file" ]] || continue
    winpid="$(cat "$pid_file")"
    if taskkill //PID "$winpid" //T //F &>/dev/null; then
        log "Stopped process tree $winpid ($(basename "$pid_file"))"
    fi
    rm -f "$pid_file"
done
