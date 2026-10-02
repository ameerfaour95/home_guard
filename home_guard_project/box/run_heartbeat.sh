#!/usr/bin/env bash
# ============================================================================
#  run_heartbeat.sh — write the box status JSON to S3
#
#  Lives in:  home_guard_project/box/run_heartbeat.sh
#  Started by the "HomeGuard-Heartbeat" scheduled task (hourly).
# ============================================================================
set -uo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

HEARTBEAT_LOG="$LOG_DIR/heartbeat.log"
ensure_venv >> "$HEARTBEAT_LOG" 2>&1 || exit 1

"$PY" -m home_guard_project.box heartbeat >> "$HEARTBEAT_LOG" 2>&1
