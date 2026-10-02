#!/usr/bin/env bash
# ============================================================================
#  run_upload.sh - move finished clips to the outbox and sync them to S3
#
#  Lives in:  home_guard_project/box/run_upload.sh
#  Started by the "HomeGuard-Upload" scheduled task (nightly). Safe to run by hand.
# ============================================================================
set -uo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

UPLOAD_LOG="$LOG_DIR/upload-$(date +%F).log"
ensure_venv >> "$UPLOAD_LOG" 2>&1 || exit 1

"$PY" -m home_guard_project.box upload >> "$UPLOAD_LOG" 2>&1
