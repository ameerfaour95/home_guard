#!/usr/bin/env bash
# ============================================================================
#  update.sh - pull the latest code on a collector box and restart collection
#
#  Lives in:  home_guard_project/box/update.sh
#  Run on the box (the install is a git checkout), or from the laptop:
#      ssh -i ~/.ssh/homeguard_box <user>@<box-ip> "\"C:\Program Files\Git\bin\bash.exe\" -lc <repo>/home_guard_project/box/update.sh"
#
#  What it does:
#    1. Stops the collector
#    2. git pull --ff-only
#    3. Refreshes the Python environment
#    4. Starts the collector task again
# ============================================================================
set -uo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

# Everything is in one function so bash has read the whole file before git replaces it.
main() {
    log "Stopping the collector..."
    schtasks //End //TN HomeGuard-Collector &>/dev/null || true
    "$SCRIPT_DIR/stop_collector.sh"

    log "Pulling..."
    # A box that lost power in the middle of an update can be left with a
    # half-written ORIG_HEAD, and git then refuses every pull. The file is only
    # git's note of where the branch was before; a new pull writes it again.
    if [[ -f .git/ORIG_HEAD ]] && ! git rev-parse -q --verify ORIG_HEAD &>/dev/null; then
        log "Removing a damaged .git/ORIG_HEAD"
        rm -f .git/ORIG_HEAD
    fi
    if ! git pull --ff-only; then
        log "git pull failed - keeping the current code"
    fi

    ensure_venv
    schtasks //Run //TN HomeGuard-Collector &>/dev/null || log "Could not start the HomeGuard-Collector task"
    log "Running $(git log --oneline -1)"
}

main "$@"
exit
