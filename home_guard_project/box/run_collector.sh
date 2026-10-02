#!/usr/bin/env bash
# ============================================================================
#  run_collector.sh - the always-on program of a box, in either mode
#
#  Lives in:  home_guard_project/box/run_collector.sh
#  Started by the "HomeGuard-Collector" scheduled task at boot.
#
#  What it does:
#    1. Waits until cameras.yaml exists (camera discovery is a one-time step)
#    2. Reads the mode from box.yaml and runs, headless (config.box.yaml overlay):
#         data_collection -> data_collection.py          (clips for tagging)
#         inference       -> home_guard_project.box.inference   (alerts)
#       logging to logs/collector-<date>.log
#    3. Touches logs/collector.alive while it runs (read by the heartbeat)
#    4. Restarts it 15 s after it exits, forever
# ============================================================================
set -uo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

export HOME_GUARD_CONFIG_OVERLAY="${HOME_GUARD_CONFIG_OVERLAY:-$SCRIPT_DIR/config.box.yaml}"

# Leave one core for reading the camera streams. With detection on all 4 cores
# of the N150 the stream readers starve and the video arrives damaged: measured
# with 6 cameras, 114 damaged-frame messages a minute on 4 threads against 23 on 3.
cores="$(nproc 2>/dev/null || echo 4)"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-$(( cores > 1 ? cores - 1 : 1 ))}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-$OMP_NUM_THREADS}"

# Lines such as "[h264 @ 000001f0] error while decoding MB 59 17" from the video decoder.
DECODER_NOISE='^\[[A-Za-z0-9_]+ @ [0-9a-fA-Fx]+\]'

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

RUNNER_PID_FILE="$LOG_DIR/runner.winpid"
MY_WINPID="$(cat "/proc/$$/winpid" 2>/dev/null || echo "$$")"

# Only one runner may exist. Ending the scheduled task kills Git's bash.exe
# launcher but not the real bash underneath it, so the newest runner stops the
# previous one (and its collector) before taking over.
kill_previous_runner() {
    if [[ -f "$RUNNER_PID_FILE" ]] && command -v tasklist &>/dev/null; then
        local winpid
        winpid="$(cat "$RUNNER_PID_FILE")"
        if [[ "$winpid" != "$MY_WINPID" ]] && tasklist //FI "PID eq $winpid" 2>/dev/null | grep -qi bash; then
            log "Stopping previous runner (PID $winpid)" >> "$RUNNER_LOG"
            taskkill //PID "$winpid" //T //F &>/dev/null || true
        fi
    fi
    echo "$MY_WINPID" > "$RUNNER_PID_FILE"
}

stop() {
    log "Stopping." >> "$RUNNER_LOG"
    [[ -n "$child" ]] && kill "$child" 2>/dev/null
    rm -f "$PID_FILE"
    exit 0
}
trap stop INT TERM

ensure_venv >> "$RUNNER_LOG" 2>&1 || exit 1
kill_previous_runner
kill_leftover

waiting_logged=false
while true; do
    if [[ ! -f "$CAMERAS_YAML" ]]; then
        if [[ "$waiting_logged" == "false" ]]; then
            log "cameras.yaml not found - run camera discovery (see box/README.md). Checking every 60 s." >> "$RUNNER_LOG"
            waiting_logged=true
        fi
        sleep 60
        continue
    fi
    waiting_logged=false

    # box.yaml says what this box runs. Read it on every start, so a mode change
    # only needs the task restarted. An unreadable box.yaml falls back to collecting.
    mode="$("$PY" -m home_guard_project.box mode 2>> "$RUNNER_LOG")" || mode="data_collection"
    case "$mode" in
        inference) entry=(-m home_guard_project.box.inference) ;;
        *)         mode="data_collection"; entry=(home_guard_project/data_collection/data_collection.py) ;;
    esac

    collector_log="$LOG_DIR/collector-$(date +%F).log"
    log "Starting $mode (overlay: $HOME_GUARD_CONFIG_OVERLAY)" >> "$RUNNER_LOG"
    # The video decoder prints a line for every damaged frame, which would bury
    # the real log lines and grow the file without limit. Drop those lines.
    "$PY" -u "${entry[@]}" \
        > >(grep --line-buffered -a -v -E "$DECODER_NOISE" >> "$collector_log") 2>&1 &
    child=$!
    cat "/proc/$child/winpid" > "$PID_FILE" 2>/dev/null || echo "$child" > "$PID_FILE"

    while kill -0 "$child" 2>/dev/null; do
        touch "$ALIVE_FILE"
        sleep 30
    done
    wait "$child"
    log "$mode exited with code $? - restarting in 15 s" >> "$RUNNER_LOG"
    child=""
    rm -f "$PID_FILE"
    sleep 15
done
