#!/usr/bin/env bash
# ============================================================================
#  start_labeling.sh — One-command annotation launcher
#
#  Lives in:  home_guard_project/start_labeling.sh
#
#  Usage:
#      ./home_guard_project/start_labeling.sh [DATASET_DIR]
#
#  Example:
#      ./home_guard_project2/start_labeling.sh ./dataset_multi
#
#  What it does:
#    1. Verifies Python >= 3.12 and installs uv if missing
#    2. Runs uv sync to install all dependencies (incl. label-studio)
#    3. Re-encodes video clips to H.264 (skips already-encoded)
#    4. Generates Label Studio config + tasks JSON
#    5. Starts file server (port 8081) and Label Studio (port 8080)
#    6. Auto-creates a Label Studio project via the API
#    7. On re-run: exports existing annotations, merges into new tasks,
#       clears old tasks, then reimports (preserves all human annotations)
#    8. Imports tasks via the API
#    9. Opens the browser
#   10. Ctrl+C cleanly shuts everything down
# ============================================================================
set -euo pipefail

# ── Always run from the project root (where pyproject.toml lives) ─────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

# ── Configuration ───────────────────────────────────────────────────────────
DATASET_DIR="${1:-./dataset_multi}"
LS_PORT=8080
FILE_SERVER_PORT=8081
PROJECT_NAME="Security Camera Annotations"
LS_EMAIL="admin@localhost"
LS_PASSWORD="admin12345678"
LS_BASE="http://localhost:${LS_PORT}"
SESSION_FILE="${DATASET_DIR}/.ls_session.json"

# PIDs for cleanup
FILE_SERVER_PID=""
LABEL_STUDIO_PID=""

# ── Colors ──────────────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

info()  { echo -e "${CYAN}[INFO]${NC}  $*"; }
ok()    { echo -e "${GREEN}[OK]${NC}    $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
err()   { echo -e "${RED}[ERROR]${NC} $*" >&2; }
step()  { echo -e "\n${BOLD}── $* ──${NC}"; }

# ── Cleanup on exit ────────────────────────────────────────────────────────
cleanup() {
    echo ""
    info "Shutting down..."
    if [[ -n "$FILE_SERVER_PID" ]]; then
        kill "$FILE_SERVER_PID" 2>/dev/null && info "File server stopped." || true
    fi
    if [[ -n "$LABEL_STUDIO_PID" ]]; then
        kill "$LABEL_STUDIO_PID" 2>/dev/null && info "Label Studio stopped." || true
    fi
    ok "All services stopped. Goodbye."
}
trap cleanup EXIT INT TERM

# ── Detect OS ───────────────────────────────────────────────────────────────
detect_os() {
    case "$(uname -s)" in
        MINGW*|MSYS*|CYGWIN*) echo "windows" ;;
        Darwin*)               echo "macos"   ;;
        *)                     echo "linux"   ;;
    esac
}
OS="$(detect_os)"

open_browser() {
    local url="$1"
    case "$OS" in
        windows) start "$url" 2>/dev/null || cmd.exe /c start "$url" 2>/dev/null || true ;;
        macos)   open "$url"   2>/dev/null || true ;;
        linux)   xdg-open "$url" 2>/dev/null || true ;;
    esac
}

# ── Find Python ─────────────────────────────────────────────────────────────
find_python() {
    # On Windows the 'py' launcher is the standard way
    if command -v py &>/dev/null; then
        echo "py"
        return
    fi
    for candidate in python3 python; do
        if command -v "$candidate" &>/dev/null; then
            # Verify it's a real Python (not Windows Store stub)
            local ver
            ver="$("$candidate" --version 2>&1)" || continue
            if [[ "$ver" == Python\ 3.* ]]; then
                echo "$candidate"
                return
            fi
        fi
    done
    return 1
}

# ============================================================================
#  STEP 1: Verify Python
# ============================================================================
step "Checking Python"

PYTHON="$(find_python)" || { err "Python >= 3.12 not found. Install it first."; exit 1; }
PY_VER="$($PYTHON --version 2>&1)"
ok "Found $PY_VER  ($PYTHON)"

# ============================================================================
#  STEP 2: Ensure uv is installed
# ============================================================================
step "Checking uv"

if ! command -v uv &>/dev/null; then
    info "uv not found — installing..."
    $PYTHON -m pip install --quiet uv
fi
ok "uv $(uv --version 2>&1 | head -1)"

# ============================================================================
#  STEP 3: Install dependencies
# ============================================================================
step "Installing dependencies (uv sync)"

uv sync --quiet
ok "All dependencies installed."

# ── Resolve the uv-managed python for running commands ──────────────────────
UVRUN="uv run"

# ============================================================================
#  STEP 4: Verify dataset directory
# ============================================================================
step "Verifying dataset"

if [[ ! -d "$DATASET_DIR" ]]; then
    err "Dataset directory not found: $DATASET_DIR"
    err "Pass the correct path:  ./start_labeling.sh /path/to/dataset_multi"
    exit 1
fi

if [[ ! -d "${DATASET_DIR}/meta" ]]; then
    err "No 'meta/' subdirectory in ${DATASET_DIR}. Is this a valid dataset_multi directory?"
    exit 1
fi

CLIP_COUNT=$(find "${DATASET_DIR}/clips" -name "*.mp4" 2>/dev/null | wc -l || echo "0")
ok "Dataset: ${DATASET_DIR}  (${CLIP_COUNT} clips found)"

# ============================================================================
#  STEP 5: Re-encode clips to H.264 + generate tasks
# ============================================================================
step "Re-encoding clips & generating tasks"

info "This re-encodes mp4v clips to H.264 (already-encoded clips are skipped)."
info "Then generates Label Studio config + tasks JSON..."

$UVRUN python -m home_guard_project.label_studio_setup \
    --dataset-dir "$DATASET_DIR" \
    --reencode

ok "Tasks generated at ${DATASET_DIR}/label_studio_tasks.json"

# ============================================================================
#  STEP 6: Start file server (background)
# ============================================================================
step "Starting file server (port ${FILE_SERVER_PORT})"

$UVRUN python -m home_guard_project.label_studio_setup \
    --serve \
    --dataset-dir "$DATASET_DIR" \
    --port "$FILE_SERVER_PORT" &
FILE_SERVER_PID=$!

sleep 1
if kill -0 "$FILE_SERVER_PID" 2>/dev/null; then
    ok "File server running (PID ${FILE_SERVER_PID})"
else
    err "File server failed to start."
    exit 1
fi

# ============================================================================
#  STEP 7: Start Label Studio (background)
# ============================================================================
step "Starting Label Studio (port ${LS_PORT})"

$UVRUN label-studio start \
    --port "$LS_PORT" \
    --no-browser \
    > /dev/null 2>&1 &
LABEL_STUDIO_PID=$!

# Wait for Label Studio API to be ready
info "Waiting for Label Studio to start..."
MAX_WAIT=120
WAITED=0
while (( WAITED < MAX_WAIT )); do
    if curl -sf "${LS_BASE}/api/version" > /dev/null 2>&1; then
        break
    fi
    sleep 2
    WAITED=$((WAITED + 2))
    printf "."
done
echo ""

if (( WAITED >= MAX_WAIT )); then
    err "Label Studio did not start within ${MAX_WAIT}s."
    exit 1
fi
ok "Label Studio ready (PID ${LABEL_STUDIO_PID})"

# ============================================================================
#  STEP 8: Auto-create user / project / import tasks via API
# ============================================================================
step "Configuring Label Studio project via API"

# Helper: JSON value extraction (pure bash, no jq dependency)
json_val() {
    # Usage: json_val '"key"' < json_string
    # Simple grep-based extraction — works for flat JSON responses
    $PYTHON -c "import sys,json; d=json.load(sys.stdin); print(d.get($1, ''))"
}

# ── Check if session exists from a previous run ────────────────────────────
TOKEN=""
PROJECT_ID=""

if [[ -f "$SESSION_FILE" ]]; then
    TOKEN=$($PYTHON -c "import json; d=json.load(open('$SESSION_FILE')); print(d.get('token',''))" 2>/dev/null || echo "")
    PROJECT_ID=$($PYTHON -c "import json; d=json.load(open('$SESSION_FILE')); print(d.get('project_id',''))" 2>/dev/null || echo "")
fi

# ── Sign up (first run) or log in ──────────────────────────────────────────
if [[ -z "$TOKEN" ]]; then
    info "Creating admin account..."
    # Try signup first (will fail if user already exists)
    SIGNUP_RESP=$(curl -sf -X POST "${LS_BASE}/api/auth/signup" \
        -H "Content-Type: application/json" \
        -d "{\"email\":\"${LS_EMAIL}\",\"password\":\"${LS_PASSWORD}\"}" 2>/dev/null || echo "{}")

    TOKEN=$(echo "$SIGNUP_RESP" | $PYTHON -c "import sys,json; d=json.load(sys.stdin); print(d.get('token',''))" 2>/dev/null || echo "")

    # If signup failed (user exists), try login
    if [[ -z "$TOKEN" ]]; then
        info "User already exists, logging in..."
        LOGIN_RESP=$(curl -sf -X POST "${LS_BASE}/api/auth/login" \
            -H "Content-Type: application/json" \
            -d "{\"email\":\"${LS_EMAIL}\",\"password\":\"${LS_PASSWORD}\"}" 2>/dev/null || echo "{}")

        # Login returns session cookie; get token from user/token endpoint
        SESSION_COOKIE=$(echo "$LOGIN_RESP" | $PYTHON -c "
import sys,json
d=json.load(sys.stdin)
print(d.get('token','') or d.get('sessionid',''))
" 2>/dev/null || echo "")

        if [[ -n "$SESSION_COOKIE" ]]; then
            TOKEN="$SESSION_COOKIE"
        else
            # Try getting token from /api/current-user/token
            TOKEN=$(curl -sf "${LS_BASE}/api/current-user/token" \
                -H "Content-Type: application/json" \
                -b <(echo "$LOGIN_RESP") 2>/dev/null \
                | $PYTHON -c "import sys,json; print(json.load(sys.stdin).get('token',''))" 2>/dev/null || echo "")
        fi
    fi
fi

if [[ -z "$TOKEN" ]]; then
    warn "Could not obtain API token automatically."
    warn "Please log in manually at ${LS_BASE} and import tasks from the UI."
    warn "  Email: ${LS_EMAIL}  Password: ${LS_PASSWORD}"
    open_browser "${LS_BASE}"
    info "Servers are running. Press Ctrl+C to stop."
    wait
    exit 0
fi

ok "Authenticated (token: ${TOKEN:0:8}...)"

AUTH_HEADER="Authorization: Token ${TOKEN}"

# ── Create or reuse project ────────────────────────────────────────────────
LABEL_CONFIG=$(cat "${DATASET_DIR}/label_studio_config.xml")
IS_RERUN=false

if [[ -z "$PROJECT_ID" ]]; then
    info "Creating project: ${PROJECT_NAME}"

    CREATE_RESP=$(curl -sf -X POST "${LS_BASE}/api/projects" \
        -H "$AUTH_HEADER" \
        -H "Content-Type: application/json" \
        -d "$($PYTHON -c "
import json, sys
config = open('${DATASET_DIR}/label_studio_config.xml').read()
print(json.dumps({
    'title': '${PROJECT_NAME}',
    'label_config': config,
}))
")" 2>/dev/null || echo "{}")

    PROJECT_ID=$(echo "$CREATE_RESP" | $PYTHON -c "import sys,json; print(json.load(sys.stdin).get('id',''))" 2>/dev/null || echo "")

    if [[ -z "$PROJECT_ID" ]]; then
        warn "Could not create project automatically. Check Label Studio UI."
        open_browser "${LS_BASE}"
        info "Servers are running. Press Ctrl+C to stop."
        wait
        exit 0
    fi

    ok "Project created (ID: ${PROJECT_ID})"
else
    IS_RERUN=true
    info "Reusing existing project (ID: ${PROJECT_ID})"

    # Update the labeling config in case it changed
    curl -sf -X PATCH "${LS_BASE}/api/projects/${PROJECT_ID}" \
        -H "$AUTH_HEADER" \
        -H "Content-Type: application/json" \
        -d "$($PYTHON -c "
import json
config = open('${DATASET_DIR}/label_studio_config.xml').read()
print(json.dumps({'label_config': config}))
")" > /dev/null 2>&1 || true

    ok "Project config updated."
fi

# ── Save session for re-runs ───────────────────────────────────────────────
$PYTHON -c "
import json
json.dump({'token': '${TOKEN}', 'project_id': ${PROJECT_ID}}, open('${SESSION_FILE}', 'w'), indent=2)
"
ok "Session saved to ${SESSION_FILE}"

# ── Export existing annotations (re-run only) ─────────────────────────────
TASKS_FILE="${DATASET_DIR}/label_studio_tasks.json"
EXPORT_BACKUP="${DATASET_DIR}/.ls_export_backup.json"

if [[ "$IS_RERUN" == "true" ]]; then
    info "Exporting existing annotations from project ${PROJECT_ID}..."

    EXPORT_STATUS=$(curl -sf -w "%{http_code}" -o "$EXPORT_BACKUP" \
        -X GET "${LS_BASE}/api/projects/${PROJECT_ID}/export?exportType=JSON" \
        -H "$AUTH_HEADER" 2>/dev/null || echo "000")

    if [[ "$EXPORT_STATUS" == "200" && -s "$EXPORT_BACKUP" ]]; then
        EXPORTED_COUNT=$($PYTHON -c "
import json
tasks = json.load(open('${EXPORT_BACKUP}'))
annotated = sum(1 for t in tasks if t.get('annotations'))
print(f'{len(tasks)} tasks, {annotated} with annotations')
" 2>/dev/null || echo "unknown")
        ok "Exported: ${EXPORTED_COUNT}"

        # Merge annotations into the regenerated tasks
        if [[ -f "$TASKS_FILE" ]]; then
            info "Merging annotations into new tasks..."
            $PYTHON -m home_guard_project.label_studio_setup \
                --merge "$EXPORT_BACKUP" \
                --dataset-dir "$DATASET_DIR"
            ok "Annotations merged."
        fi

        # Delete all existing tasks in the project before reimport
        info "Clearing existing tasks from project ${PROJECT_ID}..."
        curl -sf -X POST "${LS_BASE}/api/dm/actions?id=delete_tasks&project=${PROJECT_ID}" \
            -H "$AUTH_HEADER" \
            -H "Content-Type: application/json" \
            -d '{"selectedItems": {"all": true, "excluded": []}}' \
            > /dev/null 2>&1 || true
        ok "Existing tasks cleared."
    else
        warn "Could not export annotations (HTTP ${EXPORT_STATUS}). Proceeding without merge."
        # Clean up empty/failed export file
        rm -f "$EXPORT_BACKUP" 2>/dev/null || true
    fi
fi

# ── Import tasks ───────────────────────────────────────────────────────────
if [[ -f "$TASKS_FILE" ]]; then
    info "Importing tasks into project ${PROJECT_ID}..."

    IMPORT_RESP=$(curl -sf -X POST "${LS_BASE}/api/projects/${PROJECT_ID}/import" \
        -H "$AUTH_HEADER" \
        -H "Content-Type: application/json" \
        -d @"$TASKS_FILE" 2>/dev/null || echo "{}")

    TASK_COUNT=$(echo "$IMPORT_RESP" | $PYTHON -c "
import sys, json
d = json.load(sys.stdin)
print(d.get('task_count', d.get('total', '?')))
" 2>/dev/null || echo "?")

    ok "Imported ${TASK_COUNT} tasks."
else
    warn "Tasks file not found: ${TASKS_FILE}"
fi

# ============================================================================
#  STEP 9: Open browser
# ============================================================================
step "Ready!"

PROJECT_URL="${LS_BASE}/projects/${PROJECT_ID}"
echo ""
echo -e "  ${BOLD}Label Studio:${NC}   ${PROJECT_URL}"
echo -e "  ${BOLD}File Server:${NC}    http://localhost:${FILE_SERVER_PORT}"
echo -e "  ${BOLD}Login:${NC}          ${LS_EMAIL} / ${LS_PASSWORD}"
echo ""

open_browser "$PROJECT_URL"

ok "Browser opened. Press Ctrl+C to stop all services."

# Keep the script alive until interrupted
wait
