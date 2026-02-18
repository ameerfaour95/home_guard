#!/usr/bin/env bash
# ============================================================================
#  start.sh — One-command annotation launcher
#
#  Lives in:  home_guard_project/labeling/start.sh
#
#  Usage:
#      ./home_guard_project/labeling/start.sh [DATASET_DIR] [--force]
#
#  Examples:
#      ./home_guard_project/labeling/start.sh ./dataset_multi
#      ./home_guard_project/labeling/start.sh ./dataset_multi --force
#
#  What it does:
#    1. Verifies Python >= 3.12 and installs uv if missing
#    2. Runs uv sync to install all dependencies (incl. label-studio)
#    3. Verifies dataset directory
#    4. Checks for orphaned metadata (interactive: delete / list / skip)
#    5. Re-encodes video clips to H.264 (skips already-encoded)
#    6. Generates Label Studio config + tasks JSON
#    7. Starts file server (port 8081) and Label Studio (port 8080)
#    8. Auto-creates a Label Studio project via the API
#    9. On re-run: exports existing annotations, merges into new tasks,
#       clears old tasks, then reimports (preserves all human annotations)
#   10. Imports tasks via the API, opens the browser
#   11. Ctrl+C cleanly shuts everything down
# ============================================================================
set -euo pipefail

# ── Always run from the project root (where pyproject.toml lives) ─────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

# ── Configuration from config.yaml ─────────────────────────────────────────
CONFIG_YAML="${SCRIPT_DIR}/config.yaml"
if [[ ! -f "$CONFIG_YAML" ]]; then
    echo "[ERROR] config.yaml not found at: $CONFIG_YAML" >&2
    exit 1
fi

# Read simple YAML values using sed (no Python or pyyaml dependency needed)
# Handles: key: value  /  key: "value"  /  key: 'value'
yaml_val() {
    local key="$1"
    local raw
    raw=$(grep -E "^\s*${key}:" "$CONFIG_YAML" | head -1 | sed 's/^[^:]*:\s*//')
    # Strip surrounding quotes
    raw="${raw%\"}" ; raw="${raw#\"}"
    raw="${raw%\'}" ; raw="${raw#\'}"
    # Trim trailing whitespace
    echo "$raw" | sed 's/[[:space:]]*$//'
}

# Parse CLI args
FORCE_REBUILD=false
POSITIONAL_ARGS=()
for arg in "$@"; do
    case "$arg" in
        --force) FORCE_REBUILD=true ;;
        *)       POSITIONAL_ARGS+=("$arg") ;;
    esac
done

# Read config values (dataset_dir can be overridden by first positional arg)
DATASET_DIR="${POSITIONAL_ARGS[0]:-$(yaml_val dataset_dir)}"

# label_studio.port — grep the one directly under label_studio:
LS_PORT=$(sed -n '/^label_studio:/,/^[a-z]/{ /^\s*port:/p }' "$CONFIG_YAML" | head -1 | sed 's/^[^:]*:\s*//' | tr -d '[:space:]')
LS_EMAIL="$(yaml_val email)"
LS_PASSWORD="$(yaml_val password)"
PROJECT_NAME="$(yaml_val project_name)"

# file_server.port
FILE_SERVER_PORT=$(sed -n '/^file_server:/,/^[a-z]/{ /^\s*port:/p }' "$CONFIG_YAML" | head -1 | sed 's/^[^:]*:\s*//' | tr -d '[:space:]')

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
    # Clean up temp cookie jar
    [[ -n "${COOKIE_JAR:-}" ]] && rm -f "$COOKIE_JAR" 2>/dev/null || true
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
    err "Pass the correct path:  ./home_guard_project/labeling/start.sh /path/to/dataset_multi"
    exit 1
fi

if [[ ! -d "${DATASET_DIR}/meta" ]]; then
    err "No 'meta/' subdirectory in ${DATASET_DIR}. Is this a valid dataset_multi directory?"
    exit 1
fi

CLIP_COUNT=$(find "${DATASET_DIR}/clips" -name "*.mp4" 2>/dev/null | wc -l || echo "0")
ok "Dataset: ${DATASET_DIR}  (${CLIP_COUNT} clips found)"

# ============================================================================
#  STEP 5: Check for orphaned metadata
# ============================================================================
step "Checking for orphaned metadata"

ORPHAN_OUTPUT=$($UVRUN python -m home_guard_project.labeling \
    --cleanup-only --dry-run --dataset-dir "$DATASET_DIR" 2>/dev/null || echo "")

# Parse orphan counts from the summary output
ORPHAN_META=$(echo "$ORPHAN_OUTPUT" | grep "Orphan meta removed" | grep -oE '[0-9]+' || echo "0")
ORPHAN_RESP=$(echo "$ORPHAN_OUTPUT" | grep "Orphan resp removed" | grep -oE '[0-9]+' || echo "0")
ORPHAN_YOLO_IMG=$(echo "$ORPHAN_OUTPUT" | grep "Orphan YOLO imgs" | grep -oE '[0-9]+' || echo "0")
ORPHAN_YOLO_LBL=$(echo "$ORPHAN_OUTPUT" | grep "Orphan YOLO lbls" | grep -oE '[0-9]+' || echo "0")
ORPHAN_TOTAL=$(( ORPHAN_META + ORPHAN_RESP + ORPHAN_YOLO_IMG + ORPHAN_YOLO_LBL ))

if (( ORPHAN_TOTAL > 0 )); then
    echo ""
    warn "Found ${ORPHAN_TOTAL} orphaned files (metadata/responses/YOLO with no matching clip):"
    echo -e "    Meta files   : ${ORPHAN_META}"
    echo -e "    Response files: ${ORPHAN_RESP}"
    echo -e "    YOLO images  : ${ORPHAN_YOLO_IMG}"
    echo -e "    YOLO labels  : ${ORPHAN_YOLO_LBL}"
    echo ""
    echo -e "  ${BOLD}delete${NC}  — Remove all orphaned files and continue the pipeline"
    echo -e "  ${BOLD}list${NC}    — Print orphaned files, then stop (so you can review)"
    echo -e "  ${BOLD}skip${NC}    — Ignore orphans and continue the pipeline"
    echo ""
    read -rp "  Your choice [delete/list/skip]: " ORPHAN_CHOICE

    case "${ORPHAN_CHOICE,,}" in
        delete|d)
            info "Deleting ${ORPHAN_TOTAL} orphaned files..."
            $UVRUN python -m home_guard_project.labeling \
                --cleanup-only --dataset-dir "$DATASET_DIR" 2>&1 || true
            ok "Orphan cleanup complete."
            ;;
        list|l|print|p)
            info "Orphaned files (no matching clip exists):"
            echo ""
            $UVRUN python -m home_guard_project.labeling \
                --cleanup-only --dry-run --dataset-dir "$DATASET_DIR" -v 2>&1 \
                | grep -E "\[DRY-RUN\]" | sed 's/.*Would remove: /  /' || true
            echo ""
            warn "Pipeline stopped. Each .meta.json must have a matching .mp4 clip."
            warn "Delete the orphans (re-run and choose 'delete') or restore the missing clips."
            info "Servers were not started. Exiting."
            exit 0
            ;;
        skip|s|"")
            info "Skipping orphan cleanup."
            ;;
        *)
            warn "Unknown choice '${ORPHAN_CHOICE}'. Skipping orphan cleanup."
            ;;
    esac
    echo ""
else
    ok "No orphaned metadata found — dataset is clean."
fi

# ============================================================================
#  STEP 6: Re-encode clips to H.264 + generate tasks
# ============================================================================
step "Re-encoding clips & generating tasks"

info "This re-encodes mp4v clips to H.264 (already-encoded clips are skipped)."
info "Then generates Label Studio config + tasks JSON..."

LABELING_ARGS=(--dataset-dir "$DATASET_DIR" --reencode --no-cleanup)
if [[ "$FORCE_REBUILD" == "true" ]]; then
    LABELING_ARGS+=(--force)
fi

$UVRUN python -m home_guard_project.labeling "${LABELING_ARGS[@]}"

ok "Tasks generated at ${DATASET_DIR}/label_studio_tasks.json"

# ============================================================================
#  STEP 7: Start file server (background)
# ============================================================================
step "Starting file server (port ${FILE_SERVER_PORT})"

$UVRUN python -m home_guard_project.labeling \
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
#  STEP 8: Start Label Studio (background)
# ============================================================================
step "Starting Label Studio (port ${LS_PORT})"

# LS manages its own SECRET_KEY in $LOCALAPPDATA/label-studio/.env.
# Do NOT override it — that would invalidate existing user passwords.

# Auto-provision the admin account on first launch (when no users exist yet).
export LABEL_STUDIO_USERNAME="${LS_EMAIL}"
export LABEL_STUDIO_PASSWORD="${LS_PASSWORD}"

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
#  STEP 9: Auto-create user / project / import tasks via API
# ============================================================================
step "Configuring Label Studio project via API"

# ── Check if session exists from a previous run ────────────────────────────
PROJECT_ID=""

if [[ -f "$SESSION_FILE" ]]; then
    PROJECT_ID=$($PYTHON -c "import json; d=json.load(open('$SESSION_FILE')); print(d.get('project_id',''))" 2>/dev/null || echo "")
fi

# ── Sign up (first run) or log in ──────────────────────────────────────────
# Label Studio 1.22+ disabled legacy token auth.
# All API calls use session cookies + CSRF tokens instead.
COOKIE_JAR=$(mktemp)
AUTHENTICATED=false

_get_csrf() {
    # Fetch a page to get a CSRF cookie, then extract the token value
    curl -s -c "$COOKIE_JAR" "${LS_BASE}${1}" > /dev/null 2>&1
    grep csrftoken "$COOKIE_JAR" | awk '{print $NF}'
}

_is_authed() {
    # Check if the session cookie gives us a valid API response
    local resp
    resp=$(curl -s -b "$COOKIE_JAR" "${LS_BASE}/api/current-user/whoami" 2>/dev/null)
    echo "$resp" | $PYTHON -c "import sys,json; d=json.load(sys.stdin); sys.exit(0 if d.get('email') else 1)" 2>/dev/null
}

# Helper: authenticated curl using the session cookie jar + CSRF
_api() {
    # Usage: _api METHOD /api/endpoint [extra curl args...]
    local method="$1"; shift
    local endpoint="$1"; shift
    local csrf
    csrf=$(grep csrftoken "$COOKIE_JAR" 2>/dev/null | awk '{print $NF}')
    curl -s -b "$COOKIE_JAR" -c "$COOKIE_JAR" \
        -X "$method" "${LS_BASE}${endpoint}" \
        -H "X-CSRFToken: ${csrf}" \
        -H "Referer: ${LS_BASE}${endpoint}" \
        "$@"
}

info "Signing up / logging in..."
CSRF=$(_get_csrf "/user/signup")

# Try signup first (form-based POST with CSRF)
curl -s -o /dev/null \
    -b "$COOKIE_JAR" -c "$COOKIE_JAR" \
    -X POST "${LS_BASE}/user/signup" \
    -H "X-CSRFToken: ${CSRF}" \
    -H "Referer: ${LS_BASE}/user/signup" \
    -d "email=${LS_EMAIL}&password=${LS_PASSWORD}&csrfmiddlewaretoken=${CSRF}" \
    2>/dev/null || true

if _is_authed; then
    AUTHENTICATED=true
fi

# If signup didn't authenticate (user already exists), try login
if [[ "$AUTHENTICATED" == "false" ]]; then
    info "User exists, logging in..."
    CSRF=$(_get_csrf "/user/login")

    curl -s -o /dev/null \
        -b "$COOKIE_JAR" -c "$COOKIE_JAR" \
        -X POST "${LS_BASE}/user/login" \
        -H "X-CSRFToken: ${CSRF}" \
        -H "Referer: ${LS_BASE}/user/login" \
        -d "email=${LS_EMAIL}&password=${LS_PASSWORD}&csrfmiddlewaretoken=${CSRF}" \
        2>/dev/null || true

    if _is_authed; then
        AUTHENTICATED=true
    fi
fi

if [[ "$AUTHENTICATED" == "false" ]]; then
    warn "Could not authenticate with Label Studio."
    warn "Please log in manually at ${LS_BASE} and create a project."
    warn "  Email: ${LS_EMAIL}  Password: ${LS_PASSWORD}"
    open_browser "${LS_BASE}"
    info "Servers are running. Press Ctrl+C to stop."
    wait
    exit 0
fi

ok "Authenticated as ${LS_EMAIL}"

# ── Validate saved project ID (if any) ─────────────────────────────────────
if [[ -n "$PROJECT_ID" ]]; then
    PROJ_CHECK=$(_api GET "/api/projects/${PROJECT_ID}" 2>/dev/null || echo "{}")
    PROJ_VALID=$(echo "$PROJ_CHECK" | $PYTHON -c "import sys,json; d=json.load(sys.stdin); print('yes' if d.get('id') else 'no')" 2>/dev/null || echo "no")

    if [[ "$PROJ_VALID" != "yes" ]]; then
        warn "Saved project ID ${PROJECT_ID} is not accessible (wrong user or deleted). Creating new project."
        PROJECT_ID=""
        rm -f "$SESSION_FILE" 2>/dev/null || true
    fi
fi

# ── Create or reuse project ────────────────────────────────────────────────
IS_RERUN=false

if [[ -z "$PROJECT_ID" ]]; then
    info "Creating project: ${PROJECT_NAME}"

    CREATE_RESP=$(_api POST "/api/projects" \
        -H "Content-Type: application/json" \
        -d "$($PYTHON -c "
import json
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
    _api PATCH "/api/projects/${PROJECT_ID}" \
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
json.dump({'project_id': ${PROJECT_ID}}, open('${SESSION_FILE}', 'w'), indent=2)
"
ok "Session saved to ${SESSION_FILE}"

# ── Export existing annotations (re-run only) ─────────────────────────────
TASKS_FILE="${DATASET_DIR}/label_studio_tasks.json"
EXPORT_BACKUP="${DATASET_DIR}/.ls_export_backup.json"

if [[ "$IS_RERUN" == "true" ]]; then
    info "Exporting existing annotations from project ${PROJECT_ID}..."

    EXPORT_STATUS=$(_api GET "/api/projects/${PROJECT_ID}/export?exportType=JSON" \
        -w "%{http_code}" -o "$EXPORT_BACKUP" 2>/dev/null || echo "000")

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
            $UVRUN python -m home_guard_project.labeling \
                --merge "$EXPORT_BACKUP" \
                --dataset-dir "$DATASET_DIR"
            ok "Annotations merged."
        fi

        # Delete all existing tasks in the project before reimport
        info "Clearing existing tasks from project ${PROJECT_ID}..."
        _api POST "/api/dm/actions?id=delete_tasks&project=${PROJECT_ID}" \
            -H "Content-Type: application/json" \
            -d '{"selectedItems": {"all": true, "excluded": []}}' \
            > /dev/null 2>&1 || true
        ok "Existing tasks cleared."
    else
        warn "Could not export annotations (HTTP ${EXPORT_STATUS}). Proceeding without merge."
        rm -f "$EXPORT_BACKUP" 2>/dev/null || true
    fi
fi

# ── Import tasks ───────────────────────────────────────────────────────────
if [[ -f "$TASKS_FILE" ]]; then
    info "Importing tasks into project ${PROJECT_ID}..."

    IMPORT_RESP=$(_api POST "/api/projects/${PROJECT_ID}/import" \
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
#  STEP 10: Open browser
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
