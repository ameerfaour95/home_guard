#!/usr/bin/env bash
# ============================================================================
#  start.sh — One-command annotation launcher
#
#  Lives in:  home_guard_project/labeling/start.sh
#
#  Usage:
#      ./home_guard_project/labeling/start.sh [DATASET_DIR] [--force] [--share]
#      ./home_guard_project/labeling/start.sh [DATASET_DIR] --attach
#
#  Examples:
#      ./home_guard_project/labeling/start.sh ./dataset_multi
#      ./home_guard_project/labeling/start.sh ./dataset_multi --force
#      ./home_guard_project/labeling/start.sh ./dataset_multi --share
#
#      # Add a second project to an already-running Label Studio:
#      # 1. Edit config.yaml: set project_name and storage.s3.uri
#      # 2. Run with --attach:
#      ./home_guard_project/labeling/start.sh ./dataset_uca --attach
#
#  Flags:
#    --force   Force task regeneration even if tasks are up-to-date
#    --share   Launch a Cloudflare tunnel for remote annotator access
#    --attach  Skip starting LS/file-server/tunnel (assumes already running).
#              Creates a new project and imports tasks into the running LS.
#              Edit config.yaml (project_name, storage.s3.uri) before using.
#
#  What it does (normal mode):
#    1. Verifies Python >= 3.12 and installs uv if missing
#    2. Runs uv sync to install all dependencies (incl. label-studio)
#    3. Verifies dataset directory
#    4. Auto-syncs vlm_crops/ ↔ clips/ (cascade deletes, orphan removal)
#    5. Re-encodes video clips + VLM crops to H.264 (skips already-encoded)
#    6. Always regenerates Label Studio config + tasks JSON
#    7. Starts file server (port 8081) and Label Studio (port 8080)
#    8. Auto-creates a Label Studio project via the API
#    9. On re-run: exports existing annotations, merges into new tasks,
#       clears old tasks, then reimports (preserves all human annotations)
#   10. Auto-creates annotator accounts from config.yaml (idempotent)
#   11. Imports tasks via the API, opens the browser
#   12. (--share) Launches a Cloudflare tunnel so annotators can access remotely
#   13. Ctrl+C cleanly shuts everything down
#
#  What it does (--attach mode):
#    1. Finds Python, verifies LS is already running
#    2. Generates tasks (syncs S3 metadata if in S3 mode)
#    3. Creates or reuses a project (by project_name in config.yaml)
#    4. Imports tasks, then exits
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
SHARE_MODE=false
ATTACH_MODE=false
SKIP_GENERATE=false
POSITIONAL_ARGS=()
for arg in "$@"; do
    case "$arg" in
        --force)          FORCE_REBUILD=true ;;
        --share)          SHARE_MODE=true ;;
        --attach)         ATTACH_MODE=true ;;
        --skip-generate)  SKIP_GENERATE=true ;;
        *)                POSITIONAL_ARGS+=("$arg") ;;
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

# storage.mode (local or s3)
STORAGE_MODE="$(yaml_val mode)"
STORAGE_MODE="${STORAGE_MODE:-local}"

# share.enabled — default share mode from config (--share CLI flag overrides)
SHARE_ENABLED=$(sed -n '/^share:/,/^[a-z]/{ /^\s*enabled:/p }' "$CONFIG_YAML" | head -1 | sed 's/^[^:]*:\s*//' | tr -d '[:space:]')
if [[ "$SHARE_ENABLED" == "true" ]]; then
    SHARE_MODE=true
fi

LS_BASE="http://localhost:${LS_PORT}"

# Project-specific session file (allows multiple projects on the same dataset dir)
PROJECT_SLUG=$(echo "$PROJECT_NAME" | tr '[:upper:]' '[:lower:]' | tr -c '[:alnum:]' '_' | sed 's/__*/_/g; s/^_//; s/_$//')
SESSION_FILE="${DATASET_DIR}/.ls_session_${PROJECT_SLUG}.json"

# Migrate from the old non-project-specific session file if present
OLD_SESSION="${DATASET_DIR}/.ls_session.json"
if [[ ! -f "$SESSION_FILE" && -f "$OLD_SESSION" ]]; then
    cp "$OLD_SESSION" "$SESSION_FILE"
fi

# PIDs for cleanup
FILE_SERVER_PID=""
LABEL_STUDIO_PID=""
TUNNEL_PID=""
IMPORT_PID=""

# Initialized early so both normal and --attach paths can reference them
SHARE_URL=""
TUNNEL_LOG=""

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
    if [[ -n "$IMPORT_PID" ]]; then
        kill "$IMPORT_PID" 2>/dev/null && info "Import process stopped." || true
        wait "$IMPORT_PID" 2>/dev/null || true
    fi
    if [[ -n "$TUNNEL_PID" ]]; then
        kill "$TUNNEL_PID" 2>/dev/null && info "Cloudflare tunnel stopped." || true
    fi
    if [[ -n "$FILE_SERVER_PID" ]]; then
        kill "$FILE_SERVER_PID" 2>/dev/null && info "File server stopped." || true
    fi
    if [[ -n "$LABEL_STUDIO_PID" ]]; then
        kill "$LABEL_STUDIO_PID" 2>/dev/null && info "Label Studio stopped." || true
    fi
    # Clean up temp files
    [[ -n "${COOKIE_JAR:-}" ]] && rm -f "$COOKIE_JAR" 2>/dev/null || true
    [[ -n "${TUNNEL_LOG:-}" ]] && rm -f "$TUNNEL_LOG" 2>/dev/null || true
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

if [[ "$ATTACH_MODE" != "true" ]]; then
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

# ── Patch Label Studio: re-enable /api/dm/tasks/ endpoint ─────────────────
# LS 1.22.0 ships with the Data Manager tasks endpoint commented out,
# causing "Unexpected token '<', DOCTYPE … is not valid JSON" in the browser.
LS_DM_URLS=$($PYTHON -c "
import label_studio.data_manager as dm; import os
print(os.path.join(os.path.dirname(dm.__file__), 'urls.py'))
" 2>/dev/null || echo "")

if [[ -n "$LS_DM_URLS" && -f "$LS_DM_URLS" ]]; then
    if grep -q '# *path("api/dm/tasks/"' "$LS_DM_URLS" 2>/dev/null; then
        sed -i 's|# *path("api/dm/tasks/", api\.TaskListAPI\.as_view())|    path("api/dm/tasks/", api.TaskListAPI.as_view())|' "$LS_DM_URLS"
        ok "Patched Label Studio: /api/dm/tasks/ endpoint enabled."
    fi
fi

# ── Resolve the uv-managed python for running commands ──────────────────────
UVRUN="uv run"

# ============================================================================
#  STEP 4: Verify dataset directory
# ============================================================================
step "Verifying dataset"

info "Storage mode: ${STORAGE_MODE}"

if [[ "$STORAGE_MODE" == "s3" ]]; then
    # In S3 mode, the Python module syncs meta/ from S3 automatically.
    # Just ensure the directory exists (it will be populated by the sync).
    mkdir -p "$DATASET_DIR"
    ok "S3 mode — metadata will be synced from S3."
else
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
fi

# ============================================================================
#  STEP 5: Cleanup — sync vlm_crops ↔ clips, remove orphans (local only)
# ============================================================================
if [[ "$STORAGE_MODE" != "s3" ]]; then
    step "Synchronising dataset (cleanup)"

    info "Aligning vlm_crops/ and clips/ — cascade-deleting mismatches..."

    CLEANUP_OUTPUT=$($UVRUN python -m home_guard_project.labeling \
        --cleanup-only --dataset-dir "$DATASET_DIR" 2>&1 || echo "")

    # Parse counts from the summary
    CLEANUP_CASCADE=$(echo "$CLEANUP_OUTPUT" | grep "Cascade clip deletes" | grep -oE '[0-9]+' || echo "0")
    CLEANUP_META=$(echo "$CLEANUP_OUTPUT" | grep "Orphan meta removed" | grep -oE '[0-9]+' || echo "0")
    CLEANUP_RESP=$(echo "$CLEANUP_OUTPUT" | grep "Orphan resp removed" | grep -oE '[0-9]+' || echo "0")
    CLEANUP_YOLO_IMG=$(echo "$CLEANUP_OUTPUT" | grep "Orphan YOLO imgs" | grep -oE '[0-9]+' || echo "0")
    CLEANUP_YOLO_LBL=$(echo "$CLEANUP_OUTPUT" | grep "Orphan YOLO lbls" | grep -oE '[0-9]+' || echo "0")
    CLEANUP_VLM=$(echo "$CLEANUP_OUTPUT" | grep "Orphan VLM crops" | grep -oE '[0-9]+' || echo "0")
    CLEANUP_TOTAL=$(( CLEANUP_CASCADE + CLEANUP_META + CLEANUP_RESP + CLEANUP_YOLO_IMG + CLEANUP_YOLO_LBL + CLEANUP_VLM ))

    if (( CLEANUP_TOTAL > 0 )); then
        info "Cleaned up ${CLEANUP_TOTAL} files:"
        (( CLEANUP_CASCADE > 0 )) && echo -e "    Cascade clip deletes: ${CLEANUP_CASCADE}"
        (( CLEANUP_META > 0 ))    && echo -e "    Orphan meta files   : ${CLEANUP_META}"
        (( CLEANUP_RESP > 0 ))    && echo -e "    Orphan responses    : ${CLEANUP_RESP}"
        (( CLEANUP_YOLO_IMG > 0 )) && echo -e "    Orphan YOLO images  : ${CLEANUP_YOLO_IMG}"
        (( CLEANUP_YOLO_LBL > 0 )) && echo -e "    Orphan YOLO labels  : ${CLEANUP_YOLO_LBL}"
        (( CLEANUP_VLM > 0 ))     && echo -e "    Orphan VLM crops    : ${CLEANUP_VLM}"
        FORCE_REBUILD=true
    else
        ok "Dataset is clean — vlm_crops/ and clips/ are in sync."
    fi

    # Show final counts
    CLIP_COUNT=$(find "${DATASET_DIR}/clips" -name "*.mp4" 2>/dev/null | wc -l || echo "0")
    VLM_COUNT=$(find "${DATASET_DIR}/vlm_crops" -name "*.mp4" 2>/dev/null | wc -l || echo "0")
    ok "Clips: ${CLIP_COUNT}  |  VLM crops: ${VLM_COUNT}"
fi

else
# ── Attach mode: lightweight pre-flight ──────────────────────────────────
step "Attach mode — connecting to running Label Studio"

PYTHON="$(find_python)" || { err "Python >= 3.12 not found."; exit 1; }
UVRUN="uv run"

if ! curl -sf "${LS_BASE}/api/version" > /dev/null 2>&1; then
    err "Label Studio is not running at ${LS_BASE}."
    err "Start it first:  ./home_guard_project/labeling/start.sh"
    exit 1
fi
ok "Label Studio is running at ${LS_BASE}"

mkdir -p "$DATASET_DIR"

fi  # end ATTACH_MODE check

# ============================================================================
#  STEP 6: Re-encode clips + generate tasks  (S3: generate only, no re-encode)
# ============================================================================
if [[ "$SKIP_GENERATE" == "true" && -f "${DATASET_DIR}/label_studio_tasks.json" ]]; then
    step "Skipping task generation (--skip-generate)"
    info "Using existing ${DATASET_DIR}/label_studio_tasks.json"

    # Still need the label config XML for project creation
    if [[ ! -f "${DATASET_DIR}/label_studio_config.xml" ]]; then
        $UVRUN python -c "
from home_guard_project.labeling.tasks import build_label_config, write_label_config
write_label_config('${DATASET_DIR}')
"
    fi
    ok "Tasks file found — skipping S3 sync and task generation."
else
    step "Generating tasks"

    if [[ "$STORAGE_MODE" == "s3" ]]; then
        info "S3 mode — syncing metadata from S3 and generating tasks with pre-signed URLs..."
        LABELING_ARGS=(--dataset-dir "$DATASET_DIR" --no-cleanup --force)
    else
        info "Re-encoding mp4v clips to H.264 (already-encoded clips are skipped)."
        info "Then generating Label Studio config + tasks JSON..."
        LABELING_ARGS=(--dataset-dir "$DATASET_DIR" --reencode --no-cleanup --force)
    fi

    $UVRUN python -m home_guard_project.labeling "${LABELING_ARGS[@]}"

    ok "Tasks generated at ${DATASET_DIR}/label_studio_tasks.json"
fi

if [[ "$ATTACH_MODE" != "true" ]]; then
# ============================================================================
#  STEP 7: Start file server (local mode only)
# ============================================================================
if [[ "$STORAGE_MODE" != "s3" ]]; then
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
else
    info "S3 mode — file server not needed (videos served via pre-signed S3 URLs)."
fi

# ============================================================================
#  STEP 8: Start Cloudflare Tunnel BEFORE Label Studio (--share only)
#           The tunnel proxies traffic even before LS is listening.
#           We need the public URL so we can set LABEL_STUDIO_HOST for CSRF.
# ============================================================================
if [[ "$SHARE_MODE" == "true" ]]; then
    step "Starting Cloudflare tunnel"

    # Find cloudflared — may not be in Git Bash's PATH on Windows
    CF_BIN=""
    if command -v cloudflared &>/dev/null; then
        CF_BIN="cloudflared"
    elif [[ "$OS" == "windows" ]]; then
        for p in \
            "/c/Program Files (x86)/cloudflared/cloudflared.exe" \
            "/c/Program Files/cloudflared/cloudflared.exe" \
            "${LOCALAPPDATA:-}/Programs/cloudflared/cloudflared.exe" \
            "${PROGRAMFILES:-}/cloudflared/cloudflared.exe" \
            "${PROGRAMDATA:-}/cloudflared/cloudflared.exe" \
            "/c/Users/${USERNAME:-$USER}/AppData/Local/Programs/cloudflared/cloudflared.exe"; do
            if [[ -x "$p" ]]; then
                CF_BIN="$p"
                break
            fi
        done
    fi

    if [[ -z "$CF_BIN" ]]; then
        err "cloudflared is not installed."
        echo ""
        echo -e "  Install cloudflared:"
        echo -e "    ${BOLD}Windows:${NC}  winget install cloudflare.cloudflared"
        echo -e "    ${BOLD}macOS:${NC}    brew install cloudflared"
        echo -e "    ${BOLD}Linux:${NC}    See https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/downloads/"
        echo ""
        warn "Continuing without tunnel — Label Studio is only available locally."
    else
        TUNNEL_LOG=$(mktemp)
        "$CF_BIN" tunnel --url "http://localhost:${LS_PORT}" > "$TUNNEL_LOG" 2>&1 &
        TUNNEL_PID=$!

        info "Waiting for Cloudflare tunnel..."
        CF_WAIT=0
        while (( CF_WAIT < 30 )); do
            sleep 1
            CF_WAIT=$((CF_WAIT + 1))
            SHARE_URL=$(grep -oE 'https://[a-zA-Z0-9_-]+\.trycloudflare\.com' "$TUNNEL_LOG" 2>/dev/null | head -1 || echo "")
            if [[ -n "$SHARE_URL" ]]; then
                break
            fi
        done

        if [[ -n "$SHARE_URL" ]]; then
            ok "Cloudflare tunnel active (PID ${TUNNEL_PID}): ${SHARE_URL}"
        else
            warn "Could not detect Cloudflare tunnel URL. Check log: ${TUNNEL_LOG}"
            warn "Make sure cloudflared is installed correctly."
        fi
    fi
fi

# ============================================================================
#  STEP 9: Start Label Studio (background)
# ============================================================================
step "Starting Label Studio (port ${LS_PORT})"

# LS manages its own SECRET_KEY in $LOCALAPPDATA/label-studio/.env.
# Do NOT override it — that would invalidate existing user passwords.

# Auto-provision the admin account on first launch (when no users exist yet).
export LABEL_STUDIO_USERNAME="${LS_EMAIL}"
export LABEL_STUDIO_PASSWORD="${LS_PASSWORD}"

# Windows cp1255/cp1252 consoles choke on LS's Unicode output (→ arrow in
# version-check message).  Force UTF-8 so the print() call doesn't crash.
export PYTHONIOENCODING="utf-8"

# Tell Label Studio about the public hostname so Django CSRF trusts it.
if [[ -n "$SHARE_URL" ]]; then
    export LABEL_STUDIO_HOST="${SHARE_URL}"
    export LABEL_STUDIO_PROXY="true"
    export CSRF_TRUSTED_ORIGINS="${SHARE_URL}"
fi

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

fi  # end !ATTACH_MODE (skip service startup)

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
        ANNOTATED_COUNT=$($PYTHON -c "
import json
tasks = json.load(open('${EXPORT_BACKUP}'))
annotated = sum(1 for t in tasks if t.get('annotations'))
print(annotated)
" 2>/dev/null || echo "0")
        TOTAL_EXPORTED=$($PYTHON -c "
import json; print(len(json.load(open('${EXPORT_BACKUP}'))))
" 2>/dev/null || echo "0")
        ok "Exported: ${TOTAL_EXPORTED} tasks, ${ANNOTATED_COUNT} with annotations"

        # Only merge if there are actual human annotations to preserve
        if [[ "$ANNOTATED_COUNT" -gt 0 && -f "$TASKS_FILE" ]]; then
            info "Merging ${ANNOTATED_COUNT} annotations into new tasks..."
            $UVRUN python -m home_guard_project.labeling \
                --merge "$EXPORT_BACKUP" \
                --dataset-dir "$DATASET_DIR"
            ok "Annotations merged."
        elif [[ "$ANNOTATED_COUNT" == "0" ]]; then
            info "No annotations to merge — skipping merge step."
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

# ── Create annotator accounts (idempotent) ────────────────────────────────
ANNOTATOR_LIST=$($UVRUN python -c "
import yaml, json, sys
with open('${CONFIG_YAML}') as f:
    cfg = yaml.safe_load(f)
annotators = cfg.get('annotators') or []
json.dump(annotators, sys.stdout)
" 2>/dev/null || echo "[]")

ANNOTATOR_COUNT=$($PYTHON -c "import json,sys; print(len(json.loads(sys.argv[1])))" "$ANNOTATOR_LIST" 2>/dev/null || echo "0")

if (( ANNOTATOR_COUNT > 0 )); then
    info "Creating ${ANNOTATOR_COUNT} annotator account(s)..."

    $PYTHON -c "
import json, sys
annotators = json.loads(sys.argv[1])
for a in annotators:
    print(json.dumps(a))
" "$ANNOTATOR_LIST" | while IFS= read -r line; do
        ANN_EMAIL=$($PYTHON -c "import json,sys; print(json.loads(sys.argv[1])['email'])" "$line")
        ANN_PASS=$($PYTHON -c "import json,sys; print(json.loads(sys.argv[1])['password'])" "$line")

        # Try to create the user; LS returns 201 on success, 400 if exists
        CREATE_RESP=$(_api POST "/api/users/" \
            -H "Content-Type: application/json" \
            -d "{\"email\": \"${ANN_EMAIL}\", \"username\": \"${ANN_EMAIL}\", \"password\": \"${ANN_PASS}\"}" \
            -w "\n%{http_code}" 2>/dev/null || echo -e "\n000")

        HTTP_CODE=$(echo "$CREATE_RESP" | tail -1)
        if [[ "$HTTP_CODE" == "201" ]]; then
            ok "  Created: ${ANN_EMAIL}"
        else
            info "  Exists:  ${ANN_EMAIL} (skipped)"
        fi
    done
else
    info "No annotators configured in config.yaml — skipping account creation."
fi

# ── Import tasks (chunked for large files) ─────────────────────────────────
if [[ -f "$TASKS_FILE" ]]; then
    info "Importing tasks into project ${PROJECT_ID}..."

    IMPORT_RESULT_FILE=$(mktemp)
    $UVRUN $PYTHON -c "
import json, sys, urllib.request, http.cookiejar, ijson, re, signal
from decimal import Decimal

signal.signal(signal.SIGINT, lambda *_: sys.exit(130))
signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))

class DecimalEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, Decimal):
            return float(o)
        return super().default(o)

tasks_file  = sys.argv[1]
base_url    = sys.argv[2]
project_id  = sys.argv[3]
email       = sys.argv[4]
password    = sys.argv[5]
result_file = sys.argv[6]
BATCH_SIZE  = 200

cj = http.cookiejar.CookieJar()
opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cj))

def get_csrf():
    for c in cj:
        if c.name == 'csrftoken':
            return c.value
    return ''

def login():
    resp = opener.open(urllib.request.Request(f'{base_url}/user/signup'))
    resp.read()
    csrf = get_csrf()
    data = f'email={email}&password={password}&csrfmiddlewaretoken={csrf}'.encode()
    req = urllib.request.Request(f'{base_url}/user/login', data=data, method='POST')
    req.add_header('Content-Type', 'application/x-www-form-urlencoded')
    req.add_header('X-CSRFToken', csrf)
    req.add_header('Referer', f'{base_url}/user/login')
    try:
        opener.open(req)
    except urllib.error.HTTPError:
        pass
    whoami = json.loads(opener.open(urllib.request.Request(
        f'{base_url}/api/current-user/whoami',
        headers={'X-CSRFToken': get_csrf()},
    )).read())
    print(f'  Authenticated as: {whoami.get(\"email\", \"?\")}', flush=True)

login()

def do_import(payload, batch_num):
    csrf = get_csrf()
    url = f'{base_url}/api/projects/{project_id}/import'
    req = urllib.request.Request(url, data=payload, method='POST')
    req.add_header('Content-Type', 'application/json')
    req.add_header('X-CSRFToken', csrf)
    req.add_header('Referer', url)
    resp = opener.open(req, timeout=300)
    return json.loads(resp.read().decode('utf-8'))

imported = 0
errors   = 0
batch    = []
batch_num = 0

with open(tasks_file, 'rb') as f:
    for task in ijson.items(f, 'item'):
        batch.append(task)
        if len(batch) >= BATCH_SIZE:
            batch_num += 1
            payload = json.dumps(batch, cls=DecimalEncoder).encode('utf-8')
            try:
                body = do_import(payload, batch_num)
                imported += body.get('task_count', len(batch))
            except urllib.error.HTTPError as e:
                if e.code == 401:
                    print(f'  Session expired at batch {batch_num}, re-authenticating...', flush=True)
                    login()
                    try:
                        body = do_import(payload, batch_num)
                        imported += body.get('task_count', len(batch))
                    except Exception as e2:
                        errors += 1
                        print(f'  Error batch {batch_num} (after re-auth): {e2}', flush=True)
                else:
                    errors += 1
                    print(f'  Error batch {batch_num}: {e}', flush=True)
            except Exception as e:
                errors += 1
                print(f'  Error batch {batch_num}: {e}', flush=True)
            print(f'  Importing: batch {batch_num} — {imported} tasks', flush=True)
            batch = []

if batch:
    batch_num += 1
    payload = json.dumps(batch, cls=DecimalEncoder).encode('utf-8')
    try:
        body = do_import(payload, batch_num)
        imported += body.get('task_count', len(batch))
    except urllib.error.HTTPError as e:
        if e.code == 401:
            print(f'  Session expired at batch {batch_num}, re-authenticating...', flush=True)
            login()
            try:
                body = do_import(payload, batch_num)
                imported += body.get('task_count', len(batch))
            except Exception as e2:
                errors += 1
                print(f'  Error batch {batch_num} (after re-auth): {e2}', flush=True)
        else:
            errors += 1
            print(f'  Error batch {batch_num}: {e}', flush=True)
    except Exception as e:
        errors += 1
        print(f'  Error batch {batch_num}: {e}', flush=True)

if errors:
    print(f'  Warning: {errors} batch(es) had errors.')

open(result_file, 'w').write(str(imported))
" "$TASKS_FILE" "$LS_BASE" "$PROJECT_ID" "$LS_EMAIL" "$LS_PASSWORD" "$IMPORT_RESULT_FILE" &
    IMPORT_PID=$!

    wait "$IMPORT_PID" 2>/dev/null
    IMPORT_EXIT=$?
    IMPORT_PID=""

    if [[ "$IMPORT_EXIT" -ne 0 && "$IMPORT_EXIT" -ne 130 && "$IMPORT_EXIT" -ne 143 ]]; then
        warn "Import process exited with code ${IMPORT_EXIT}"
    fi

    TASK_COUNT=$(cat "$IMPORT_RESULT_FILE" 2>/dev/null || echo "?")
    rm -f "$IMPORT_RESULT_FILE"

    ok "Imported ${TASK_COUNT} tasks."
else
    warn "Tasks file not found: ${TASKS_FILE}"
fi

# ============================================================================
#  STEP 12: Open browser
# ============================================================================
step "Ready!"

if [[ -n "$SHARE_URL" ]]; then
    PROJECT_URL="${SHARE_URL}/projects/${PROJECT_ID}"
else
    PROJECT_URL="${LS_BASE}/projects/${PROJECT_ID}"
fi

echo ""
echo -e "  ${BOLD}Label Studio:${NC}   ${PROJECT_URL}"
if [[ -n "$SHARE_URL" ]]; then
    echo -e "  ${BOLD}Local URL:${NC}      ${LS_BASE}/projects/${PROJECT_ID}"
fi
if [[ "$STORAGE_MODE" != "s3" ]]; then
    echo -e "  ${BOLD}File Server:${NC}    http://localhost:${FILE_SERVER_PORT}"
else
    echo -e "  ${BOLD}Storage:${NC}        S3 (pre-signed URLs)"
fi
echo -e "  ${BOLD}Login:${NC}          ${LS_EMAIL} / ${LS_PASSWORD}"

# Print annotator credentials if any
if (( ANNOTATOR_COUNT > 0 )); then
    echo ""
    echo -e "  ${BOLD}Annotator accounts:${NC}"
    $PYTHON -c "
import json, sys
annotators = json.loads(sys.argv[1])
for a in annotators:
    print(f\"    {a['email']}  /  {a['password']}\")
" "$ANNOTATOR_LIST" 2>/dev/null || true
fi

echo ""

if [[ -n "$SHARE_URL" ]]; then
    echo -e "  ${YELLOW}Share this URL with your annotators:${NC}"
    echo -e "  ${BOLD}${SHARE_URL}/projects/${PROJECT_ID}${NC}"
    echo ""
fi

open_browser "$PROJECT_URL"

if [[ "$ATTACH_MODE" == "true" ]]; then
    ok "Project attached. You can close this terminal."
else
    ok "Press Ctrl+C to stop all services."
    # Keep the script alive until interrupted
    wait
fi
