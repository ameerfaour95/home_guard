#!/usr/bin/env bash
# ============================================================================
#  review_tags.sh — Review all completed tagging batches in Label Studio
#
#  Downloads export JSONs from S3, regenerates tasks with fresh pre-signed
#  URLs, merges existing annotations (YOLO tracks + VLM text), and launches
#  Label Studio with one project per batch for visual review.
#
#  Usage:
#      ./review_tags.sh
#
#  Press Ctrl+C to stop Label Studio when done reviewing.
# ============================================================================
set -euo pipefail

# ── Always run from the project root (where pyproject.toml lives) ──────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

CONFIG_YAML="home_guard_project/labeling/config.yaml"
START_SH="./home_guard_project/labeling/start.sh"

# ── Batch definitions ──────────────────────────────────────────────────────
BATCH_NAMES=(
    "ameer_house_batch_1"
    "ameer_house_batch_2"
    "uca_dataset_batch"
)
BATCH_S3_URIS=(
    "s3://security-camera-project-v1/tagging/ameer_house_batch_1/dataset_multi"
    "s3://security-camera-project-v1/tagging/ameer_house_batch_2/dataset_multi"
    "s3://security-camera-project-v1/tagging/uca_dataset_batch/dataset_uca"
)
BATCH_PROJECTS=(
    "Review: Ameer House Batch 1"
    "Review: Ameer House Batch 2"
    "Review: UCA Dataset Batch"
)
BATCH_DIRS=(
    "./dataset_review/batch1"
    "./dataset_review/batch2"
    "./dataset_review/batch3"
)
BATCH_S3_EXPORTS=(
    "s3://security-camera-project-v1/tagging/ameer_house_batch_1/ameer_house_batch_1.json"
    "s3://security-camera-project-v1/tagging/ameer_house_batch_2/ameer_house_batch_2.json"
    "s3://security-camera-project-v1/tagging/uca_dataset_batch/uca_dataset_batch.json"
)
BATCH_EXPORT_FILES=(
    "review_exports/ameer_house_batch_1.json"
    "review_exports/ameer_house_batch_2.json"
    "review_exports/uca_dataset_batch.json"
)

S3_REGION="us-east-1"
LS_PORT=8080
LS_BASE="http://localhost:${LS_PORT}"
LS_EMAIL="ameerfaour95@gmail.com"
LS_PASSWORD="Amer1967"

# ── Colors ─────────────────────────────────────────────────────────────────
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
CONFIG_BACKUP=""
LS_PID=""

cleanup() {
    echo ""
    info "Shutting down..."
    if [[ -n "$LS_PID" ]]; then
        kill "$LS_PID" 2>/dev/null && info "Label Studio stopped." || true
    fi
    if [[ -n "$CONFIG_BACKUP" && -f "$CONFIG_BACKUP" ]]; then
        cp "$CONFIG_BACKUP" "$CONFIG_YAML"
        rm -f "$CONFIG_BACKUP"
        ok "config.yaml restored from backup."
    fi
    ok "Done. Goodbye."
}
trap cleanup EXIT INT TERM

# ── Find Python ────────────────────────────────────────────────────────────
find_python() {
    if command -v py &>/dev/null; then echo "py"; return; fi
    for c in python3 python; do
        if command -v "$c" &>/dev/null; then
            local ver
            ver="$("$c" --version 2>&1)" || continue
            if [[ "$ver" == Python\ 3.* ]]; then echo "$c"; return; fi
        fi
    done
    return 1
}

PYTHON="$(find_python)" || { err "Python >= 3.12 not found."; exit 1; }
PY_VER="$($PYTHON --version 2>&1)"
ok "Found $PY_VER  ($PYTHON)"

UVRUN="uv run"
AWS="$UVRUN aws"

# ── Helper: patch config.yaml for a batch ──────────────────────────────────
patch_config() {
    local s3_uri="$1"
    local project_name="$2"
    local dataset_dir="$3"

    $UVRUN python -c "
import yaml, sys

config_path = sys.argv[1]
project_name = sys.argv[2]
dataset_dir = sys.argv[3]
s3_uri = sys.argv[4]

with open(config_path, 'r') as f:
    cfg = yaml.safe_load(f)

cfg['label_studio']['project_name'] = project_name
cfg['dataset_dir'] = dataset_dir
cfg['storage']['mode'] = 's3'
cfg['storage']['s3']['uri'] = s3_uri
cfg['storage']['s3']['region'] = 'us-east-1'
cfg['storage']['s3']['url_expiry_sec'] = 604800
cfg['share']['enabled'] = False

with open(config_path, 'w') as f:
    yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
" "$CONFIG_YAML" "$project_name" "$dataset_dir" "$s3_uri"
}

# ============================================================================
#  STEP 1: Prerequisites
# ============================================================================
step "Prerequisites"

if ! command -v uv &>/dev/null; then
    info "Installing uv..."
    $PYTHON -m pip install --quiet uv
fi

uv sync --quiet
ok "Dependencies installed."

CONFIG_BACKUP=$(mktemp)
cp "$CONFIG_YAML" "$CONFIG_BACKUP"
ok "config.yaml backed up."

if curl -sf "${LS_BASE}/api/version" > /dev/null 2>&1; then
    err "Label Studio is already running on port ${LS_PORT}."
    err "Stop it first (Ctrl+C in the running terminal), then re-run this script."
    CONFIG_BACKUP=""
    exit 1
fi

# ============================================================================
#  STEP 2: Download export JSONs from S3
# ============================================================================
step "Downloading export JSONs from S3"

mkdir -p review_exports

for i in "${!BATCH_NAMES[@]}"; do
    export_file="${BATCH_EXPORT_FILES[$i]}"
    if [[ -f "$export_file" ]]; then
        ok "Already downloaded: ${export_file}"
    else
        info "Downloading: ${BATCH_S3_EXPORTS[$i]}"
        $AWS s3 cp "${BATCH_S3_EXPORTS[$i]}" "$export_file" --region "$S3_REGION"
        ok "Downloaded: ${export_file}"
    fi
done

# ============================================================================
#  STEP 3: Generate tasks and merge annotations for each batch
# ============================================================================
for i in "${!BATCH_NAMES[@]}"; do
    batch_name="${BATCH_NAMES[$i]}"
    s3_uri="${BATCH_S3_URIS[$i]}"
    project="${BATCH_PROJECTS[$i]}"
    dataset_dir="${BATCH_DIRS[$i]}"
    export_file="${BATCH_EXPORT_FILES[$i]}"

    step "Preparing batch: ${batch_name}"

    mkdir -p "$dataset_dir"
    patch_config "$s3_uri" "$project" "$dataset_dir"

    info "Syncing metadata from S3 and building tasks with fresh pre-signed URLs..."
    $UVRUN python -m home_guard_project.labeling \
        --dataset-dir "$dataset_dir" --force --no-cleanup

    info "Merging annotations from export..."
    $UVRUN python -m home_guard_project.labeling \
        --merge "$export_file" --dataset-dir "$dataset_dir"

    info "Cleaning annotation metadata for import compatibility..."
    $UVRUN python -c "
import json, sys

tasks_path = sys.argv[1]
with open(tasks_path, 'r', encoding='utf-8') as f:
    tasks = json.load(f)

KEEP_KEYS = {'result', 'was_cancelled', 'ground_truth'}
for task in tasks:
    if 'annotations' in task:
        cleaned = []
        for ann in task['annotations']:
            cleaned.append({k: v for k, v in ann.items() if k in KEEP_KEYS})
        task['annotations'] = cleaned

with open(tasks_path, 'w', encoding='utf-8') as f:
    json.dump(tasks, f, ensure_ascii=False)
" "${dataset_dir}/label_studio_tasks.json"

    TASK_COUNT=$($PYTHON -c "
import json, sys
with open(sys.argv[1]) as f:
    tasks = json.load(f)
annotated = sum(1 for t in tasks if t.get('annotations'))
print(f'{len(tasks)} tasks, {annotated} with annotations')
" "${dataset_dir}/label_studio_tasks.json" 2>/dev/null || echo "unknown")

    ok "${project}: ${TASK_COUNT}"
done

# ============================================================================
#  STEP 4: Patch Label Studio DM endpoint (same fix as start.sh)
# ============================================================================
step "Patching Label Studio"

LS_DM_URLS=$($UVRUN python -c "
import label_studio.data_manager as dm; import os
print(os.path.join(os.path.dirname(dm.__file__), 'urls.py'))
" 2>/dev/null || echo "")

if [[ -n "$LS_DM_URLS" && -f "$LS_DM_URLS" ]]; then
    if grep -q '# *path("api/dm/tasks/"' "$LS_DM_URLS" 2>/dev/null; then
        sed -i 's|# *path("api/dm/tasks/", api\.TaskListAPI\.as_view())|    path("api/dm/tasks/", api.TaskListAPI.as_view())|' "$LS_DM_URLS"
        ok "Patched LS Data Manager endpoint."
    else
        ok "LS Data Manager endpoint already patched."
    fi
else
    ok "LS DM patch skipped (could not locate urls.py)."
fi

# ============================================================================
#  STEP 5: Start Label Studio
# ============================================================================
step "Starting Label Studio (port ${LS_PORT})"

export LABEL_STUDIO_USERNAME="${LS_EMAIL}"
export LABEL_STUDIO_PASSWORD="${LS_PASSWORD}"
export PYTHONIOENCODING="utf-8"

$UVRUN label-studio start \
    --port "$LS_PORT" \
    --no-browser \
    > /dev/null 2>&1 &
LS_PID=$!

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
ok "Label Studio running (PID ${LS_PID})"

# ============================================================================
#  STEP 6: Create projects and import tasks (using start.sh --attach)
# ============================================================================
for i in "${!BATCH_NAMES[@]}"; do
    project="${BATCH_PROJECTS[$i]}"
    s3_uri="${BATCH_S3_URIS[$i]}"
    dataset_dir="${BATCH_DIRS[$i]}"

    step "Importing project: ${project}"
    patch_config "$s3_uri" "$project" "$dataset_dir"

    $START_SH "$dataset_dir" --skip-generate --attach

    ok "Imported: ${project}"
done

# ============================================================================
#  STEP 7: Summary
# ============================================================================
step "All review projects loaded!"

echo ""
echo -e "  ${BOLD}Label Studio:${NC}   ${LS_BASE}"
echo -e "  ${BOLD}Login:${NC}          ${LS_EMAIL} / ${LS_PASSWORD}"
echo ""
echo -e "  ${BOLD}Projects:${NC}"
for i in "${!BATCH_PROJECTS[@]}"; do
    echo -e "    $((i + 1)). ${BATCH_PROJECTS[$i]}"
done
echo ""

case "$(uname -s)" in
    MINGW*|MSYS*|CYGWIN*) start "${LS_BASE}" 2>/dev/null || true ;;
    Darwin*)               open "${LS_BASE}"  2>/dev/null || true ;;
    *)                     xdg-open "${LS_BASE}" 2>/dev/null || true ;;
esac

ok "Press Ctrl+C to stop Label Studio."
wait "$LS_PID" 2>/dev/null || true
