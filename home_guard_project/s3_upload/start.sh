#!/usr/bin/env bash
# ============================================================================
#  start.sh — One-command S3 upload launcher
#
#  Lives in:  home_guard_project/s3_upload/start.sh
#
#  Usage:
#      ./home_guard_project/s3_upload/start.sh [DATASET_DIR] [--skip-reencode] [--dry-run] [--no-cleanup]
#
#  What it does:
#    1. Verifies Python >= 3.12 and installs uv if missing
#    2. Runs uv sync to install all dependencies (incl. boto3)
#    3. Verifies dataset directory
#    4. Auto-syncs vlm_crops/ ↔ clips/ (cascade deletes, orphan removal)
#    5. Verifies AWS credentials exist at ~/.aws/credentials
#    6. Re-encodes MP4 clips to H.264 (skips already-encoded)
#    7. Uploads all files to S3 (skips files with matching size)
# ============================================================================
set -euo pipefail

# ── Always run from the project root (where pyproject.toml lives) ─────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

# ── Configuration from config.yaml ────────────────────────────────────────
CONFIG_YAML="${SCRIPT_DIR}/config.yaml"

yaml_val() {
    local key="$1"
    local raw
    raw=$(grep -E "^\s*${key}:" "$CONFIG_YAML" | head -1 | sed 's/^[^:]*:\s*//')
    raw="${raw%\"}" ; raw="${raw#\"}"
    raw="${raw%\'}" ; raw="${raw#\'}"
    echo "$raw" | sed 's/[[:space:]]*$//'
}

S3_BUCKET="$(yaml_val bucket)"
S3_PREFIX="$(yaml_val prefix)"
S3_REGION="$(yaml_val region)"
S3_WORKERS="$(yaml_val workers)"
DEFAULT_DATASET="$(yaml_val dataset_dir)"

# Read pipeline behaviour defaults from config.yaml
CFG_SKIP_REENCODE="$(yaml_val skip_reencode)"
CFG_CLEANUP="$(yaml_val cleanup)"
CFG_FORCE="$(yaml_val force)"

# ── Parse CLI args (override config.yaml defaults) ────────────────────────
SKIP_REENCODE=$([[ "$CFG_SKIP_REENCODE" == "true" ]] && echo true || echo false)
DRY_RUN=false
FORCE=$([[ "$CFG_FORCE" == "true" ]] && echo true || echo false)
NO_CLEANUP=$([[ "$CFG_CLEANUP" == "false" ]] && echo true || echo false)
POSITIONAL_ARGS=()

for arg in "$@"; do
    case "$arg" in
        --skip-reencode) SKIP_REENCODE=true ;;
        --dry-run)       DRY_RUN=true ;;
        --force)         FORCE=true ;;
        --no-cleanup)    NO_CLEANUP=true ;;
        -h|--help)
            echo "Usage: $0 [DATASET_DIR] [--skip-reencode] [--dry-run] [--force] [--no-cleanup]"
            echo ""
            echo "  DATASET_DIR      Path to the dataset directory (default: ${DEFAULT_DATASET})"
            echo "  --skip-reencode  Skip H.264 re-encoding step (config: skip_reencode=${CFG_SKIP_REENCODE})"
            echo "  --dry-run        Show what would be uploaded without uploading"
            echo "  --force          Re-upload all files even if they already exist on S3 (config: force=${CFG_FORCE})"
            echo "  --no-cleanup     Skip orphan cleanup (config: cleanup=${CFG_CLEANUP})"
            exit 0 ;;
        *)  POSITIONAL_ARGS+=("$arg") ;;
    esac
done

DATASET_DIR="${POSITIONAL_ARGS[0]:-$DEFAULT_DATASET}"

# ── Colors ────────────────────────────────────────────────────────────────
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

# ============================================================================
#  STEP 1: Ensure uv is installed
# ============================================================================
step "Checking uv"

if ! command -v uv &>/dev/null; then
    info "uv not found — installing..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
    if ! command -v uv &>/dev/null; then
        err "Failed to install uv. Install it manually: https://docs.astral.sh/uv/"
        exit 1
    fi
fi
ok "$(uv --version 2>&1)"

# ============================================================================
#  STEP 2: Ensure Python 3.12
# ============================================================================
step "Checking Python 3.12"

REQUIRED_PY="3.12"
uv python install "$REQUIRED_PY" 2>/dev/null || true

ACTUAL_VER="$(uv run --python "$REQUIRED_PY" python --version 2>&1)" || {
    err "Python $REQUIRED_PY could not be installed. Install it manually."
    exit 1
}
ok "$ACTUAL_VER (managed by uv)"

# ============================================================================
#  STEP 3: Install dependencies
# ============================================================================
step "Installing dependencies (uv sync)"

uv sync --python "$REQUIRED_PY" --quiet
ok "All dependencies installed."

UVRUN="uv run"

# ============================================================================
#  STEP 4: Verify dataset directory
# ============================================================================
step "Verifying dataset"

if [[ ! -d "$DATASET_DIR" ]]; then
    err "Dataset directory not found: $DATASET_DIR"
    err "Pass the correct path:  $0 /path/to/dataset_multi"
    exit 1
fi

CLIP_COUNT=$(find "${DATASET_DIR}/clips" -name "*.mp4" 2>/dev/null | wc -l || echo "0")
VLM_COUNT=$(find "${DATASET_DIR}/vlm_crops" -name "*.mp4" 2>/dev/null | wc -l || echo "0")
ok "Dataset: ${DATASET_DIR}  (${CLIP_COUNT} clips, ${VLM_COUNT} VLM crops)"

# ============================================================================
#  STEP 5: Cleanup — sync vlm_crops ↔ clips, remove orphans
# ============================================================================
if [[ "$NO_CLEANUP" == "true" ]]; then
    info "Skipping orphan cleanup (--no-cleanup)."
else
    step "Synchronising dataset (cleanup)"

    info "Aligning vlm_crops/ and clips/ — cascade-deleting mismatches..."

    CLEANUP_OUTPUT=$($UVRUN python -m home_guard_project.labeling.utils.cleanup \
        "$DATASET_DIR" 2>&1 || echo "")

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
    else
        ok "Dataset is clean — vlm_crops/ and clips/ are in sync."
    fi

    CLIP_COUNT=$(find "${DATASET_DIR}/clips" -name "*.mp4" 2>/dev/null | wc -l || echo "0")
    VLM_COUNT=$(find "${DATASET_DIR}/vlm_crops" -name "*.mp4" 2>/dev/null | wc -l || echo "0")
    ok "Post-cleanup: ${CLIP_COUNT} clips, ${VLM_COUNT} VLM crops"
fi

# ============================================================================
#  STEP 6: AWS credentials (renumbered from 5)
# ============================================================================
step "Checking AWS credentials"

AWS_CREDS="$HOME/.aws/credentials"
AWS_CONFIG="$HOME/.aws/config"

if [[ ! -f "$AWS_CREDS" ]]; then
    err "AWS credentials not found at ${AWS_CREDS}"
    err "Create them first with:  aws configure"
    err "Or manually create ~/.aws/credentials with your access key and secret key."
    exit 1
fi
ok "AWS credentials found at ${AWS_CREDS}"

# Quick connectivity check
info "Verifying S3 access to s3://${S3_BUCKET}/ ..."
if $UVRUN python -c "
import boto3, sys
try:
    boto3.client('s3').head_bucket(Bucket='${S3_BUCKET}')
    print('OK')
except Exception as e:
    print(f'FAIL: {e}', file=sys.stderr)
    sys.exit(1)
" 2>&1; then
    ok "S3 bucket ${S3_BUCKET} is accessible."
else
    err "Cannot access S3 bucket ${S3_BUCKET}. Check your credentials and bucket name."
    exit 1
fi

# ============================================================================
#  STEP 7: Run the upload pipeline
# ============================================================================
step "Starting S3 upload"

UPLOAD_ARGS=("$DATASET_DIR" --bucket "$S3_BUCKET" --prefix "$S3_PREFIX" --workers "$S3_WORKERS" --no-cleanup)

if [[ "$SKIP_REENCODE" == "true" ]]; then
    UPLOAD_ARGS+=(--skip-reencode)
fi

if [[ "$DRY_RUN" == "true" ]]; then
    UPLOAD_ARGS+=(--dry-run)
fi

if [[ "$FORCE" == "true" ]]; then
    UPLOAD_ARGS+=(--force)
fi

info "Bucket:  s3://${S3_BUCKET}/${S3_PREFIX}/"
info "Dataset: ${DATASET_DIR}"
info "Workers: ${S3_WORKERS}"
[[ "$SKIP_REENCODE" == "true" ]] && info "Re-encode: SKIPPED" || info "Re-encode: enabled (skip already H.264)"
[[ "$FORCE" == "true" ]] && info "Mode: FORCE RE-UPLOAD" || true
[[ "$DRY_RUN" == "true" ]] && info "Mode: DRY RUN" || info "Mode: LIVE UPLOAD"
echo ""

$UVRUN python -m home_guard_project.s3_upload "${UPLOAD_ARGS[@]}"

echo ""
ok "S3 upload complete."
