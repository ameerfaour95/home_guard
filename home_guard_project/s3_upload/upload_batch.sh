#!/usr/bin/env bash
# ============================================================================
#  upload_batch.sh — Upload a tagged annotation batch to S3
#
#  Lives in:  home_guard_project/s3_upload/upload_batch.sh
#
#  Usage:
#      ./home_guard_project/s3_upload/upload_batch.sh \
#          --batch ameer_house_batch_2 \
#          --export "C:\Users\hp\Downloads\export.json" \
#          --analysis-dir ./ameer_house_batch_2_analysis_output \
#          [--source-prefix dataset_multi] \
#          [--dry-run]
#
#  What it does:
#    1. Verifies AWS credentials
#    2. S3-to-S3 copies the dataset from source prefix to the batch path
#    3. Uploads analysis output
#    4. Uploads the raw LS export JSON
#    5. Prints a summary
# ============================================================================
set -euo pipefail

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
S3_REGION="$(yaml_val region)"

# ── Parse CLI args ─────────────────────────────────────────────────────────
BATCH_NAME=""
EXPORT_PATH=""
ANALYSIS_DIR=""
SOURCE_PREFIX="dataset_multi"
DRY_RUN=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --batch)        BATCH_NAME="$2";     shift 2 ;;
        --export)       EXPORT_PATH="$2";    shift 2 ;;
        --analysis-dir) ANALYSIS_DIR="$2";   shift 2 ;;
        --source-prefix) SOURCE_PREFIX="$2"; shift 2 ;;
        --dry-run)      DRY_RUN=true;        shift ;;
        -h|--help)
            echo "Usage: $0 --batch NAME --export EXPORT.json --analysis-dir DIR [--source-prefix PREFIX] [--dry-run]"
            echo ""
            echo "  --batch NAME         Batch name (e.g. ameer_house_batch_2)"
            echo "  --export FILE        Path to the Label Studio JSON export"
            echo "  --analysis-dir DIR   Path to the analysis output directory"
            echo "  --source-prefix PRE  S3 prefix to copy dataset from (default: dataset_multi)"
            echo "  --dry-run            Show what would be done without uploading"
            exit 0 ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

if [[ -z "$BATCH_NAME" || -z "$EXPORT_PATH" || -z "$ANALYSIS_DIR" ]]; then
    echo "ERROR: --batch, --export, and --analysis-dir are required." >&2
    echo "Run with --help for usage." >&2
    exit 1
fi

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

# Ensure uv is available for running aws CLI from the project venv
if ! command -v uv &>/dev/null; then
    export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
fi
AWS="uv run aws"

DEST_PREFIX="tagging/${BATCH_NAME}"

info "Batch:           ${BATCH_NAME}"
info "S3 bucket:       s3://${S3_BUCKET}"
info "Source prefix:   ${SOURCE_PREFIX}"
info "Dest prefix:     ${DEST_PREFIX}"
info "Export file:     ${EXPORT_PATH}"
info "Analysis dir:    ${ANALYSIS_DIR}"
[[ "$DRY_RUN" == "true" ]] && info "Mode: DRY RUN"

# ============================================================================
#  STEP 1: Verify prerequisites
# ============================================================================
step "Checking prerequisites"

if [[ ! -f "$EXPORT_PATH" ]]; then
    err "Export file not found: $EXPORT_PATH"
    exit 1
fi
ok "Export file exists ($(wc -c < "$EXPORT_PATH" | tr -d ' ') bytes)"

if [[ ! -d "$ANALYSIS_DIR" ]]; then
    err "Analysis directory not found: $ANALYSIS_DIR"
    exit 1
fi
ANALYSIS_FILE_COUNT=$(find "$ANALYSIS_DIR" -type f 2>/dev/null | wc -l | tr -d ' ')
ok "Analysis directory exists (${ANALYSIS_FILE_COUNT} files)"

AWS_CREDS="$HOME/.aws/credentials"
if [[ ! -f "$AWS_CREDS" ]]; then
    err "AWS credentials not found at ${AWS_CREDS}. Run: aws configure"
    exit 1
fi
ok "AWS credentials found"

# ============================================================================
#  STEP 2: S3-to-S3 copy of dataset
# ============================================================================
step "Copying dataset: s3://${S3_BUCKET}/${SOURCE_PREFIX}/ -> s3://${S3_BUCKET}/${DEST_PREFIX}/${SOURCE_PREFIX}/"

SYNC_ARGS=(
    "s3://${S3_BUCKET}/${SOURCE_PREFIX}/"
    "s3://${S3_BUCKET}/${DEST_PREFIX}/${SOURCE_PREFIX}/"
    --region "$S3_REGION"
)

if [[ "$DRY_RUN" == "true" ]]; then
    SYNC_ARGS+=(--dryrun)
fi

$AWS s3 sync "${SYNC_ARGS[@]}"
ok "Dataset copy complete."

# ============================================================================
#  STEP 3: Upload analysis output
# ============================================================================
step "Uploading analysis output -> s3://${S3_BUCKET}/${DEST_PREFIX}/analysis_output/"

ANALYSIS_ARGS=(
    "$ANALYSIS_DIR/"
    "s3://${S3_BUCKET}/${DEST_PREFIX}/analysis_output/"
    --region "$S3_REGION"
)

if [[ "$DRY_RUN" == "true" ]]; then
    ANALYSIS_ARGS+=(--dryrun)
fi

$AWS s3 sync "${ANALYSIS_ARGS[@]}"
ok "Analysis output uploaded."

# ============================================================================
#  STEP 4: Upload raw export JSON
# ============================================================================
step "Uploading export JSON -> s3://${S3_BUCKET}/${DEST_PREFIX}/${BATCH_NAME}.json"

CP_ARGS=(
    "$EXPORT_PATH"
    "s3://${S3_BUCKET}/${DEST_PREFIX}/${BATCH_NAME}.json"
    --region "$S3_REGION"
)

if [[ "$DRY_RUN" == "true" ]]; then
    CP_ARGS+=(--dryrun)
fi

$AWS s3 cp "${CP_ARGS[@]}"
ok "Export JSON uploaded."

# ============================================================================
#  STEP 5: Summary
# ============================================================================
step "Upload Summary"

echo ""
info "Batch:  ${BATCH_NAME}"
info "S3 paths:"
echo "  s3://${S3_BUCKET}/${DEST_PREFIX}/${BATCH_NAME}.json"
echo "  s3://${S3_BUCKET}/${DEST_PREFIX}/analysis_output/"
echo "  s3://${S3_BUCKET}/${DEST_PREFIX}/${SOURCE_PREFIX}/"
echo ""
ok "Batch upload complete."
