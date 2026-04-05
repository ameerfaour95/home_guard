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
#          [--selective] \
#          [--dry-run]
#
#  What it does:
#    1. Verifies AWS credentials
#    2. S3-to-S3 copies the dataset from source prefix to the batch path
#       (full sync or --selective: only clips listed in vlm_training.jsonl)
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
SELECTIVE=false
DRY_RUN=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --batch)        BATCH_NAME="$2";     shift 2 ;;
        --export)       EXPORT_PATH="$2";    shift 2 ;;
        --analysis-dir) ANALYSIS_DIR="$2";   shift 2 ;;
        --source-prefix) SOURCE_PREFIX="$2"; shift 2 ;;
        --selective)    SELECTIVE=true;       shift ;;
        --dry-run)      DRY_RUN=true;        shift ;;
        -h|--help)
            echo "Usage: $0 --batch NAME --export EXPORT.json --analysis-dir DIR [--source-prefix PREFIX] [--selective] [--dry-run]"
            echo ""
            echo "  --batch NAME         Batch name (e.g. ameer_house_batch_2)"
            echo "  --export FILE        Path to the Label Studio JSON export"
            echo "  --analysis-dir DIR   Path to the analysis output directory"
            echo "  --source-prefix PRE  S3 prefix to copy dataset from (default: dataset_multi)"
            echo "  --selective          Only copy clips listed in analysis vlm_training.jsonl (not full prefix)"
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
[[ "$SELECTIVE" == "true" ]] && info "Mode: SELECTIVE (only kept clips)"
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

if [[ "$SELECTIVE" == "true" ]]; then
    step "Selective copy: only kept clips -> s3://${S3_BUCKET}/${DEST_PREFIX}/${SOURCE_PREFIX}/"

    VLM_JSONL="${ANALYSIS_DIR}/vlm_training.jsonl"
    if [[ ! -f "$VLM_JSONL" ]]; then
        err "vlm_training.jsonl not found in ${ANALYSIS_DIR}. Run analysis first."
        exit 1
    fi

    CLIP_COUNT=$(wc -l < "$VLM_JSONL" | tr -d ' ')
    info "Found ${CLIP_COUNT} kept clips in vlm_training.jsonl"

    UVRUN="uv run py"
    $UVRUN -c "
import json, sys, re, os
import boto3

jsonl_path = sys.argv[1]
bucket = sys.argv[2]
source_prefix = sys.argv[3]
dest_prefix = sys.argv[4]
dry_run = sys.argv[5] == 'true'

s3 = boto3.client('s3')
paginator = s3.get_paginator('list_objects_v2')

clips = []
with open(jsonl_path, 'r', encoding='utf-8') as f:
    for line in f:
        rec = json.loads(line)
        s3_path = rec['video_s3_path']
        rel = s3_path.split(source_prefix + '/')[-1]
        meta_rel = re.sub(r'^clips/', 'meta/', rel)
        meta_rel = re.sub(r'\.mp4\$', '.meta.json', meta_rel)
        parts = meta_rel.replace('\\\\', '/').split('/')
        cam_date = '/'.join(parts[1:3])
        clips.append((rec['clip_id'], cam_date))

print(f'Processing {len(clips)} clips...')

copied = 0
skipped = 0
errors = 0

for i, (clip_id, cam_date) in enumerate(clips, 1):
    exact_keys = [
        f'{source_prefix}/clips/{cam_date}/{clip_id}.mp4',
        f'{source_prefix}/meta/{cam_date}/{clip_id}.meta.json',
        f'{source_prefix}/responses/{cam_date}/{clip_id}.model_raw.txt',
        f'{source_prefix}/vlm_crops/{cam_date}/{clip_id}.mp4',
    ]
    for src_key in exact_keys:
        dst_key = f'{dest_prefix}/{src_key}'
        try:
            if dry_run:
                print(f'  [DRY] {src_key}')
            else:
                s3.copy_object(Bucket=bucket, CopySource={'Bucket': bucket, 'Key': src_key}, Key=dst_key)
            copied += 1
        except Exception as e:
            if 'NoSuchKey' in str(e) or '404' in str(e):
                skipped += 1
            else:
                print(f'  WARN: {src_key}: {e}', file=sys.stderr)
                errors += 1

    for subdir in ['yolo/images', 'yolo/labels']:
        yolo_prefix = f'{source_prefix}/{subdir}/{cam_date}/{clip_id}_f'
        for page in paginator.paginate(Bucket=bucket, Prefix=yolo_prefix):
            for obj in page.get('Contents', []):
                src_key = obj['Key']
                dst_key = f'{dest_prefix}/{src_key}'
                try:
                    if dry_run:
                        print(f'  [DRY] {src_key}')
                    else:
                        s3.copy_object(Bucket=bucket, CopySource={'Bucket': bucket, 'Key': src_key}, Key=dst_key)
                    copied += 1
                except Exception as e:
                    print(f'  WARN: {src_key}: {e}', file=sys.stderr)
                    errors += 1

    if i % 10 == 0 or i == len(clips):
        print(f'  Progress: {i}/{len(clips)} clips ({copied} copied, {skipped} skipped, {errors} errors)')

print(f'Selective copy complete: {copied} objects copied, {skipped} skipped (not found), {errors} errors')
" "$VLM_JSONL" "$S3_BUCKET" "$SOURCE_PREFIX" "$DEST_PREFIX" "$DRY_RUN"

    ok "Selective dataset copy complete."
else
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
fi

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
