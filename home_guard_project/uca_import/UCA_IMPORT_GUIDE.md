# UCA Dataset Import Guide

Step-by-step instructions for importing the UCA (UCF Crime Annotation) dataset,
uploading to S3, and launching Label Studio for annotation. Every step is
crash-resilient -- this guide explains how to resume if anything fails.

---

## Prerequisites

- Python 3.12+ installed
- `uv` package manager
- ffmpeg in PATH
- AWS credentials configured (`~/.aws/credentials`)
- The UCA Dataset zip file (default: `C:\Users\hp\Downloads\UCA Dataset.zip`)

---

## Pipeline Overview

```
Step 1: Import UCA dataset (windowed clips + YOLO + VLM text)
    |
Step 2: Upload to S3
    |
Step 3: Generate Label Studio tasks + launch
```

---

## Step 1: Import the UCA Dataset

This step extracts videos from the zip, splits each into 10-second windows,
runs YOLO detection on each clip, and writes metadata.

### Run

```bash
./home_guard_project/uca_import/start.sh --zip "C:\Users\hp\Downloads\UCA Dataset.zip"
```

Or with explicit options:

```bash
uv run python -m home_guard_project.uca_import \
    --zip "C:\Users\hp\Downloads\UCA Dataset.zip" \
    --window-size 10.0 \
    --splits UCFCrime_Train UCFCrime_Test UCFCrime_Val
```

### What it produces

```
dataset_uca/
  clips/<Category>/<VideoName>/<clip_id>.mp4        (~22,000 clips)
  meta/<Category>/<VideoName>/<clip_id>.meta.json
  responses/<Category>/<VideoName>/<clip_id>.model_raw.txt
  yolo/images/<Category>/<VideoName>/<clip_id>_f<NNNN>.jpg
  yolo/labels/<Category>/<VideoName>/<clip_id>_f<NNNN>.txt
  _processed_manifest.json                          (progress tracker)
```

### If it crashes

**Just re-run the same command.** The pipeline tracks progress in
`_processed_manifest.json`. On restart it logs:

```
Resuming: manifest has N clips already processed.
```

Already-processed clips are skipped automatically. No data is lost or duplicated.

### If you need a complete fresh start

```bash
# Delete local data + manifest
rm -rf dataset_uca/

# Or just reset the manifest (keeps existing clips but reprocesses all)
uv run python -m home_guard_project.uca_import --zip "..." --reset-manifest
```

### Useful options

| Flag | Effect |
|------|--------|
| `--no-yolo` | Skip YOLO detection (much faster, add later via labeling pipeline) |
| `--reencode` | Re-encode to H.264 CRF 18 (frame-accurate cuts, slower) |
| `--window-size 15` | Use 15-second windows instead of default 10 |
| `--splits UCFCrime_Test` | Import only the test split |
| `--reset-manifest` | Force full reprocessing from scratch |

### Runtime estimates

With YOLO enabled on GPU:

| Split | Est. videos | Est. windows | Approx. time |
|-------|-------------|-------------|---------------|
| UCFCrime_Test | 310 | ~3,700 | ~18 hours |
| UCFCrime_Train | ~1,165 | ~14,000 | ~3 days |
| UCFCrime_Val | ~379 | ~4,500 | ~20 hours |
| **Total** | **~1,854** | **~22,000** | **~5 days** |

Without YOLO (`--no-yolo`): roughly 10x faster.

---

## Step 2: Upload to S3

Uses `aws s3 sync` which only uploads new/changed files.

### Run

```bash
uv run python home_guard_project/s3_upload/s3_sync_progress.py \
    ./dataset_uca \
    s3://security-camera-project-v1/dataset_uca
```

### If it crashes

**Just re-run the same command.** `aws s3 sync` is idempotent -- it skips
files that already exist on S3 with matching size. Only the delta is uploaded.

### Verify upload

```bash
uv run python -c "
import boto3
s3 = boto3.client('s3', region_name='us-east-1')
resp = s3.list_objects_v2(
    Bucket='security-camera-project-v1',
    Prefix='dataset_uca/',
    Delimiter='/',
)
for p in resp.get('CommonPrefixes', []):
    print(p['Prefix'])
print(f'KeyCount: {resp[\"KeyCount\"]}')
"
```

Expected output: `clips/`, `meta/`, `responses/`, `yolo/` prefixes.

---

## Incremental Upload + Disk Cleanup (import in batches)

When disk space or memory is limited, you can import in batches: stop the
import partway, upload what you have to S3, delete the heavy local files
(clips + YOLO images) to reclaim disk, then continue importing.

### Workflow

```
1. Run import (Ctrl+C to stop when disk gets full or you want to pause)
2. Upload to S3 (syncs everything produced so far)
3. Delete local clips + YOLO images (keeps meta + manifest)
4. Re-run import (resumes from manifest, regenerates only new clips)
5. Repeat from step 2 until all videos are processed
```

### Step-by-step commands

```bash
# 1. Stop the import (Ctrl+C in the terminal running it, or just let it finish)

# 2. Check progress
uv run py -c "import json; d=json.load(open('dataset_uca/_processed_manifest.json')); print(f'{d[\"total_processed\"]} clips done')"

# 3. Upload everything to S3 (only uploads new/changed files)
uv run python home_guard_project/s3_upload/s3_sync_progress.py \
    ./dataset_uca \
    s3://security-camera-project-v1/dataset_uca

# 4. Free disk space — delete heavy files but KEEP meta + manifest
#    (meta = tiny JSON files, manifest = resume tracker)
rm -rf dataset_uca/clips/
rm -rf dataset_uca/yolo/
rm -rf dataset_uca/responses/
rm -rf dataset_uca/_uca_raw/

# 5. Resume the import (picks up where it left off via manifest)
uv run python -m home_guard_project.uca_import \
    --zip "C:\Users\hp\Downloads\UCA Dataset.zip" \
    --splits UCFCrime_Train UCFCrime_Test UCFCrime_Val

# 6. When done, do one final S3 upload
uv run python home_guard_project/s3_upload/s3_sync_progress.py \
    ./dataset_uca \
    s3://security-camera-project-v1/dataset_uca
```

### What is safe to delete locally

| Path | Size | Safe to delete? | Why |
|------|------|-----------------|-----|
| `dataset_uca/clips/` | Large (MP4s) | Yes, after S3 upload | Videos are on S3 |
| `dataset_uca/yolo/` | Large (JPGs + TXTs) | Yes, after S3 upload | YOLO data is on S3 |
| `dataset_uca/responses/` | Small (TXTs) | Yes, after S3 upload | VLM text is on S3 |
| `dataset_uca/_uca_raw/` | Temp extraction | Yes, always | Temp files, re-extracted on next run |
| `dataset_uca/meta/` | Small (JSONs) | **Keep locally** | Needed by labeling pipeline for task generation |
| `dataset_uca/_processed_manifest.json` | Tiny | **Never delete** | Resume tracker -- deleting it restarts from zero |

---

## Step 3: Generate Tasks and Launch Label Studio

### Run

```bash
./home_guard_project/labeling/start.sh ./dataset_uca
```

This will:
1. Sync metadata from S3 (downloads `.meta.json` files)
2. Generate `label_studio_tasks.json` with pre-signed S3 URLs
3. Start Cloudflare tunnel + Label Studio
4. Create/reuse a project and import tasks

### If it crashes during metadata sync

**Re-run the same command.** Already-downloaded metadata files are preserved.

### If it crashes during task import

**Re-run the same command.** The script detects the existing session file,
exports any previously imported tasks' annotations, merges them into the
regenerated tasks, clears old tasks, and reimports everything. No annotations
are lost.

### If Label Studio shows 0 tasks

This can happen if the task import failed silently (e.g., the JSON file was
too large). The pipeline now uses chunked streaming import via `ijson` to
handle arbitrarily large task files. Re-run `start.sh`.

### What annotators see

Two-panel layout per clip:

```
+-------------------------------+-------------------------------+
| Full Frame -- YOLO Re-tagging | VLM Text Panel                |
|                               |                               |
| [Video player with YOLO       | [Same video at original       |
|  bounding box pre-annotations]|  resolution]                  |
| Person  Car  Dog  ...         |                               |
| [Bounding box drawing tool]   | Scene Description:            |
|                               | [Pre-filled with aggregated   |
|                               |  UCA annotation texts]        |
+-------------------------------+-------------------------------+
```

The VLM text field is pre-filled with all UCA annotation descriptions that
overlap with the 10-second window, separated by `----` dividers.

---

## Deleting and Re-importing

If you need to wipe everything and start over:

```bash
# 1. Delete from S3
uv run python -c "
import boto3
s3 = boto3.resource('s3', region_name='us-east-1')
bucket = s3.Bucket('security-camera-project-v1')
bucket.objects.filter(Prefix='dataset_uca/').delete()
print('S3 data deleted.')
"

# 2. Delete local data
rm -rf dataset_uca/

# 3. Re-import
./home_guard_project/uca_import/start.sh --zip "C:\Users\hp\Downloads\UCA Dataset.zip"

# 4. Upload to S3
uv run python home_guard_project/s3_upload/s3_sync_progress.py \
    ./dataset_uca s3://security-camera-project-v1/dataset_uca

# 5. Launch Label Studio
./home_guard_project/labeling/start.sh ./dataset_uca
```

---

## Windowing Algorithm

For each source video:

- **Duration <= 10s**: the entire video becomes one clip
- **Duration > 10s**: non-overlapping 10-second windows; the last window
  slides back to ensure a full 10 seconds

Examples:

| Video length | Windows |
|-------------|---------|
| 5s | `[0-5]` (1 clip) |
| 10s | `[0-10]` (1 clip) |
| 11s | `[0-10, 1-11]` (2 clips) |
| 20s | `[0-10, 10-20]` (2 clips) |
| 25s | `[0-10, 10-20, 15-25]` (3 clips) |

### VLM text per window

For each window, all UCA annotation descriptions whose time range overlaps
with the window are concatenated with `\n----\n` separators. Windows with
no overlapping annotations have empty VLM text (annotators fill it in).
