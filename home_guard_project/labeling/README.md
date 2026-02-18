# Label Studio Annotation Pipeline

## Overview

Annotate security camera clips with:
- **Video playback** — watch the clip
- **YOLO bounding boxes** — draw/edit object boxes on video frames (VideoRectangle)
- **Free text** — write scene descriptions for VLM training

---

## Quick Start (one command)

```bash
./home_guard_project/labeling/start.sh ./dataset_multi
```

That's it. The script automatically:
1. Checks Python >= 3.12 and installs `uv` if needed
2. Installs all dependencies from `pyproject.toml`
3. Verifies the dataset directory
4. Checks for orphaned metadata (interactive prompt: delete / list / skip)
5. Re-encodes video clips to H.264 (skips already-encoded)
6. Generates Label Studio config and tasks JSON
7. Starts the file server and Label Studio (ports from `config.yaml`)
8. Authenticates via session cookies + CSRF (LS 1.22+)
9. Creates a project and imports tasks via the API
10. On re-run: exports annotations, merges into new tasks, then reimports
11. Opens your browser to the project

Press **Ctrl+C** to stop everything.

### Force rebuild

If tasks were previously generated with `--limit` or you want to regenerate from scratch:

```bash
./home_guard_project/labeling/start.sh ./dataset_multi --force
```

### Fresh clone on a new machine

```bash
git clone <repo-url>
cd security_camera_project
./home_guard_project/labeling/start.sh ./dataset_multi
```

**Prerequisites:** Python >= 3.12, ffmpeg, and Git Bash (Windows) or bash (Linux/macOS).

---

## Configuration

All settings live in `home_guard_project/labeling/config.yaml`:

```yaml
label_studio:
  port: 8080
  email: "ameerfaour95@gmail.com"
  password: "Amer1967"
  project_name: "Security Camera Annotations"

file_server:
  port: 8081

coco_labels:
  0: "person"
  1: "bicycle"
  ...

dataset_dir: "./dataset_multi"
```

Both the Python code and the bash script read from this file.

---

## Project Structure

```
home_guard_project/labeling/
  config.yaml        # All settings (ports, credentials, labels)
  config.py          # Python loader for config.yaml
  tasks.py           # Task scanning, building, export
  start.sh           # One-command launcher (bash)
  README.md          # This file
  __init__.py
  __main__.py        # CLI entry point (python -m home_guard_project.labeling)
  utils/
    __init__.py
    cleanup.py       # Orphan detection & removal (meta/yolo with no matching clip)
    ffmpeg.py        # FFmpeg detection, H.264 re-encoding
    file_server.py   # CORS-enabled HTTP server for video files
    merge.py         # Merge annotations from LS export into new tasks
    yolo.py          # YOLO <-> Label Studio coordinate conversion
```

---

## What the script does (step by step)

```
./home_guard_project/labeling/start.sh [DATASET_DIR] [--force]
        |
        v
  Read config.yaml (ports, credentials, project name)
        |
        v
  Check Python >= 3.12
        |
        v
  Check / install uv
        |
        v
  uv sync  (install all deps)
        |
        v
  Verify dataset directory (clips/, meta/)
        |
        v
  Check for orphaned metadata  ──┐
  (meta/yolo with no clip)       │
        |                        ├─ delete: remove orphans, continue
        v                        ├─ list:   print orphans, stop pipeline
  Interactive prompt ────────────├─ skip:   ignore, continue
        |                        └─ (none found): continue
        v
  Re-encode clips to H.264  (skips already done)
        |
        v
  Generate config XML + tasks JSON  (skipped if up-to-date, unless --force)
        |
        v
  Start file server (background)
        |
        v
  Start Label Studio (background)
        |
        v
  Wait for LS API ready
        |
        v
  Authenticate (form-based signup/login with CSRF)
        |
        v
  Validate saved project ID (if stale/wrong user, create new)
        |
        v
  Create project + import tasks (API)
  (re-run: export -> merge -> clear -> reimport)
        |
        v
  Open browser
        |
        v
  Running...  Ctrl+C to stop all services
```

---

## Orphan Cleanup

When you delete video clips, their metadata, VLM responses, and YOLO exports become
orphaned. The pipeline detects this automatically and prompts you:

```
[WARN] Found 42 orphaned files (metadata/responses/YOLO with no matching clip):
    Meta files   : 12
    Response files: 12
    YOLO images  : 9
    YOLO labels  : 9

  delete  — Remove all orphaned files and continue the pipeline
  list    — Print orphaned files, then stop (so you can review)
  skip    — Ignore orphans and continue the pipeline

  Your choice [delete/list/skip]:
```

You can also run cleanup standalone:

```bash
# Preview what would be deleted (safe)
uv run python -m home_guard_project.labeling --cleanup-only --dry-run

# Actually delete orphans
uv run python -m home_guard_project.labeling --cleanup-only
```

---

## Adding New Data

When `data_collection.py` captures new clips, just re-run the launcher:

```bash
./home_guard_project/labeling/start.sh ./dataset_multi
```

The script will:
- Skip re-encoding clips that are already H.264
- Skip task generation if no new clips were added (freshness check)
- Export existing annotations, merge them into the new task list, and reimport
- Use `--force` to bypass the freshness check and rebuild everything

---

## Manual Workflow (alternative)

If you prefer to run each step separately:

### Terminal 1: File server

```bash
uv run python -m home_guard_project.labeling --serve --dataset-dir ./dataset_multi
```

### Terminal 2: Label Studio

```bash
uv run label-studio start
```

### Browser

Open http://localhost:8080, create a project, paste the XML config, and import tasks.

---

## Exporting Annotations

In the Label Studio UI:
1. Open your project
2. Click **Export**
3. Choose format: **JSON** (for VLM text) or **YOLO** (for bounding boxes)

---

## CLI Reference

```
uv run python -m home_guard_project.labeling [OPTIONS]

Options:
  --dataset-dir DIR     Path to dataset_multi/ (default from config.yaml)
  --serve               Start the file server only
  --port PORT           File server port (default from config.yaml)
  --reencode            Re-encode clips from mp4v to H.264 (requires ffmpeg)
  --ffmpeg PATH         Explicit path to ffmpeg executable
  --force               Force rebuild even if tasks are up-to-date
  --merge EXPORTED.json Merge annotations from a LS export into new tasks
  --cleanup-only        Run orphan cleanup only, then exit
  --dry-run             With --cleanup-only: preview without deleting
  --no-cleanup          Skip automatic orphan cleanup
  --cameras CAM1,CAM2   Filter by camera names
  --kinds KIND1,KIND2   Filter by clip kind (trigger, random)
  --limit N             Max number of tasks to generate
  --no-predictions      Skip YOLO box and VLM text pre-annotations
  --workers N           Parallel workers for re-encoding/task generation
  -v, --verbose         Enable debug-level logging
```

---

## Credentials

Configured in `config.yaml` and used by the automated script.

Session data is saved to `dataset_multi/.ls_session.json` so re-runs
skip project creation. If the session becomes stale (e.g. different user,
deleted project), the script detects this and creates a new project.

---

## Authentication (LS 1.22+)

Label Studio 1.22 disabled legacy token-based API auth. The script uses:
- **Form-based signup/login** with CSRF tokens (`/user/signup`, `/user/login`)
- **Session cookies** for all subsequent API calls (project creation, import, export)
- `LABEL_STUDIO_USERNAME` / `LABEL_STUDIO_PASSWORD` env vars to auto-provision
  the admin account on first launch

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Videos don't play | Run with `--reencode` or let `start.sh` handle it |
| ffmpeg not found | Install via `winget install ffmpeg` or download from ffmpeg.org |
| Port already in use | Change ports in `config.yaml` |
| Slow task generation | Use `--no-predictions` or `--limit 100` for testing |
| Only 3 tasks imported | Previous run used `--limit`; re-run with `--force` |
| Auth fails | Delete LS database: `rm "$LOCALAPPDATA/label-studio/label-studio/label_studio.sqlite3"` |
| Project not found | Delete stale session: `rm dataset_multi/.ls_session.json` and re-run |
| Orphaned metadata | Re-run `start.sh` — the pipeline prompts you to clean up |
