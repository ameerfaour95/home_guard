# Label Studio Annotation Guide

## Overview

Annotate security camera clips with:
- **Video playback** -- watch the clip
- **YOLO bounding boxes** -- draw/edit object boxes on video frames (VideoRectangle)
- **Free text** -- write scene descriptions for VLM training

---

## Quick Start (one command)

```bash
./home_guard_smolvlm2/labeling/start.sh ./dataset_multi
```

That's it. The script automatically:
1. Checks Python >= 3.12 and installs `uv` if needed
2. Installs all dependencies from `pyproject.toml`
3. Re-encodes video clips to H.264 (skips already-encoded)
4. Generates Label Studio config and tasks JSON
5. Starts the file server and Label Studio (ports from `config.yaml`)
6. Creates a project and imports tasks via the API
7. On re-run: exports annotations, merges into new tasks, then reimports
8. Opens your browser to the project

Press **Ctrl+C** to stop everything.

### Fresh clone on a new machine

```bash
git clone <repo-url>
cd security_camera_project
./home_guard_smolvlm2/labeling/start.sh ./dataset_multi
```

**Prerequisites:** Python >= 3.12, ffmpeg, and Git Bash (Windows) or bash (Linux/macOS).

---

## Configuration

All settings live in `home_guard_smolvlm2/labeling/config.yaml`:

```yaml
label_studio:
  port: 8080
  email: "admin@localhost"
  password: "admin12345678"
  project_name: "Security Camera Annotations"

file_server:
  port: 8081

coco_labels:
  0: "person"
  1: "bicycle"
  ...
```

Both the Python code and the bash script read from this file.

---

## Project Structure

```
home_guard_smolvlm2/labeling/
  config.yaml        # All settings (ports, credentials, labels)
  config.py          # Python loader for config.yaml
  tasks.py           # Task scanning, building, export
  start.sh           # One-command launcher (bash)
  README.md          # This file
  __init__.py
  __main__.py        # CLI entry point (python -m home_guard_smolvlm2.labeling)
  utils/
    __init__.py
    ffmpeg.py        # FFmpeg detection, H.264 re-encoding
    file_server.py   # CORS-enabled HTTP server for video files
    merge.py         # Merge annotations from LS export into new tasks
    yolo.py          # YOLO <-> Label Studio coordinate conversion
```

---

## What the script does (step by step)

```
./home_guard_smolvlm2/labeling/start.sh [DATASET_DIR]
        |
        v
  Read config.yaml (ports, credentials)
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
  Re-encode clips to H.264  (skips already done)
        |
        v
  Generate config XML + tasks JSON
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
  Create project + import tasks (API)
  (re-run: export -> merge -> clear -> reimport)
        |
        v
  Open browser
        |
        v
  Running...  Ctrl+C to stop
```

---

## Manual Workflow (alternative)

If you prefer to run each step separately:

### Terminal 1: File server

```bash
uv run python -m home_guard_smolvlm2.labeling --serve --dataset-dir ./dataset_multi
```

### Terminal 2: Label Studio

```bash
uv run label-studio start
```

### Browser

Open http://localhost:8080, create a project, paste the XML config, and import tasks.

---

## Adding New Data

When `data_collection.py` captures new clips, just re-run the launcher:

```bash
./home_guard_smolvlm2/labeling/start.sh ./dataset_multi
```

The script will:
- Skip re-encoding clips that are already H.264
- Regenerate all tasks
- Export existing annotations, merge them into the new task list, and reimport

---

## Exporting Annotations

In the Label Studio UI:
1. Open your project
2. Click **Export**
3. Choose format: **JSON** (for VLM text) or **YOLO** (for bounding boxes)

Or via the API:

```bash
curl -X GET "http://localhost:8080/api/projects/<ID>/export?exportType=JSON" \
  -H "Authorization: Token <YOUR_TOKEN>" \
  -o annotations.json
```

---

## CLI Reference

```
uv run python -m home_guard_smolvlm2.labeling [OPTIONS]

Options:
  --dataset-dir DIR     Path to dataset_multi/ (default from config.yaml)
  --serve               Start the file server only
  --port PORT           File server port (default from config.yaml)
  --reencode            Re-encode clips from mp4v to H.264 (requires ffmpeg)
  --ffmpeg PATH         Explicit path to ffmpeg executable
  --merge EXPORTED.json Merge annotations from a LS export into new tasks
  --cameras CAM1,CAM2   Filter by camera names
  --kinds KIND1,KIND2   Filter by clip kind (trigger, random)
  --limit N             Max number of tasks to generate
  --no-predictions      Skip YOLO box and VLM text pre-annotations
  --workers N           Parallel workers for re-encoding/task generation
  -v, --verbose         Enable debug-level logging
```

---

## Default Credentials

Configured in `config.yaml` and used by the automated script:
- **Email:** admin@localhost
- **Password:** admin12345678

Session data is saved to `dataset_multi/.ls_session.json` so re-runs
skip account/project creation.

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Videos don't play | Run `--reencode` to convert to H.264 |
| ffmpeg not found | Install via `winget install ffmpeg` or download from ffmpeg.org |
| Port already in use | Change ports in `config.yaml` |
| Slow task generation | Use `--no-predictions` or `--limit 100` for testing |
