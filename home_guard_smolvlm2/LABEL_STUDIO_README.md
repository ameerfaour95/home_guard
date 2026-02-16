# Label Studio Annotation Guide

## Overview

This setup lets you annotate security camera clips with:
- **Video playback** -- watch the clip
- **YOLO bounding boxes** -- draw/edit object boxes on video frames (VideoRectangle)
- **Free text** -- write scene descriptions for VLM training

---

## First-Time Setup (do once)

### 1. Install dependencies

```bash
pip install label-studio
```

### 2. Re-encode videos to H.264

Label Studio requires H.264 video. Your clips are recorded with mp4v codec
and must be converted. This only needs to be done once (or when new clips are added).

To re-encode **all** clips:

```bash
py -m home_guard_smolvlm2.label_studio_setup --dataset-dir ./dataset_multi --reencode
```

To re-encode only specific cameras or a subset:

```bash
py -m home_guard_smolvlm2.label_studio_setup --dataset-dir ./dataset_multi --reencode --cameras main_door --limit 50
```

### 3. Generate tasks (first time)

```bash
py -m home_guard_smolvlm2.label_studio_setup --dataset-dir ./dataset_multi
```

This creates:
- `dataset_multi/label_studio_config.xml` -- the labeling interface config
- `dataset_multi/label_studio_tasks.json` -- tasks with pre-annotations

### 4. Create the Label Studio project

1. Open **http://localhost:8080** (after starting Label Studio, see below)
2. Sign up / log in
3. Click **Create Project**, give it a name (e.g. "Security Camera Annotations")
4. Go to **Settings -> Labeling Interface -> Code**
5. Paste the contents of `dataset_multi/label_studio_config.xml`
6. Save

### 5. Import tasks

1. In the project, click **Import**
2. Upload `dataset_multi/label_studio_tasks.json`

---

## Every Time You Want to Annotate

You need **two terminals** running side by side.

### Terminal 1: Start the file server

```bash
cd C:\Users\hp\Desktop\ameer\data-science\security_camera_project
py -m home_guard_smolvlm2.label_studio_setup --serve --dataset-dir ./dataset_multi
```

Keep this running. It serves video files on **port 8081**.

### Terminal 2: Start Label Studio

```bash
cd C:\Users\hp\Desktop\ameer\data-science\security_camera_project
label-studio start
```

### Open the browser

Go to **http://localhost:8080**, open your project, and start annotating.

### When you're done

Press `Ctrl+C` in both terminals to stop.

---

## Adding New Data

When the data collection script (`data_collection.py`) captures new clips:

### 1. Re-encode the new clips

```bash
py -m home_guard_smolvlm2.label_studio_setup --dataset-dir ./dataset_multi --reencode
```

(Already-encoded clips are overwritten harmlessly -- but for large datasets
you may want to filter by camera/date to save time.)

### 2. Regenerate and re-import tasks

```bash
py -m home_guard_smolvlm2.label_studio_setup --dataset-dir ./dataset_multi
```

Then in Label Studio:
1. Go to your project
2. Click **Import**
3. Upload the new `dataset_multi/label_studio_tasks.json`

(Label Studio will add new tasks; duplicates are handled by matching the data fields.)

---

## CLI Reference

```
py -m home_guard_smolvlm2.label_studio_setup [OPTIONS]

Options:
  --dataset-dir DIR     Path to dataset_multi/ (default: ./dataset_multi)
  --serve               Start the file server only (port 8081)
  --port PORT           File server port (default: 8081)
  --reencode            Re-encode clips from mp4v to H.264 (requires ffmpeg)
  --ffmpeg PATH         Explicit path to ffmpeg.exe
  --cameras CAM1,CAM2   Filter by camera names
  --kinds KIND1,KIND2   Filter by clip kind (trigger, random)
  --limit N             Max number of tasks to generate
  --no-predictions      Skip YOLO box and VLM text pre-annotations
```

---

## Quick Reference (cheat sheet)

```
# === EVERY SESSION ===

# Terminal 1 - file server
py -m home_guard_smolvlm2.label_studio_setup --serve --dataset-dir ./dataset_multi

# Terminal 2 - label studio
label-studio start

# Browser: http://localhost:8080

# === AFTER NEW DATA ===

# Re-encode + regenerate
py -m home_guard_smolvlm2.label_studio_setup --dataset-dir ./dataset_multi --reencode
py -m home_guard_smolvlm2.label_studio_setup --dataset-dir ./dataset_multi

# Then import the new label_studio_tasks.json in the Label Studio UI
```
