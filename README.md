# Home Guard — Smart Security Camera System

Real-time home security monitoring using RTSP IP cameras, YOLO object detection, and a Vision-Language Model (VLM). The system watches your cameras, detects people and vehicles, saves high-quality evidence clips, and (in the inference pipeline) describes scenes in natural language.

---

## Architecture

```
RTSP Cameras ──► Data Collection ──► Dataset ──► Labeling ──► Training
                  (YOLO detection)     (clips +    (Label      (fine-tune
                                        meta +      Studio)     YOLO + VLM)
                                        YOLO labels)
```

The project has two main pipelines, each self-contained with its own config, launcher, and README:

| Pipeline | Purpose | Launcher |
|----------|---------|----------|
| **Data Collection** | Capture video from RTSP cameras, run YOLO, save high-quality clips + metadata | `./home_guard_project/data_collection/start.sh` |
| **Labeling** | Re-encode clips, launch Label Studio for human annotation with YOLO pre-annotations | `./home_guard_project/labeling/start.sh` |

---

## Quick Start

### Prerequisites

- **Python >= 3.12**
- **Git Bash** (Windows) or **bash** (Linux/macOS)
- **ffmpeg** (for the labeling pipeline — `winget install ffmpeg` on Windows)

### 1. Collect Data

```bash
./home_guard_project/data_collection/start.sh
```

On first run, the script will:
- Install all dependencies via `uv`
- Guide you through camera discovery (auto-detect or manual RTSP entry)
- Optionally set up ROI zones (polygon regions to focus detection on your property)
- Start recording clips when people are detected

Press **Ctrl+C** to stop.

### 2. Label Data

```bash
./home_guard_project/labeling/start.sh ./dataset_multi
```

This will:
- Re-encode clips to H.264 (browser-playable)
- Generate Label Studio tasks with YOLO pre-annotations
- Launch Label Studio in your browser for human annotation
- On re-runs, preserve all existing annotations

Press **Ctrl+C** to stop.

---

## Repository Structure

```
security_camera_project/
│
├── home_guard_project/
│   ├── data_collection/                # Data collection pipeline
│   │   ├── start.sh                    #   One-command launcher
│   │   ├── data_collection.py          #   Multi-camera RTSP capture + YOLO
│   │   ├── discover.py                 #   Camera auto-discovery (ONVIF / scan / manual)
│   │   ├── roi_editor.py              #   OpenCV GUI for ROI polygon drawing
│   │   ├── config.yaml                #   Pipeline settings
│   │   ├── config.py                  #   YAML → Config dataclass
│   │   └── README.md                  #   Detailed documentation
│   │
│   ├── labeling/                      # Labeling pipeline
│   │   ├── start.sh                   #   One-command launcher
│   │   ├── __main__.py                #   CLI entry point
│   │   ├── tasks.py                   #   Task generation for Label Studio
│   │   ├── config.yaml                #   Labeling settings (ports, creds)
│   │   ├── config.py                  #   YAML → config loader
│   │   ├── utils/                     #   ffmpeg, file server, merge, YOLO conversion
│   │   └── README.md                  #   Detailed documentation
│   │
│   └── fortified_security_smolvlm.py  # Inference script (YOLO + VLM)
│
├── dataset_multi/                     # Output dataset (gitignored contents)
│   ├── clips/                         #   Saved video clips
│   ├── meta/                          #   Metadata JSON per clip
│   ├── yolo/                          #   Weak-label images + YOLO .txt files
│   └── responses/                     #   VLM responses (future)
│
├── pyproject.toml                     # Dependencies (all versions pinned)
└── README.md                          # This file
```

---

## How Data Collection Works

```
RTSP Camera Stream
       │
       ▼
  Buffer frames at native resolution (e.g. 1920×1080)
       │
       ▼
  YOLO detection on downscaled frame (960×544)
       │
       ├── ROI filter: only count detections inside your property polygon
       │
       ▼
  Hysteresis scoring (prevents spurious triggers)
       │
       ▼
  Trigger fires → wait 5s post-roll → save 10s clip
       │
       ├── clips/<camera>/<date>/<clip>.mp4        (high-res video)
       ├── meta/<camera>/<date>/<clip>.meta.json   (detection summary)
       └── yolo/images/ + labels/                   (weak training labels)
```

Key features:
- **High-quality saving**: YOLO runs on downscaled frames for speed, but clips are saved at full camera resolution
- **ROI zones**: Draw polygons to ignore public areas (sidewalks, roads, neighbors)
- **Post-roll recording**: The trigger event appears in the middle of each clip, not at the end
- **Camera auto-discovery**: ONVIF, subnet scanning, or manual entry
- **YOLO weak labels**: Auto-generated training data from every saved clip

---

## How Labeling Works

```
dataset_multi/
       │
       ▼
  Re-encode mp4v → H.264 (browser-playable)
       │
       ▼
  Scan .meta.json files → build Label Studio tasks
       │
       ├── YOLO bounding box pre-annotations
       └── Metadata display (timestamps, camera name, detection info)
       │
       ▼
  Label Studio (browser UI)
       │
       ├── Watch video clips
       ├── Edit/verify YOLO boxes
       └── Write scene descriptions (for VLM training)
       │
       ▼
  Export annotations (JSON / YOLO format)
```

Key features:
- **One-command launch**: handles deps, re-encoding, LS setup, project creation, task import
- **Incremental sync**: re-runs preserve all existing human annotations
- **Orphan cleanup**: detects metadata without matching clips
- **Pre-annotations**: YOLO boxes appear automatically for faster labeling

---

## Inference Architecture (future)

```
RTSP stream
  │
  ▼
YOLO (full frame) ─── person detected? ─── NO → skip
  │                                         
  YES
  │
  ▼
Adaptive crop-and-pad (high-res square around all persons)
  │
  ▼
VLM (SmolVLM2) — receives cropped, high-quality video
  │
  ▼
{ summary, alert_reason, alert_command }
```

The VLM never sees the full frame. It receives a tightly cropped square patch around detected persons, with adaptive padding:
- **Far subjects** (small in frame): 25% padding — maximize pixel density
- **Close subjects** (large in frame): 50% padding — maximize context

---

## Configuration

Each pipeline has its own `config.yaml`:

| File | Purpose |
|------|---------|
| `data_collection/config.yaml` | Clip length, FPS, YOLO thresholds, trigger tuning, RTSP settings, display |
| `data_collection/cameras.yaml` | RTSP URLs with credentials **(gitignored)** |
| `data_collection/zones.yaml` | ROI polygon coordinates **(gitignored)** |
| `labeling/config.yaml` | Label Studio port, credentials, COCO labels, file server port |

---

## Dependencies

All pinned in `pyproject.toml`. Managed via [`uv`](https://github.com/astral-sh/uv) (auto-installed by the launcher scripts).

Key packages: `ultralytics` (YOLO), `opencv-python`, `torch`, `transformers` (VLM), `label-studio`, `pyyaml`, `onvif-zeep` (camera discovery).

```bash
# Manual install (not needed if using start.sh)
pip install uv
uv sync
```

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `python` opens Windows Store | Use `py` or `py -m` instead |
| No cameras found | Run `start.sh --discover` or edit `cameras.yaml` manually |
| Too many false triggers | Define ROI zones to ignore public areas |
| Videos won't play in Label Studio | The labeling pipeline re-encodes to H.264 automatically |
| Port already in use | Change ports in the relevant `config.yaml` |
| Label Studio auth fails | Delete `$LOCALAPPDATA/label-studio/label-studio/label_studio.sqlite3` |

See the per-pipeline READMEs for detailed troubleshooting:
- [Data Collection README](home_guard_project/data_collection/README.md)
- [Labeling README](home_guard_project/labeling/README.md)
