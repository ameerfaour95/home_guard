"""Load pipeline configuration from config.yaml + cameras.yaml."""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
import yaml

log = logging.getLogger(__name__)

_DIR = os.path.dirname(os.path.abspath(__file__))
_CONFIG_PATH = os.path.join(_DIR, "config.yaml")
_CAMERAS_PATH = os.path.join(_DIR, "cameras.yaml")
_ZONES_PATH = os.path.join(_DIR, "zones.yaml")


COCO_NAMES: List[str] = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic_light", "fire_hydrant", "stop_sign",
    "parking_meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag",
    "tie", "suitcase", "frisbee", "skis", "snowboard", "sports_ball", "kite",
    "baseball_bat", "baseball_glove", "skateboard", "surfboard",
    "tennis_racket", "bottle", "wine_glass", "cup", "fork", "knife", "spoon",
    "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
    "hot_dog", "pizza", "donut", "cake", "chair", "couch", "potted_plant",
    "bed", "dining_table", "toilet", "tv", "laptop", "mouse", "remote",
    "keyboard", "cell_phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy_bear",
    "hair_drier", "toothbrush",
]
COCO_NAME_TO_ID: Dict[str, int] = {n: i for i, n in enumerate(COCO_NAMES)}


def _resolve_class_names(names: List[str]) -> List[int]:
    """Convert a list of COCO class names to integer IDs."""
    ids: List[int] = []
    for name in names:
        key = name.strip().lower().replace(" ", "_")
        if key in COCO_NAME_TO_ID:
            ids.append(COCO_NAME_TO_ID[key])
        else:
            log.warning("Unknown YOLO class name '%s' — skipping", name)
    return sorted(set(ids))


def _deep_get(d: Dict[str, Any], *keys: str, default: Any = None) -> Any:
    for k in keys:
        if not isinstance(d, dict):
            return default
        d = d.get(k, default)
    return d


@dataclass
class Config:
    # ── Paths ─────────────────────────────────────────────────────────────
    OUT_DIR: str = "./dataset_multi"

    # ── Cameras ───────────────────────────────────────────────────────────
    CAMERAS: Dict[str, str] = field(default_factory=dict)       # sub-stream URLs
    CAMERAS_MAIN: Dict[str, str] = field(default_factory=dict)  # main-stream URLs

    # ── Models ────────────────────────────────────────────────────────────
    YOLO_MODEL: str = "yolov8n.pt"
    VLM_MODEL_ID: str = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"

    # ── Hardware (auto-detected, not in YAML) ─────────────────────────────
    DEVICE: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")
    DTYPE: torch.dtype = field(default_factory=lambda: torch.bfloat16 if torch.cuda.is_available() else torch.float32)

    # ── Clip & buffer (sub-stream) ────────────────────────────────────────
    CLIP_SECONDS: float = 10.0
    STORE_FPS: float = 10.0
    STORE_SIZE: Optional[Tuple[int, int]] = None
    YOLO_IMGSZ: int = 640

    # ── Main stream (high-res VLM crops) ───────────────────────────────
    MAIN_STREAM_ENABLED: bool = True
    MAIN_STORE_FPS: float = 5.0
    MAIN_JPEG_QUALITY: int = 85
    CROP_PADDING: float = 0.3
    CROP_MIN_SIZE: int = 384
    CROP_EMA_ALPHA: float = 0.3

    # ── Trigger hysteresis ────────────────────────────────────────────────
    SCORE_MAX: float = 10.0
    SCORE_REWARD: float = 3.0
    SCORE_PENALTY: float = 2.0
    COOLDOWN_TRIGGER_SEC: float = 5.0
    POST_ROLL_SEC: float = 5.0

    # ── Random sampling ───────────────────────────────────────────────────
    RANDOM_CLIP_ENABLED: bool = True
    RANDOM_CLIP_INTERVAL_SEC: float = 3600.0
    RANDOM_JITTER_FRAC: float = 0.25
    RANDOM_ALLOW_PERSON: bool = True

    # ── VLM ───────────────────────────────────────────────────────────────
    RUN_VLM_ON_SAVED_CLIPS: bool = False
    VLM_SAMPLE_FPS: int = 1
    VLM_PROMPT: str = "Describe what happens in the video"

    # ── YOLO export ───────────────────────────────────────────────────────
    EXPORT_YOLO_TRAINING_DATA: bool = True
    YOLO_EXPORT_FPS: float = 2.0
    YOLO_EXPORT_CONF: float = 0.25
    YOLO_EXPORT_JPEG_QUALITY: int = 90
    YOLO_EXPORT_SUBDIR: str = "yolo"

    # ── RTSP stability ────────────────────────────────────────────────────
    OPENCV_FFMPEG_CAPTURE_OPTIONS: str = (
        "rtsp_transport;tcp|stimeout;5000000|max_delay;500000|fflags;nobuffer"
    )
    OPEN_TIMEOUT_MSEC: int = 5000
    READ_TIMEOUT_MSEC: int = 5000
    FREEZE_RECONNECT_AFTER_SEC: float = 3.0
    RECONNECT_BACKOFF_START: float = 1.0
    RECONNECT_BACKOFF_MAX: float = 10.0

    # ── Camera discovery ──────────────────────────────────────────────────
    DISCOVERY_MAX_CHANNELS: int = 16
    DISCOVERY_CONSECUTIVE_FAIL_STOP: int = 2
    DISCOVERY_PROBE_TIMEOUT: float = 5.0

    # ── ROI zones (normalised polygons per camera) ─────────────────────────
    ROI_ZONES: Dict[str, List[Tuple[float, float]]] = field(default_factory=dict)

    # ── Detection & display ───────────────────────────────────────────────
    TRIGGER_CLASS_IDS: List[int] = field(default_factory=lambda: [0])
    YOLO_TRIGGER_CONF: float = 0.5
    YOLO_EVERY_N_FRAMES_CPU: int = 6
    SHOW_WINDOWS: bool = True
    SHOW_PLOTTED_BOXES: bool = False
    SAVE_WITH_PLOTTED_BOXES: bool = False


def _derive_sub_url(main_url: str) -> str:
    """Derive sub-stream URL from main-stream URL (/s0/ -> /s1/)."""
    sub = re.sub(r"/s0/", "/s1/", main_url, count=1)
    if sub == main_url:
        log.warning("Could not derive sub-stream URL from %s — using as-is", main_url)
    return sub


def _parse_size(raw: Any) -> Optional[Tuple[int, int]]:
    if raw is None:
        return None
    if isinstance(raw, (list, tuple)) and len(raw) == 2:
        return (int(raw[0]), int(raw[1]))
    return None


def _load_zones(path: str) -> Dict[str, List[Tuple[float, float]]]:
    """Load normalised polygon zones from zones.yaml."""
    if not os.path.isfile(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    raw = data.get("zones", {})
    if not isinstance(raw, dict):
        return {}
    zones: Dict[str, List[Tuple[float, float]]] = {}
    for cam, pts in raw.items():
        if isinstance(pts, list) and len(pts) >= 3:
            zones[str(cam)] = [(float(p[0]), float(p[1])) for p in pts]
    return zones


def load_config(
    config_path: str = _CONFIG_PATH,
    cameras_path: str = _CAMERAS_PATH,
    zones_path: str = _ZONES_PATH,
) -> Config:
    """Build a Config from the YAML files, falling back to defaults."""
    cfg_data: Dict[str, Any] = {}
    cam_data: Dict[str, Any] = {}

    if os.path.isfile(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            cfg_data = yaml.safe_load(f) or {}
        log.info("Loaded config from %s", config_path)
    else:
        log.warning("Config file not found: %s — using defaults", config_path)

    if os.path.isfile(cameras_path):
        with open(cameras_path, "r", encoding="utf-8") as f:
            cam_data = yaml.safe_load(f) or {}
        log.info("Loaded cameras from %s", cameras_path)
    else:
        log.warning("Cameras file not found: %s — no cameras configured", cameras_path)

    roi_zones = _load_zones(zones_path)
    if roi_zones:
        log.info("Loaded ROI zones for %d camera(s)", len(roi_zones))

    cameras_raw = cam_data.get("cameras", {})
    cameras_main = {str(k): str(v) for k, v in cameras_raw.items()} if cameras_raw else {}
    cameras_sub = {name: _derive_sub_url(url) for name, url in cameras_main.items()}

    store_size = _parse_size(_deep_get(cfg_data, "clip", "store_size"))

    return Config(
        OUT_DIR=cfg_data.get("output_dir", "./dataset_multi"),
        CAMERAS=cameras_sub,
        CAMERAS_MAIN=cameras_main,

        YOLO_MODEL=_deep_get(cfg_data, "models", "yolo", default="yolov8n.pt"),
        VLM_MODEL_ID=_deep_get(cfg_data, "models", "vlm", default="HuggingFaceTB/SmolVLM2-500M-Video-Instruct"),

        CLIP_SECONDS=float(_deep_get(cfg_data, "clip", "seconds", default=10.0)),
        STORE_FPS=float(_deep_get(cfg_data, "clip", "store_fps", default=10.0)),
        STORE_SIZE=store_size,
        YOLO_IMGSZ=int(_deep_get(cfg_data, "clip", "yolo_imgsz", default=640)),

        MAIN_STREAM_ENABLED=bool(_deep_get(cfg_data, "main_stream", "enabled", default=True)),
        MAIN_STORE_FPS=float(_deep_get(cfg_data, "main_stream", "store_fps", default=5.0)),
        MAIN_JPEG_QUALITY=int(_deep_get(cfg_data, "main_stream", "jpeg_quality", default=85)),
        CROP_PADDING=float(_deep_get(cfg_data, "main_stream", "crop_padding", default=0.3)),
        CROP_MIN_SIZE=int(_deep_get(cfg_data, "main_stream", "crop_min_size", default=384)),
        CROP_EMA_ALPHA=float(_deep_get(cfg_data, "main_stream", "crop_ema_alpha", default=0.3)),

        SCORE_MAX=float(_deep_get(cfg_data, "trigger", "score_max", default=10.0)),
        SCORE_REWARD=float(_deep_get(cfg_data, "trigger", "score_reward", default=3.0)),
        SCORE_PENALTY=float(_deep_get(cfg_data, "trigger", "score_penalty", default=2.0)),
        COOLDOWN_TRIGGER_SEC=float(_deep_get(cfg_data, "trigger", "cooldown_sec", default=5.0)),
        POST_ROLL_SEC=float(_deep_get(cfg_data, "trigger", "post_roll_sec", default=5.0)),

        RANDOM_CLIP_ENABLED=bool(_deep_get(cfg_data, "random_clip", "enabled", default=True)),
        RANDOM_CLIP_INTERVAL_SEC=float(_deep_get(cfg_data, "random_clip", "interval_sec", default=3600.0)),
        RANDOM_JITTER_FRAC=float(_deep_get(cfg_data, "random_clip", "jitter_frac", default=0.25)),
        RANDOM_ALLOW_PERSON=bool(_deep_get(cfg_data, "random_clip", "allow_person", default=True)),

        RUN_VLM_ON_SAVED_CLIPS=bool(_deep_get(cfg_data, "vlm", "enabled", default=False)),
        VLM_SAMPLE_FPS=int(_deep_get(cfg_data, "vlm", "sample_fps", default=1)),
        VLM_PROMPT=str(_deep_get(cfg_data, "vlm", "prompt", default="Describe what happens in the video")),

        EXPORT_YOLO_TRAINING_DATA=bool(_deep_get(cfg_data, "yolo_export", "enabled", default=True)),
        YOLO_EXPORT_FPS=float(_deep_get(cfg_data, "yolo_export", "fps", default=2.0)),
        YOLO_EXPORT_CONF=float(_deep_get(cfg_data, "yolo_export", "confidence", default=0.25)),
        YOLO_EXPORT_JPEG_QUALITY=int(_deep_get(cfg_data, "yolo_export", "jpeg_quality", default=90)),
        YOLO_EXPORT_SUBDIR=str(_deep_get(cfg_data, "yolo_export", "subdir", default="yolo")),

        OPENCV_FFMPEG_CAPTURE_OPTIONS=str(_deep_get(cfg_data, "rtsp", "ffmpeg_options",
            default="rtsp_transport;tcp|stimeout;5000000|max_delay;500000|fflags;nobuffer")),
        OPEN_TIMEOUT_MSEC=int(_deep_get(cfg_data, "rtsp", "open_timeout_ms", default=5000)),
        READ_TIMEOUT_MSEC=int(_deep_get(cfg_data, "rtsp", "read_timeout_ms", default=5000)),
        FREEZE_RECONNECT_AFTER_SEC=float(_deep_get(cfg_data, "rtsp", "freeze_reconnect_sec", default=3.0)),
        RECONNECT_BACKOFF_START=float(_deep_get(cfg_data, "rtsp", "reconnect_backoff_start", default=1.0)),
        RECONNECT_BACKOFF_MAX=float(_deep_get(cfg_data, "rtsp", "reconnect_backoff_max", default=10.0)),

        DISCOVERY_MAX_CHANNELS=int(_deep_get(cfg_data, "discovery", "max_channels", default=16)),
        DISCOVERY_CONSECUTIVE_FAIL_STOP=int(_deep_get(cfg_data, "discovery", "consecutive_fail_stop", default=2)),
        DISCOVERY_PROBE_TIMEOUT=float(_deep_get(cfg_data, "discovery", "probe_timeout_sec", default=5.0)),

        ROI_ZONES=roi_zones,

        TRIGGER_CLASS_IDS=_resolve_class_names(
            _deep_get(cfg_data, "detection", "trigger_classes", default=["person"]) or ["person"]
        ),
        YOLO_TRIGGER_CONF=float(_deep_get(cfg_data, "detection", "yolo_trigger_confidence", default=0.5)),
        YOLO_EVERY_N_FRAMES_CPU=int(_deep_get(cfg_data, "detection", "yolo_every_n_frames_cpu", default=6)),
        SHOW_WINDOWS=bool(_deep_get(cfg_data, "display", "show_windows", default=True)),
        SHOW_PLOTTED_BOXES=bool(_deep_get(cfg_data, "display", "show_plotted_boxes", default=False)),
        SAVE_WITH_PLOTTED_BOXES=bool(_deep_get(cfg_data, "display", "save_with_plotted_boxes", default=False)),
    )
