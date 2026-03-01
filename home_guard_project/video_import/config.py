"""Load video_import/config.yaml into a frozen dataclass."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import yaml

_DIR = os.path.dirname(os.path.abspath(__file__))
_CONFIG_PATH = os.path.join(_DIR, "config.yaml")

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
    ids: List[int] = []
    for name in names:
        key = name.strip().lower().replace(" ", "_")
        if key in COCO_NAME_TO_ID:
            ids.append(COCO_NAME_TO_ID[key])
    return sorted(set(ids))


def _deep_get(d: Dict[str, Any], *keys: str, default: Any = None) -> Any:
    for k in keys:
        if not isinstance(d, dict):
            return default
        d = d.get(k, default)
    return d


@dataclass(frozen=True)
class ImportConfig:
    camera_name: str = "external"
    kind: str = "import"

    # Clip splitting
    clip_seconds: float = 10.0
    store_fps: float = 7.0

    # YOLO export
    yolo_enabled: bool = True
    yolo_fps: float = 3.0
    yolo_conf: float = 0.25
    yolo_jpeg_quality: int = 90
    yolo_model: str = "yolo11s.pt"
    yolo_imgsz: int = 640
    trigger_class_ids: List[int] = field(default_factory=lambda: [0])

    # VLM crop
    vlm_crop_enabled: bool = True
    vlm_crop_padding: float = 0.3
    vlm_crop_min_size: int = 384
    vlm_crop_ema_alpha: float = 0.3

    # Output
    output_dir: str = "./dataset_multi"


def load_config(path: Optional[str] = None) -> ImportConfig:
    cfg_path = path or _CONFIG_PATH
    with open(cfg_path, encoding="utf-8") as f:
        raw: Dict[str, Any] = yaml.safe_load(f) or {}

    project_root = os.path.abspath(os.path.join(_DIR, "..", ".."))
    ds_rel = raw.get("output_dir", "./dataset_multi")
    output_dir = os.path.normpath(os.path.join(project_root, ds_rel))

    trigger_names = _deep_get(raw, "yolo_export", "trigger_classes", default=["person"])
    trigger_ids = _resolve_class_names(trigger_names) or [0]

    return ImportConfig(
        camera_name=raw.get("camera_name", "external"),
        kind=raw.get("kind", "import"),

        clip_seconds=float(_deep_get(raw, "clip", "seconds", default=10.0)),
        store_fps=float(_deep_get(raw, "clip", "store_fps", default=7.0)),

        yolo_enabled=bool(_deep_get(raw, "yolo_export", "enabled", default=True)),
        yolo_fps=float(_deep_get(raw, "yolo_export", "fps", default=3.0)),
        yolo_conf=float(_deep_get(raw, "yolo_export", "confidence", default=0.25)),
        yolo_jpeg_quality=int(_deep_get(raw, "yolo_export", "jpeg_quality", default=90)),
        yolo_model=str(_deep_get(raw, "yolo_export", "model", default="yolo11s.pt")),
        yolo_imgsz=int(_deep_get(raw, "yolo_export", "imgsz", default=640)),
        trigger_class_ids=trigger_ids,

        vlm_crop_enabled=bool(_deep_get(raw, "vlm_crop", "enabled", default=True)),
        vlm_crop_padding=float(_deep_get(raw, "vlm_crop", "padding", default=0.3)),
        vlm_crop_min_size=int(_deep_get(raw, "vlm_crop", "min_size", default=384)),
        vlm_crop_ema_alpha=float(_deep_get(raw, "vlm_crop", "ema_alpha", default=0.3)),

        output_dir=output_dir,
    )
