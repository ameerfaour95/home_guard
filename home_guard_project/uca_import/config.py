"""Load uca_import/config.yaml into a frozen dataclass."""

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
class UCAConfig:
    camera_name_from: str = "category"
    kind: str = "uca_import"

    splits: List[str] = field(
        default_factory=lambda: ["UCFCrime_Train", "UCFCrime_Test", "UCFCrime_Val"],
    )

    # YOLO export
    yolo_enabled: bool = True
    yolo_fps: float = 3.0
    yolo_conf: float = 0.25
    yolo_jpeg_quality: int = 95
    yolo_model: str = "yolo11s.pt"
    yolo_imgsz: int = 640
    trigger_class_ids: List[int] = field(default_factory=lambda: [0])

    # VLM crop (disabled by default for UCA)
    vlm_crop_enabled: bool = False

    # Windowing: split each source video into fixed-duration windows
    window_size_sec: float = 10.0

    # Category filtering: skip videos whose category matches these names
    exclude_categories: List[str] = field(default_factory=list)

    reencode: bool = False
    output_dir: str = "./dataset_uca"


def load_config(path: Optional[str] = None) -> UCAConfig:
    cfg_path = path or _CONFIG_PATH
    with open(cfg_path, encoding="utf-8") as f:
        raw: Dict[str, Any] = yaml.safe_load(f) or {}

    project_root = os.path.abspath(os.path.join(_DIR, "..", ".."))
    ds_rel = raw.get("output_dir", "./dataset_uca")
    output_dir = os.path.normpath(os.path.join(project_root, ds_rel))

    trigger_names = _deep_get(raw, "yolo_export", "trigger_classes", default=["person"])
    trigger_ids = _resolve_class_names(trigger_names) or [0]

    return UCAConfig(
        camera_name_from=raw.get("camera_name_from", "category"),
        kind=raw.get("kind", "uca_import"),
        splits=raw.get("splits", ["UCFCrime_Train", "UCFCrime_Test", "UCFCrime_Val"]),

        yolo_enabled=bool(_deep_get(raw, "yolo_export", "enabled", default=True)),
        yolo_fps=float(_deep_get(raw, "yolo_export", "fps", default=3.0)),
        yolo_conf=float(_deep_get(raw, "yolo_export", "confidence", default=0.25)),
        yolo_jpeg_quality=int(_deep_get(raw, "yolo_export", "jpeg_quality", default=95)),
        yolo_model=str(_deep_get(raw, "yolo_export", "model", default="yolo11s.pt")),
        yolo_imgsz=int(_deep_get(raw, "yolo_export", "imgsz", default=640)),
        trigger_class_ids=trigger_ids,

        vlm_crop_enabled=bool(_deep_get(raw, "vlm_crop", "enabled", default=False)),

        window_size_sec=float(_deep_get(raw, "windowing", "window_size_sec", default=10.0)),

        exclude_categories=[
            s.strip() for s in raw.get("exclude_categories", []) if s.strip()
        ],

        reencode=bool(raw.get("reencode", False)),
        output_dir=output_dir,
    )
