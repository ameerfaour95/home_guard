"""Load and expose labeling pipeline configuration from config.yaml."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import yaml

_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.yaml")

_cache: Dict[str, Any] | None = None


def _load() -> Dict[str, Any]:
    global _cache
    if _cache is None:
        with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
            _cache = yaml.safe_load(f)
    return _cache


def get_ls_port() -> int:
    return int(_load()["label_studio"]["port"])


def get_ls_email() -> str:
    return str(_load()["label_studio"]["email"])


def get_ls_password() -> str:
    return str(_load()["label_studio"]["password"])


def get_project_name() -> str:
    return str(_load()["label_studio"]["project_name"])


def get_file_server_port() -> int:
    return int(_load()["file_server"]["port"])


def get_coco_labels() -> Dict[int, str]:
    """Return the COCO class-id -> label name mapping."""
    raw = _load()["coco_labels"]
    return {int(k): str(v) for k, v in raw.items()}


def get_all_label_names() -> List[str]:
    """Return sorted unique label names from COCO config."""
    return sorted(set(get_coco_labels().values()))


def get_tracking_config() -> Dict[str, Any]:
    """Return tracking settings for pre-annotation object merging."""
    defaults: Dict[str, Any] = {"enabled": True, "iou_threshold": 0.2}
    raw = _load().get("tracking", {})
    if not isinstance(raw, dict):
        return defaults
    return {
        "enabled": bool(raw.get("enabled", defaults["enabled"])),
        "iou_threshold": float(raw.get("iou_threshold", defaults["iou_threshold"])),
    }


def get_default_dataset_dir() -> str:
    return str(_load().get("dataset_dir", "./dataset_multi"))


# ---------------------------------------------------------------------------
# Storage config (local vs S3)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class S3StorageConfig:
    bucket: str
    prefix: str
    region: str
    url_expiry_sec: int


def get_storage_mode() -> str:
    """Return ``"local"`` or ``"s3"``."""
    storage = _load().get("storage", {})
    return str(storage.get("mode", "local")).lower()


def get_s3_config() -> Optional[S3StorageConfig]:
    """Return S3 settings if storage mode is ``"s3"``, else ``None``."""
    if get_storage_mode() != "s3":
        return None

    s3 = _load().get("storage", {}).get("s3", {})
    uri = str(s3.get("uri", ""))
    if not uri.startswith("s3://"):
        return None

    stripped = uri[len("s3://"):]
    parts = stripped.split("/", 1)
    bucket = parts[0]
    prefix = parts[1].rstrip("/") if len(parts) > 1 else ""

    return S3StorageConfig(
        bucket=bucket,
        prefix=prefix,
        region=str(s3.get("region", "us-east-1")),
        url_expiry_sec=int(s3.get("url_expiry_sec", 604800)),
    )
