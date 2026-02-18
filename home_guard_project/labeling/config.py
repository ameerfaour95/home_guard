"""Load and expose labeling pipeline configuration from config.yaml."""

from __future__ import annotations

import os
from typing import Any, Dict, List

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


def get_default_dataset_dir() -> str:
    return str(_load().get("dataset_dir", "./dataset_multi"))
