"""Load s3_upload/config.yaml into a simple namespace."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet

import yaml

_DIR = os.path.dirname(os.path.abspath(__file__))
_CONFIG_PATH = os.path.join(_DIR, "config.yaml")


@dataclass(frozen=True)
class S3Config:
    bucket: str
    prefix: str
    region: str
    workers: int
    dataset_dir: str
    skip_reencode: bool
    cleanup: bool
    force: bool
    allowed_labels: FrozenSet[str] = field(default_factory=frozenset)


def load_config(path: str = _CONFIG_PATH) -> S3Config:
    with open(path, encoding="utf-8") as f:
        raw: Dict[str, Any] = yaml.safe_load(f) or {}

    s3 = raw.get("s3", {})

    project_root = os.path.abspath(os.path.join(_DIR, "..", ".."))
    ds_rel = raw.get("dataset_dir", "./dataset_multi")
    dataset_dir = os.path.normpath(os.path.join(project_root, ds_rel))

    coco_raw = raw.get("coco_labels", {})
    allowed_labels = frozenset(str(v) for v in coco_raw.values()) if coco_raw else frozenset()

    return S3Config(
        bucket=s3.get("bucket", "security-camera-project-v1"),
        prefix=s3.get("prefix", "dataset_multi"),
        region=s3.get("region", "us-east-1"),
        workers=int(s3.get("workers", 4)),
        dataset_dir=dataset_dir,
        skip_reencode=bool(raw.get("skip_reencode", False)),
        cleanup=bool(raw.get("cleanup", True)),
        force=bool(raw.get("force", False)),
        allowed_labels=allowed_labels,
    )
