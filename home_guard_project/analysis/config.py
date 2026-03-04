"""Load analysis/config.yaml into a simple namespace."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict

import yaml

_DIR = os.path.dirname(os.path.abspath(__file__))
_CONFIG_PATH = os.path.join(_DIR, "config.yaml")


@dataclass(frozen=True)
class AnalysisConfig:
    s3_bucket: str
    s3_prefix: str
    s3_region: str
    coco_labels: Dict[int, str]
    output_dir: str


def load_config(path: str = _CONFIG_PATH) -> AnalysisConfig:
    with open(path, encoding="utf-8") as f:
        raw: Dict[str, Any] = yaml.safe_load(f) or {}

    s3 = raw.get("s3", {})

    project_root = os.path.abspath(os.path.join(_DIR, "..", ".."))
    out_rel = raw.get("output_dir", "./analysis_output")
    output_dir = os.path.normpath(os.path.join(project_root, out_rel))

    coco_raw = raw.get("coco_labels", {})
    coco_labels = {int(k): str(v) for k, v in coco_raw.items()}

    return AnalysisConfig(
        s3_bucket=s3.get("bucket", "security-camera-project-v1"),
        s3_prefix=s3.get("prefix", "dataset_multi"),
        s3_region=s3.get("region", "us-east-1"),
        coco_labels=coco_labels,
        output_dir=output_dir,
    )
