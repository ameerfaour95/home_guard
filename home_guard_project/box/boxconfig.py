"""Paths and per-box settings (box.yaml) for the collector box."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass

import yaml

_DIR = os.path.dirname(os.path.abspath(__file__))

PROJECT_ROOT = os.path.abspath(os.path.join(_DIR, "..", ".."))
LIVE_DIR = os.path.join(PROJECT_ROOT, "dataset_multi")      # written by the collector
OUTBOX_DIR = os.path.join(PROJECT_ROOT, "dataset_outbox")   # finished clips waiting for S3
LOG_DIR = os.path.join(PROJECT_ROOT, "logs")
ALIVE_FILE = os.path.join(LOG_DIR, "collector.alive")       # touched by run_collector.sh
BOX_YAML = os.path.join(_DIR, "box.yaml")

_SITE_RE = re.compile(r"^[a-z0-9_]+$")


class BoxConfigError(Exception):
    """box.yaml is missing or invalid."""


@dataclass(frozen=True)
class BoxConfig:
    site: str
    min_age_minutes: float


def load_box_config(path: str = BOX_YAML) -> BoxConfig:
    if not os.path.isfile(path):
        raise BoxConfigError(f"{path} not found — run setup_box.ps1 or create it with a 'site:' entry")

    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    site = str(raw.get("site") or "")
    if not _SITE_RE.match(site):
        raise BoxConfigError(
            f"{path}: 'site' must be lowercase letters, digits or underscores (got {site!r})"
        )

    return BoxConfig(site=site, min_age_minutes=float(raw.get("min_age_minutes", 10.0)))


def s3_prefix(site: str) -> str:
    """Top-level S3 dataset prefix for a site, e.g. ``dataset_house2``."""
    return f"dataset_{site}"
