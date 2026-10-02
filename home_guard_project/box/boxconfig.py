"""Paths and per-box settings (box.yaml) for the collector box."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any, Dict

import yaml

_DIR = os.path.dirname(os.path.abspath(__file__))

PROJECT_ROOT = os.path.abspath(os.path.join(_DIR, "..", ".."))
LIVE_DIR = os.path.join(PROJECT_ROOT, "dataset_multi")      # written by the collector
OUTBOX_DIR = os.path.join(PROJECT_ROOT, "dataset_outbox")   # finished clips waiting for S3
LOG_DIR = os.path.join(PROJECT_ROOT, "logs")
ALIVE_FILE = os.path.join(LOG_DIR, "collector.alive")       # touched by run_collector.sh
BOX_YAML = os.path.join(_DIR, "box.yaml")

_SITE_RE = re.compile(r"^[a-z0-9_]+$")

# What the box runs. data_collection saves clips for tagging; inference sends alerts.
MODE_DATA_COLLECTION = "data_collection"
MODE_INFERENCE = "inference"
MODES = (MODE_DATA_COLLECTION, MODE_INFERENCE)


class BoxConfigError(Exception):
    """box.yaml is missing or invalid."""


@dataclass(frozen=True)
class BoxConfig:
    site: str
    min_age_minutes: float
    mode: str = MODE_DATA_COLLECTION


def load_box_settings(path: str = BOX_YAML) -> Dict[str, Any]:
    """Everything in box.yaml as a dict, for code that reads its own keys from it."""
    if not os.path.isfile(path):
        raise BoxConfigError(f"{path} not found — run setup_box.ps1 or create it with a 'site:' entry")
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_box_config(path: str = BOX_YAML) -> BoxConfig:
    raw = load_box_settings(path)

    site = str(raw.get("site") or "")
    if not _SITE_RE.match(site):
        raise BoxConfigError(
            f"{path}: 'site' must be lowercase letters, digits or underscores (got {site!r})"
        )

    mode = str(raw.get("mode") or MODE_DATA_COLLECTION)
    if mode not in MODES:
        raise BoxConfigError(f"{path}: 'mode' must be one of {', '.join(MODES)} (got {mode!r})")

    return BoxConfig(site=site, min_age_minutes=float(raw.get("min_age_minutes", 10.0)), mode=mode)


def set_site(site: str, path: str = BOX_YAML) -> None:
    """Set the site in box.yaml, keeping every other line. Creates the file if it is missing.

    The site is the house the box is installed in. It names the S3 folder the
    clips go to (``dataset_<site>``), so the installer chooses it at setup time.
    """
    if not _SITE_RE.match(site):
        raise BoxConfigError(f"'site' must be lowercase letters, digits or underscores (got {site!r})")

    lines = []
    if os.path.isfile(path):
        with open(path, encoding="utf-8") as f:
            lines = f.read().splitlines()

    new_line = f'site: "{site}"'
    for i, line in enumerate(lines):
        if re.match(r"^\s*site\s*:", line):
            lines[i] = new_line
            break
    else:
        lines.append(new_line)

    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines) + "\n")


def s3_prefix(site: str) -> str:
    """Top-level S3 dataset prefix for a site, e.g. ``dataset_house2``."""
    return f"dataset_{site}"
