"""Paths and per-box settings (box.yaml) for the collector box."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any, Dict, Optional, Union

import yaml

_DIR = os.path.dirname(os.path.abspath(__file__))

PROJECT_ROOT = os.path.abspath(os.path.join(_DIR, "..", ".."))
LIVE_DIR = os.path.join(PROJECT_ROOT, "dataset_multi")      # written by the collector
OUTBOX_DIR = os.path.join(PROJECT_ROOT, "dataset_outbox")   # finished clips waiting for S3
LOG_DIR = os.path.join(PROJECT_ROOT, "logs")
ALIVE_FILE = os.path.join(LOG_DIR, "collector.alive")       # touched by run_collector.sh
BOX_YAML = os.path.join(_DIR, "box.yaml")

# Inference (production) mode saves its alert clips in folders of its own, in the
# same layout. They are uploaded like the collector's clips, but to an S3 folder
# that the bucket empties after PRODUCTION_RETENTION_DAYS (see retention.py).
PRODUCTION_LIVE_DIR = os.path.join(PROJECT_ROOT, "production_multi")
PRODUCTION_ARCHIVE_DIR = os.path.join(PROJECT_ROOT, "production_archive")
PRODUCTION_PREFIX_ROOT = "production_"
PRODUCTION_RETENTION_DAYS = 14

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


def _set_line(key: str, rendered: str, path: str) -> None:
    """Replace the ``key: ...`` line of box.yaml (or append it), keeping every other line."""
    lines = []
    if os.path.isfile(path):
        with open(path, encoding="utf-8") as f:
            lines = f.read().splitlines()

    new_line = f"{key}: {rendered}"
    for i, line in enumerate(lines):
        if re.match(rf"^\s*{re.escape(key)}\s*:", line):
            lines[i] = new_line
            break
    else:
        lines.append(new_line)

    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines) + "\n")


def set_site(site: str, path: str = BOX_YAML) -> None:
    """Set the site in box.yaml, keeping every other line. Creates the file if it is missing.

    The site is the house the box is installed in. It names the S3 folder the
    clips go to (``dataset_<site>``), so the installer chooses it at setup time.
    """
    if not _SITE_RE.match(site):
        raise BoxConfigError(f"'site' must be lowercase letters, digits or underscores (got {site!r})")
    _set_line("site", f'"{site}"', path)


# Choices the installer makes in the setup program.
#   show_cameras:      on the box's own screen, open a window per camera (true) or only the log (false).
#   notify_dry_run:    inference mode logs its alerts instead of sending them.
#   alert_start_hour,
#   alert_end_hour:    the hours of the day between which inference mode alerts the owner.
#   alert_cooldown_sec: the shortest time between two alerts from one camera, in seconds.
#   mode:              what the box runs.
#   alert_channel:     how inference mode delivers its alerts.
#   telegram_chat_ids: who receives the Telegram alerts, e.g. -1001234567,987654.
#   alert_on:          the house default of what the owner is alerted about: person, vehicle,
#                      animal - one or more (person,vehicle). A camera can have its own choice
#                      (camera_alerts.yaml, find_cameras set-camera-alerts).
BOOLEAN_OPTIONS = ("show_cameras", "notify_dry_run")
# Whole numbers, each with the smallest and largest value it may take.
NUMBER_OPTIONS = {"alert_start_hour": (0, 23), "alert_end_hour": (0, 23), "alert_cooldown_sec": (10, 86400)}
# Decimal numbers, each with its range.
#   inference_conf:    how sure the detector must be before a person or vehicle counts
#                      (0.05 reacts to almost anything, 0.95 only to what it is certain of).
DECIMAL_OPTIONS = {"inference_conf": (0.05, 0.95)}
#   owner_language: the language of alerts and announcements, en or he (replies follow each person's own language).
CHOICE_OPTIONS = {"mode": MODES, "alert_channel": ("telegram", "twilio", "both"), "owner_language": ("en", "he")}
CHAT_IDS_OPTION = "telegram_chat_ids"
# Options holding a set of choices, written comma-separated in a fixed order (e.g. person,vehicle).
SET_OPTIONS = {"alert_on": ("person", "vehicle", "animal")}
OPTIONS = (BOOLEAN_OPTIONS + tuple(NUMBER_OPTIONS) + tuple(DECIMAL_OPTIONS) + tuple(CHOICE_OPTIONS)
           + tuple(SET_OPTIONS) + (CHAT_IDS_OPTION,))
# Options the running program re-reads while it runs (inference.LiveSettings): a change applies
# within seconds, without a restart.
LIVE_OPTIONS = ("alert_start_hour", "alert_end_hour", "alert_cooldown_sec", "inference_conf", "alert_on", "owner_language")
# Options the running program reads only when it starts. show_cameras is read by the screen, not by it.
RESTART_OPTIONS = tuple(key for key in OPTIONS if key != "show_cameras" and key not in LIVE_OPTIONS)

_TRUE = ("true", "yes", "y", "1", "on")
_FALSE = ("false", "no", "n", "0", "off")
_CHAT_IDS_RE = re.compile(r"^-?\d+(,-?\d+)*$")

OptionValue = Union[bool, int, float, str]


def _check_option(key: str) -> None:
    if key not in OPTIONS:
        raise BoxConfigError(f"unknown option {key!r}; known: {', '.join(OPTIONS)}")


def set_option(key: str, value: str, path: str = BOX_YAML) -> OptionValue:
    """Set an option in box.yaml, keeping every other line. Returns the value stored."""
    _check_option(key)
    text = str(value).strip()

    if key in BOOLEAN_OPTIONS:
        if text.lower() not in _TRUE + _FALSE:
            raise BoxConfigError(f"{key} must be true or false (got {value!r})")
        flag = text.lower() in _TRUE
        _set_line(key, "true" if flag else "false", path)
        return flag

    if key in NUMBER_OPTIONS:
        low, high = NUMBER_OPTIONS[key]
        if not text.isdigit() or not low <= int(text) <= high:
            raise BoxConfigError(f"{key} must be a whole number from {low} to {high} (got {value!r})")
        _set_line(key, str(int(text)), path)
        return int(text)

    if key in DECIMAL_OPTIONS:
        low, high = DECIMAL_OPTIONS[key]
        try:
            number = round(float(text), 3)
        except ValueError:
            number = None
        if number is None or not low <= number <= high:
            raise BoxConfigError(f"{key} must be a number from {low} to {high} (got {value!r})")
        _set_line(key, repr(number), path)
        return number

    if key in CHOICE_OPTIONS:
        if text not in CHOICE_OPTIONS[key]:
            raise BoxConfigError(f"{key} must be one of {', '.join(CHOICE_OPTIONS[key])} (got {value!r})")
        _set_line(key, text, path)
        return text

    if key in SET_OPTIONS:
        allowed = SET_OPTIONS[key]
        picked = {part.strip().lower() for part in text.split(",") if part.strip()}
        if not picked or not picked <= set(allowed):
            raise BoxConfigError(f"{key} must be one or more of {', '.join(allowed)}, separated by commas "
                                 f"(got {value!r})")
        joined = ",".join(choice for choice in allowed if choice in picked)
        _set_line(key, f'"{joined}"', path)
        return joined

    if not _CHAT_IDS_RE.match(text):
        raise BoxConfigError(f"{key} must be numbers separated by commas, without spaces (got {value!r})")
    _set_line(key, f'"{text}"', path)
    return text


def get_option(key: str, path: str = BOX_YAML) -> Optional[OptionValue]:
    """An option from box.yaml. Unset: yes/no options are false, mode is data_collection, the rest None."""
    _check_option(key)
    try:
        settings = load_box_settings(path)
    except BoxConfigError:
        settings = {}
    if key in BOOLEAN_OPTIONS:
        return bool(settings.get(key, False))
    if key == "mode":
        return str(settings.get(key) or MODE_DATA_COLLECTION)
    return settings.get(key)


def s3_prefix(site: str) -> str:
    """Top-level S3 dataset prefix for a site, e.g. ``dataset_house2``."""
    return f"dataset_{site}"


def production_prefix(site: str) -> str:
    """Top-level S3 prefix for a site's production (inference mode) clips, e.g. ``production_house2``."""
    return f"{PRODUCTION_PREFIX_ROOT}{site}"
