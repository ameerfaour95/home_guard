"""The owner's memories follow a camera across a site rename, once, at start (2026-10-10).

12:31 the box's site was renamed (ameer_week_0_1_* -> ameer_v2_*). The camera search carries zones, scene maps,
alert choices and the family's names along (find_cameras.carry_settings), but what the owner TOLD the box stayed
under the old ids: the pergola workers' marks (events/known.json), the electricians' activity fact (activity.json),
the camera facts and roles (events/camera_profiles.json) and the investigator's precedents (cases.jsonl). Every
lookup there is by the exact camera id, so none of them applied to the renamed cameras any more.

:func:`carry_memory` rewrites each store once: an id that is no longer a camera moves to the one current camera on
the same channel (``find_cameras.orphan_renames``: one old id and one camera per channel, else nothing moves and it
is logged). Only whole camera ids move - as a value (``"camera": "ameer_week_0_1_ch3"``, a list of cameras) or as a
key (camera_profiles' ``cameras``); an alert id that merely starts with the old name is history and stays. Each
file is copied to ``<file>.bak-rename-<YYYYmmdd-HHMMSS>`` before it is written (through a temp file). Never raises:
a store that cannot be read or written is logged and left as it was.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import os
import re
import shutil
import uuid
from typing import Any, Dict, List, Optional, Sequence, Set

log = logging.getLogger("box.memory_rename")

_CAMERA_ID = re.compile(r"^[A-Za-z][A-Za-z0-9]*(?:_[A-Za-z0-9]+)*_ch\d+$")
JSON_STORES = (os.path.join("events", "known.json"), os.path.join("events", "camera_profiles.json"), "activity.json")
JSONL_STORES = ("cases.jsonl",)


def _ids(value: Any, found: Set[str]) -> None:
    if isinstance(value, str):
        if _CAMERA_ID.match(value):
            found.add(value)
    elif isinstance(value, dict):
        for k, v in value.items():
            _ids(k, found)
            _ids(v, found)
    elif isinstance(value, list):
        for v in value:
            _ids(v, found)


def _renamed(value: Any, renames: Dict[str, str]) -> Any:
    if isinstance(value, str):
        return renames.get(value, value)
    if isinstance(value, dict):
        out: Dict[Any, Any] = {}
        for k, v in value.items():
            key = renames.get(k, k) if isinstance(k, str) else k
            if key in out and isinstance(out[key], dict) and isinstance(v, dict):
                continue                     # the camera already has its own entry: the old one is not merged in
            out[key] = _renamed(v, renames)
        return out
    if isinstance(value, list):
        return [_renamed(v, renames) for v in value]
    return value


def renames_for(ids: Set[str], cameras: Sequence[str]) -> Dict[str, str]:
    """``{old: new}`` for the ids of one store that are no longer cameras (find_cameras.orphan_renames)."""
    from .find_cameras import orphan_renames  # noqa: PLC0415

    current = [str(c) for c in cameras if c]
    if not current:
        return {}
    return orphan_renames(current, current, ids)


def _backup(path: str, stamp: str) -> None:
    shutil.copy2(path, f"{path}.bak-rename-{stamp}")


def _write(path: str, text: str) -> None:
    tmp = f"{path}.{uuid.uuid4().hex[:6]}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def carry_json(path: str, cameras: Sequence[str], stamp: str) -> Dict[str, str]:
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        return {}
    except (OSError, ValueError) as exc:
        log.warning("%s not read for the camera rename: %s", path, exc)
        return {}
    found: Set[str] = set()
    _ids(data, found)
    renames = renames_for(found, cameras)
    if not renames:
        return {}
    _backup(path, stamp)
    _write(path, json.dumps(_renamed(data, renames), ensure_ascii=False, indent=1))
    log.info("Carried %s to the renamed cameras: %s", os.path.basename(path), renames)
    return renames


def carry_jsonl(path: str, cameras: Sequence[str], stamp: str) -> Dict[str, str]:
    try:
        with open(path, encoding="utf-8") as f:
            lines = f.read().splitlines()
    except FileNotFoundError:
        return {}
    except OSError as exc:
        log.warning("%s not read for the camera rename: %s", path, exc)
        return {}
    rows: List[Optional[Any]] = []
    found: Set[str] = set()
    for line in lines:
        try:
            row = json.loads(line) if line.strip() else None
        except ValueError:
            row = None
        rows.append(row)
        if row is not None:
            _ids(row, found)
    renames = renames_for(found, cameras)
    if not renames:
        return {}
    out = [json.dumps(_renamed(row, renames), ensure_ascii=False) if row is not None else line
           for row, line in zip(rows, lines)]
    _backup(path, stamp)
    _write(path, "\n".join(out) + ("\n" if out else ""))
    log.info("Carried %s to the renamed cameras: %s", os.path.basename(path), renames)
    return renames


def carry_memory(cameras: Sequence[str], state_dir: str, now: Optional[float] = None) -> Dict[str, Dict[str, str]]:
    """Move the owner's memories kept under old camera ids to the renamed cameras (see the module docstring).
    Returns ``{store: {old: new}}`` for what moved. Never raises."""
    stamp = dt.datetime.fromtimestamp(now if now is not None else dt.datetime.now().timestamp()).strftime(
        "%Y%m%d-%H%M%S")
    moved: Dict[str, Dict[str, str]] = {}
    for rel in JSON_STORES + JSONL_STORES:
        path = os.path.join(state_dir, rel)
        try:
            done = (carry_jsonl if rel in JSONL_STORES else carry_json)(path, cameras, stamp)
        except Exception as exc:  # noqa: BLE001 - a store left as it was never stops the box
            log.warning("%s not carried over the camera rename: %s", rel, exc)
            continue
        if done:
            moved[rel] = done
    return moved


def box_cameras() -> List[str]:
    """Every camera of the box, on or off (cameras.yaml ``cameras`` and ``disabled``); [] when unreadable."""
    try:
        from .find_cameras import CAMERAS_PATH, _read_cameras_raw  # noqa: PLC0415

        raw = _read_cameras_raw(CAMERAS_PATH)
        return [str(c) for c in list(dict(raw.get("cameras") or {})) + list(dict(raw.get("disabled") or {}))]
    except Exception as exc:  # noqa: BLE001
        log.warning("Cameras not read for the memory rename: %s", exc)
        return []
