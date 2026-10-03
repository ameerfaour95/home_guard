"""Parse legacy S3 keys and site-relative artifact paths without filesystem I/O."""

from dataclasses import dataclass
import datetime
import ntpath
from pathlib import PurePosixPath
import re
from typing import Literal, Optional


_KINDS = {"alert": "alert", "fp": "false_positive", "paused": "paused",
          "trigger": "trigger", "random": "random"}


@dataclass(frozen=True)
class KeyInfo:
    key: str
    root: Literal["dataset", "production"]
    site: str
    area: Literal["meta", "clips", "feedback", "status", "responses", "vlm_crops", "yolo_images", "yolo_labels", "other"]
    camera: Optional[str]
    day: Optional[str]
    stem: Optional[str]
    kind: Optional[str]
    ext: str


def stem_kind(stem: str) -> tuple[str, Optional[int], Optional[str]]:
    """Split from the right so underscores in camera names remain intact."""
    parts = stem.rsplit("_", 2)
    kind = _KINDS.get(parts[-1])
    if len(parts) == 3 and parts[0] and re.fullmatch(r"[0-9]+", parts[1]):
        try:
            return parts[0], int(parts[1]), kind
        except ValueError:  # An unreasonably long integer in malformed metadata.
            pass
    return stem, None, kind


def normalize_rel(path: str) -> Optional[str]:
    """Return a portable relative path; never resolve away parent traversal."""
    if not isinstance(path, str) or not path or "\x00" in path:
        return None
    path = path.replace("\\", "/")
    while path.startswith("./"):
        path = path[2:]
    if path.startswith("/") or ntpath.splitdrive(path)[0]:
        return None
    parts = path.split("/")
    if ".." in parts:
        return None
    return "/".join(part for part in parts if part not in ("", ".")) or None


_CAMERA = re.compile(r"[A-Za-z0-9_.-]{1,80}")
_DAY = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}")
_FILENAME = re.compile(r"[^\x00-\x1f\x7f]{1,255}")
# Exact key depth of each indexed area: <root>_<site>/<area>[/<sub>]/<camera>/<day>/<file>.
_DEPTH = {"meta": 5, "clips": 5, "feedback": 5, "responses": 5, "vlm_crops": 5,
          "yolo_images": 6, "yolo_labels": 6, "status": 3}
_CLIP_AREAS = {"meta", "clips", "responses", "vlm_crops", "yolo_images", "yolo_labels"}


def _real_day(day: str) -> bool:
    """`YYYY-MM-DD` that is a real calendar date (not 2026-99-99 or 2026-02-30)."""
    if not _DAY.fullmatch(day):
        return False
    try:
        datetime.date.fromisoformat(day)
    except ValueError:
        return False
    return True


def _clip_stem(camera: str, stem: str) -> bool:
    """`<camera>_<epoch>_<kind>`: the stem belongs to its folder's camera and carries a trigger time."""
    return stem.startswith(camera + "_") and stem_kind(stem)[1] is not None


def _stem_fits(area: str, camera: str, stem: str) -> bool:
    if area in _CLIP_AREAS:
        return _clip_stem(camera, stem)
    if area == "feedback":  # `<alert stem>_<ms>`; general feedback (`_general/` or `general_<ms>`) is exempt
        if camera == "_general" or re.fullmatch(r"general_[0-9]+", stem):
            return True
        alert = re.fullmatch(r"(.+)_[0-9]+", stem)
        return alert is not None and _clip_stem(camera, alert.group(1))
    return True


def parse_key(key: str) -> Optional[KeyInfo]:
    """Parse a legacy S3 key; None for keys outside the layout.

    Keys with `..`, backslashes, NULs, empty or `.` segments are rejected anywhere. Keys in an indexed area must
    also have that area's exact depth, a camera matching `[A-Za-z0-9_.-]{1,80}`, a `YYYY-MM-DD` day that is a
    real calendar date, and a stem of that camera: `<camera>_<epoch>_<kind>` (feedback: `<that stem>_<ms>`,
    except general feedback).
    """
    if not isinstance(key, str) or ".." in key or "\\" in key or "\x00" in key:
        return None
    parts = key.split("/")
    if len(parts) < 2 or any(part in ("", ".") for part in parts):
        return None
    prefix = re.fullmatch(r"(dataset|production)_(.+)", parts[0])
    if prefix is None:
        return None
    root, site = prefix.groups()
    area = {"_status": "status", "meta": "meta", "clips": "clips",
            "feedback": "feedback", "responses": "responses",
            "vlm_crops": "vlm_crops"}.get(parts[1], "other")
    camera_index = 2
    if parts[1] == "yolo" and len(parts) > 2 and parts[2] in ("images", "labels"):
        area = "yolo_" + parts[2]
        camera_index = 3
    if area != "other" and len(parts) != _DEPTH[area]:
        return None
    camera = day = None
    if area != "status" and area != "other":
        camera, day = parts[camera_index], parts[camera_index + 1]
        if not _CAMERA.fullmatch(camera) or not _real_day(day):
            return None
    filename = parts[-1]
    if not _FILENAME.fullmatch(filename):
        return None
    ext = PurePosixPath(filename).suffix
    stem = None
    if filename:
        stem = filename
        for suffix in (".meta.json", ".feedback.json", ".model_raw.txt"):
            if filename.endswith(suffix):
                stem = filename[:-len(suffix)]
                break
        else:
            stem = filename[:-len(ext)] if ext else filename
            if ext in (".jpg", ".txt"):
                stem = re.sub(r"_f[0-9]+$", "", stem)
    if camera is not None and not _stem_fits(area, camera, stem):
        return None
    kind = stem_kind(stem)[2] if stem else None
    return KeyInfo(key, root, site, area, camera, day, stem, kind, ext)
