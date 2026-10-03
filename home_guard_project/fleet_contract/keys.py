"""Parse legacy S3 keys and site-relative artifact paths without filesystem I/O."""

from dataclasses import dataclass
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


def parse_key(key: str) -> Optional[KeyInfo]:
    if not isinstance(key, str):
        return None
    parts = key.split("/")
    if len(parts) < 2:
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
    camera = day = None
    if area not in ("status", "other"):
        if len(parts) > camera_index + 1:
            camera = parts[camera_index] or None
        if len(parts) > camera_index + 2:
            day = parts[camera_index + 1] or None
    filename = parts[-1]
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
    kind = stem_kind(stem)[2] if stem else None
    return KeyInfo(key, root, site, area, camera, day, stem, kind, ext)
