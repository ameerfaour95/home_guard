"""Move finished clips from the live dataset to the outbox.

The uploader re-encodes clips in place and deletes local files, so it must
never see a clip the collector is still writing.  The collector writes a
clip's ``.meta.json`` last; a meta older than the minimum age marks a clip
whose files are all on disk.
"""

from __future__ import annotations

import logging
import os
import time
from typing import List, Optional, Tuple

log = logging.getLogger(__name__)

META_SUFFIX = ".meta.json"

# Folders that hold a clip's files under <camera>/<date>/, besides meta/.
_CLIP_DIRS = ("clips", "vlm_crops", "responses", "yolo/images", "yolo/labels")


def finished_clip_metas(
    live_dir: str,
    min_age_sec: float,
    now: Optional[float] = None,
) -> List[str]:
    """Return meta files under *live_dir* that are at least *min_age_sec* old."""
    now = time.time() if now is None else now
    cutoff = now - min_age_sec
    metas: List[str] = []
    for dirpath, _, filenames in os.walk(os.path.join(live_dir, "meta")):
        for name in filenames:
            path = os.path.join(dirpath, name)
            if name.endswith(META_SUFFIX) and os.path.getmtime(path) <= cutoff:
                metas.append(path)
    return sorted(metas)


def _belongs_to_clip(filename: str, stem: str) -> bool:
    # "<stem>.mp4", "<stem>.model_raw.txt", "<stem>_f0012.jpg" — but not "<stem>0_...".
    return filename.startswith(stem + ".") or filename.startswith(stem + "_f")


def move_clip(meta_path: str, live_dir: str, outbox_dir: str) -> int:
    """Move every file of the clip described by *meta_path*. Returns files moved.

    The meta moves last, so an interrupted move is picked up again next run.
    """
    stem = os.path.basename(meta_path)[: -len(META_SUFFIX)]
    camera_date = os.path.relpath(os.path.dirname(meta_path), os.path.join(live_dir, "meta"))

    sources: List[str] = []
    for sub in _CLIP_DIRS:
        folder = os.path.join(live_dir, sub, camera_date)
        if not os.path.isdir(folder):
            continue
        sources += [
            os.path.join(folder, name)
            for name in sorted(os.listdir(folder))
            if _belongs_to_clip(name, stem)
        ]
    sources.append(meta_path)

    for src in sources:
        dst = os.path.join(outbox_dir, os.path.relpath(src, live_dir))
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        os.replace(src, dst)
    return len(sources)


def move_finished_clips(
    live_dir: str,
    outbox_dir: str,
    min_age_sec: float,
    now: Optional[float] = None,
) -> Tuple[int, int]:
    """Move all finished clips to the outbox. Returns ``(clips_moved, files_moved)``."""
    clips = files = 0
    for meta_path in finished_clip_metas(live_dir, min_age_sec, now):
        try:
            files += move_clip(meta_path, live_dir, outbox_dir)
            clips += 1
        except OSError as exc:
            log.warning("Could not move clip %s (will retry next run): %s", meta_path, exc)
    return clips, files
