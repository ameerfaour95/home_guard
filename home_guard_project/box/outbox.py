"""Move finished clips from the live dataset to the outbox.

The uploader re-encodes clips in place and deletes local files, so it must
never see a clip the collector is still writing.  The collector writes a
clip's ``.meta.json`` last; a meta older than the minimum age marks a clip
whose files are all on disk.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import logging
import math
import os
import shutil
import time
from typing import Any, Dict, List, Optional, Tuple

log = logging.getLogger(__name__)

META_SUFFIX = ".meta.json"

# Folders that hold a clip's files under <camera>/<date>/, besides meta/.
_CLIP_DIRS = ("clips", "vlm_crops", "responses", "yolo/images", "yolo/labels")

# Files with no meta after this long belong to a clip the collector never finished.
ORPHAN_AGE_SEC = 3600.0


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


def orphan_files(
    live_dir: str,
    min_age_sec: float,
    now: Optional[float] = None,
) -> List[str]:
    """Return files under *live_dir* that belong to no meta and are at least *min_age_sec* old.

    The meta is written last, a minute or more after the clip.  A collector that
    is stopped in between leaves the clip's files without one, and nothing would
    ever pick them up.  The age limit keeps clips whose meta is still on its way.
    """
    now = time.time() if now is None else now
    cutoff = now - min_age_sec
    orphans: List[str] = []
    for sub in _CLIP_DIRS:
        root = os.path.join(live_dir, sub)
        for dirpath, _, filenames in os.walk(root):
            meta_dir = os.path.join(live_dir, "meta", os.path.relpath(dirpath, root))
            stems = (
                [m[: -len(META_SUFFIX)] for m in os.listdir(meta_dir) if m.endswith(META_SUFFIX)]
                if os.path.isdir(meta_dir)
                else []
            )
            for name in filenames:
                path = os.path.join(dirpath, name)
                if os.path.getmtime(path) > cutoff:
                    continue
                if not any(_belongs_to_clip(name, stem) for stem in stems):
                    orphans.append(path)
    return sorted(orphans)


def move_orphans(
    live_dir: str,
    outbox_dir: str,
    min_age_sec: float,
    now: Optional[float] = None,
) -> int:
    """Move the files of interrupted clips to the outbox. Returns files moved."""
    moved = 0
    for src in orphan_files(live_dir, min_age_sec, now):
        dst = os.path.join(outbox_dir, os.path.relpath(src, live_dir))
        try:
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            os.replace(src, dst)
            moved += 1
        except OSError as exc:
            log.warning("Could not move %s (will retry next run): %s", src, exc)
    return moved


def move_feedback(live_dir: str, outbox_dir: str) -> int:
    """Move the owner-feedback files (``feedback/``, written by inference mode) to the outbox. Returns files moved.

    Each file is complete when it appears (it is written through a ``.tmp``
    file), so there is no age to wait for.
    """
    moved = 0
    for dirpath, _, filenames in os.walk(os.path.join(live_dir, "feedback")):
        for name in filenames:
            if name.endswith(".tmp"):
                continue
            src = os.path.join(dirpath, name)
            dst = os.path.join(outbox_dir, os.path.relpath(src, live_dir))
            try:
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                os.replace(src, dst)
                moved += 1
            except OSError as exc:
                log.warning("Could not move %s (will retry next run): %s", src, exc)
    return moved


CHAT_DIR = "chat"                 # in the site outbox: chat/<YYYY-MM-DD>.jsonl and chat/images/
CHAT_STATE_NAME = "chat_upload_state.json"


def _line_hash(line: str) -> str:
    return hashlib.sha1(line.encode("utf-8")).hexdigest()


def _read_chat_state(path: str) -> Tuple[float, set]:
    try:
        with open(path, encoding="utf-8") as f:
            state = json.load(f)
        last = float(state.get("last_ts"))
        if not math.isfinite(last):
            raise ValueError("last_ts must be finite")
        return last, set(state.get("at_last_ts") or ())
    except (OSError, ValueError, TypeError, AttributeError):
        return float("-inf"), set()


def _write_chat_state(path: str, last_ts: float, at_last: set) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump({"last_ts": last_ts, "at_last_ts": sorted(at_last), "updated": time.time()}, f)
    os.replace(tmp, path)


def copy_chat_feed(feed_path: str, site_outbox: str, state_path: str) -> int:
    """Copy the Telegram conversation's new lines (chat_feed.ChatFeed's file) into *site_outbox* for the uploader.

    Lines go to ``chat/<YYYY-MM-DD>.jsonl`` by the local date of their ``ts``, appended to the day's file as they
    are (UTF-8, Hebrew unescaped); the pictures they name (``logs/chat_images/``) are copied to ``chat/images/``.
    The live feed is only read: the assistant and the window read it, and ChatFeed trims it itself.

    Progress is the last copied ``ts`` (and the lines at exactly that ts) in *state_path*, not a line count, so a
    trimmed feed neither repeats nor loses lines; a line already in its day file is never appended twice (a run
    stopped between the copy and the state). A line still being written (no newline yet) waits for the next run.
    Returns the lines copied; never raises (the clips' upload goes on).
    """
    try:
        with open(feed_path, encoding="utf-8") as f:
            raw = f.readlines()
    except OSError:
        return 0
    last_ts, at_last = _read_chat_state(state_path)
    new: List[Tuple[float, str, Dict[str, Any]]] = []
    for line in raw:
        if not line.endswith("\n"):
            continue
        text = line.rstrip("\r\n")
        try:
            entry = json.loads(text)
            ts = float(entry["ts"])
            if not math.isfinite(ts):
                raise ValueError("ts must be finite")
            day = dt.datetime.fromtimestamp(ts).strftime("%Y-%m-%d")
        except (ValueError, TypeError, KeyError, OverflowError, OSError):
            continue                       # a damaged line, or one without a usable time
        if not isinstance(entry, dict) or ts < last_ts or (ts == last_ts and _line_hash(text) in at_last):
            continue
        new.append((ts, text, dict(entry, _day=day)))
    if not new:
        return 0
    chat_dir = os.path.join(site_outbox, CHAT_DIR)
    image_src = os.path.join(os.path.dirname(feed_path) or ".", "chat_images")
    copied = 0
    try:
        by_day: Dict[str, List[str]] = {}
        for _, text, entry in new:
            by_day.setdefault(entry["_day"], []).append(text)
        os.makedirs(chat_dir, exist_ok=True)
        for day, lines in sorted(by_day.items()):
            day_path = os.path.join(chat_dir, f"{day}.jsonl")
            try:
                with open(day_path, encoding="utf-8") as f:
                    there = {x.rstrip("\r\n") for x in f}
            except OSError:
                there = set()
            fresh = [x for x in lines if x not in there]
            if fresh:
                with open(day_path, "a", encoding="utf-8", newline="\n") as f:
                    f.write("".join(x + "\n" for x in fresh))
                copied += len(fresh)
        for _, _, entry in new:
            name = os.path.basename(str(entry.get("image") or ""))
            src = os.path.join(image_src, name)
            if name and os.path.isfile(src):
                dst = os.path.join(chat_dir, "images", name)
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                try:
                    shutil.copy2(src, dst)
                except OSError as exc:
                    log.warning("Chat picture %s not copied (the next run does not retry it): %s", name, exc)
        top = max(ts for ts, _, _ in new)
        hashes = {_line_hash(text) for ts, text, _ in new if ts == top}
        _write_chat_state(state_path, top, hashes | (at_last if top == last_ts else set()))
    except OSError as exc:
        log.warning("Telegram conversation not copied to the outbox (will retry next run): %s", exc)
    return copied


USAGE_DIR = "usage"               # in the site outbox: usage/<YYYY-MM-DD>.jsonl (usage_ledger.py)
USAGE_STATE_NAME = "usage_upload_state.json"


def copy_usage_ledger(usage_dir: str, site_outbox: str, state_path: str) -> int:
    """Copy the AI usage ledger's day files (usage_ledger.py, ``<state_dir>/usage/<day>.jsonl``) into *site_outbox*
    ``usage/`` for the uploader, as the chat goes: a day file is copied whole whenever it grew since the last copy
    (the uploader re-sends a file whose size changed), and a day already copied at its size is never copied again,
    so the archive's retention deleting an old copy does not bring it back. Progress is ``{day: size}`` in
    *state_path*. Returns the files copied; never raises (the clips' upload goes on)."""
    try:
        names = sorted(n for n in os.listdir(usage_dir) if n.endswith(".jsonl"))
    except OSError:
        return 0
    try:
        with open(state_path, encoding="utf-8") as f:
            state = {str(k): int(v) for k, v in (json.load(f).get("sizes") or {}).items()}
    except (OSError, ValueError, TypeError, AttributeError):
        state = {}
    copied = 0
    try:
        for name in names:
            src = os.path.join(usage_dir, name)
            size = os.path.getsize(src)
            if size <= state.get(name, -1):
                continue
            dst = os.path.join(site_outbox, USAGE_DIR, name)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            with open(src, "rb") as f:
                data = f.read(size)          # what is there now; a line still being appended waits for the next run
            data = data[: data.rfind(b"\n") + 1]
            tmp = dst + ".tmp"
            with open(tmp, "wb") as f:
                f.write(data)
            os.replace(tmp, dst)
            state[name] = len(data)
            copied += 1
        if copied:
            os.makedirs(os.path.dirname(state_path) or ".", exist_ok=True)
            tmp = state_path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump({"sizes": state, "updated": time.time()}, f)
            os.replace(tmp, state_path)
    except OSError as exc:
        log.warning("AI usage ledger not copied to the outbox (will retry next run): %s", exc)
    return copied


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
