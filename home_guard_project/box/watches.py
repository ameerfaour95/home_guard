"""One-shot watches the owner asks for in the chat: "ברגע שהיא יוצאת תן לי דיווח" (tell me when the white car leaves).

2026-10-11 07:57: the owner asked "is the white car still at the main entrance? when she leaves, tell me" (his wife has
no phone; her car leaving is how he knows she went out). The assistant had no such thing and saved "אשת בעל הבית" as
a camera name instead. A watch is that request, kept until it fires once or expires:

- ``leaves``: the vehicles seen at the camera when he asked (the baseline boxes) are checked every ``CHECK_SEC`` on a
  new picture by the box's own detector (no AI, no cost). A baseline vehicle that has no vehicle on it on
  ``MISSES_TO_FIRE`` checks in a row has left: one photo goes to the chat, "🚗 הרכב הלבן יצא מכניסה ראשית · 08:41".
- ``arrives``: a vehicle where none of the baseline ones stands, on two checks in a row.

Presence is lenient (``PRESENT_CONF``) so absence is conservative: a car half in shadow still counts as there. The
watch ends when it fires, or quietly at its end (he gave one, else ``DEFAULT_HOURS``). Stored in one JSON file; the
checker thread is started by the v3 assistant.
"""
from __future__ import annotations

import datetime as dt
import json
import logging
import os
import threading
import time
import uuid
from typing import Any, Callable, Dict, List, Optional, Sequence

log = logging.getLogger("box.watches")

CHECK_SEC = 60.0
TICK_SEC = 15.0
MISSES_TO_FIRE = 2
DEFAULT_HOURS = 24.0
PRESENT_CONF = 0.25
VEHICLES = ("car", "truck", "bus", "motorcycle")
LEAVES, ARRIVES = "leaves", "arrives"
_lock = threading.Lock()


def default_path() -> str:
    from .boxconfig import LOG_DIR  # noqa: PLC0415

    return os.path.join(LOG_DIR, "watches.json")


def _read(path: str) -> List[Dict[str, Any]]:
    try:
        with open(path, encoding="utf-8") as f:
            rows = json.load(f)
        return [r for r in rows if isinstance(r, dict)] if isinstance(rows, list) else []
    except (OSError, ValueError):
        return []


def _write(path: str, rows: Sequence[Dict[str, Any]]) -> None:
    tmp = path + ".tmp"
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(list(rows), f, ensure_ascii=False, indent=1)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def vehicles(objects: Any, conf: float = PRESENT_CONF) -> List[List[float]]:
    """The vehicle boxes (0..1) among the detector's objects ``[{"label", "conf", "box"}]``."""
    out = []
    for o in objects or []:
        try:
            if str(o.get("label")) in VEHICLES and float(o.get("conf") or 0) >= conf:
                out.append([float(v) for v in o["box"][:4]])
        except (AttributeError, KeyError, TypeError, ValueError):
            continue
    return out


def overlaps(a: Sequence[float], b: Sequence[float]) -> bool:
    """The same parked vehicle: IoU at least 0.25, or one box's centre inside the other."""
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    if union > 0 and inter / union >= 0.25:
        return True

    def inside(p: Sequence[float], box: Sequence[float]) -> bool:
        cx, cy = (p[0] + p[2]) / 2, (p[1] + p[3]) / 2
        return box[0] <= cx <= box[2] and box[1] <= cy <= box[3]

    return inside(a, b) or inside(b, a)


class Watches:
    """The watch file and the checks. *grab(camera)* -> ``{"image"}`` | ``{"error"}``; *detect(path)* -> objects or
    None; *send(chat_id, image, caption)* -> bool."""

    def __init__(self, path: str, grab: Callable[[str], Dict[str, Any]], detect: Callable[[str], Any],
                 send: Callable[[str, str, str], bool], now: Callable[[], float] = time.time) -> None:
        self.path, self.grab, self.detect, self.send, self.now = path, grab, detect, send, now
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()

    # -- what the chat does -----------------------------------------------------------------------------------------
    def look(self, camera: str) -> Optional[Dict[str, Any]]:
        """A new picture and its vehicle boxes: ``{"image", "vehicles"}``; None when there is no picture or no
        detector answer."""
        shot = self.grab(camera)
        if not isinstance(shot, dict) or not isinstance(shot.get("image"), str):
            return None
        objects = self.detect(shot["image"])
        if objects is None:
            return None
        return {"image": shot["image"], "vehicles": vehicles(objects)}

    def add(self, chat_id: str, camera: str, name: str, change: str, subject: str, baseline: Sequence[Sequence[float]],
            until: Optional[float] = None) -> Dict[str, Any]:
        now = float(self.now())
        row = {"id": uuid.uuid4().hex[:8], "chat_id": str(chat_id), "camera": camera, "name": name,
               "change": change if change in (LEAVES, ARRIVES) else LEAVES, "subject": subject[:60],
               "baseline": [list(map(float, b[:4])) for b in baseline], "created": now,
               "until": float(until) if until and until > now else now + DEFAULT_HOURS * 3600,
               "checked": now, "misses": 0, "seen_new": 0}
        with _lock:
            rows = [r for r in _read(self.path) if not (r.get("chat_id") == row["chat_id"] and
                                                        r.get("camera") == camera and r.get("change") == row["change"])]
            rows.append(row)
            _write(self.path, rows)
        log.info("Watch %s added: %s %s at %s (%d baseline vehicle(s)) until %s", row["id"], row["subject"] or "vehicle",
                 row["change"], camera, len(row["baseline"]), dt.datetime.fromtimestamp(row["until"]).strftime("%H:%M"))
        return row

    def active(self, chat_id: str = "") -> List[Dict[str, Any]]:
        now = float(self.now())
        return [r for r in _read(self.path) if float(r.get("until") or 0) > now
                and (not chat_id or r.get("chat_id") == str(chat_id))]

    # -- the checks ---------------------------------------------------------------------------------------------------
    def tick(self) -> None:
        now = float(self.now())
        with _lock:
            rows = _read(self.path)
        keep, changed = [], False
        for row in rows:
            if float(row.get("until") or 0) <= now:
                log.info("Watch %s ended without a change (%s at %s)", row.get("id"), row.get("change"), row.get("camera"))
                changed = True
                continue
            if now - float(row.get("checked") or 0) < CHECK_SEC:
                keep.append(row)
                continue
            changed = True
            row["checked"] = now
            try:
                fired = self._check(row, now)
            except Exception as exc:  # noqa: BLE001 - one watch's failure never stops the others
                log.warning("Watch %s check failed: %s", row.get("id"), exc)
                fired = False
            if not fired:
                keep.append(row)
        if changed:
            with _lock:
                fresh = {r.get("id") for r in _read(self.path)}
                ours = {r.get("id") for r in rows}
                added = [r for r in _read(self.path) if r.get("id") in fresh - ours]    # added during the check
                _write(self.path, keep + added)

    def _check(self, row: Dict[str, Any], now: float) -> bool:
        seen = self.look(str(row["camera"]))
        if seen is None:
            return False                       # no picture: neither there nor gone
        base = row.get("baseline") or []
        if row["change"] == LEAVES:
            gone = [b for b in base if not any(overlaps(b, v) for v in seen["vehicles"])]
            row["misses"] = int(row.get("misses") or 0) + 1 if gone else 0
            if row["misses"] < MISSES_TO_FIRE:
                return False
            subject = row.get("subject") if len(base) == 1 else ""
            caption = f"🚗 {subject or 'אחד הרכבים'} יצא מ{row['name']} · {_hhmm(now)}"
        else:
            new = [v for v in seen["vehicles"] if not any(overlaps(v, b) for b in base)]
            row["seen_new"] = int(row.get("seen_new") or 0) + 1 if new else 0
            if row["seen_new"] < MISSES_TO_FIRE:
                return False
            caption = f"🚗 {row.get('subject') or 'רכב'} הגיע ל{row['name']} · {_hhmm(now)}"
        ok = bool(self.send(str(row["chat_id"]), seen["image"], caption))
        log.info("Watch %s fired (%s): %r -> %s", row.get("id"), row["change"], caption, "sent" if ok else "NOT sent")
        return ok                              # not sent: tried again on the next check

    # -- the thread ---------------------------------------------------------------------------------------------------
    def start(self) -> None:
        if self._thread is not None:
            return

        def run() -> None:
            while not self._stop.wait(TICK_SEC):
                try:
                    self.tick()
                except Exception as exc:  # noqa: BLE001
                    log.warning("Watches tick failed: %s", exc)

        self._thread = threading.Thread(target=run, name="owner-watches", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()


def _hhmm(ts: float) -> str:
    return dt.datetime.fromtimestamp(ts).strftime("%H:%M")


_current: Optional[Watches] = None


def current() -> Optional[Watches]:
    return _current


def start(services: Any, deliverer: Any, path: Optional[str] = None) -> Optional[Watches]:
    """The box's watches on the assistant's services (grab_photo, detect) and Telegram deliverer; None without them."""
    global _current
    if services is None or deliverer is None or services.grab_photo is None or services.detect is None:
        log.info("Watches off (no live picture, detector or Telegram)")
        return None

    def send(chat_id: str, image: str, caption: str) -> bool:
        got = deliverer.photo(chat_id, image, caption=caption)
        return bool(got.get("ok")) if isinstance(got, dict) else bool(got)

    _current = Watches(path or default_path(), services.grab_photo, services.detect, send, now=services.now)
    _current.start()
    log.info("Watches on (%d active)", len(_current.active()))
    return _current
