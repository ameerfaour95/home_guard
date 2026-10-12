"""A picture of a moment that has passed: "תראה לי תמונה שהם היו שם ב-10:24" is answered from the box's recordings,
never with a live picture of now.

2026-10-12 11:43-11:45: the owner asked four times for a picture of the pergola workers "when they were here" (10:24,
"בשעות האלה"); each time the assistant took a LIVE picture of the empty pergola and said no picture of those times
was kept, while the box held the event (10:18-10:24, five alert clips) and its keyframe. A past moment is found in the
event memory (at the camera he means, the one with people nearest the time he gives, else the times the assistant
itself just cited), and a still of that alert's clip at the moment the detector fired is sent with the camera's name
and the time it was taken.
"""
from __future__ import annotations

import datetime as dt
import glob
import json
import logging
import os
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

log = logging.getLogger("box.assistant_v3.past_photo")

MAX_PHOTOS = 3
NEAR_SEC = 20 * 60.0          # a moment with no event around it takes the nearest one this close
EDGE_SEC = 2 * 60.0           # an event "covers" a moment a little before its start and after its end

_CLOCK = re.compile(r"(?<![\d:])([01]?\d|2[0-3])[:.]([0-5]\d)(?![\d:])")
_PAST = re.compile(r"(?<![א-ת])(?:ש?היו|ש?היה|ש?היתה|ש?הייתה|ש?הגיעו|ש?הגיע|ש?הגיעה|עבדו|בשעות האלה|בשעה הזאת|"
                   r"בשעה הזו|באותה שעה|באותו זמן|בזמנים האלה|מהבוקר|קודם|של אז|לא של עכשיו|לא עכשיו)(?![א-ת])"
                   r"|\b(?:were|was|earlier|this morning|at that time)\b", re.IGNORECASE)
_THOSE = re.compile(r"(?<![א-ת])(?:האלה|האלו|ההן|ההם|האלה)(?![א-ת])|\bthose\b", re.IGNORECASE)


def asks_past(*texts: str) -> bool:
    """True when the words ask about a moment that has passed (a past verb, "those hours", a clock time)."""
    return any(_PAST.search(t or "") or _CLOCK.search(t or "") for t in texts)


def clock_times(text: str, now: float) -> List[float]:
    """Every HH:MM in *text* as the latest such moment not after *now* (today, else yesterday)."""
    out = []
    today = dt.datetime.fromtimestamp(now)
    for m in _CLOCK.finditer(text or ""):
        at = today.replace(hour=int(m.group(1)), minute=int(m.group(2)), second=0, microsecond=0)
        if at.timestamp() > now + 60:
            at -= dt.timedelta(days=1)
        if at.timestamp() not in out:
            out.append(at.timestamp())
    return out


def moments(text: str, earlier_owner: Sequence[str], last_reply: str, now: float) -> List[float]:
    """The moments he asks to see: the times in his message, else in his last messages, else the times the
    assistant's last reply cited ("בשעות האלה": those are the reply's times first). [] when no time is given anywhere
    (the newest event is shown)."""
    sources = [text, *earlier_owner[:2], last_reply]
    if _THOSE.search(text or ""):
        sources = [text, last_reply, *earlier_owner[:2]]
    for source in sources:
        found = clock_times(source, now)
        if found:
            return found
    return []


def _covers(record: Dict[str, Any], at: float) -> float:
    """0 when the event spans *at*, else how far it is from it (seconds)."""
    start = float(record.get("start") or 0)
    end = float(record.get("end") or start)
    if start - EDGE_SEC <= at <= end + EDGE_SEC:
        return 0.0
    return min(abs(at - start), abs(at - end))


def pick_events(records: Sequence[Dict[str, Any]], at_times: Sequence[float], now: float) -> List[Tuple[Dict[str, Any], float]]:
    """(event, moment) for each moment: the event with people that spans it, else the nearest within NEAR_SEC.
    With no moment, the newest event with people. One event is shown once."""
    people = [r for r in records if int(r.get("people_max") or 0) > 0 and float(r.get("start") or 0) <= now]
    if not people:
        return []
    if not at_times:
        newest = max(people, key=lambda r: float(r.get("start") or 0))
        return [(newest, float(newest.get("start") or now))]
    out: List[Tuple[Dict[str, Any], float]] = []
    seen = set()
    for at in at_times:
        ranked = sorted(people, key=lambda r: (_covers(r, at), -float(r.get("start") or 0)))
        if _covers(ranked[0], at) > NEAR_SEC:
            continue
        # the nearest event, and each other camera's event that spans the moment too (no camera named: the owner
        # sees every place people were at that time)
        cameras = set()
        for record in ranked:
            if record is not ranked[0] and (_covers(record, at) > 0 or record.get("camera") in cameras):
                continue
            cameras.add(record.get("camera"))
            if record.get("event_id") in seen:
                continue
            seen.add(record.get("event_id"))
            start, end = float(record.get("start") or at), float(record.get("end") or at)
            out.append((record, min(max(at, start), end)))
    return out[:MAX_PHOTOS]


def _alert_ts(alert_id: str) -> float:
    m = re.search(r"_(\d{9,11})_", alert_id + "_")
    return float(m.group(1)) if m else 0.0


def observation_at(record: Dict[str, Any], at: float) -> Dict[str, Any]:
    obs = [o for o in record.get("observations") or [] if isinstance(o, dict)]
    with_people = [o for o in obs if int(o.get("people") or 0) > 0] or obs
    return min(with_people, key=lambda o: abs(float(o.get("ts") or 0) - at)) if with_people else {}


def _meta(roots: Sequence[str], alert_id: str) -> Tuple[str, Dict[str, Any]]:
    for root in roots:
        for path in glob.glob(os.path.join(root, "meta", "*", "*", alert_id + ".meta.json")):
            try:
                with open(path, encoding="utf-8") as f:
                    return root, json.load(f)
            except (OSError, ValueError):
                continue
    return "", {}


def _frame(clip: str, offset: float, out_path: str) -> bool:
    import cv2  # noqa: PLC0415

    cap = cv2.VideoCapture(clip)
    try:
        if not cap.isOpened():
            return False
        cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, offset) * 1000.0)
        ok, frame = cap.read()
        if not ok or frame is None:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = cap.read()
        return bool(ok and frame is not None and cv2.imwrite(out_path, frame))
    finally:
        cap.release()


def still_of(record: Dict[str, Any], at: float, roots: Sequence[str], work_dir: str) -> Tuple[str, float]:
    """(jpeg path, when it shows) for the event at *at*: a frame of the alert clip nearest *at*, at the moment its
    detector fired (someone is in it); else the event's keyframe. ("", 0) when the box has neither."""
    alert_ids = [a for a in record.get("alert_ids") or [] if a]
    for alert_id in sorted(alert_ids, key=lambda a: abs(_alert_ts(a) - at)):
        root, meta = _meta(roots, alert_id)
        clip = os.path.join(root, str(meta.get("clip_path") or "").replace("\\", os.sep)) if meta else ""
        if not (meta.get("clip_path") and os.path.isfile(clip)):
            continue
        start = float(meta.get("clip_start_ts") or 0) or _alert_ts(alert_id)
        trigger = float(meta.get("trigger_ts") or 0) or start + 3.0
        shown = trigger + 1.0
        if start <= at <= float(meta.get("clip_end_ts") or start) and at - trigger > 0:
            shown = at
        os.makedirs(work_dir, exist_ok=True)
        out = os.path.join(work_dir, f"{alert_id}_still_{int(shown)}.jpg")
        try:
            if os.path.isfile(out) or _frame(clip, shown - start, out):
                return out, shown
        except Exception as exc:  # noqa: BLE001 - the keyframe still answers
            log.warning("still of %s not made: %s", alert_id, exc)
    keyframe = str(record.get("keyframe") or "")
    if keyframe and os.path.isfile(keyframe):
        return keyframe, float(record.get("start") or at)
    return "", 0.0
