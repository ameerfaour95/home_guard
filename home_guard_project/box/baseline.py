"""What is usual at each camera: the historian (task 2.9 of the alert fix, 2026-10-08).

Owner: "Does the memory know about each camera individually? That something usually happens at the main entrance,
but if it happens in the backyard it's not usual?" and "knocking on the door is usual at the main door but not at
the back door". Code, never text in the Eye's prompt (Frigate: "what's normal here" text biases the model).

**The ledger** (``<state>/events/baseline.json``): per camera, per local day, how many person events were going on in
each hour (an event counts once in every hour it spans, so an all-day work crew counts at 13:00 too), which activity
tags were seen in which hour (activities.py), the event's dwell when known and the scene-map areas people went
through. One event per session (a rolled-over chain of event-book sessions is one event; imported clips closer than
SESSION_GAP_SEC on one camera are one event). False positives (the VLM saw nobody) are skipped. Days the source was
watching with no event are kept as zero days: ``days_of_data`` is how many days a camera has.

- Rebuilt nightly at ~03:30 local (:class:`NightlyBuild`, ticked by the guard loop) and on demand
  (``python -m home_guard_project.box.baseline build``), from the event archive (event_memory.py). The archive keeps
  30 days; the ledger keeps KEEP_DAYS, so older days stay.
- An offline importer (``import --raw <folder>``) adds history from production meta (``production_<site>``,
  ``*.meta.json`` with ``alert``) and data-collection meta (``dataset_<site>/meta``, ``yolo.class_counts``). Old
  site prefixes map to today's cameras by the ``_chN`` ending when exactly one current camera has that channel.
- A day from the archive is never overwritten by an import (archive > production > collector).

**surprise(camera, ts, tags)** combines three layers and says which one said what:

- *time*: the bucket is (weekday or weekend) x hour. ``seen`` = how many of the last WINDOW_DAYS same-type days had
  an event in that hour or the one before / after; ``expected_per_hour`` = the bucket's events per day, smoothed with
  the neighbouring hours (half weight) and one pseudo-day at the camera's own average (the add-one prior).
  ``rare`` needs MIN_DAYS of data, at least RARE_MIN_WINDOWS same-type days (else every day type is used), ``seen``
  <= RARE_MAX_SEEN and seen/windows <= RARE_MAX_SHARE (so 1 in 10+ or 0 in 7+). ``common``: seen/windows >=
  COMMON_SHARE or expected_per_hour >= COMMON_SHARE. Otherwise ``uncommon``; without MIN_DAYS ``unknown``.
- *activity*: per tag, on how many of the camera's last WINDOW_DAYS days it was seen. With MIN_DAYS of data, at most
  ACTIVITY_RARE_MAX events of that tag in the window is ``rare``, seen on ACTIVITY_COMMON_SHARE of the days is
  ``common``. Cameras where the tag is usual are named ("usually only at the main entrance").
- *facts* (camera_profiles.py): the owner's taught rules for this camera ("nobody at night", "family only") and
  other cameras' "only here" rules ("deliveries only at the main entrance") make it ``rare`` regardless of data.
- *house* (camera_profiles.house_profile): known people, what the family expects, the family's dog / kids and the
  camera's "usual" facts EXPLAIN an activity: then nothing is raised.

The decision (inference.py, events.decide ``baseline``) may only RAISE: a rare "normal" becomes a quiet message
with box.yaml ``baseline_alerts: on``; ``shadow`` (the default) only logs; a suspicious or an escalation is never
changed (it only gets the rarity line). Owner-facing text uses display names, never camera ids.
"""

from __future__ import annotations

import argparse
import datetime as dt
import glob
import json
import logging
import math
import os
import statistics
import sys
import threading
import time
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

from . import activities
from .camera_names import channel_of
from .camera_profiles import CameraProfiles, house_profile, propose_role, resolve_key, role_of, ROLE_NAMES

log = logging.getLogger("box.baseline")

BASELINE_NAME = "baseline.json"
VERSION = 1
MIN_DAYS = 14                    # box.yaml baseline_min_days: before this, rarity is "unknown" (facts still count)
WINDOW_DAYS = 30                 # the last 30 same-bucket days
KEEP_DAYS = 400
SESSION_GAP_SEC = 300.0          # imported clips of one camera closer than this are one event (alert cooldown 120 s)
MAX_EVENT_SEC = 86400.0
DEFAULT_WEEKEND = (4, 5)         # Friday and Saturday (Israel); box.yaml baseline_weekend_days
RARE_MAX_SEEN = 1
RARE_MIN_WINDOWS = 7
RARE_MAX_SHARE = 0.1
COMMON_SHARE = 0.3
ACTIVITY_RARE_MAX = 1            # events of this activity at this camera in the window
ACTIVITY_COMMON_SHARE = 0.2      # of the window's days
USUAL_AT_MIN_DAYS = 3            # another camera where the activity is usual: on at least this many days
NIGHTLY_AT = "03:30"
MODES = ("off", "shadow", "on")
SOURCE_RANK = {"collector": 1, "production": 2, "archive": 3}
# Day phases (fixed local hours; simple and the same all year).
PHASE_HOURS: Dict[str, Tuple[int, ...]] = {"late_night": tuple(range(0, 6)), "day": tuple(range(6, 17)),
                                            "evening": tuple(range(17, 21)), "night": tuple(range(21, 24))}
PHASE_NAMES = {"late_night": {"he": "לפנות בוקר (00-06)", "en": "late night (00-06)"},
               "day": {"he": "ביום (06-17)", "en": "by day (06-17)"},
               "evening": {"he": "בערב (17-21)", "en": "in the evening (17-21)"},
               "night": {"he": "בלילה (21-24)", "en": "at night (21-24)"}}
_PLURAL = {"late_night": ("הלילות", "nights"), "night": ("הלילות", "nights"), "evening": ("הערבים", "evenings"),
           "day": ("הימים", "days")}


def phase_of_hour(hour: int) -> str:
    for phase, hours in PHASE_HOURS.items():
        if hour in hours:
            return phase
    return "day"


def default_path() -> str:
    from . import paths  # noqa: PLC0415

    return os.path.join(paths.state_dir(), "events", BASELINE_NAME)


def mode_of(settings: Optional[Mapping[str, Any]]) -> str:
    """box.yaml ``baseline_alerts``: off | shadow | on (default shadow; anything else is shadow)."""
    value = (settings or {}).get("baseline_alerts", "shadow") if isinstance(settings, Mapping) else "shadow"
    if value is False:
        return "off"
    if value is True:
        return "on"
    value = str(value or "shadow").strip().lower()
    return value if value in MODES else "shadow"


def min_days_of(settings: Optional[Mapping[str, Any]]) -> int:
    try:
        return max(1, int((settings or {}).get("baseline_min_days", MIN_DAYS)))
    except (TypeError, ValueError):
        return MIN_DAYS


_DAY_NAMES = {"mon": 0, "tue": 1, "wed": 2, "thu": 3, "fri": 4, "sat": 5, "sun": 6}


def weekend_of(settings: Optional[Mapping[str, Any]]) -> Tuple[int, ...]:
    """box.yaml ``baseline_weekend_days``: [4, 5] or "fri,sat" (Python weekdays, Monday = 0)."""
    value = (settings or {}).get("baseline_weekend_days") if isinstance(settings, Mapping) else None
    if value in (None, "", []):
        return DEFAULT_WEEKEND
    items = value.split(",") if isinstance(value, str) else list(value) if isinstance(value, (list, tuple)) else []
    out = []
    for item in items:
        text = str(item).strip().lower()[:3]
        if text.isdigit() and 0 <= int(text) <= 6:
            out.append(int(text))
        elif text in _DAY_NAMES:
            out.append(_DAY_NAMES[text])
    return tuple(sorted(set(out))) or DEFAULT_WEEKEND


def remap(camera: str, cameras: Sequence[str]) -> Optional[str]:
    """Today's camera for a (possibly pre-rename) id: itself, else the one current camera with the same ``_chN``
    ending; None when none or two match. Without a list of current cameras the id is kept as it is."""
    camera = str(camera or "")
    if not cameras:
        return camera or None
    if camera in cameras:
        return camera
    ch = channel_of(camera)
    if ch is None:
        return None
    same = [c for c in cameras if channel_of(c) == ch]
    return same[0] if len(same) == 1 else None


# ---------- events from the sources ----------
def _date(ts: float) -> str:
    return dt.datetime.fromtimestamp(ts).strftime("%Y-%m-%d")


def _num(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _int(value: Any) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def events_from_archive(records: Iterable[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """One event per chain of archived sessions (a roll-over's ``parent`` links them), person events only."""
    rows = [dict(r) for r in records if isinstance(r, Mapping) and r.get("event_id")]
    by_id = {str(r["event_id"]): r for r in rows}

    def root(r: Dict[str, Any]) -> str:
        seen: Set[str] = set()
        while r.get("parent") and str(r["parent"]) in by_id and str(r["parent"]) not in seen:
            seen.add(str(r["parent"]))
            r = by_id[str(r["parent"])]
        return str(r["event_id"])

    chains: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        chains.setdefault(root(r), []).append(r)
    out = []
    for chain in chains.values():
        observations = [o for r in chain for o in r.get("observations") or [] if isinstance(o, dict)]
        entities = [e for r in chain for e in (r.get("entities") or []) if isinstance(e, dict)]
        person = (any(int(r.get("people_max") or 0) > 0 for r in chain)
                  or any(int(o.get("people") or 0) > 0 for o in observations)
                  or any(e.get("kind") == "person" for e in entities))
        if not person:
            continue                       # a car alone, or the AI did not answer: not a person event
        start = min(float(r.get("start") or 0.0) for r in chain)
        end = max(float(r.get("end") or r.get("start") or 0.0) for r in chain)
        if start <= 0:
            continue
        closed = all(r.get("outcome") != "open" for r in chain)
        areas = [a for e in entities if e.get("kind") == "person" and e.get("mapped")
                 for a in dict.fromkeys(e.get("path") or []) if a]
        texts = [(float(o.get("ts") or start), str(o.get("summary") or "")) for o in observations]
        texts += [(start, k) for r in chain for k in r.get("owner_known") or [] if k]
        out.append({"camera": str(chain[0].get("camera") or ""), "start": start, "end": max(start, end),
                    "dwell": (end - start) if closed and end > start else None, "texts": texts, "areas": areas})
    return out


def _meta_files(folder: str) -> List[str]:
    return sorted(glob.glob(os.path.join(folder, "**", "*.meta.json"), recursive=True))


def read_meta(path: str) -> Optional[Dict[str, Any]]:
    """One clip as a person observation: ``{camera, ts, end, text, session, source, person}`` or None (unreadable)."""
    try:
        with open(path, encoding="utf-8") as f:
            meta = json.load(f)
    except (OSError, ValueError):
        return None
    if not isinstance(meta, dict) or not meta.get("camera_name"):
        return None
    alert = meta.get("alert") if isinstance(meta.get("alert"), dict) else None
    yolo = meta.get("yolo") if isinstance(meta.get("yolo"), dict) else {}
    counts = yolo.get("class_counts") if isinstance(yolo.get("class_counts"), dict) else {}
    yolo_person = (_int(counts.get("person")) or 0) > 0
    ts = _num(meta.get("trigger_ts")) or _num(meta.get("clip_start_ts"))
    if ts is None:
        return None
    end = _num(meta.get("clip_end_ts")) or ts
    kind = str(meta.get("kind") or "")
    if alert is not None or kind in ("alert", "quiet"):
        source = "production"
        alert = alert or {}
        person = yolo_person
        people = _int(alert.get("people"))
        vlm = alert.get("vlm") if isinstance(alert.get("vlm"), dict) else {}
        if alert.get("false_positive") or _int(vlm.get("people")) == 0:
            person = False                 # the VLM looked and saw nobody to alert on
        elif people is not None:
            person = people > 0
        text = " ".join(str(alert.get(k) or "") for k in ("summary", "summary_owner"))
        event = alert.get("event") if isinstance(alert.get("event"), dict) else {}
        session = str(event.get("session_id") or "")
    else:
        source = "collector"
        person = yolo_person
        response = meta.get("model_response")
        text = response if isinstance(response, str) else ""
        session = ""
    return {"camera": str(meta["camera_name"]), "ts": ts, "end": max(ts, end), "text": text, "session": session,
            "source": source, "person": person}


def events_from_clips(clips: Sequence[Dict[str, Any]], gap: float = SESSION_GAP_SEC) -> List[Dict[str, Any]]:
    """Person clips grouped into events: by the event book's session id when the meta has one, else clips of one
    camera closer than *gap* seconds."""
    out: List[Dict[str, Any]] = []
    by_cam: Dict[str, List[Dict[str, Any]]] = {}
    for c in clips:
        if c.get("person"):
            by_cam.setdefault(c["camera"], []).append(c)
    for camera, rows in by_cam.items():
        rows.sort(key=lambda c: c["ts"])
        current: Optional[Dict[str, Any]] = None
        for c in rows:
            same = current is not None and (
                (c["session"] and c["session"] == current["session"])
                or (not (c["session"] and current["session"]) and c["ts"] - current["end"] <= gap))
            if not same:
                current = {"camera": camera, "start": c["ts"], "end": c["end"], "texts": [], "areas": [],
                           "session": c["session"], "clips": 0}
                out.append(current)
            current["end"] = max(current["end"], c["end"])
            current["clips"] += 1
            current["texts"].append((c["ts"], c["text"]))
    for e in out:
        e["dwell"] = (e["end"] - e["start"]) if e.pop("clips") > 1 else None   # one clip says nothing of the stay
        e.pop("session", None)
    return out


def _new_day(source: str) -> Dict[str, Any]:
    return {"src": source, "hours": [0] * 24, "tags": {}, "dwell": [], "areas": {}}


def ledger_days(events: Sequence[Dict[str, Any]], covered: Mapping[str, Set[str]], source: str,
                cameras: Sequence[str] = ()) -> Tuple[Dict[str, Dict[str, Dict[str, Any]]], Dict[str, int]]:
    """``({camera: {date: day}}, unmapped counts)`` for one source. *covered* names the days each camera was watched
    (zero days included); an event's days are added to it."""
    out: Dict[str, Dict[str, Dict[str, Any]]] = {}
    unmapped: Dict[str, int] = {}
    for camera, dates in covered.items():
        cam = remap(camera, cameras)
        if cam is None:
            continue
        for d in dates:
            out.setdefault(cam, {}).setdefault(d, _new_day(source))
    for e in events:
        cam = remap(e["camera"], cameras)
        if cam is None:
            unmapped[e["camera"]] = unmapped.get(e["camera"], 0) + 1
            continue
        days = out.setdefault(cam, {})
        start, end = float(e["start"]), min(float(e["end"]), float(e["start"]) + MAX_EVENT_SEC)
        hours: Set[Tuple[str, int]] = set()
        t = start
        while t <= end:
            moment = dt.datetime.fromtimestamp(t)
            hours.add((moment.strftime("%Y-%m-%d"), moment.hour))
            t = (moment.replace(minute=0, second=0, microsecond=0) + dt.timedelta(hours=1)).timestamp()
        for d, h in hours:
            days.setdefault(d, _new_day(source))["hours"][h] += 1
        tagged: Set[Tuple[str, int, str]] = set()
        for ts, text in e.get("texts") or []:
            moment = dt.datetime.fromtimestamp(float(ts) if ts else start)
            for tag in activities.tags_of(text):
                tagged.add((moment.strftime("%Y-%m-%d"), moment.hour, tag))
        for d, h, tag in tagged:
            row = days.setdefault(d, _new_day(source))["tags"].setdefault(tag, {})
            row[str(h)] = row.get(str(h), 0) + 1
        first = days.setdefault(_date(start), _new_day(source))
        if e.get("dwell"):
            first["dwell"].append(round(float(e["dwell"]), 1))
        for area in dict.fromkeys(e.get("areas") or []):
            first["areas"][area] = first["areas"].get(area, 0) + 1
    return out, unmapped


def _span(ts_values: Iterable[float], through: Optional[float] = None) -> Set[str]:
    values = [t for t in ts_values if t]
    if not values:
        return set()
    first = dt.datetime.fromtimestamp(min(values)).date()
    last = dt.datetime.fromtimestamp(max(values + ([through] if through else []))).date()
    out = set()
    while first <= last:
        out.add(first.isoformat())
        first += dt.timedelta(days=1)
    return out


# ---------- the ledger ----------
class Baseline:
    """The per-camera ledger in ``baseline.json`` and the questions asked of it. Thread-safe."""

    def __init__(self, path: Optional[str] = None, clock: Callable[[], float] = time.time) -> None:
        self.path = path or default_path()
        self.clock = clock
        self._lock = threading.RLock()
        self._cache: Optional[Tuple[Tuple[float, int], Dict[str, Any]]] = None
        self._memory: Optional[Dict[str, Any]] = None        # a ledger that is never written (the report)

    # -- file --
    def load(self) -> Dict[str, Any]:
        with self._lock:
            if self._memory is not None:
                return self._memory
            try:
                st = os.stat(self.path)
            except OSError:
                return {"version": VERSION, "cameras": {}}
            key = (st.st_mtime, st.st_size)
            if self._cache is not None and self._cache[0] == key:
                return self._cache[1]
            try:
                with open(self.path, encoding="utf-8") as f:
                    data = json.load(f)
            except (OSError, ValueError) as exc:
                log.warning("baseline not read (%s)", exc)
                data = {}
            if not isinstance(data, dict) or not isinstance(data.get("cameras"), dict):
                data = {"version": VERSION, "cameras": {}}
            self._cache = (key, data)
            return data

    def _save(self, data: Dict[str, Any]) -> None:
        if self._memory is not None:
            self._memory = data
            return
        os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, separators=(",", ":"))
        os.replace(tmp, self.path)
        self._cache = None

    @classmethod
    def in_memory(cls, clock: Callable[[], float] = time.time) -> "Baseline":
        b = cls(path=os.devnull, clock=clock)
        b._memory = {"version": VERSION, "cameras": {}}
        return b

    # -- building --
    def merge(self, days: Mapping[str, Mapping[str, Dict[str, Any]]], source: str, cameras: Sequence[str] = (),
              now: Optional[float] = None, unmapped: Optional[Mapping[str, int]] = None) -> Dict[str, Any]:
        """Add one source's days. A day is replaced only by a source of the same or a higher rank. Old ids are moved
        to today's cameras (``_chN``); days older than KEEP_DAYS are dropped."""
        now = float(self.clock()) if now is None else float(now)
        rank = SOURCE_RANK.get(source, 0)
        with self._lock:
            data = json.loads(json.dumps(self.load()))
            cams: Dict[str, Any] = data.setdefault("cameras", {})
            if cameras:
                for old in [k for k in cams if k not in cameras]:
                    new = remap(old, cameras)
                    if new and new != old:
                        target = cams.setdefault(new, {"days": {}})
                        for d, day in (cams[old].get("days") or {}).items():
                            target["days"].setdefault(d, day)
                        del cams[old]
            replaced = added = 0
            for cam, cam_days in days.items():
                stored = cams.setdefault(cam, {"days": {}}).setdefault("days", {})
                for d, day in cam_days.items():
                    old = stored.get(d)
                    if old is None:
                        added += 1
                    elif SOURCE_RANK.get(str(old.get("src")), 0) > rank:
                        continue
                    else:
                        replaced += 1
                    stored[d] = day
            cutoff = (dt.datetime.fromtimestamp(now).date() - dt.timedelta(days=KEEP_DAYS)).isoformat()
            for cam in cams.values():
                cam["days"] = {d: v for d, v in sorted((cam.get("days") or {}).items()) if d >= cutoff}
            data["version"] = VERSION
            data["built"] = now
            data["built_local"] = dt.datetime.fromtimestamp(now).isoformat(timespec="seconds")
            sources = data.setdefault("sources", {})
            sources[source] = {"at": now, "days_added": added, "days_replaced": replaced}
            if unmapped:
                data.setdefault("unmapped", {}).update({k: int(v) for k, v in unmapped.items()})
            self._save(data)
            return {"source": source, "days_added": added, "days_replaced": replaced,
                    "cameras": sorted(days), "unmapped": dict(unmapped or {})}

    def build_from_archive(self, records: Iterable[Mapping[str, Any]], cameras: Sequence[str] = (),
                           now: Optional[float] = None) -> Dict[str, Any]:
        """The nightly rebuild: every day from the archive's first record through yesterday, for every camera."""
        now = float(self.clock()) if now is None else float(now)
        rows = [r for r in records if isinstance(r, Mapping)]
        events = events_from_archive(rows)
        yesterday = (dt.datetime.fromtimestamp(now).replace(hour=12) - dt.timedelta(days=1)).timestamp()
        starts = [float(r.get("start") or 0) for r in rows if float(r.get("start") or 0) < now]
        span = {d for d in _span(starts, through=yesterday) if d < _date(now)} if starts else set()
        names = set(cameras) | {str(r.get("camera") or "") for r in rows if r.get("camera")}
        days, unmapped = ledger_days([e for e in events if _date(e["start"]) < _date(now)],
                                     {c: span for c in names}, "archive", cameras)
        return self.merge(days, "archive", cameras, now, unmapped)

    def import_raw(self, folder: str, cameras: Sequence[str] = (), now: Optional[float] = None) -> Dict[str, Any]:
        """History from saved meta under *folder* (production and data-collection). Each source covers the days from
        its first to its last clip; days the archive already has are left alone."""
        clips = [c for c in (read_meta(p) for p in _meta_files(folder)) if c is not None]
        results = {}
        for source in ("collector", "production"):
            mine = [c for c in clips if c["source"] == source]
            if not mine:
                continue
            span_by_cam: Dict[str, Set[str]] = {}
            for c in mine:
                span_by_cam.setdefault(c["camera"], set())
            # A source watched every camera it has from its first to its last day (zero days count).
            span = _span(c["ts"] for c in mine)
            days, unmapped = ledger_days(events_from_clips(mine), {c: span for c in span_by_cam}, source, cameras)
            results[source] = self.merge(days, source, cameras, now, unmapped)
            results[source]["clips"] = len(mine)
        return results

    # -- reading --
    def camera_days(self, camera: str) -> Tuple[str, Dict[str, Dict[str, Any]]]:
        cams = self.load().get("cameras") or {}
        key = resolve_key(camera, list(cams)) or ""
        return key, dict((cams.get(key) or {}).get("days") or {}) if key else {}

    def cameras(self) -> List[str]:
        return sorted(self.load().get("cameras") or {})


# ---------- the questions ----------
def _hours(day: Mapping[str, Any]) -> List[int]:
    hours = day.get("hours") if isinstance(day, Mapping) else None
    return [int(x or 0) for x in hours] if isinstance(hours, list) and len(hours) == 24 else [0] * 24


def _tag_count(day: Mapping[str, Any], tag: str, hours: Optional[Iterable[int]] = None) -> int:
    row = (day.get("tags") or {}).get(tag) if isinstance(day, Mapping) else None
    if not isinstance(row, Mapping):
        return 0
    if hours is None:
        return sum(int(v or 0) for v in row.values())
    wanted = {str(h) for h in hours}
    return sum(int(v or 0) for k, v in row.items() if k in wanted)


def _is_weekend(date: str, weekend: Sequence[int]) -> bool:
    return dt.date.fromisoformat(date).weekday() in weekend


def _in(name: str) -> str:
    """Hebrew "in <name>": ב + הכניסה -> בכניסה."""
    name = str(name or "").strip()
    return f"ב{name[1:]}" if name.startswith("ה") and len(name) > 2 else f"ב{name}"


def _default_names(camera: str, lang: str) -> str:
    try:
        from .camera_names import display_name  # noqa: PLC0415

        return display_name(camera, lang)
    except Exception:  # noqa: BLE001
        ch = channel_of(camera)
        return (f"מצלמה {ch}" if lang == "he" else f"Camera {ch}") if ch else "this camera"


def _times(count: int, lang: str) -> str:
    if lang == "he":
        return "אף פעם" if count == 0 else "פעם אחת" if count == 1 else f"{count} פעמים"
    return "never" if count == 0 else "once" if count == 1 else f"{count} times"


def _per_week(share: float, lang: str) -> str:
    """How often per week a day with this happens: "בערך 3 פעמים בשבוע"."""
    n = share * 7.0
    if lang == "he":
        if n >= 6.5:
            return "כמעט כל יום"
        if n >= 1.5:
            return f"בערך {round(n)} פעמים בשבוע"
        if n >= 0.75:
            return "בערך פעם בשבוע"
        return "פחות מפעם בשבוע"
    if n >= 6.5:
        return "almost every day"
    if n >= 1.5:
        return f"about {round(n)} times a week"
    if n >= 0.75:
        return "about once a week"
    return "less than once a week"


def _rate_text(events: int, days: int, lang: str) -> str:
    """How often an activity happens: "בערך פעם ביום", "בערך 2 פעמים בשבוע", "אף פעם בחודש האחרון"."""
    if days <= 0:
        return "אין עדיין נתונים" if lang == "he" else "no data yet"
    per_day = events / days
    he = lang == "he"
    if events == 0:
        return f"אף פעם ב-{days} הימים האחרונים" if he else f"never in the last {days} days"
    if per_day >= 1.5:
        return f"כמה פעמים ביום (בערך {per_day:.0f})" if he else f"several times a day (about {per_day:.0f})"
    if per_day >= 0.75:
        return "בערך פעם ביום" if he else "about once a day"
    if per_day * 7 >= 1.5:
        return f"בערך {round(per_day * 7)} פעמים בשבוע" if he else f"about {round(per_day * 7)} times a week"
    return (f"{_times(events, 'he')} ב-{days} הימים האחרונים" if he
            else f"{_times(events, 'en')} in the last {days} days")


def time_layer(days: Mapping[str, Mapping[str, Any]], ts: float, min_days: int = MIN_DAYS,
               weekend: Sequence[int] = DEFAULT_WEEKEND) -> Dict[str, Any]:
    """The hour-of-the-week layer (see the module docstring for the thresholds)."""
    moment = dt.datetime.fromtimestamp(ts)
    today, hour = moment.strftime("%Y-%m-%d"), moment.hour
    before = sorted(d for d in days if d < today)
    kind = "weekend" if moment.weekday() in weekend else "weekday"
    windows = [d for d in before if _is_weekend(d, weekend) == (kind == "weekend")][-WINDOW_DAYS:]
    bucket = kind
    if len(windows) < RARE_MIN_WINDOWS:
        windows, bucket = before[-WINDOW_DAYS:], "all days"
    n = len(windows)
    near = ((hour - 1) % 24, hour, (hour + 1) % 24)
    seen = sum(1 for d in windows if any(_hours(days[d])[h] > 0 for h in near))
    exact = sum(1 for d in windows if _hours(days[d])[hour] > 0)
    smoothed = sum(_hours(days[d])[hour] + 0.5 * (_hours(days[d])[near[0]] + _hours(days[d])[near[2]])
                   for d in windows) / 2.0
    total = sum(sum(_hours(days[d])) for d in windows)
    mean = total / (24.0 * n) if n else 0.0
    expected = (smoothed + mean) / (n + 1.0)                 # one pseudo-day at the camera's own average
    share = seen / n if n else 0.0
    if len(before) < min_days:
        rarity = "unknown"
    elif n >= RARE_MIN_WINDOWS and seen <= RARE_MAX_SEEN and share <= RARE_MAX_SHARE:
        rarity = "rare"
    elif share >= COMMON_SHARE or expected >= COMMON_SHARE:
        rarity = "common"
    else:
        rarity = "uncommon"
    phase = phase_of_hour(hour)
    plural_he, plural_en = _PLURAL[phase]
    if bucket == "weekend":
        plural_he, plural_en = "סופי השבוע", "weekend days"
    if rarity == "rare":
        text_he = f"לא רגיל למצלמה הזו בשעה הזו ({_times(seen, 'he')} ב-{n} {plural_he} האחרונים)"
        text_en = f"Not usual for this camera at this hour ({_times(seen, 'en')} in the last {n} {plural_en})"
    elif rarity == "unknown":
        text_he = f"עדיין אין מספיק היסטוריה למצלמה הזו ({len(before)} ימים מתוך {min_days})"
        text_en = f"Not enough history for this camera yet ({len(before)} of {min_days} days)"
    else:
        text_he = f"זה קורה כאן {_per_week(exact / n if n else 0.0, 'he')} בשעה הזו"
        text_en = f"This happens here {_per_week(exact / n if n else 0.0, 'en')} at this hour"
    return {"rarity": rarity, "expected_per_hour": round(expected, 3), "seen_in_last_30_days_same_bucket": seen,
            "windows": n, "bucket": bucket, "hour": hour, "phase": phase, "days_of_data": len(before),
            "text_he": text_he, "text_en": text_en}


def _window(days: Mapping[str, Any], today: str) -> List[str]:
    return sorted(d for d in days if d < today)[-WINDOW_DAYS:]


def activity_layer(camera_key: str, all_days: Mapping[str, Mapping[str, Mapping[str, Any]]], ts: float,
                   tags: Sequence[str], names: Callable[[str, str], str], min_days: int = MIN_DAYS) -> Dict[str, Any]:
    """Which of *tags* is rare for THIS camera, and where it is usual instead."""
    today = _date(ts)
    mine = all_days.get(camera_key) or {}
    window = _window(mine, today)
    enough = len([d for d in mine if d < today]) >= min_days
    rows = []
    for tag in [t for t in tags if t in activities.TAGS]:
        events = sum(_tag_count(mine[d], tag) for d in window)
        days_with = sum(1 for d in window if _tag_count(mine[d], tag) > 0)
        usual_at = []
        for other, other_days in all_days.items():
            if other == camera_key:
                continue
            ow = _window(other_days, today)
            n_with = sum(1 for d in ow if _tag_count(other_days[d], tag) > 0)
            if ow and n_with >= max(USUAL_AT_MIN_DAYS, ACTIVITY_COMMON_SHARE * len(ow)):
                usual_at.append(other)
        if not enough:
            rarity = "unknown"
        elif events <= ACTIVITY_RARE_MAX:
            rarity = "rare"
        elif window and days_with / len(window) >= ACTIVITY_COMMON_SHARE:
            rarity = "common"
        else:
            rarity = "uncommon"
        rows.append({"tag": tag, "rarity": rarity, "events": events, "days_with": days_with, "window": len(window),
                     "usual_at": usual_at})
    order = {"rare": 0, "uncommon": 1, "common": 2, "unknown": 3}
    rows.sort(key=lambda r: (order[r["rarity"]], -len(r["usual_at"])))
    top = rows[0] if rows else None
    out: Dict[str, Any] = {"rarity": top["rarity"] if top else "unknown", "tags": rows}
    if top:
        out.update(tag=top["tag"], usual_at=[names(c, "he") for c in top["usual_at"]],
                   usual_at_en=[names(c, "en") for c in top["usual_at"]])
        here_he, here_en = names(camera_key, "he"), names(camera_key, "en")
        what_he, what_en = activities.name_at(top["tag"], here_he, "he"), activities.name_at(top["tag"], here_en, "en")
        if top["rarity"] == "rare":
            if top["usual_at"]:
                where_he = " או ".join(_in(n) for n in out["usual_at"][:2])
                where_en = " or ".join(out["usual_at_en"][:2])
                out["text_he"] = f"{what_he} {_in(here_he)} - לא רגיל למצלמה הזו (בדרך כלל רק {where_he})"
                out["text_en"] = f"{what_en} at {here_en} - not usual for this camera (usually only at {where_en})"
            else:
                out["text_he"] = (f"{what_he} {_in(here_he)} - לא רגיל למצלמה הזו ({_times(top['events'], 'he')} "
                                  f"ב-{top['window']} הימים האחרונים)")
                out["text_en"] = (f"{what_en} at {here_en} - not usual for this camera ({_times(top['events'], 'en')} "
                                  f"in the last {top['window']} days)")
        else:
            out["text_he"] = f"{what_he} {_in(here_he)}: {_rate_text(top['events'], top['window'], 'he')}"
            out["text_en"] = f"{what_en} at {here_en}: {_rate_text(top['events'], top['window'], 'en')}"
    return out


def _fact_applies(rule: Mapping[str, Any], tags: Sequence[str], phase: str) -> bool:
    phases = rule.get("phases") or []
    if phases and phase not in phases:
        return False
    rule_tags = set(rule.get("tags") or [])
    return not rule_tags or bool(rule_tags & set(tags))


def fact_layer(camera: str, ts: float, tags: Sequence[str], facts: Mapping[str, Sequence[Mapping[str, Any]]],
               names: Callable[[str, str], str], house_state: str = "") -> Dict[str, Any]:
    """The owner's taught rules: this camera's (nobody / family only) and other cameras' "only here"."""
    phase = phase_of_hour(dt.datetime.fromtimestamp(ts).hour)
    key = resolve_key(camera, list(facts)) or ""
    here_he, here_en = names(camera, "he"), names(camera, "en")
    explains: List[str] = []
    for fact in facts.get(key, ()) if key else ():
        rule = fact.get("rule") or {}
        text = str(fact.get("text") or "")
        kind = rule.get("kind")
        if kind == "nobody" and _fact_applies(rule, tags, phase):
            return {"rarity": "rare", "fact": text, "kind": kind,
                    "text_he": f"לא רגיל {_in(here_he)}: לימדת אותי \"{text}\"",
                    "text_en": f"Not usual at {here_en}: you told me \"{text}\""}
        if kind == "family_only" and (set(tags) & {"knock", "delivery"} or house_state in ("away", "vacation")):
            return {"rarity": "rare", "fact": text, "kind": kind,
                    "text_he": f"לא רגיל {_in(here_he)}: לימדת אותי \"{text}\"",
                    "text_en": f"Not usual at {here_en}: you told me \"{text}\""}
        if kind in ("usual", "only_here") and set(rule.get("tags") or []) & set(tags):
            explains.append(text)
    for other, rows in facts.items():
        if other == key:
            continue
        for fact in rows:
            rule = fact.get("rule") or {}
            hit = set(rule.get("tags") or []) & set(tags)
            if rule.get("kind") == "only_here" and hit:
                tag = sorted(hit, key=activities.TAGS.index)[0]
                there_he, there_en = names(other, "he"), names(other, "en")
                return {"rarity": "rare", "fact": str(fact.get("text") or ""), "kind": "only_elsewhere",
                        "tag": tag, "usual_at": [there_he],
                        "text_he": f"{activities.name_at(tag, here_he, 'he')} {_in(here_he)} - לא רגיל למצלמה הזו "
                                   f"(בדרך כלל רק {_in(there_he)})",
                        "text_en": f"{activities.name_at(tag, here_en, 'en')} at {here_en} - not usual for this camera "
                                   f"(usually only at {there_en})"}
    return {"rarity": "", "explains": explains}


def house_layer(camera: str, ts: float, tags: Sequence[str], house: Mapping[str, Any],
                usual_facts: Sequence[str] = ()) -> Dict[str, Any]:
    """What the house memory knows that explains this: known people, what the family expects, the family's dog or
    kids, the camera's own "usual" facts."""
    reasons: List[str] = []
    ch = channel_of(camera)
    for k in house.get("known") or []:
        cam = str(k.get("camera") or "")
        if not cam or cam == camera or (ch is not None and channel_of(cam) == ch):
            reasons.append(f"known: {k.get('who')}")
    for e in house.get("expecting") or []:
        cam = str(e.get("camera") or "")
        if not cam or cam == camera or (ch is not None and channel_of(cam) == ch):
            reasons.append(f"expecting: {e.get('text')}")
    household = house.get("household") or {}
    tagset = set(tags)
    if (household.get("dog") or household.get("cat")) and tagset and tagset <= {"animal"}:
        reasons.append("the family's pet")
    if household.get("kids") and "kids" in tagset and tagset <= {"kids", "walk_past", "standing_talking",
                                                                 "enter_leave"}:
        reasons.append("the family's kids")
    reasons += [f"usual here: {f}" for f in usual_facts]
    return {"explains": reasons, "state": str(house.get("state") or "")}


class Historian:
    """The two layers together for the guard and the assistant: the ledger, the camera personalities and the house
    memory. Every answer is computed in code; nothing here is shown to the Eye."""

    def __init__(self, baseline: Optional[Baseline] = None, profiles: Optional[CameraProfiles] = None,
                 names: Optional[Callable[[str, str], str]] = None,
                 house: Optional[Callable[[float], Mapping[str, Any]]] = None,
                 settings: Optional[Mapping[str, Any]] = None) -> None:
        self.baseline = baseline or Baseline()
        self.profiles = profiles
        self.names = names or _default_names
        self.house = house
        self.settings = dict(settings or {})

    def _house(self, ts: float) -> Mapping[str, Any]:
        if self.house is not None:
            try:
                return self.house(ts) or {}
            except Exception as exc:  # noqa: BLE001
                log.debug("house memory not read: %s", exc)
                return {}
        events_dir = os.path.dirname(os.path.abspath(self.baseline.path))
        return house_profile(now=ts, events_dir=events_dir, profiles=self.profiles)

    def _facts(self) -> Dict[str, List[Dict[str, Any]]]:
        if self.profiles is None:
            return {}
        try:
            return self.profiles.all_facts()
        except Exception as exc:  # noqa: BLE001
            log.debug("camera facts not read: %s", exc)
            return {}

    def surprise(self, camera: str, ts: float, tags: Sequence[str] = ()) -> Dict[str, Any]:
        """How unusual an event at *camera* at *ts* with activity *tags* is; which layer said what. Never raises."""
        try:
            return self._surprise(camera, float(ts), list(tags or ()))
        except Exception as exc:  # noqa: BLE001 - the baseline only adds; it never stops an alert
            log.warning("[%s] baseline surprise failed: %s", camera, exc)
            return {"rarity": "unknown", "raise": False, "text_he": "", "text_en": "", "error": str(exc)}

    def _surprise(self, camera: str, ts: float, tags: List[str]) -> Dict[str, Any]:
        min_days = min_days_of(self.settings)
        weekend = weekend_of(self.settings)
        data = self.baseline.load().get("cameras") or {}
        key = resolve_key(camera, list(data)) or camera
        all_days = {cam: dict((v or {}).get("days") or {}) for cam, v in data.items()}
        days = all_days.get(key, {})
        names = lambda cam, lang: self.names(camera if cam == key else cam, lang)  # noqa: E731
        when = time_layer(days, ts, min_days, weekend)
        what = activity_layer(key, all_days, ts, tags, names, min_days) if tags else {"rarity": "", "tags": []}
        house = self._house(ts)
        facts = fact_layer(camera, ts, tags, self._facts(), names, str(house.get("state") or ""))
        usual = list(facts.pop("explains", []) or [])
        why = house_layer(camera, ts, tags, house, usual)
        rarities = [x for x in (facts.get("rarity"), what.get("rarity"), when["rarity"]) if x]
        if "rare" in rarities:
            rarity = "rare"
        elif "uncommon" in rarities:
            rarity = "uncommon"
        elif "common" in rarities:
            rarity = "common"
        else:
            rarity = "unknown"
        said_by = ("facts" if facts.get("rarity") == "rare" else "activity" if what.get("rarity") == "rare"
                   else "time" if when["rarity"] == "rare" else "")
        explained = bool(why["explains"])
        primary = {"facts": facts, "activity": what, "time": when}.get(said_by) or (
            what if what.get("text_he") and what.get("rarity") in ("common", "uncommon") else when)
        out = {
            "camera": camera,
            "expected_per_hour": when["expected_per_hour"],
            "seen_in_last_30_days_same_bucket": when["seen_in_last_30_days_same_bucket"],
            "windows": when["windows"],
            "bucket": when["bucket"],
            "rarity": rarity,
            "days_of_data": when["days_of_data"],
            "enough_data": when["days_of_data"] >= min_days,
            "tags": tags,
            "said_by": said_by,
            "explained_by": why["explains"],
            "raise": rarity == "rare" and not explained,
            "text_he": primary.get("text_he", ""),
            "text_en": primary.get("text_en", ""),
            "layers": {"time": when, "activity": what, "facts": facts, "house": why},
        }
        return out

    # -- for the assistant (how_usual) --
    def describe(self, camera: str, ts: Optional[float] = None, phase: str = "", tag: str = "",
                 lang: str = "he") -> Dict[str, Any]:
        """What is usual at *camera*: days of data, busiest and quietest hours, each part of the day, each activity
        (and where else it happens), typical stay and areas, the role and the owner's facts. Owner names only."""
        now = float(self.baseline.clock()) if ts is None else float(ts)
        today = _date(now)
        he = lang == "he"
        data = self.baseline.load().get("cameras") or {}
        key = resolve_key(camera, list(data)) or camera
        all_days = {cam: dict((v or {}).get("days") or {}) for cam, v in data.items()}
        days = all_days.get(key, {})
        window = _window(days, today)
        n = len(window)
        name = self.names(camera, lang)
        by_hour = [sum(_hours(days[d])[h] for d in window) for h in range(24)]
        busiest = [h for h in sorted(range(24), key=lambda h: (-by_hour[h], h)) if by_hour[h] > 0][:3]
        quietest = [h for h in sorted(range(24), key=lambda h: (by_hour[h], h))][:3]
        phases = {}
        for p, hours in PHASE_HOURS.items():
            with_event = sum(1 for d in window if any(_hours(days[d])[h] > 0 for h in hours))
            phases[p] = {"name": PHASE_NAMES[p]["he" if he else "en"], "days_with_people": with_event,
                         "often": _rate_text(with_event, n, lang) if n else ("אין עדיין נתונים" if he else "no data")}
        tag_rows = {}
        for t in activities.TAGS:
            events = sum(_tag_count(days[d], t) for d in window)
            if events or t == tag:
                tag_rows[t] = {"name": activities.name(t, lang), "events": events,
                               "often": _rate_text(events, n, lang)}
        dwell = [float(x) for d in window for x in days[d].get("dwell") or []]
        areas: Dict[str, int] = {}
        for d in window:
            for a, c in (days[d].get("areas") or {}).items():
                areas[a] = areas.get(a, 0) + int(c or 0)
        tag_days = {t: sum(1 for d in window if _tag_count(days[d], t) > 0) for t in activities.TAGS}
        role, role_from = role_of(camera, self.settings, self.profiles, propose_role(tag_days, n))
        facts = self.profiles.facts(camera) if self.profiles is not None else []
        out: Dict[str, Any] = {
            "camera": name, "days_of_data": len([d for d in days if d < today]), "window_days": n,
            "enough_data": len([d for d in days if d < today]) >= min_days_of(self.settings),
            "busiest_hours": [f"{h:02d}:00-{(h + 1) % 24:02d}:00" for h in busiest],
            "quietest_hours": [f"{h:02d}:00-{(h + 1) % 24:02d}:00" for h in quietest],
            "parts_of_day": phases,
            "activities": tag_rows,
            "typical_stay_min": round(statistics.median(dwell) / 60.0, 1) if dwell else None,
            "long_stay_min_p90": round(sorted(dwell)[max(0, math.ceil(0.9 * len(dwell)) - 1)] / 60.0, 1)
            if dwell else None,
            "common_areas": [a for a, _ in sorted(areas.items(), key=lambda x: -x[1])[:3]],
            "role": ROLE_NAMES[role]["he" if he else "en"] if role else "",
            "role_from": {"setup": "setup", "owner": "the owner", "map": "the scene map",
                          "proposed": "proposed from the statistics (not confirmed)"}.get(role_from, ""),
            "owner_facts": [str(f.get("text") or "") for f in facts],
        }
        if phase in PHASE_HOURS:
            out["asked_part_of_day"] = phases[phase]
        if tag in activities.TAGS:
            elsewhere = {}
            for other, other_days in all_days.items():
                if other == key:
                    continue
                ow = _window(other_days, today)
                events = sum(_tag_count(other_days[d], tag) for d in ow)
                elsewhere[self.names(other, lang)] = _rate_text(events, len(ow), lang)
            out["asked_activity"] = dict(tag_rows.get(tag) or {}, elsewhere=elsewhere)
        if ts is not None:
            moment = self.surprise(camera, now, [tag] if tag in activities.TAGS else [])
            out["at_that_time"] = {"rarity": moment["rarity"],
                                   "says": moment["text_he"] if he else moment["text_en"]}
        return out


# ---------- the box: nightly build ----------
def build_for_box(events_dir: str, cameras: Sequence[str] = (), now: Optional[float] = None) -> Dict[str, Any]:
    """Rebuild the box's baseline from its event archive."""
    from .event_memory import memory_for  # noqa: PLC0415

    records = memory_for(events_dir).records()
    return Baseline(os.path.join(events_dir, BASELINE_NAME)).build_from_archive(records, cameras, now)


def _seconds_until(at: str, now: float) -> float:
    hour, minute = (int(x) for x in at.split(":"))
    moment = dt.datetime.fromtimestamp(now)
    target = moment.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if target <= moment:
        target += dt.timedelta(days=1)
    return max(1.0, target.timestamp() - now)


class NightlyBuild:
    """The nightly rebuild, driven by the guard loop's tick (no thread of its own waits): due at once when the
    baseline is missing or older than a day, then every night at *at* (~03:30 local). The build itself runs in a
    short daemon thread so the tick never waits for it; one at a time."""

    def __init__(self, events_dir: str, cameras: Sequence[str] = (), at: str = NIGHTLY_AT,
                 now: Optional[float] = None, builder: Optional[Callable[[], Dict[str, Any]]] = None) -> None:
        self.events_dir = events_dir
        self.cameras = list(cameras)
        self.at = at
        self.builder = builder or (lambda: build_for_box(self.events_dir, self.cameras))
        self._running = threading.Lock()
        now = time.time() if now is None else float(now)
        try:
            stale = now - os.path.getmtime(os.path.join(events_dir, BASELINE_NAME)) > 26 * 3600
        except OSError:
            stale = True
        self.next = now if stale else now + _seconds_until(at, now)

    def build(self) -> None:
        if not self._running.acquire(blocking=False):
            return
        try:
            result = self.builder()
            log.info("baseline rebuilt: %d day(s) added, %d replaced", result.get("days_added", 0),
                     result.get("days_replaced", 0))
        except Exception as exc:  # noqa: BLE001 - the baseline only adds; the guard goes on without it
            log.warning("baseline not rebuilt: %s", exc)
        finally:
            self._running.release()

    def tick(self, now: float) -> bool:
        """Start the build when it is due; True when one was started. Never raises."""
        if now < self.next:
            return False
        self.next = now + _seconds_until(self.at, now)
        try:
            threading.Thread(target=self.build, name="baseline-build", daemon=True).start()
            return True
        except Exception as exc:  # noqa: BLE001
            log.warning("baseline build not started: %s", exc)
            return False


# ---------- the lead's sanity check ----------
def report(folder: str, cameras: Sequence[str] = (), now: Optional[float] = None, out: Any = None) -> Dict[str, Any]:
    """Import *folder* into a ledger that is never written and print, per camera, the days of data, the busiest
    and quietest hours, each part of the day and the top activity tags."""
    out = out or sys.stdout
    clips = [c for c in (read_meta(p) for p in _meta_files(folder)) if c is not None]
    last = max([c["end"] for c in clips] or [time.time()])
    now = float(now) if now is not None else (dt.datetime.fromtimestamp(last).replace(hour=0, minute=0, second=0)
                                              + dt.timedelta(days=1)).timestamp()
    ledger = Baseline.in_memory(clock=lambda: now)
    results = ledger.import_raw(folder, cameras, now)
    historian = Historian(ledger, names=lambda cam, lang: cam)
    summary: Dict[str, Any] = {"imports": results, "cameras": {}}
    print(f"baseline report for {folder} ({len(clips)} clips; person clips: {sum(1 for c in clips if c['person'])})",
          file=out)
    for source, r in results.items():
        if r.get("unmapped"):
            print(f"  {source}: not mapped to a current camera: {r['unmapped']}", file=out)
    for cam in ledger.cameras():
        d = historian.describe(cam, None, lang="en")
        _, days = ledger.camera_days(cam)
        by_hour = [sum(_hours(v)[h] for v in days.values()) for h in range(24)]
        tag_totals = {t: sum(_tag_count(v, t) for v in days.values()) for t in activities.TAGS}
        top_tags = [(t, n) for t, n in sorted(tag_totals.items(), key=lambda x: -x[1]) if n][:4]
        phase_counts = {p: sum(by_hour[h] for h in hours) for p, hours in PHASE_HOURS.items()}
        summary["cameras"][cam] = {"days_of_data": d["days_of_data"], "busiest_hours": d["busiest_hours"],
                                   "phase_event_hours": phase_counts, "top_tags": top_tags,
                                   "event_hours": sum(by_hour)}
        print(f"\n{cam}: {d['days_of_data']} day(s) of data, {sum(by_hour)} event-hours", file=out)
        print(f"  busiest hours: {', '.join(d['busiest_hours']) or '-'}", file=out)
        print(f"  quietest hours: {', '.join(d['quietest_hours'])}", file=out)
        print("  by part of day (event-hours): " + ", ".join(f"{p} {n}" for p, n in phase_counts.items()), file=out)
        print("  hours 00-23: " + " ".join(str(x) for x in by_hour), file=out)
        print("  top activities: " + (", ".join(f"{t} {n}" for t, n in top_tags) or "-"), file=out)
        if d.get("typical_stay_min") is not None:
            print(f"  typical stay: {d['typical_stay_min']} min (p90 {d['long_stay_min_p90']} min)", file=out)
    return summary


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m home_guard_project.box.baseline",
                                     description="The per-camera baseline (what is usual at each camera).")
    sub = parser.add_subparsers(dest="command", required=True)
    rep = sub.add_parser("report", help="print what is usual per camera for a folder of saved meta (no writes)")
    rep.add_argument("--raw", required=True, help="a production_<site> or dataset_<site> folder, or their parent")
    rep.add_argument("--cameras", default="", help="today's camera ids, comma separated, to map old ids by _chN")
    sub.add_parser("build", help="rebuild the box's baseline from its event archive now")
    imp = sub.add_parser("import", help="add a folder's history to the box's baseline")
    imp.add_argument("--raw", required=True)
    imp.add_argument("--cameras", default="")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    cameras = [c.strip() for c in getattr(args, "cameras", "").split(",") if c.strip()]
    if args.command == "report":
        report(args.raw, cameras)
        return 0
    from . import paths  # noqa: PLC0415

    events_dir = os.path.join(paths.state_dir(), "events")
    if not cameras:
        try:
            from .alert_settings import camera_names  # noqa: PLC0415

            cameras = list(camera_names())
        except Exception:  # noqa: BLE001
            cameras = []
    if args.command == "build":
        print(json.dumps(build_for_box(events_dir, cameras), ensure_ascii=False, indent=1))
    else:
        result = Baseline(os.path.join(events_dir, BASELINE_NAME)).import_raw(args.raw, cameras)
        print(json.dumps(result, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
