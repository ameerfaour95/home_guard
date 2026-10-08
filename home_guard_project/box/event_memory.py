"""The event memory: a long-term archive of the box's events, and a search over it (stage 2b, 2026-10-08).

Why (owner, 2026-10-08): "it doesn't learn and doesn't have a good memory"; the assistant could not say which video it
talked about and made context up. Modelled on EventMemAgent (arXiv 2602.15329, Apache-2.0, lingcco/EventMemAgent):
short-term memory is the active window (the open sessions of events.EventBook); long-term memory is an archive where
each event keeps its first keyframe, a caption (who, what they did, where, the outcome; no timestamps), a text
embedding and its change from the previous event. Their histogram event boundaries are not used: our boundary is
the event session (events.py). Their search returns the top 3 with similarity >= 0.3; so does :meth:`search`.

- **Archive** (:meth:`EventMemory.archive`): called by the event book when a session closes
  (``EventBook.archive_hooks``). One JSON line per event in ``events_archive.jsonl`` beside ``events.jsonl`` (which
  stays as it is). The caption and ``change_from_previous`` are written by code from the session's fields, never
  by a model. ``change_to_next`` is the next event's ``change_from_previous``, filled in when the archive is read.
  Kept ``RETENTION_DAYS``.
- **Keyframe**: the guard loop saves the snapshot of an event's first alert job as ``keyframes/<event id>.jpg``
  (:func:`save_keyframe`); the record points at it when it is on disk.
- **Embedding**: lazy. Only at search time, only with an embedder (embeddings.Embedder, which caches every vector on
  disk by the text's hash); a failure (no key, no credit, offline) turns embeddings off for ``EMBED_RETRY_SEC`` and
  the search falls back to keywords silently.
- **Search** (:meth:`EventMemory.search`): embedding similarity when it works, else a BM25 score over the caption and
  the observations. The owner writes Hebrew and the captions are English, so query words go through a small
  Hebrew->English map, the camera's names match, and time words ("בצהריים", "at night", "אתמול") become filters.

Never raises out of the archive hook: a memory that cannot be written must not stop the event book.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import math
import os
import re
import threading
import time
from dataclasses import asdict, is_dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple

log = logging.getLogger("box.event_memory")

ARCHIVE_NAME = "events_archive.jsonl"
KEYFRAMES_DIR = "keyframes"
RETENTION_DAYS = 30.0
PRUNE_EVERY_SEC = 6 * 3600.0
MIN_SIMILARITY = 0.3            # EventMemAgent's threshold
DEFAULT_K = 3                   # EventMemAgent's top 3
MAX_K = 8
MAX_DID = 3                     # what they did: at most this many distinct observation summaries in the caption
MAX_OBSERVATIONS = 12           # per archived event (the session keeps 40)
EMBED_RETRY_SEC = 3600.0        # after a failed embeddings call, keywords only for this long
SAME_TEXT = 0.6                 # two summaries this alike (word overlap) say the same thing
SIMILAR_ACTIVITY = 0.3
LEVELS = {"none": 0, "normal": 1, "suspicious": 2, "escalation": 3}


# ---------- words ----------
_WORD = re.compile(r"[a-z0-9]+|[א-ת]+(?:['׳\"״][א-ת]+)?")
_HE_PREFIX = "והבלמשכ"
_EN_STOP = frozenset("""
a an the at in on of to for from by with and or but is are was were be been being do does did done have has had
what whats happened happen happening who whom when where which how many much any anyone anything someone something
there here this that these those it its they them their he she his her him we our us you your i me my
again still yet already about around near just please tell show me can could would will should
""".split())
_HE_STOP = frozenset("""
מה קרה היה היו האם פה כאן את של עם על מי מתי כמה זה זו זאת יש אין אצלי אצלנו הם הן הוא היא עוד כבר שוב גם
אם או כל איפה איך למה תגיד תראה תראי תגידי לי שלי שלנו אתה את אני אנחנו בבית ליד הייתה
""".split())

# The owner writes Hebrew; captions and observations are English. Common words only (a model is not asked).
HE_EN: Dict[str, Tuple[str, ...]] = {
    "אנשים": ("people", "person", "men", "group"), "אדם": ("person", "man"), "מישהו": ("person", "someone"),
    "בנאדם": ("person", "man"), "עובדים": ("workers", "worker", "work"), "עובד": ("worker", "work"),
    "פועלים": ("workers", "worker", "work"), "פועל": ("worker", "work"),
    "רכב": ("car", "vehicle"), "מכונית": ("car", "vehicle"), "אוטו": ("car", "vehicle"),
    "טנדר": ("pickup", "van", "truck"), "משאית": ("truck", "lorry"), "ואן": ("van",),
    "אופנוע": ("motorcycle", "motorbike", "scooter"), "קטנוע": ("scooter", "motorcycle"), "אופניים": ("bicycle", "bike"),
    "אישה": ("woman", "lady"), "אשה": ("woman", "lady"), "נשים": ("women", "woman"), "גבר": ("man", "men"),
    "גברים": ("men", "man"), "ילד": ("child", "kid", "boy"), "ילדה": ("child", "girl", "kid"),
    "ילדים": ("children", "child", "kids"), "כלב": ("dog",), "כלבים": ("dogs", "dog"), "חתול": ("cat",),
    "חבילה": ("package", "parcel", "box"), "חבילות": ("packages", "package", "parcel"),
    "שליח": ("delivery", "courier"), "משלוח": ("delivery", "package"), "דואר": ("mail", "post", "delivery"),
    "לבן": ("white",), "לבנה": ("white",), "שחור": ("black",), "שחורה": ("black",), "אדום": ("red",),
    "אדומה": ("red",), "כחול": ("blue",), "כחולה": ("blue",), "אפור": ("grey", "gray"), "אפורה": ("grey", "gray"),
    "כסוף": ("silver",), "כסופה": ("silver",),
    "יצא": ("left", "leave", "exit"), "יצאו": ("left", "leave", "exit"), "נסע": ("left", "drove", "drive"),
    "נסעו": ("left", "drove"), "עזב": ("left", "leave"), "עזבו": ("left", "leave"), "הלך": ("left", "walked", "walk"),
    "הלכו": ("left", "walked"), "הגיע": ("arrived", "came", "arrive"), "הגיעו": ("arrived", "came", "arrive"),
    "חנה": ("parked", "park"), "חונה": ("parked", "park"), "סולם": ("ladder",), "גנן": ("gardener", "garden"),
    "שער": ("gate",), "דלת": ("door",), "חצר": ("yard",), "חניה": ("driveway", "parking", "parked"),
    "חשוד": ("suspicious",), "חשודה": ("suspicious",), "שכן": ("neighbour", "neighbor"),
}
# The same idea for English words whose other forms do not share a stem.
EN_ALSO: Dict[str, Tuple[str, ...]] = {
    "car": ("vehicle",), "cars": ("vehicle",), "vehicle": ("car",), "pickup": ("truck", "van"), "van": ("vehicle",),
    "truck": ("pickup", "vehicle"), "leave": ("left", "drove"), "left": ("leave", "drove"), "gone": ("left",),
    "workers": ("work",), "worker": ("work",), "delivery": ("courier", "package"), "courier": ("delivery",),
    "package": ("parcel", "box"), "kid": ("child",), "child": ("kid", "boy", "girl"), "people": ("person", "men"),
}
# Time words become a filter on the local hour the event was going on (start, end), wrapping midnight.
TIME_WORDS: Dict[str, Tuple[int, int]] = {
    "בבוקר": (6, 12), "בוקר": (6, 12), "morning": (6, 12),
    "בצהריים": (11, 15), "צהריים": (11, 15), "בצהרים": (11, 15), "צהרים": (11, 15), "noon": (11, 15), "midday": (11, 15),
    "אחה\"צ": (13, 18), "אחה״צ": (13, 18), "afternoon": (13, 18),
    "בערב": (17, 22), "ערב": (17, 22), "evening": (17, 22),
    "בלילה": (21, 6), "לילה": (21, 6), "night": (21, 6), "tonight": (21, 6),
}
DAY_WORDS: Dict[str, int] = {"היום": 0, "today": 0, "אתמול": 1, "yesterday": 1, "שלשום": 2}


def _stem(word: str) -> str:
    """A light English stem ("working" -> "work", "parked" -> "park", "cars" -> "car"); Hebrew is left as is."""
    if not word.isascii():
        return word
    for suffix in ("ing", "ed", "es", "s"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            word = word[: -len(suffix)]
            break
    if len(word) > 4 and word.endswith("e"):
        word = word[:-1]
    return word


def _variants(word: str) -> Set[str]:
    """The word, its stem, and for Hebrew the word without one or two prefix letters ("והרכב" -> "רכב")."""
    out = {word, _stem(word)}
    if not word.isascii():
        for n in (1, 2):
            if len(word) - n >= 2 and all(ch in _HE_PREFIX for ch in word[:n]):
                out.add(word[n:])
    return out


def words(text: str) -> List[str]:
    return _WORD.findall(str(text or "").casefold().replace("_", " "))


def doc_terms(text: str) -> List[str]:
    """The searchable terms of a document: every word's variants."""
    out: List[str] = []
    for w in words(text):
        out.extend(sorted(_variants(w)))
    return out


def query_groups(text: str) -> List[Set[str]]:
    """One group of alternative terms per content word of the query (stop, time and day words dropped)."""
    groups: List[Set[str]] = []
    for w in words(text):
        forms = _variants(w)
        if forms & (_EN_STOP | _HE_STOP) or forms & set(TIME_WORDS) or forms & set(DAY_WORDS):
            continue
        alts = set(forms)
        for form in forms:
            for en in HE_EN.get(form, ()) + EN_ALSO.get(form, ()):
                alts |= _variants(en)
        groups.append(alts)
    return groups


def english_of(text: str) -> str:
    """The query's mapped English words, to help an embedding match a Hebrew question to English captions."""
    out: List[str] = []
    for w in words(text):
        for form in _variants(w):
            for en in HE_EN.get(form, ()):
                if en not in out:
                    out.append(en)
    return " ".join(out)


def time_filter(text: str) -> Tuple[Optional[Tuple[int, int]], Optional[int]]:
    """``(hour range, days ago)`` named by the query's time words, each None when not named."""
    hours: Optional[Tuple[int, int]] = None
    day: Optional[int] = None
    found = set()
    for w in words(text):
        found |= _variants(w)
    raw = str(text or "")
    for word, span in TIME_WORDS.items():
        if word in found or ('"' in word or "״" in word) and word in raw:
            hours = span
            break
    for word, ago in DAY_WORDS.items():
        if word in found:
            day = ago
            break
    return hours, day


def _hours_of(start: float, end: float) -> Set[int]:
    """The local hours an event was going on in (at most a day of them)."""
    out: Set[int] = set()
    t = start
    end = max(start, min(end, start + 86400.0))
    while t <= end:
        out.add(dt.datetime.fromtimestamp(t).hour)
        t += 1800.0
    out.add(dt.datetime.fromtimestamp(end).hour)
    return out


def _in_hours(hours: Set[int], span: Tuple[int, int]) -> bool:
    a, b = span
    wanted = set(range(a, b)) if a < b else set(range(a, 24)) | set(range(0, b))
    return bool(hours & wanted)


def _jaccard(a: Iterable[str], b: Iterable[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def _content(text: str) -> Set[str]:
    return {_stem(w) for w in words(text) if w not in _EN_STOP and w not in _HE_STOP and len(w) > 1}


# ---------- the record ----------
def _session_dict(session: Any) -> Dict[str, Any]:
    if is_dataclass(session):
        return asdict(session)
    return dict(session or {})


def _people(n: int) -> str:
    if n <= 0:
        return "Activity (no person counted)"
    return "1 person" if n == 1 else f"{n} people"


def distinct_summaries(summaries: Sequence[str], limit: int = MAX_DID) -> List[str]:
    """The observation summaries without repeats (the same words over and over from an all-day crew), at most
    *limit*: the first ones and the last, so the outcome is kept."""
    kept: List[str] = []
    sets: List[Set[str]] = []
    for text in summaries:
        text = " ".join(str(text or "").split()).rstrip(". ")
        if not text:
            continue
        mine = _content(text)
        if any(_jaccard(mine, other) >= SAME_TEXT for other in sets) or text in kept:
            continue
        kept.append(text)
        sets.append(mine)
    if len(kept) > limit:
        kept = kept[: limit - 1] + kept[-1:]
    return kept


def _gap(seconds: float) -> str:
    seconds = max(0.0, seconds)
    if seconds < 60:
        return "less than a minute"
    if seconds < 3600:
        return f"{int(seconds // 60)} min"
    if seconds < 86400:
        hours = seconds / 3600.0
        return f"{hours:.0f} h" if hours >= 10 else f"{hours:.1f} h".replace(".0 h", " h")
    return f"{int(seconds // 86400)} days" if seconds >= 2 * 86400 else "a day"


def outcome_of(session: Dict[str, Any], idle_sec: float) -> str:
    """``left`` (closed after nobody was seen for the idle time), ``continued`` (rolled into a linked event while
    still going on), or ``open`` (not closed yet)."""
    closed = float(session.get("closed") or 0.0)
    if not closed:
        return "open"
    last = float(session.get("last_active") or session.get("opened") or closed)
    return "left" if closed - last >= idle_sec - 1.0 else "continued"


def make_caption(session: Dict[str, Any], place: str, outcome: str) -> str:
    """Who / how many, what they did (max MAX_DID distinct observations), where (the camera's name), the outcome.
    English, no timestamps (EventMemAgent's caption fields: subjects, actions, location, outcome)."""
    observations = [o for o in session.get("observations") or [] if isinstance(o, dict)]
    did = distinct_summaries([str(o.get("summary") or "") for o in observations])
    people = int(session.get("people_max") or 0)
    parts = [f"{_people(people)} at {place or 'a camera'}."]
    if did:
        parts.append("Seen: " + "; ".join(did) + ".")
    levels = [str(o.get("label") or "") for o in observations]
    top = max(levels, key=lambda x: LEVELS.get(x, 0)) if levels else ""
    if LEVELS.get(top, 0) >= LEVELS["suspicious"]:
        parts.append(f"Judged {top}.")
    reported = str(session.get("reported_level") or "none")
    parts.append("The owner was told." if reported != "none" else "Not sent to the owner.")
    known = [str(k.get("text") or "") for k in session.get("known") or [] if isinstance(k, dict) and k.get("text")]
    if known:
        parts.append("The owner said: " + "; ".join(dict.fromkeys(known)) + ".")
    parts.append({"left": "They left.", "continued": "Still there; it goes on in the next event.",
                  "open": "Still going on."}.get(outcome, ""))
    return " ".join(p for p in parts if p)


def change_from_previous(record: Dict[str, Any], previous: Optional[Dict[str, Any]]) -> str:
    """What is new against the camera's previous event, in code from the fields: the gap, more / fewer / the same
    people, the same group back (the owner's own words, or the same count and activity), a different activity."""
    if not previous:
        return "The first event at this camera in the memory."
    parts: List[str] = []
    if record.get("parent") and record.get("parent") == previous.get("event_id"):
        parts.append("Continues the previous event")
    else:
        parts.append(f"{_gap(float(record.get('start') or 0) - float(previous.get('end') or 0))} after the previous one")
    now_n, before_n = int(record.get("people_max") or 0), int(previous.get("people_max") or 0)
    if now_n > before_n:
        parts.append(f"more people ({now_n}, before {before_n})")
    elif now_n < before_n:
        parts.append(f"fewer people ({now_n}, before {before_n})")
    elif now_n:
        parts.append(f"the same number of people ({now_n})")
    similar = _jaccard(_content(" ".join(o.get("summary", "") for o in record.get("observations") or [])),
                       _content(" ".join(o.get("summary", "") for o in previous.get("observations") or [])))
    same_words = set(record.get("owner_known") or []) & set(previous.get("owner_known") or [])
    if same_words:
        parts.append(f"the same group the owner named ({sorted(same_words)[0]})")
    elif now_n and now_n == before_n and similar >= SIMILAR_ACTIVITY:
        parts.append("likely the same group back")
    elif now_n > before_n and before_n:
        parts.append("new people")
    parts.append("similar activity" if similar >= SIMILAR_ACTIVITY else "different activity")
    if LEVELS.get(record.get("level", ""), 0) > LEVELS.get(previous.get("level", ""), 0) >= LEVELS["normal"]:
        parts.append(f"now judged {record.get('level')}")
    return "; ".join(parts) + "."


def build_record(session: Any, previous: Optional[Dict[str, Any]], directory: str, place: str,
                 idle_sec: float = 60.0) -> Dict[str, Any]:
    """The archive record of one closed (or open, for a search) session."""
    s = _session_dict(session)
    observations = [o for o in s.get("observations") or [] if isinstance(o, dict)]
    labels = list(dict.fromkeys(str(o.get("label") or "") for o in observations if o.get("label")))
    level = max(labels, key=lambda x: LEVELS.get(x, 0)) if labels else "none"
    outcome = outcome_of(s, idle_sec)
    event_id = str(s.get("id") or "")
    keyframe = keyframe_path(directory, event_id) if event_id else ""
    end = float(s.get("closed") or s.get("last_active") or s.get("opened") or 0.0)
    kept = observations if len(observations) <= MAX_OBSERVATIONS else (
        observations[: MAX_OBSERVATIONS // 2] + observations[-(MAX_OBSERVATIONS - MAX_OBSERVATIONS // 2):])
    record: Dict[str, Any] = {
        "event_id": event_id,
        "camera": str(s.get("camera") or ""),
        "camera_name": place,
        "start": float(s.get("opened") or 0.0),
        "end": end,
        "parent": str(s.get("parent") or ""),
        "labels": labels,
        "level": level,
        "people_max": int(s.get("people_max") or 0),
        "reported": str(s.get("reported_level") or "none") != "none",
        "reported_level": str(s.get("reported_level") or "none"),
        "owner_known": list(dict.fromkeys(str(k.get("text") or "") for k in s.get("known") or []
                                          if isinstance(k, dict) and k.get("text"))),
        "alert_ids": list(dict.fromkeys(str(o.get("alert_id")) for o in observations if o.get("alert_id"))),
        "keyframe": keyframe if keyframe and os.path.isfile(keyframe) else "",
        "outcome": outcome,
        "observations": [{"ts": float(o.get("ts") or 0.0), "label": str(o.get("label") or ""),
                          "people": int(o.get("people") or 0), "summary": str(o.get("summary") or "")[:400]}
                         for o in kept],
        "caption": make_caption(s, place, outcome),
    }
    entities = s.get("entities")
    if isinstance(entities, (list, dict)) and entities:
        record["entities"] = entities           # stage 2 entities, kept as the session had them
    record["change_from_previous"] = change_from_previous(record, previous)
    return record


def keyframe_path(directory: str, event_id: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_-]", "_", str(event_id or ""))
    return os.path.join(directory, KEYFRAMES_DIR, f"{safe}.jpg")


def has_keyframe(directory: str, event_id: str) -> bool:
    return bool(event_id) and os.path.isfile(keyframe_path(directory, event_id))


def save_keyframe(directory: str, event_id: str, jpeg: bytes) -> str:
    """Keep *jpeg* as the event's keyframe unless it already has one (the first alert job's snapshot wins).
    Returns the path, or "" when nothing was written. Never raises."""
    if not event_id or not jpeg:
        return ""
    path = keyframe_path(directory, event_id)
    if os.path.isfile(path):
        return ""
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "wb") as f:
            f.write(jpeg)
        os.replace(tmp, path)
        return path
    except OSError as exc:
        log.warning("keyframe of event %s not saved: %s", event_id, exc)
        return ""


def _default_place(camera: str) -> str:
    try:
        from .camera_names import display_name  # noqa: PLC0415

        return display_name(camera, "en")
    except Exception:  # noqa: BLE001 - a name is never worth a lost record
        return ""


# ---------- the memory ----------
class EventMemory:
    """The archive in ``<directory>/events_archive.jsonl`` and its search. Thread-safe."""

    def __init__(self, directory: str, place: Optional[Callable[[str], str]] = None,
                 retention_days: float = RETENTION_DAYS, idle_sec: float = 60.0,
                 clock: Callable[[], float] = time.time) -> None:
        self.directory = directory
        self.place = place or _default_place
        self.retention_days = float(retention_days)
        self.idle_sec = float(idle_sec)
        self.clock = clock
        self._lock = threading.RLock()
        self._cache: Optional[Tuple[Tuple[float, int], List[Dict[str, Any]]]] = None
        self._last_prune = 0.0
        self._embed_off_until = 0.0
        os.makedirs(directory, exist_ok=True)

    @property
    def path(self) -> str:
        return os.path.join(self.directory, ARCHIVE_NAME)

    # -- writing --
    def archive(self, session: Any) -> Optional[Dict[str, Any]]:
        """Write one closed session's record. Never raises (the event book's lock is held around this)."""
        try:
            s = _session_dict(session)
            camera = str(s.get("camera") or "")
            with self._lock:
                previous = next((r for r in reversed(self._read()) if r.get("camera") == camera), None)
                try:
                    place = self.place(camera) or ""
                except Exception:  # noqa: BLE001
                    place = ""
                record = build_record(s, previous, self.directory, place, self.idle_sec)
                with open(self.path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
                self._cache = None
                now = float(self.clock())
                if now - self._last_prune >= PRUNE_EVERY_SEC:
                    self.prune(now)
            return record
        except Exception as exc:  # noqa: BLE001
            log.warning("event %s not added to the memory: %s", getattr(session, "id", "?"), exc)
            return None

    def prune(self, now: Optional[float] = None) -> int:
        """Drop records (and their keyframes) older than the retention; returns how many were dropped."""
        now = float(self.clock()) if now is None else now
        cutoff = now - self.retention_days * 86400.0
        with self._lock:
            self._last_prune = now
            rows = self._read()
            keep = [r for r in rows if float(r.get("end") or r.get("start") or 0.0) >= cutoff]
            dropped = len(rows) - len(keep)
            if dropped:
                tmp = self.path + ".tmp"
                with open(tmp, "w", encoding="utf-8") as f:
                    for r in keep:
                        row = {key: value for key, value in r.items() if key != "change_to_next"}   # derived on read
                        f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
                os.replace(tmp, self.path)
                self._cache = None
            folder = os.path.join(self.directory, KEYFRAMES_DIR)
            alive = {os.path.basename(keyframe_path(self.directory, r.get("event_id", ""))) for r in keep}
            try:
                for name in os.listdir(folder):
                    full = os.path.join(folder, name)
                    if name not in alive and os.path.getmtime(full) < cutoff:
                        os.remove(full)
            except OSError:
                pass
            return dropped

    # -- reading --
    def _read(self) -> List[Dict[str, Any]]:
        try:
            st = os.stat(self.path)
        except OSError:
            return []
        key = (st.st_mtime, st.st_size)
        if self._cache is not None and self._cache[0] == key:
            return self._cache[1]
        rows: List[Dict[str, Any]] = []
        try:
            with open(self.path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                    except ValueError:
                        continue
                    if isinstance(row, dict) and row.get("event_id"):
                        rows.append(row)
        except OSError:
            return []
        rows.sort(key=lambda r: float(r.get("start") or 0.0))
        nxt: Dict[str, Dict[str, Any]] = {}
        for row in reversed(rows):            # change_to_next: the camera's next event's change_from_previous
            later = nxt.get(row.get("camera", ""))
            row["change_to_next"] = later.get("change_from_previous", "") if later else ""
            nxt[row.get("camera", "")] = row
        self._cache = (key, rows)
        return rows

    def records(self) -> List[Dict[str, Any]]:
        with self._lock:
            return [dict(r) for r in self._read()]

    def get(self, event_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            row = next((r for r in self._read() if r.get("event_id") == event_id), None)
            return dict(row) if row else None

    def live_record(self, session: Any) -> Dict[str, Any]:
        """A record for a session that is still open (searchable before it is archived)."""
        s = _session_dict(session)
        camera = str(s.get("camera") or "")
        with self._lock:
            previous = next((r for r in reversed(self._read()) if r.get("camera") == camera), None)
        try:
            place = self.place(camera) or ""
        except Exception:  # noqa: BLE001
            place = ""
        return build_record(s, previous, self.directory, place, self.idle_sec)

    # -- search --
    def search(self, query: str, since: Optional[float] = None, until: Optional[float] = None,
               camera: Any = None, k: int = DEFAULT_K, embedder: Any = None,
               names: Optional[Dict[str, Sequence[str]]] = None, boost: Sequence[str] = (),
               extra: Sequence[Dict[str, Any]] = (), now: Optional[float] = None) -> List[Dict[str, Any]]:
        """The *k* events that best match *query*, newest first among equals; each a record copy with ``score`` and
        ``match`` ("embedding" / "keywords" / "time and place").

        *since*/*until* bound the event's time, *camera* (an id or a set of ids) filters; with neither, the query's
        day words ("אתמול") bound it. Time words ("בצהריים") filter the local hour. *names* maps a camera id to the
        family's names for it and *boost* lists cameras the query named: their events rank first. *extra* are
        records of events still going on (EventMemory.live_record)."""
        now = float(self.clock()) if now is None else float(now)
        k = max(1, min(MAX_K, int(k or DEFAULT_K)))
        hours, days_ago = time_filter(query)
        if since is None and until is None and days_ago is not None:
            day = dt.datetime.fromtimestamp(now).replace(hour=0, minute=0, second=0, microsecond=0)
            day = day - dt.timedelta(days=days_ago)
            since, until = day.timestamp(), (day + dt.timedelta(days=1)).timestamp()
        cams: Optional[Set[str]] = None
        if camera:
            cams = {camera} if isinstance(camera, str) else set(camera)
        rows = self.records() + [dict(r) for r in extra]
        pool: List[Dict[str, Any]] = []
        for r in rows:
            start, end = float(r.get("start") or 0.0), float(r.get("end") or r.get("start") or 0.0)
            if since is not None and end < since:
                continue
            if until is not None and start > until:
                continue
            if cams is not None and r.get("camera") not in cams:
                continue
            if hours is not None and not _in_hours(_hours_of(start, end), hours):
                continue
            pool.append(r)
        if not pool:
            return []
        named = set(boost) | self._named_cameras(query, pool, names or {})
        groups = query_groups(self._without_names(query, pool, names or {}))
        scored: List[Tuple[float, Dict[str, Any], str]] = []
        if groups and embedder is not None and now >= self._embed_off_until:
            scored = self._by_embedding(query, pool, embedder, named, now)
        if not scored and groups:
            scored = self._by_keywords(groups, pool, named)
        if not groups:
            # Only a place or a time was asked ("what happened at the pergola at noon?"): the newest there and then.
            chosen = [r for r in pool if r.get("camera") in named] if named else pool
            scored = [(1.0, r, "time and place") for r in chosen]
        scored.sort(key=lambda x: (x[0], float(x[1].get("start") or 0.0)), reverse=True)
        out = []
        for score, row, how in scored[:k]:
            out.append(dict(row, score=round(score, 3), match=how))
        return out

    @staticmethod
    def _names_of(row: Dict[str, Any], names: Dict[str, Sequence[str]]) -> List[str]:
        own = list(names.get(str(row.get("camera") or ""), ()))
        if row.get("camera_name"):
            own.append(str(row["camera_name"]))
        return [n for n in own if n and len(n.strip()) > 1]

    def _named_cameras(self, query: str, pool: Sequence[Dict[str, Any]], names: Dict[str, Sequence[str]]) -> Set[str]:
        q = " " + " ".join(words(query)) + " "
        found: Set[str] = set()
        for row in pool:
            for name in self._names_of(row, names):
                key = " ".join(words(name))
                if key and re.search(rf"(?<![\wא-ת])[{_HE_PREFIX}]{{0,2}}{re.escape(key)}(?![\wא-ת])", q):
                    found.add(str(row.get("camera")))
        return found

    def _without_names(self, query: str, pool: Sequence[Dict[str, Any]], names: Dict[str, Sequence[str]]) -> str:
        """The query without the cameras' names: a place is matched by camera, not by caption words."""
        q = " ".join(words(query))
        for name in sorted({n for row in pool for n in self._names_of(row, names)}, key=len, reverse=True):
            key = " ".join(words(name))
            if key:
                q = re.sub(rf"(?<![\wא-ת])[{_HE_PREFIX}]{{0,2}}{re.escape(key)}(?![\wא-ת])", " ", q)
        return q

    @staticmethod
    def _text(row: Dict[str, Any]) -> str:
        obs = " ".join(str(o.get("summary") or "") for o in row.get("observations") or [])
        return " ".join([str(row.get("caption") or ""), obs, " ".join(row.get("owner_known") or []),
                         " ".join(row.get("labels") or [])])

    def _by_keywords(self, groups: List[Set[str]], pool: Sequence[Dict[str, Any]],
                     named: Set[str]) -> List[Tuple[float, Dict[str, Any], str]]:
        """BM25 (k1 1.2, b 0.75); a query word scores by its best alternative; a named camera adds 1."""
        docs = [doc_terms(self._text(r)) for r in pool]
        n = len(docs)
        avg = sum(len(d) for d in docs) / n if n else 1.0
        counts = [{} for _ in docs]
        df: Dict[str, int] = {}
        for i, d in enumerate(docs):
            for term in d:
                counts[i][term] = counts[i].get(term, 0) + 1
            for term in set(d):
                df[term] = df.get(term, 0) + 1
        # A longer question must match at least half its words: "They left." is in every caption, so "did the white
        # pickup leave?" must not find every event that ended.
        needed = 1 if len(groups) <= 2 else math.ceil(len(groups) / 2)
        out = []
        for i, row in enumerate(pool):
            length = len(docs[i]) or 1
            total = 0.0
            matched = 0
            for group in groups:
                best = 0.0
                for term in group:
                    tf = counts[i].get(term, 0)
                    if not tf:
                        continue
                    idf = math.log(1.0 + (n - df[term] + 0.5) / (df[term] + 0.5))
                    best = max(best, idf * tf * 2.2 / (tf + 1.2 * (0.25 + 0.75 * length / avg)))
                total += best
                matched += best > 0.0
            if total <= 0.0 or matched < needed:
                continue
            if row.get("camera") in named:
                total += 1.0
            out.append((total, row, "keywords"))
        return out

    def _by_embedding(self, query: str, pool: Sequence[Dict[str, Any]], embedder: Any, named: Set[str],
                      now: float) -> List[Tuple[float, Dict[str, Any], str]]:
        from .embeddings import cosine  # noqa: PLC0415

        try:
            texts = [str(r.get("caption") or "") for r in pool]
            asked = f"{query} ({english_of(query)})" if english_of(query) else query
            vectors = embedder.embed([asked] + texts)
        except Exception as exc:  # noqa: BLE001
            log.debug("event memory embeddings failed: %s", exc)
            vectors = None
        if not vectors or vectors[0] is None:
            self._embed_off_until = now + EMBED_RETRY_SEC
            log.info("event memory: no embeddings (no key, no credit or offline); keyword search for %.0f min",
                     EMBED_RETRY_SEC / 60)
            return []
        out = []
        for row, vec in zip(pool, vectors[1:]):
            sim = cosine(vectors[0], vec)
            if sim >= MIN_SIMILARITY:
                out.append((sim + (0.1 if row.get("camera") in named else 0.0), row, "embedding"))
        return out


_MEMORIES: Dict[str, EventMemory] = {}
_MEMORIES_LOCK = threading.Lock()


def memory_for(directory: str) -> EventMemory:
    """The one EventMemory of an events folder (the guard loop writes it, the assistant reads it)."""
    key = os.path.abspath(directory)
    with _MEMORIES_LOCK:
        memory = _MEMORIES.get(key)
        if memory is None:
            memory = _MEMORIES[key] = EventMemory(directory)
        return memory


def attach(book: Any, memory: Optional[EventMemory] = None) -> EventMemory:
    """Archive every session *book* (events.EventBook) closes from now on. Idempotent."""
    memory = memory or memory_for(book.directory)
    memory.idle_sec = float(getattr(book, "idle_sec", memory.idle_sec))
    hooks = getattr(book, "archive_hooks", None)
    if hooks is not None and not any(getattr(h, "__self__", None) is memory for h in hooks):
        hooks.append(memory.archive)
    return memory
