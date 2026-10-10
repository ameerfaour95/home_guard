# home_guard_project/box/brain/day_story.py
""""סיכום יום" told like a guard: what happened at the house, in order, not alert statistics.

Owner, 2026-10-10 (after "היום נרשמו 84 התרעות, מתוכן 7 נחשבות נורמליות ו-2 חשודות ..."): "When I ask for a summary
of the day it needs to tell what happened: in the morning at 7:00 a person with a red hat came to the house, knocked on
the door and left; at 10:00 a woman came out of the house to the car and left the yard; at 14:00 it looks like a
delivery guy... Not alerts and numbers. I don't care about that."

How:
1. The period's episodes come from the event memory (``events_archive.jsonl``, one record per closed camera session)
   plus the open sessions; ``events.jsonl`` adds the cross-camera links and the Hebrew notes. Noise is dropped: no AI
   description, nobody and no vehicle (birds, the lamp), "no special activity".
2. Sessions that are one story are merged: the same camera within 3 minutes, or two cameras within 3 minutes that
   the cross-camera link joins or whose people look alike (the same colour and garment).
3. What the owner already explained (a live mark: "העובדים של הפרגולה" 07:00-18:00, an activity fact: "the
   electricians") collapses into ONE line with its span; the family's own comings and goings when an entity carries
   the owner's label.
4. ONE model call (box.yaml ``day_story_model``, else the assistant's model) writes 4-10 Hebrew lines from compact
   1-line records (about 3k input tokens at most), and code checks them: no counts, no "התרעות", no labels, no
   coverage, no English. Without a model, or when its lines fail the check, code writes the lines from the Hebrew
   notes.
5. Each line keeps its clip's handle in the chat state, so "שלח את 10:15" / "תראה לי את זה" sends that clip.

The usage ledger counts the call as agent ``day_story``.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from ..camera_names import channel_of
from .mode import hhmm
from .registry import display

log = logging.getLogger("box.brain.day_story")

MERGE_SEC = 180.0            # two sessions this close are one story when they belong together
MAX_RECORDS = 16             # episodes shown to the model
MAX_RECORD_CHARS = 360       # one episode's text
MAX_INPUT_CHARS = 6500       # all the records: about 2.5k tokens with the instructions
MAX_LINES = 10
STORY_KEEP_SEC = 6 * 3600.0  # "שלח את 10:15" works this long after the story
OFFER_KEEP_SEC = 1800.0      # a bare "כן" answers the story's offer this long

_PLACEHOLDER = re.compile(
    r"^\s*(?:a person or vehicle was detected|no (?:special |unusual |notable )?(?:activity|motion|movement|one)"
    r"(?: (?:is |was )?(?:visible|seen|detected))?|nothing (?:unusual|special|notable|to report)|"
    r"(?:the )?(?:scene|yard|area) is (?:empty|quiet|clear)|אין פעילות(?: מיוחדת)?|לא נראית? פעילות.*)\s*\.?\s*$",
    re.IGNORECASE)
_VEHICLE = re.compile(r"\b(?:car|cars|truck|pickup|van|vehicle|vehicles|motorcycle|motorbike|scooter|bicycle|bike|"
                      r"taxi|bus|jeep|suv)\b", re.IGNORECASE)
_PERSON = re.compile(r"\b(?:person|people|man|men|woman|women|boy|girl|child|children|kid|kids|someone|worker|"
                     r"workers|courier|delivery|guy|individual|figure)\b", re.IGNORECASE)
_COLOURS = ("white", "black", "dark", "light", "red", "blue", "green", "yellow", "orange", "grey", "gray", "brown",
            "pink", "purple", "beige", "navy", "khaki")
_GARMENTS = ("shirt", "t-shirt", "hat", "cap", "helmet", "jacket", "hoodie", "sweater", "pants", "trousers", "jeans",
             "shorts", "dress", "skirt", "vest", "coat", "uniform", "bag", "backpack", "head covering", "scarf",
             "shoes", "top", "clothing", "clothes")
_LOOK = re.compile(r"\b(" + "|".join(_COLOURS) + r")(?:[- ]colou?red)?\s+(?:\w+\s+)?(" +
                   "|".join(re.escape(g) for g in _GARMENTS) + r")s?\b", re.IGNORECASE)
_LEVELS = {"none": 0, "normal": 1, "suspicious": 2, "escalation": 3}

# What a story line may never say (owner, 2026-10-10): counts, alerts, labels, coverage.
_BANNED = re.compile(
    r"התר[עא](?:ה|ות|ת)|"
    r"(?<!\w)[והש]?(?:חשוד|חשודה|חשודים|חשודות|נורמלי|נורמלית|נורמליים|נורמליות|רגיל|רגילה|רגילים|רגילות)(?!\w)|"
    r"\d+\s+(?:אירועים|אירוע|מקרים|פעמים|ביקורים|זיהויים)|"
    r"(?<!\w)(?:ה)?כיסוי(?!\w)|הוערכ|לא\s+נבדק|\b(?:alerts?|suspicious|normal|coverage)\b",
    re.IGNORECASE)
_ENGLISH = re.compile(r"[A-Za-z]{2,}")


# ---------------------------------------------------------------------------------------------- the owner's words
_PERIOD_WORDS = (r"היום|הבוקר|בבוקר|הלילה|בלילה|אתמול|מאתמול|בצהריים|אחר\s+הצהריים|אחה\"?צ|הערב|בערב|"
                 r"בשע(?:ה|תיים|ות)\s+האחרונ(?:ה|ות)|עד\s+עכשיו|מאז\s+הבוקר")
_ASKS = re.compile(
    r"(?<!\w)(?:סיכום|סכם|תסכם|תסכמי|סכמי)(?:\s+(?:לי|לנו|את|של|ה|יומי|היום|יום|הבוקר|הלילה|אתמול))*"
    r"(?:\s+(?:" + _PERIOD_WORDS + r"|יום|יומי|היום))|"
    r"(?<!\w)(?:סיכום\s+יומי|סיכום\s+היום|סיכום\s+יום)(?!\w)|"
    r"(?<!\w)(?:מה|מה\s+בעצם)\s+(?:היה|קרה|הלך|התרחש)(?:\s+(?:פה|כאן|בבית|בחצר|אצלנו))?\s+(?:" + _PERIOD_WORDS + r")|"
    r"(?<!\w)(?:ת?ספר|תספרי|ספרי)\s+(?:לי|לנו)\s+מה\s+(?:היה|קרה)|"
    r"(?<!\w)איך\s+היה\s+(?:היום|הבוקר|הלילה|אתמול)(?!\w)|"
    r"\b(?:summary\s+of\s+(?:the|my|to)\s*day|daily\s+summary|day\s+summary|summari[sz]e\s+(?:the|my|to)\s*day|"
    r"what\s+(?:happened|went\s+on)\s+(?:here\s+|at\s+home\s+)?(?:today|this\s+morning|last\s+night|yesterday|"
    r"tonight|this\s+afternoon|this\s+evening)|tell\s+me\s+what\s+happened)\b",
    re.IGNORECASE)
_COUNT_QUESTION = re.compile(r"(?<!\w)כמה(?!\w)|\bhow\s+many\b", re.IGNORECASE)


def asks_day_story(text: str) -> bool:
    """"סיכום יום", "מה היה היום", "מה קרה הבוקר", "תספר לי מה היה", "מה היה מאתמול בערב" (not "how many")."""
    text = " ".join(str(text or "").split())
    return bool(text) and len(text) <= 120 and bool(_ASKS.search(text)) and not _COUNT_QUESTION.search(text)


def _at(day: dt.date, hour: int, minute: int = 0) -> float:
    return dt.datetime.combine(day, dt.time(hour % 24, minute)).timestamp() + (86400.0 if hour >= 24 else 0.0)


def period_of(text: str, now: float, args: Optional[Dict[str, Any]] = None) -> Tuple[float, float, str]:
    """``(since, until, the period in Hebrew)`` for the owner's words (or a tool call's day / time_from / time_to /
    last_hours). Today since midnight by default."""
    args = args or {}
    text = " ".join(str(text or "").split())
    today = dt.datetime.fromtimestamp(now).date()
    yesterday = today - dt.timedelta(days=1)
    hours = args.get("last_hours")
    m = re.search(r"(?:ב|מ)?(?:-)?(\d{1,2})\s+השעות\s+האחרונות|בשעתיים\s+האחרונות|בשעה\s+האחרונה|"
                  r"\b(?:last|past)\s+(\d{1,2})\s+hours?\b", text)
    if hours in (None, "") and m:
        hours = 2 if "שעתיים" in m.group(0) else 1 if "בשעה" in m.group(0) else int(m.group(1) or m.group(2))
    if hours not in (None, ""):
        try:
            h = min(48.0, max(0.5, float(hours)))
        except (TypeError, ValueError):
            h = 24.0
        return now - h * 3600.0, now, "בשעה האחרונה" if h <= 1 else f"ב-{h:g} השעות האחרונות"
    day_arg = str(args.get("day") or "").strip().lower()
    t_from, t_to = str(args.get("time_from") or "").strip(), str(args.get("time_to") or "").strip()
    if day_arg or t_from or t_to:
        base = yesterday if day_arg in ("yesterday", "אתמול") else today
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", day_arg):
            base = dt.date.fromisoformat(day_arg)
        since, until = _at(base, 0), min(now, _at(base, 24))
        for value, which in ((t_from, "from"), (t_to, "to")):
            mm = re.fullmatch(r"(\d{1,2}):(\d{2})", value)
            if mm and int(mm.group(1)) < 24 and int(mm.group(2)) < 60:
                ts = _at(base, int(mm.group(1)), int(mm.group(2)))
                since, until = (ts, until) if which == "from" else (since, min(now, ts))
        name = "אתמול" if base == yesterday else "היום" if base == today else base.strftime("%d.%m")
        if t_from or t_to:
            name += f" בין {hhmm(since)} ל-{hhmm(until)}"
        return since, until, name
    if re.search(r"מאתמול\s+(?:ב)?ערב|since\s+(?:last|yesterday)\s+evening", text, re.IGNORECASE):
        return _at(yesterday, 18), now, "מאתמול בערב"
    if re.search(r"(?<!מ)אתמול\s+(?:ב)?ערב|yesterday\s+evening", text, re.IGNORECASE):
        return _at(yesterday, 17), _at(yesterday, 24), "אתמול בערב"
    if re.search(r"(?<!\w)מאתמול(?!\w)|since\s+yesterday", text, re.IGNORECASE):
        return _at(yesterday, 0), now, "מאתמול"
    if re.search(r"(?<!\w)אתמול(?!\w)|\byesterday\b", text, re.IGNORECASE):
        return _at(yesterday, 0), _at(yesterday, 24), "אתמול"
    hour = dt.datetime.fromtimestamp(now).hour
    if re.search(r"(?<!\w)(?:ה|ב)לילה(?!\w)|last\s+night|\btonight\b", text, re.IGNORECASE):
        if hour < 15:
            return _at(yesterday, 20), min(now, _at(today, 7)), "הלילה"
        return _at(today, 20), now, "הלילה"
    if re.search(r"(?<!\w)(?:ה|ב)בוקר(?!\w)|מאז\s+הבוקר|this\s+morning", text, re.IGNORECASE):
        return _at(today, 5), min(now, _at(today, 12)), "הבוקר"
    if re.search(r"אחר\s+הצהריים|אחה\"?צ|this\s+afternoon", text, re.IGNORECASE):
        return _at(today, 12), min(now, _at(today, 18)), "אחר הצהריים"
    if re.search(r"(?<!\w)בצהריים(?!\w)|\bat\s+noon\b", text, re.IGNORECASE):
        return _at(today, 11), min(now, _at(today, 15)), "בצהריים"
    if re.search(r"(?<!\w)(?:ה|ב)ערב(?!\w)|this\s+evening", text, re.IGNORECASE):
        return _at(today, 17), now, "הערב"
    return _at(today, 0), now, "היום"


# ------------------------------------------------------------------------------------------------- the episodes
@dataclass
class Episode:
    start: float
    end: float
    cameras: List[str]
    records: List[Dict[str, Any]]
    summaries: List[str]
    notes_he: List[str]
    people: int
    level: str
    vehicles: bool
    alert_ids: List[str]
    owner_known: List[str]
    who: List[str] = field(default_factory=list)       # the owner's labels of people seen (family)
    known: str = ""                                     # the owner's words that explain it ("העובדים של הפרגולה")
    top_note: str = ""                                  # the Hebrew note of its most serious session

    @property
    def concern(self) -> bool:
        return _LEVELS.get(self.level, 0) >= _LEVELS["suspicious"] and not self.known


def _same_camera(a: str, b: str) -> bool:
    """Two ids of one camera: equal, or the same channel under two site names (a rename: ameer_week_0_1_ch3 is
    ameer_v2_ch3)."""
    if a == b:
        return True
    ca, cb = channel_of(a), channel_of(b)
    return ca is not None and ca == cb


def _summaries(record: Dict[str, Any]) -> List[str]:
    out: List[str] = []
    for o in record.get("observations") or []:
        if not isinstance(o, dict):
            continue
        text = " ".join(str(o.get("summary") or "").split())
        if text and not _PLACEHOLDER.match(text) and text not in out:
            out.append(text)
    return out


def is_noise(record: Dict[str, Any]) -> bool:
    """No AI description, or nobody and no vehicle (a bird, the lamp, a shadow), or "no special activity"."""
    said = _summaries(record)
    if not said:
        return True
    people = int(record.get("people_max") or 0) or max(
        [int(o.get("people") or 0) for o in record.get("observations") or [] if isinstance(o, dict)] or [0])
    text = " ".join(said)
    if people <= 0 and not _VEHICLE.search(text) and not _PERSON.search(text):
        return True
    return False


def _looks(texts: Iterable[str]) -> set:
    return {(c.lower().replace("gray", "grey"), g.lower()) for t in texts for c, g in _LOOK.findall(t)}


def _vehicles(record: Dict[str, Any]) -> bool:
    if any(isinstance(e, dict) and e.get("kind") == "vehicle" for e in record.get("entities") or []):
        return True
    return bool(_VEHICLE.search(" ".join(_summaries(record))))


def _labels(record: Dict[str, Any]) -> List[str]:
    return [str(e.get("owner_label")) for e in record.get("entities") or []
            if isinstance(e, dict) and e.get("owner_label")]


def _linked(a: Dict[str, Any], b: Dict[str, Any], sessions: Dict[str, Dict[str, Any]]) -> bool:
    """The cross-camera link joins the two sessions (an incident, or the link seen in shadow mode)."""
    sa, sb = sessions.get(str(a.get("event_id") or "")) or {}, sessions.get(str(b.get("event_id") or "")) or {}
    if sa.get("incident_id") and sa.get("incident_id") == sb.get("incident_id"):
        return True
    for x, y in ((sa, b), (sb, a)):
        for key in ("incident_from", "cross_seen"):
            link = x.get(key) if isinstance(x.get(key), dict) else {}
            if link.get("session") and link.get("session") == y.get("event_id"):
                return True
    for x, y in ((a, b), (b, a)):
        for e in x.get("entities") or []:
            if isinstance(e, dict) and e.get("from_session") and e.get("from_session") == y.get("event_id"):
                return True
    return False


def episodes_of(records: Sequence[Dict[str, Any]], sessions: Optional[Dict[str, Dict[str, Any]]] = None
                ) -> List[Episode]:
    """The period's records (noise already out) merged into episodes, oldest first."""
    sessions = sessions or {}
    out: List[Episode] = []
    for r in sorted(records, key=lambda x: float(x.get("start") or 0)):
        start = float(r.get("start") or 0)
        end = float(r.get("end") or start)
        said = _summaries(r)
        notes = [str(o.get("note") or "").strip() for o in (sessions.get(str(r.get("event_id") or "")) or {})
                 .get("observations") or [] if isinstance(o, dict) and str(o.get("note") or "").strip()]
        joined: Optional[Episode] = None
        for ep in reversed(out[-6:]):
            if start - ep.end > MERGE_SEC:
                continue
            same = any(_same_camera(str(r.get("camera") or ""), c) for c in ep.cameras)
            if (same or str(r.get("parent") or "") in {str(x.get("event_id") or "") for x in ep.records}
                    or any(_linked(r, x, sessions) for x in ep.records)
                    or (_looks(said) & _looks(ep.summaries))):
                joined = ep
                break
        level = str(r.get("level") or "none")
        if joined is None:
            out.append(Episode(start=start, end=end, cameras=[str(r.get("camera") or "")], records=[r],
                               summaries=list(said), notes_he=notes, people=int(r.get("people_max") or 0),
                               level=level, vehicles=_vehicles(r), alert_ids=list(r.get("alert_ids") or []),
                               owner_known=list(r.get("owner_known") or []), who=_labels(r),
                               top_note=notes[0] if notes else ""))
            continue
        ep = joined
        ep.end = max(ep.end, end)
        if not any(_same_camera(str(r.get("camera") or ""), c) for c in ep.cameras):
            ep.cameras.append(str(r.get("camera") or ""))
        ep.records.append(r)
        ep.summaries += [s for s in said if s not in ep.summaries]
        ep.notes_he += [n for n in notes if n not in ep.notes_he]
        ep.people = max(ep.people, int(r.get("people_max") or 0))
        if _LEVELS.get(level, 0) > _LEVELS.get(ep.level, 0):
            ep.level = level
            ep.top_note = notes[0] if notes else ep.top_note
        ep.vehicles = ep.vehicles or _vehicles(r)
        ep.alert_ids += [a for a in r.get("alert_ids") or [] if a not in ep.alert_ids]
        ep.owner_known += [k for k in r.get("owner_known") or [] if k not in ep.owner_known]
        ep.who += [w for w in _labels(r) if w not in ep.who]
    return out


# ------------------------------------------------------------------------------------- what the owner explained
def _clock(ts: float) -> str:
    return dt.datetime.fromtimestamp(ts).strftime("%H:%M")


def _mark_covers(mark: Dict[str, Any], camera: str, ts: float) -> bool:
    if not (float(mark.get("at") or 0) - 86400.0 <= ts < float(mark.get("until") or 0)):
        return False
    where = str(mark.get("camera") or "")
    if where and not _same_camera(where, camera):
        return False
    lo, hi = str(mark.get("daily_from") or ""), str(mark.get("daily_to") or "")
    return not (lo and hi) or lo <= _clock(ts) < hi


def _fact_covers(fact: Any, camera: str, ts: float) -> bool:
    cameras = [str(c) for c in (getattr(fact, "cameras", None) or [])]
    if cameras and not any(_same_camera(c, camera) for c in cameras):
        return False
    try:
        return bool(fact.in_window(ts))
    except Exception:  # noqa: BLE001
        return False


def explain(episodes: List[Episode], marks: Sequence[Dict[str, Any]], facts: Sequence[Any] = ()) -> None:
    """Sets ``known`` on each episode the owner already explained: words said in it, or a mark / an activity fact
    that covers every one of its sessions. A judged-suspicious episode is explained only by words said in it."""
    for ep in episodes:
        if ep.owner_known:
            ep.known = ep.owner_known[0]
            continue
        if _LEVELS.get(ep.level, 0) >= _LEVELS["suspicious"]:
            continue
        for mark in marks:
            if all(_mark_covers(mark, str(r.get("camera") or ""), float(r.get("start") or 0)) for r in ep.records):
                ep.known = str(mark.get("text") or "")
                break
        if ep.known:
            continue
        for fact in facts:
            if all(_fact_covers(fact, str(r.get("camera") or ""), float(r.get("start") or 0)) for r in ep.records):
                ep.known = str(getattr(fact, "who", "") or getattr(fact, "cause", "") or "")
                break


# ------------------------------------------------------------------------------------------------ the records
@dataclass
class Row:
    """One line of the model's input: an episode, or the owner's known activity collapsed into one."""
    n: int
    start: float
    end: float
    text: str
    episode: Optional[Episode] = None
    known: str = ""
    places: List[str] = field(default_factory=list)


def _place(snapshot: Any, cameras: Sequence[str]) -> str:
    names: List[str] = []
    for c in cameras:
        name = display(snapshot, c, "he")
        if name not in names:
            names.append(name)
    return " → ".join(names)


def _weight(ep: Episode) -> float:
    text = " ".join(ep.summaries).lower()
    w = 3.0 * _LEVELS.get(ep.level, 0) + min(ep.people, 4) + (1.5 if ep.vehicles else 0.0)
    w += 2.0 if re.search(r"door|knock|package|parcel|deliver|gate|enter|leav|drive|window|carr", text) else 0.0
    w += min(2.0, (ep.end - ep.start) / 600.0) + (1.0 if ep.who else 0.0)
    return w


def rows_of(episodes: List[Episode], snapshot: Any) -> List[Row]:
    """Known activity collapsed per who (one row with its span and places), the rest one row each; at most
    MAX_RECORDS rows (the most telling ones), oldest first, numbered from 1."""
    groups: Dict[str, List[Episode]] = {}
    rest: List[Episode] = []
    for ep in episodes:
        if ep.known:
            groups.setdefault(ep.known, []).append(ep)
        else:
            rest.append(ep)
    if len(rest) > MAX_RECORDS - len(groups):
        keep = sorted(rest, key=_weight, reverse=True)[: max(1, MAX_RECORDS - len(groups))]
        rest = [ep for ep in rest if ep in keep]
    rows: List[Row] = []
    for who, eps in groups.items():
        cams: List[str] = []
        for ep in eps:
            cams += [c for c in ep.cameras if not any(_same_camera(c, x) for x in cams)]
        places = [p for p in dict.fromkeys(display(snapshot, c, "he") for c in cams)]
        rows.append(Row(0, min(e.start for e in eps), max(e.end for e in eps), "", known=who, places=places,
                        episode=max(eps, key=_weight)))
    for ep in rest:
        rows.append(Row(0, ep.start, ep.end, "", episode=ep, places=[_place(snapshot, ep.cameras)]))
    rows.sort(key=lambda r: r.start)
    for i, row in enumerate(rows, 1):
        row.n = i
        span = f"{_clock(row.start)}–{_clock(row.end)}" if row.end - row.start >= 120 else _clock(row.start)
        if row.known:
            row.text = (f"[{i}] {span} · KNOWN (the owner said who they are: \"{row.known}\") · places: "
                        f"{', '.join(row.places)}")
        else:
            ep = row.episode
            bits = [f"[{i}] {span}", row.places[0] or "camera"]
            if ep.people:
                bits.append(f"{ep.people} {'person' if ep.people == 1 else 'people'}")
            if ep.who:
                bits.append("who: " + ", ".join(ep.who))
            if ep.concern:
                bits.append("CHECK")
            what = " | ".join(ep.summaries)
            if len(what) > MAX_RECORD_CHARS:
                what = what[: MAX_RECORD_CHARS - 1].rstrip() + "…"
            row.text = " · ".join(bits) + " · " + what
    total = 0
    kept: List[Row] = []
    for row in rows:                          # never more than about 2.5k tokens of records
        total += len(row.text) + 1
        if total > MAX_INPUT_CHARS:
            break
        kept.append(row)
    return kept


# ------------------------------------------------------------------------------------------------- the writer
SYSTEM = """You are the guard of a family home. The owner asked what happened at the house {period}. You get one line per episode the cameras saw, oldest first (what people did, as the camera's AI described it in English). Tell the owner the story in Hebrew, the way a human guard reports at the end of a shift.

Write only lines, each in this form:
[n] HH:MM · <place>: <what happened>
- [n] is the episode's number from the input, HH:MM its start time, <place> the place name exactly as given.
- At most {max_lines} lines; usually fewer. Join episodes that are one story into one line (use the first number). Leave out episodes that add nothing new.
- Past tense, short: one sentence per line, about 20 words at most.
- Tell what people did, plainly: came in, walked to the door, knocked, waited, left, got into a car and drove out, carried something, worked. Mention one detail that helps recognise them (a red hat, a white shirt, a helmet). When the description only suggests who it was, say "כנראה" (כנראה שליח, כנראה עובד).
- A KNOWN line is the household's routine that the owner already explained: write it as ONE short line, ending with "כרגיל" ("[n] HH:MM–HH:MM · <places>: <who> עבדו כאן, כרגיל.").
- A line marked CHECK: when what was seen really deserves a look (someone at a window or a door that is not theirs, trying a handle, a face hidden on purpose, taking things away), write "[n] HH:MM · <place>: משהו שכדאי לראות: ..." and say exactly what made it worth a look - at most two such lines. Workers with helmets, hats or tools, people carrying bags in daylight, are ordinary: tell those like the others.
{busy}- Never write: how many alerts or events there were, the words התרעה/התרעות/התראה, labels such as חשוד/רגיל/נורמלי, anything about what the cameras could or could not cover, camera ids, English words, or any detail that is not in the input.
- No greeting, no title, no closing line."""


def _prompt(rows: Sequence[Row], period: str) -> List[Dict[str, str]]:
    busy = ("- It was a busy day: put the one line that matters most first (a CHECK, else the most unusual), then "
            "the others oldest first.\n" if len(rows) > MAX_LINES else "")
    system = SYSTEM.format(period=period, max_lines=MAX_LINES, busy=busy)
    return [{"role": "system", "content": system},
            {"role": "user", "content": f"Period: {period}\n" + "\n".join(r.text for r in rows)}]


_LINE = re.compile(r"^\s*[-*•]?\s*\[(\d+)\]\s*(.+?)\s*$")
_TIME = re.compile(r"(?<!\d)([01]?\d|2[0-3]):([0-5]\d)(?!\d)")


def _clean_line(text: str) -> str:
    return " ".join(text.split()).strip(" -*•")


_HEAD = re.compile(r"^\s*\d{1,2}:\d{2}(?:\s*[–-]\s*\d{1,2}:\d{2})?\s*(?:·\s*[^:·]{1,60}?:)?\s*")


def moment_of(row: Row) -> float:
    """When a line happened: the start, or for a line worth a look the start of its most serious session."""
    ep = row.episode
    if ep is None or not ep.concern:
        return row.start
    top = max(ep.records, key=lambda r: _LEVELS.get(str(r.get("level") or ""), 0))
    return float(top.get("start") or row.start)


def head_of(row: Row) -> str:
    """A line's time and place, written by code (the model copied an example's 07:40-17:50 on 2026-10-10)."""
    if row.known:
        return f"{_clock(row.start)}–{_clock(row.end)} · {', '.join(row.places)}"
    return f"{_clock(moment_of(row))} · {row.places[0] if row.places else ''}".rstrip(" ·")


def with_head(text: str, row: Optional[Row]) -> str:
    if row is None:
        return text
    body = _HEAD.sub("", text, count=1).strip()
    return f"{head_of(row)}: {body}" if body else ""


def check_lines(raw: str, rows: Sequence[Row]) -> List[Tuple[str, Optional[Row]]]:
    """The model's lines that may go out, each with its row: tagged ``[n]`` (or matched by its time), without
    counts, alert words, labels, coverage or English. At most MAX_LINES."""
    by_n = {r.n: r for r in rows}
    out: List[Tuple[str, Optional[Row]]] = []
    for line in str(raw or "").splitlines():
        if not line.strip():
            continue
        m = _LINE.match(line)
        row: Optional[Row] = None
        if m:
            row = by_n.get(int(m.group(1)))
            text = _clean_line(m.group(2))
        else:
            text = _clean_line(line)
            t = _TIME.search(text)
            if t:
                minute = int(t.group(1)) * 60 + int(t.group(2))
                row = next((r for r in rows if abs(_minute(r.start) - minute) <= 2), None)
        if not text or not _TIME.search(text):
            continue                               # a title, a greeting, a closing line
        text = with_head(text, row)
        if not text:
            continue
        if _BANNED.search(text):
            log.warning("day story: a line says what it may not (%s); dropped", _BANNED.search(text).group(0))
            continue
        if _ENGLISH.search(text):
            log.warning("day story: a line has English words; dropped")
            continue
        out.append((text, row))
    return out[:MAX_LINES]


def _minute(ts: float) -> int:
    d = dt.datetime.fromtimestamp(ts)
    return d.hour * 60 + d.minute


def _first_sentence(text: str, limit: int = 140) -> str:
    text = " ".join(str(text or "").split())
    m = re.search(r"^(.+?[.!?])(?:\s|$)", text)
    text = m.group(1) if m else text
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def code_lines(rows: Sequence[Row]) -> List[Tuple[str, Optional[Row]]]:
    """The story without a model: the known routine in one line each, the rest from the Hebrew notes."""
    out: List[Tuple[str, Optional[Row]]] = []
    for row in rows:
        span = f"{_clock(row.start)}–{_clock(row.end)}" if row.end - row.start >= 120 else _clock(row.start)
        if row.known:
            out.append((f"{span} · {', '.join(row.places)}: {row.known} היו כאן, כרגיל.", row))
            continue
        ep = row.episode
        notes = ([ep.top_note] if ep.concern else []) + ep.notes_he if ep else []
        note = next((n for n in notes if n and not _ENGLISH.search(n)), "")
        if not note:
            what = "רכב" if ep and ep.vehicles and not ep.people else (
                "אדם אחד" if ep and ep.people == 1 else "כמה אנשים")
            note = f"{what} נראה כאן." if what != "כמה אנשים" else "כמה אנשים נראו כאן."
        head = "משהו שכדאי לראות: " if ep is not None and ep.concern else ""
        out.append((f"{head_of(row)}: {head}{_first_sentence(note)}", row))
    if len(out) > MAX_LINES:
        keep = sorted(out, key=lambda x: _weight(x[1].episode) if x[1] and x[1].episode else 0.0,
                      reverse=True)[:MAX_LINES]
        out = [x for x in out if x in keep]
    return [(text, row) for text, row in out if not _BANNED.search(text)]


# ------------------------------------------------------------------------------------------------- the sources
def _read_jsonl(path: str, since: float, key: str, tail_bytes: int = 4_000_000) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(max(0, size - tail_bytes))
            data = f.read().decode("utf-8", errors="replace")
    except OSError:
        return rows
    for line in data.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if isinstance(row, dict) and float(row.get(key) or row.get("closed") or 0) >= since - 6 * 3600.0:
            rows.append(row)
    return rows


def gather(book: Any, since: float, until: float, camera: str = "") -> Tuple[List[Dict[str, Any]],
                                                                            Dict[str, Dict[str, Any]]]:
    """The archive records of the period (and the open sessions as records) with the sessions by id."""
    directory = str(getattr(book, "directory", "") or "")
    if not directory:
        return [], {}
    from ..event_memory import memory_for  # noqa: PLC0415

    memory = memory_for(directory)
    sessions = {str(s.get("id")): s for s in _read_jsonl(os.path.join(directory, "events.jsonl"), since, "opened")
                if s.get("id")}
    records = [r for r in memory.records()]
    seen = {str(r.get("event_id") or "") for r in records}
    try:
        for s in book.recent(since):
            sid = str(s.get("id") or "")
            sessions.setdefault(sid, s)
            if sid and sid not in seen:
                records.append(memory.live_record(s))
    except Exception as exc:  # noqa: BLE001 - the archive alone still tells the day
        log.debug("open sessions not read: %s", exc)
    out = []
    for r in records:
        start, end = float(r.get("start") or 0), float(r.get("end") or r.get("start") or 0)
        if end < since or start > until:
            continue
        if camera and not _same_camera(str(r.get("camera") or ""), camera):
            continue
        out.append(r)
    return out, sessions


def marks_of(book: Any) -> List[Dict[str, Any]]:
    """Every mark in known.json (expired ones too: they explained the day they were live)."""
    path = os.path.join(str(getattr(book, "directory", "") or ""), "known.json")
    try:
        with open(path, encoding="utf-8") as f:
            rows = json.load(f)
        return [r for r in rows if isinstance(r, dict) and r.get("text")]
    except (OSError, ValueError, TypeError):
        return []


def facts_of(activities: Any, since: float) -> List[Any]:
    try:
        activities.live(since)        # loads the file
        return [f for f in getattr(activities, "_facts", []) if not getattr(f, "cancelled_at", 0)]
    except Exception:  # noqa: BLE001
        return []


# ------------------------------------------------------------------------------------------------- the story
@dataclass
class Story:
    text: str
    lines: List[Dict[str, Any]]           # {"at", "start", "end", "handle", "concern"}
    offer: str = ""                       # the handle offered ("רוצה שאשלח את הסרטון של 13:35?")
    usage: Tuple[int, int] = (0, 0)
    model_used: bool = False


def _handle_for(add_handle: Optional[Callable[..., str]], row: Optional[Row]) -> str:
    if add_handle is None or row is None or row.episode is None:
        return ""
    ep = row.episode
    top = max(ep.records, key=lambda r: _LEVELS.get(str(r.get("level") or ""), 0))
    alert_id = next((a for a in top.get("alert_ids") or [] if a), "") or next((a for a in ep.alert_ids if a), "")
    if not alert_id:
        return ""
    return add_handle("event", alert_id, str(top.get("camera") or ""), float(top.get("start") or ep.start),
                      " | ".join(ep.summaries)[:200])


def tell(book: Any, activities: Any, snapshot: Any, text: str, now: float, model: Any = None,
         add_handle: Optional[Callable[..., str]] = None, args: Optional[Dict[str, Any]] = None,
         camera: str = "") -> Story:
    """The story of the period the owner asked about. One model call at most; never raises."""
    since, until, period = period_of(text, now, args)
    try:
        records, sessions = gather(book, since, until, camera)
    except Exception as exc:  # noqa: BLE001
        log.warning("day story: the event memory could not be read: %s", exc)
        records, sessions = [], {}
    kept = [r for r in records if not is_noise(r)]
    episodes = episodes_of(kept, sessions)
    explain(episodes, marks_of(book), facts_of(activities, since) if activities is not None else [])
    rows = rows_of(episodes, snapshot)
    log.info("day story %s: %d records, %d after noise, %d episodes, %d rows", period, len(records), len(kept),
             len(episodes), len(rows))
    if not rows:
        return Story(text=f"{period} היה שקט, לא היה משהו לספר.", lines=[])
    lines: List[Tuple[str, Optional[Row]]] = []
    usage = (0, 0)
    used = False
    if model is not None:
        try:
            from .. import usage_ledger  # noqa: PLC0415

            with usage_ledger.scope(agent="day_story"):
                msg = model.chat(_prompt(rows, period), [])
            usage = tuple(int(x or 0) for x in (getattr(msg, "usage", None) or (0, 0)))[:2]  # type: ignore[assignment]
            if getattr(msg, "error", ""):
                log.warning("day story: the model failed: %s", msg.error)
            else:
                lines = check_lines(getattr(msg, "content", "") or "", rows)
                used = bool(lines)
                if len(lines) < min(2, len(rows)):
                    log.warning("day story: the model's lines did not pass the check (%d of %d rows)", len(lines),
                                len(rows))
                    lines, used = [], False
        except Exception as exc:  # noqa: BLE001 - the code's lines then
            log.warning("day story: the model call failed: %s", exc)
            lines = []
    if not lines:
        lines = code_lines(rows)
    out_lines: List[Dict[str, Any]] = []
    offer = offer_at = ""
    texts: List[str] = []
    for line_text, row in lines:
        handle = _handle_for(add_handle, row)
        concern = "שכדאי לראות" in line_text
        t = _TIME.search(line_text)
        out_lines.append({"at": f"{int(t.group(1)):02d}:{t.group(2)}" if t else (_clock(row.start) if row else ""),
                          "start": row.start if row else 0.0, "end": row.end if row else 0.0,
                          "handle": handle, "concern": concern})
        if concern and handle and not offer:
            offer = handle
            offer_at = out_lines[-1]["at"]
        texts.append(line_text)
    story = "\n".join(texts)
    if offer:
        story += f"\nרוצה שאשלח את הסרטון של {offer_at}?"
    return Story(text=story, lines=out_lines, offer=offer, usage=usage, model_used=used)  # type: ignore[arg-type]


# ----------------------------------------------------------------------------------------- "שלח את 10:15"
def remember(state: Any, story: Story, now: float) -> None:
    """The story's lines and their clips, for a follow-up ("שלח את 10:15", "תראה לי את זה")."""
    try:      # a JSON string: the chat file keeps only plain values in prefs (memory._prefs)
        state.prefs["day_story"] = json.dumps({"ts": float(now), "offer": story.offer,
                                               "lines": [dict(x) for x in story.lines if x.get("handle")]})
    except Exception as exc:  # noqa: BLE001
        log.warning("day story not kept for a follow-up: %s", exc)


_SEND = re.compile(r"(?<!\w)(?:ת?שלח|תשלחי|שלחי|תראה|תראי|הראה|תן|תני|אפשר|רוצה|לראות|סרטון|וידאו|קליפ)(?!\w)|"
                   r"\b(?:send|show|video|clip|see)\b", re.IGNORECASE)
_THAT = re.compile(r"(?<!\w)(?:את\s+זה|אותו|אותה|זה|הזה|הזאת|כן|בטח|יאללה|תשלח|שלח)(?!\w)|\b(?:it|that|this|yes)\b",
                   re.IGNORECASE)


def send_target(text: str, state: Any, now: float) -> str:
    """The handle of the story line the owner asks for ("שלח את 10:15", "תראה לי את 13:35"), or of the story's
    offer ("תראה לי את זה", "כן" right after it); "" when the message is not that."""
    try:
        story = state.prefs.get("day_story") if isinstance(getattr(state, "prefs", None), dict) else None
        story = json.loads(story) if isinstance(story, str) and story else story
        if not isinstance(story, dict) or now - float(story.get("ts") or 0) > STORY_KEEP_SEC:
            return ""
        text = " ".join(str(text or "").split())
        if not text or len(text) > 80 or asks_day_story(text):
            return ""
        lines = [x for x in story.get("lines") or [] if isinstance(x, dict) and x.get("handle")]
        times = _TIME.findall(text)
        if times:
            if not _SEND.search(text) and not re.fullmatch(r"(?:את\s+)?(?:של\s+)?\d{1,2}:\d{2}\??", text):
                return ""
            minute = int(times[0][0]) * 60 + int(times[0][1])
            best, gap = "", 1e9
            for x in lines:                        # the line that starts nearest the time asked for
                m = _TIME.search(str(x.get("at") or ""))
                if not m:
                    continue
                d = abs(int(m.group(1)) * 60 + int(m.group(2)) - minute)
                if d < gap:
                    best, gap = str(x["handle"]), d
            if gap <= 5:
                return best
            inside = [x for x in lines if x.get("start") and x.get("end")
                      and _minute(float(x["start"])) <= minute <= _minute(float(x["end"]))]
            # else a line whose span holds it (the shortest: "the workers 12:53-17:15" holds everything)
            return str(min(inside, key=lambda x: float(x["end"]) - float(x["start"]))["handle"]) if inside else ""
        offer = str(story.get("offer") or "")
        if offer and now - float(story.get("ts") or 0) <= OFFER_KEEP_SEC and _THAT.search(text) and (
                _SEND.search(text) or re.fullmatch(r"(?:כן|בטח|יאללה|yes|sure)[.!]?\s*(?:שלח|תשלח)?[.!]?", text,
                                                   re.IGNORECASE)):
            return offer
        return ""
    except Exception as exc:  # noqa: BLE001
        log.warning("day story follow-up not read: %s", exc)
        return ""


def story_from_results(results: Sequence[str]) -> str:
    """The story a ``day_story`` tool call returned this turn (it goes out as written), or ""."""
    for raw in reversed(list(results or ())):
        try:
            data = json.loads(raw)
        except (TypeError, ValueError):
            continue
        if isinstance(data, dict) and data.get("ok") and data.get("day_story") and data.get("story"):
            return str(data["story"])
    return ""


def day_story_tool(ctx: Any, args: Dict[str, Any]) -> Dict[str, Any]:
    """The model's way in (the code route catches the usual words first): the story of the period."""
    try:
        args = args if isinstance(args, dict) else {}
        now = float(ctx.services.now())
        camera = ""
        if str(args.get("camera") or "").strip() and ctx.snapshot is not None:
            from .registry import resolve_camera  # noqa: PLC0415

            camera = resolve_camera(ctx.snapshot, str(args["camera"])).camera or ""
        words = str(args.get("words") or "") or str(ctx.text or "")
        story = tell(ctx.services.events, getattr(ctx.services, "activities", None), ctx.snapshot, words, now,
                     model=getattr(ctx.services, "story_model", None), add_handle=ctx.state.add_handle,
                     args={k: args.get(k) for k in ("day", "time_from", "time_to", "last_hours") if args.get(k)},
                     camera=camera)
        remember(ctx.state, story, now)
        return {"ok": True, "day_story": True, "story": story.text,
                "note": "This story is the whole answer: reply with it exactly as written, add nothing."}
    except Exception as exc:  # noqa: BLE001
        log.warning("day_story failed: %s", exc)
        return {"ok": False, "error": "the day's story could not be written"}
