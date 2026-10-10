""""Why don't you explain?" right after live photos: what they show, per camera, with its name and time (2026-10-09).

13:00:53 "תגיד לי יש מישהו בחוץ אצלי?" got two photos; 13:02:01 "למה אתה לא נותן הסבר ?" got "האם יש משהו מסוים
שאתה רוצה לדעת?" (a question back) in the replay, and the memory status on the box. The looks were already made:
every photo the brain sent (check_camera, look_around) keeps what the vision model saw on its handle. A request
for an explanation ("הסבר", "תסביר", "מה אתה רואה") while those photos are fresh is answered from them:

    בפרגולה ב-13:00: שלושה עובדים ליד הטנדר, כנראה העובדים שסימנת.
    בכניסה הראשית ב-13:00: רכב לבן עם תא מטען פתוח, אין אנשים.

One model call writes the lines in the owner's language from those looks only (no tools, no question). When it
fails, or still asks, the lines are written in code from the people counts.
"""

from __future__ import annotations

import datetime as dt
import logging
import re
from typing import Any, Dict, List, Optional, Sequence

from ...prompts import render
from .i18n import LANGUAGE_NAMES, t
from .registry import display

log = logging.getLogger("box.brain.look_explain")

FRESH_SEC = 300.0          # photos sent these 5 minutes are "the photos" the owner asks about
MAX_CAMERAS = 4

_EXPLAIN = re.compile(
    r"(?<![א-ת])(?:[ול]?הסבר|הסברים|תסביר|תסבירי|להסביר|מסביר|תפרט|פירוט)(?![א-ת])|"
    r"(?<![א-ת])מה\s+(?:אתה\s+|את\s+)?(?:רואה|רואים|יש\s+שם|יש\s+בתמונה|בתמונות|בתמונה|זה\s+אומר)(?![א-ת])|"
    r"\bexplain|\bexplanation|\bwhat\s+do\s+you\s+see\b|\bwhat(?:'s|\s+is)\s+in\s+(?:the|these|this)\s+(?:photo|picture)",
    re.IGNORECASE)


def asks_explanation(text: str) -> bool:
    return bool(_EXPLAIN.search(str(text or "")))


def recent_looks(state: Any, now: float, within: float = FRESH_SEC) -> List[Dict[str, Any]]:
    """The latest described photo of each camera these *within* seconds (oldest camera first), from the chat's
    handles: ``{"camera", "ts", "description", "people"}``."""
    latest: Dict[str, Dict[str, Any]] = {}
    for handle, entry in (getattr(state, "handles", None) or {}).items():
        if not isinstance(entry, dict) or entry.get("kind") != "photo" or not entry.get("camera"):
            continue
        description = str(entry.get("observation") or "").strip()
        try:
            ts = float(entry.get("ts") or 0)
        except (TypeError, ValueError):
            continue
        if not description or not 0 <= now - ts <= within:
            continue
        cam = str(entry["camera"])
        if cam not in latest or ts >= latest[cam]["ts"]:
            people = entry.get("people")
            latest[cam] = {"camera": cam, "ts": ts, "description": description, "handle": handle,
                           "people": people if isinstance(people, int) else None,
                           "explained": bool(entry.get("explained"))}
    return sorted(latest.values(), key=lambda x: x["ts"])[-MAX_CAMERAS:]


def _hm(ts: float) -> str:
    return dt.datetime.fromtimestamp(float(ts)).strftime("%H:%M")


def _place(snapshot: Any, camera: str, lang: str) -> str:
    from .same_people import _place as place  # noqa: PLC0415

    out = place(snapshot, camera, lang)
    return out[:1].upper() + out[1:] if not str(lang).startswith("he") else out


def _marks_for(marks: Sequence[Dict[str, Any]], camera: str) -> List[str]:
    return [str(k.get("text") or "").strip() for k in marks
            if isinstance(k, dict) and k.get("camera") in (camera, "", None) and str(k.get("text") or "").strip()]


def code_lines(looks: Sequence[Dict[str, Any]], snapshot: Any, lang: str) -> str:
    """The plain lines, from the counts only (no model): "בפרגולה ב-13:00: 2 אנשים"."""
    rows = []
    for look in looks:
        n = look.get("people")
        what = (t("look_nobody", lang) if n == 0 else t("look_one_person", lang) if n == 1
                else t("look_people", lang, n=n) if isinstance(n, int) else t("look_no_picture", lang))
        rows.append(t("look_explain_line", lang, place=_place(snapshot, look["camera"], lang),
                      time=_hm(look["ts"]), what=what))
    return "\n".join(rows)


def explain(model: Any, looks: Sequence[Dict[str, Any]], snapshot: Any, marks: Sequence[Dict[str, Any]], lang: str,
            usage: Dict[str, List[int]]) -> str:
    """The lines for *looks*. Never raises; never a question (the code lines replace one)."""
    fallback = code_lines(looks, snapshot, lang)
    if model is None:
        return fallback
    rows = []
    for look in looks:
        known = _marks_for(marks, look["camera"])
        head = t("look_explain_line", lang, place=_place(snapshot, look["camera"], lang), time=_hm(look["ts"]),
                 what="").strip()
        people = look.get("people")
        rows.append(f'- PLACE "{head}"; people counted: {people if people is not None else "unknown"}; '
                    f'description: {look["description"]}' + (f'; KNOWN: {"; ".join(known)}' if known else ""))
    language = LANGUAGE_NAMES.get(lang, "English")
    try:
        msg = model.chat([{"role": "system", "content": render("brain_look_explain.system_prompt", language=language)},
                          {"role": "user", "content": "PHOTOS:\n" + "\n".join(rows)}], [])
        spent = usage.setdefault("big", [0, 0])
        spent[0] += int(msg.usage[0] or 0)
        spent[1] += int(msg.usage[1] or 0)
        if getattr(msg, "error", "") or getattr(msg, "refused", False):
            raise RuntimeError(getattr(msg, "error", "") or "refused")
        out = "\n".join(line.strip() for line in str(msg.content or "").splitlines() if line.strip())
    except Exception as exc:  # noqa: BLE001 - the plain lines still explain
        log.warning("Photo explanation not written: %s", exc)
        return fallback
    if not out or "?" in out or len(out) > 900:
        log.warning("Photo explanation unusable (%r); the plain lines go out", out[:80])
        return fallback
    return out


def answer(model: Any, state: Any, snapshot: Any, events: Any, now: float, lang: str,
           usage: Dict[str, List[int]]) -> Optional[str]:
    """The explanation of the photos just sent, or None when there are none fresh, or they were explained already
    (2026-10-09 replay: the 13:02 explanation again at 13:04 was a repeat; the models then take a new look)."""
    looks = recent_looks(state, now)
    if not looks or all(look["explained"] for look in looks):
        return None
    from . import known_memory as km  # noqa: PLC0415

    out = explain(model, looks, snapshot, km.live_marks(events, now), lang, usage)
    for look in looks:
        entry = state.handles.get(look["handle"])
        if isinstance(entry, dict):
            entry["explained"] = True
    return out
