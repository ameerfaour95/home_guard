"""Owner-facing lines the guard loop adds to an alert (stage 1 of the alert fix, 2026-10-08).

Kept here, not in brain/i18n.py, because the assistant side edits that file in parallel. Hebrew and English; any
other language reads English.
"""

from __future__ import annotations

from typing import Any

_NOT = {
    "weapon": {"he": "לא נשק", "en": "not a weapon"},
    "vehicle": {"he": "לא פריצה לרכב", "en": "not a car break-in"},
    "violence": {"he": "לא אלימות", "en": "not violence"},
}


def _he(lang: str) -> bool:
    return str(lang or "").startswith("he")


def more_people(n: int, lang: str) -> str:
    """The first line of an update in an event's thread: more people came than the owner was told about."""
    n = int(n)
    if _he(lang):
        return "עוד אדם אחד הגיע" if n == 1 else f"עוד {n} אנשים הגיעו"
    return "1 more person arrived" if n == 1 else f"{n} more people arrived"


def second_look(kind: str, what_it_is: str, frame: Any, lang: str) -> str:
    """The line under a red the second look did not confirm: what it really was, and in which frame."""
    what = " ".join(str(what_it_is or "").split())[:80]
    not_it = _NOT.get(kind, _NOT["weapon"])["he" if _he(lang) else "en"]
    try:
        frame_no = int(frame)
    except (TypeError, ValueError):
        frame_no = 0
    where = (f" (פריים {frame_no})" if _he(lang) else f" (frame {frame_no})") if frame_no > 0 else ""
    head = "בדקתי שוב" if _he(lang) else "Second look"
    return f"{head}: {what}{where}, {not_it}" if what else f"{head}: {not_it}"


_SEEN = (("people", "אדם", "a person"), ("vehicles", "רכב", "a vehicle"), ("animals", "בעל חיים", "an animal"))


def ai_unavailable(kinds: Any, lang: str, with_picture: bool = True) -> str:
    """What the owner reads when neither vision model answered (2026-10-09 ch1: the bare "a person or vehicle was
    detected"): what the detector saw, and plainly that the AI check did not finish. *kinds* are
    inference.detected_fact_kinds' words (people / vehicles / animals)."""
    found = set(kinds or ())
    he = _he(lang)
    names = [name_he if he else name_en for kind, name_he, name_en in _SEEN if kind in found]
    if he:
        if not names:
            seen = "זוהתה תנועה"
        elif len(names) == 1:
            seen = f"זוהה {names[0]}"
        else:
            seen = "זוהו " + ", ".join(names[:-1]) + f" ו{names[-1]}"
        tail = "הבדיקה של ה-AI לא הספיקה, הנה התמונה." if with_picture else "הבדיקה של ה-AI לא הספיקה."
        return f"{seen}. {tail}"
    if not names:
        seen = "Movement was detected"
    else:
        listed = names[0] if len(names) == 1 else ", ".join(names[:-1]) + f" and {names[-1]}"
        seen = f"{listed[0].upper()}{listed[1:]} {'was' if len(names) == 1 else 'were'} detected"
    tail = ("The AI check did not finish in time; here is the picture." if with_picture
            else "The AI check did not finish in time.")
    return f"{seen}. {tail}"
