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
