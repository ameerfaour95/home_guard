"""An end time from the owner's OWN words only (a quoted slot), never a default: "עד 18:00", "עד שש", "ב17",
"ב 12 בלילה", "עד חצות", "היום בלילה", "כל השבוע". None when the words give no time."""
from __future__ import annotations

import datetime as dt
import re
from typing import Optional

from ..brain import known_memory as km

_PART = r"(בערב|אחה\"צ|אחר הצהריים|בלילה|בבוקר|pm|am)?"
_AT_NUM = re.compile(r"(?<![\d:])(?:(?<![א-ת])(?:ב|ב-|בשעה|לשעה|ל|מסיימים ב|מסיים ב|גומרים ב)\s*)"
                     r"([01]?\d|2[0-3])(?::([0-5]\d))?(?![\d:])\s*" + _PART, re.IGNORECASE)
_AT_WORD = re.compile(r"(?<![א-ת])(?:ב|בשעה\s+)(" + "|".join(sorted(km._HE_HOURS, key=len, reverse=True))
                      + r")(?![א-ת])\s*" + _PART)
_MIDNIGHT = re.compile(r"(?<![א-ת])(?:עד\s+)?(?:חצות|היום בלילה|הלילה)(?![א-ת])|\b(?:midnight|tonight)\b",
                       re.IGNORECASE)


def _hour(now: float, hour: int, minute: int, part: str) -> float:
    part = str(part or "").lower()
    if hour == 12 and part == "בלילה":
        hour = 0
    elif hour < 12 and part in ("pm", "בערב", 'אחה"צ', "אחר הצהריים") or (part == "בלילה" and 6 <= hour < 12):
        hour += 12
    elif not part and hour <= 12:
        return km._hour_today(now, hour, "") + minute * 60
    return km._at(now, hour % 24, minute)


_HOURS_WORDS = {"שעה": 1.0, "חצי שעה": 0.5, "שעה וחצי": 1.5, "שעתיים": 2.0, "שלוש שעות": 3.0, "ארבע שעות": 4.0,
                "חמש שעות": 5.0, "שש שעות": 6.0}
_DURATION = re.compile(r"(?<![א-ת\d])(?:עוד|בערך|כ-?|ל-?)?\s*(" + "|".join(sorted(_HOURS_WORDS, key=len, reverse=True))
                       + r"|(\d{1,2})\s*שעות)(?![א-ת])")
_MORNING = re.compile(r"(?<![א-ת])עד (?:ה)?בוקר(?![א-ת])")


def _duration(text: str, now: float) -> Optional[float]:
    """"שעה בערך" / "עוד שעתיים" / "3 שעות": *now* plus that long (2026-10-19 22:43: "שעה בערך" was no time, and
    "עד מתי הם פה היום?" came back three times). A clock hour ("ב-3") is not a duration."""
    m = _DURATION.search(text)
    if not m:
        return None
    hours = float(m.group(2)) if m.group(2) else _HOURS_WORDS[m.group(1)]
    return now + hours * 3600.0 if 0 < hours <= 12 else None


def parse_until(words: str, now: float) -> Optional[float]:
    text = str(words or "")
    if not text.strip():
        return None
    found = km.until_from_words(text, now)
    if found is not None:
        # "ב 12 בלילה" is midnight even when "עד היום בלילה" came first in the same words
        m = _AT_NUM.search(text)
        if m and m.group(3) == "בלילה" and int(m.group(1)) == 12:
            return _hour(now, 12, 0, "בלילה")
        return found
    found = _duration(text, now)
    if found is not None:
        return found
    if _MORNING.search(text):
        return km._at(now, 7)                        # the next 07:00
    m = _AT_NUM.search(text)
    if m:
        return _hour(now, int(m.group(1)), int(m.group(2) or 0), m.group(3) or "")
    m = _AT_WORD.search(text)
    if m:
        return _hour(now, km._HE_HOURS[m.group(1)], 0, m.group(2) or "")
    if _MIDNIGHT.search(text):
        return km._at(now, 0)
    return None


def clock(ts: float) -> str:
    return dt.datetime.fromtimestamp(ts).strftime("%H:%M")
