# home_guard_project/box/brain/human.py
"""Talk like a person (owner, 2026-10-10): the memory is private, no robotic questions, an insult gets an apology
and the fix.

The chat of 2026-10-10 13:48-15:14: "זה הבית של השכן" -> "עד מתי לזכור את הבית של השכן?"; "מה קשר ? לא הבנתי" ->
"שמור אצלי עכשיו: העובדים של הפרגולה, ... מה לתקן?"; "מה הקשר העבדים של הפרגולה יא חתיכת מטומטם" -> "מה לתקן?";
"הזכרון שלך שמור אצלך אתה לא צריך לחשוף לי אותו" -> "העובדים של הפרגולה כבר מסומנים ..."; "אתה יכול לשאול בצורה
דרך אגב אבל לא לחשוף" -> "הבנתי אותך."; "מה הקשר עובדים יא מטומטם" -> "מה תרצה לשנות או להוסיף?". The owner: "if it
was a person it wouldn't talk like that", "why even mention the workers!!!".

- :func:`memory_related`: does this message (or the event it is about) concern a saved mark? Only then may the
  model read the marks or a reply mention them.
- :func:`generic_sentences` / :func:`drop_generic`: "מה לתקן?", "מה תרצה לשנות או להוסיף?", "במה אוכל לעזור?",
  an empty "הבנתי אותך." - never sent.
- :func:`is_angry`: an insult or a stretched "מה הקשררררר".
- :func:`private_memory_request`: "don't reveal your memory to me" - agreed in one line, and kept (code).
- :func:`not_understood`: ONE specific line about the thing at hand, when nothing better is left.

Pure functions; never raise.
"""

from __future__ import annotations

import datetime as dt
import re
from typing import Any, Dict, Iterable, List, Optional, Sequence

_HE = "א-ת"

# -- robotic questions and empty acknowledgements ------------------------------------------------------------------
_GENERIC = re.compile(
    rf"^\s*(?:ו?(?:אז\s+)?מה\s+(?:לתקן|לשנות|לעשות|תרצה|תרצי|אתה\s+רוצה|את\s+רוצה|ברצונך)(?:\s+(?:לתקן|לשנות|לעשות|"
    rf"להוסיף|שאעשה|שאתקן|או\s+להוסיף|או\s+לתקן|או\s+לשנות))*|במה\s+(?:עוד\s+)?(?:אוכל|אני\s+יכול)\s+לעזור(?:\s+לך)?|"
    rf"איך\s+(?:עוד\s+)?(?:אוכל|אני\s+יכול)\s+לעזור(?:\s+לך)?|יש\s+משהו\s+(?:שתרצה|שאתה\s+רוצה)\s+ש?(?:אשנה|אתקן|"
    rf"אעשה)|What\s+should\s+I\s+(?:fix|change|do)|What\s+would\s+you\s+like\s+(?:me\s+)?to\s+(?:change|add|fix|do)"
    rf"(?:\s+or\s+(?:add|change))?|How\s+can\s+I\s+help(?:\s+you)?)\s*\??\s*[.!]?\s*$"
    rf"|^\s*(?:הבנתי|אני\s+מבין|מבין)(?:\s+אותך)?\s*[.!]?\s*$|^\s*I\s+understand(?:\s+you)?\s*[.!]?\s*$",
    re.IGNORECASE)
_SENTENCE = re.compile(r"(?<=[.!?\n])\s+")


def generic_sentences(text: str) -> List[str]:
    """The sentences of *text* that are a robotic question or an empty acknowledgement."""
    return [s for s in _SENTENCE.split(str(text or "")) if s.strip() and _GENERIC.match(s.strip())]


def drop_generic(text: str) -> str:
    """*text* without its robotic sentences ("" when it was nothing else)."""
    bad = set(generic_sentences(text))
    if not bad:
        return str(text or "")
    return " ".join(s.strip() for s in _SENTENCE.split(str(text or "")) if s.strip() and s not in bad).strip()


# -- anger -------------------------------------------------------------------------------------------------------
_STRETCHED = re.compile(rf"([{_HE}a-zA-Z])\1{{3,}}")                 # "הקשררררר", "עובדייים" is only 3
_ANGRY = re.compile(rf"(?<![{_HE}])[ושה]{{0,2}}(?:מטומט\w*|טיפש\w*|מטופש\w*|אידיוט\w*|דביל\w*|סתום|סתומה|חמור|"
                    rf"מפגר\w*|דפוק\w*|חתיכת)(?![{_HE}])|\b(?:stupid|idiot\w*|dumb|useless|moron)\b", re.IGNORECASE)
_WHAT_RELATION = re.compile(rf"(?<![{_HE}])מה\s+ה?קשר", re.IGNORECASE)


def is_angry(text: str) -> bool:
    """An insult ("יא מטומטם", "חתיכת"), or a stretched "מה הקשררררר" (not a plain "מה הקשר?")."""
    text = str(text or "")
    return bool(_ANGRY.search(text) or (_WHAT_RELATION.search(text) and _STRETCHED.search(text)))


# -- "keep your memory to yourself" ----------------------------------------------------------------------------
_PRIVATE = re.compile(rf"(?<![{_HE}])(?:לא|אל|בלי|תפסיק)\s+(?:\S+\s+){{0,4}}?(?:לחשוף|תחשוף|לפרט|תפרט|לגלות|תגלה)"
                      rf"(?![{_HE}])|\b(?:don'?t|stop)\s+(?:tell|show|list|reveal)\w*\s+(?:me\s+)?(?:your\s+|what'?s?\s+in\s+"
                      rf"your\s+)?memory", re.IGNORECASE)
_ASK_BY_THE_WAY = re.compile(rf"(?<![{_HE}])ל?(?:שאול|תשאל)(?![{_HE}])|דרך\s+אגב|\bask\b|\bby\s+the\s+way\b",
                             re.IGNORECASE)


def private_memory_request(text: str) -> str:
    """"ask" when the owner says to ask in passing without revealing ("אתה יכול לשאול בצורה דרך אגב אבל לא לחשוף"),
    "keep" when he says not to reveal the memory ("הזכרון שלך שמור אצלך אתה לא צריך לחשוף לי אותו"), else ""."""
    text = str(text or "")
    if not _PRIVATE.search(text):
        return ""
    return "ask" if _ASK_BY_THE_WAY.search(text) else "keep"


# -- is the memory part of THIS message? -----------------------------------------------------------------------
def memory_related(text: str, marks: Sequence[Dict[str, Any]], event_camera: str = "",
                   covers: Any = None) -> List[Dict[str, Any]]:
    """The live marks this message is about: the owner names their people ("העובדים"), or the conversation is about
    a camera the mark covers (*covers(mark, camera)*). [] when the memory has nothing to do with it."""
    out = []
    for k in marks or ():
        here = bool(event_camera) and covers is not None and covers(k, event_camera)
        if names_mark(text, k) or here:
            out.append(k)
    return out


_WHERE = re.compile(rf"\s+(?:של|ב|על|ליד|at|of|in|by)(?:\s|(?=[{_HE}]))")


def squeeze(text: str) -> str:
    """Stretched letters back to one, and the owner's typo for the workers: "מה הקשררררר העבדים" -> "מה הקשר
    העובדים", "עובדייים" -> "עובדים"."""
    out = re.sub(rf"([{_HE}])\1{{2,}}", r"\1", str(text or ""))
    return re.sub(rf"(?<![{_HE}])([ושהבלמכ]{{0,2}})עבדים(?![{_HE}])", r"\1עובדים", out)


def names_mark(sentence: str, mark: Dict[str, Any]) -> bool:
    """The sentence names the PEOPLE of *mark* ("העובדים", "עמיר") - not merely the camera they are at ("בפרגולה"
    is not "העובדים של הפרגולה")."""
    from . import known_memory as km  # noqa: PLC0415

    who = str((mark or {}).get("text") or "")
    if not who:
        return False
    sentence = squeeze(sentence)
    if km.work_group(who):
        return km.work_group(sentence)
    words = km._content_words(_WHERE.split(who, maxsplit=1)[0])
    return bool(words) and bool(words & km._content_words(sentence))


def drop_unrelated(text: str, unrelated: Iterable[Dict[str, Any]]) -> str:
    """*text* without the sentences that bring up a mark this conversation is not about."""
    marks = [m for m in unrelated if isinstance(m, dict)]
    if not marks:
        return str(text or "")
    kept = [s.strip() for s in _SENTENCE.split(str(text or "")) if s.strip()
            and not any(names_mark(s, m) for m in marks)]
    return " ".join(kept).strip()


# -- when nothing better is left ---------------------------------------------------------------------------------
def not_understood(lang: str, event: Optional[Dict[str, Any]] = None, camera_name: str = "") -> str:
    """ONE line about the thing at hand: the alert being discussed by its time and camera, else plainly."""
    from .i18n import t  # noqa: PLC0415

    if event and camera_name:
        try:
            when = dt.datetime.fromtimestamp(float(event.get("ts") or 0)).strftime("%H:%M")
        except (TypeError, ValueError, OverflowError, OSError):
            when = ""
        if when:
            return t("not_understood_event", lang, time=when, camera=camera_name)
    return t("not_understood", lang)
