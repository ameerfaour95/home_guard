"""Code guards on the model's label, before an alert reaches the owner (stage 1 of the alert fix, 2026-10-08).

Why: in two days of real alerts the model called workers "suspicious" for a mask, a hoodie or a covered face, and
gave a red "possible weapon" for a long work tool and a red "smashed the car window" for a man who parked. The
owner's rules:

- Appearance alone (mask, hood, covered face, dark clothes, hat, sunglasses, a blurred or pixelated face) is never
  suspicious; only actions are. :func:`appearance_only` finds a "suspicious" whose reason names nothing but looks,
  and the guard loop lowers it to normal. It never touches an escalation.
- A red for a weapon, a car break-in or violence that comes from one model answer gets a second look before it goes
  out red (:func:`verify_class` picks the question). Clear serious things - a break-in through the house's door or
  window, climbing in, fire or smoke, a person lying motionless (:func:`clear_class`) - go out at once.

Matching is on words, in English and Hebrew. English words are matched at a word start (so "hat" does not match
"that"); Hebrew words as substrings, so a prefix letter (ב, ה, ו, ש, ל) still matches. Pure functions, no imports
beyond ``re``: unit-tested on their own.
"""

from __future__ import annotations

import re
from typing import List, Optional, Sequence

_FLAGS = re.IGNORECASE


def _any(patterns: Sequence[str]) -> "re.Pattern[str]":
    return re.compile("|".join(f"(?:{p})" for p in patterns), _FLAGS)


# ---------- appearance vs action ----------
APPEARANCE = _any([
    r"\bmask", r"\bhood", r"\bbalaclava", r"\bski[- ]mask", r"\bhats?\b", r"\bcaps?\b", r"\bbeanie",
    r"\bsunglasses", r"\bpixelat", r"\bblur", r"\bdark (?:cloth|outfit|attire|jacket|hoodie|garment)",
    r"\bcover(?:ed|ing|s)? (?:his |her |their |the )?faces?", r"\bfaces? (?:is |are |was |were )?(?:covered|hidden|obscured|concealed|not visible)",
    r"\b(?:hidden|obscured|concealed) faces?", r"\bhid(?:es|ing)? (?:his|her|their) faces?",
    r"מסכה", r"מסיכה", r"קפוצ['׳]?ון", r"ברדס", r"פנים מכוסות", r"פנים מוסתרות", r"פניו מכוסות", r"פניו מוסתרות",
    r"מכסה את פניו", r"בגדים כהים", r"לבוש כהה", r"כובע", r"משקפי שמש", r"פיקסל", r"מפוקסל", r"מטושטש", r"רעול",
])

# Anything here keeps the "suspicious": an action, or night (walking around the property at night stays suspicious).
ACTION = _any([
    r"\bhandles?\b", r"\bdoors?\b", r"\bwindows?\b", r"\bgates?\b", r"\bfence", r"\blocks?\b", r"\bclimb",
    r"\bhid(?:e|es|ing)\b(?! (?:his|her|their) faces?)",r"\bcrouch", r"\bsteal", r"\bstole",
    r"\btak(?:e|es|ing|en)\b", r"\btook\b", r"\btamper", r"\bpeek", r"\bpeer", r"\blook(?:s|ed|ing)? (?:in|into|inside)\b",
    r"\bloiter", r"\blurk", r"\bbreak", r"\bforc", r"\bpry", r"\bentrance", r"\bnight\b",
    r"ידית", r"דלת", r"חלון", r"שער", r"גדר", r"מנעול", r"טיפוס", r"מטפס", r"מסתתר", r"מתחבא", r"כורע", r"גונב",
    r"לוקח", r"לקח", r"מציץ", r"פוגע במצלמה", r"פורץ", r"כניסה", r"לילה",
])


def appearance_only(text: str) -> bool:
    """True when *text* (the model's why / alert_reason) names appearance and no action: such a "suspicious" is
    lowered to normal. Empty text, or text naming neither, is False (nothing to judge, the label stays)."""
    text = str(text or "")
    return bool(APPEARANCE.search(text)) and not ACTION.search(text)


# ---------- the second look before a red ----------
_VEHICLE = r"(?:\bcars?\b|\bvehicles?\b|\btrucks?\b|\bvans?\b|רכב|מכונית)"
VERIFY_PATTERNS = {
    "weapon": _any([r"\bweapon", r"\bguns?\b", r"\bknife", r"\bknives", r"\brifle", r"\bpistol", r"\bfirearm",
                    r"\bmachete", r"נשק", r"אקדח", r"סכין", r"רובה"]),
    # A car must be named: "smashed a window" alone may be the house's, and that goes out red at once.
    "vehicle": _any([rf"(?:\bsmash|\bshatter|\bbreak|\bbroke|\bforc|\bpry|ניפץ|מנפץ|שובר|שבר|פורץ|פריצה).{{0,40}}{_VEHICLE}",
                     rf"{_VEHICLE}.{{0,30}}(?:\bsmash|\bshatter|\bbroken into|\bbreak-in|\bforced|נופץ|נפרץ|פריצה)"]),
    # Hitting a person, not a window: the person must be named.
    "violence": _any([r"\bfight", r"\battack", r"\bassault", r"\bpunch", r"\bbeat(?:s|ing)? (?:up )?(?:a |an |the |another )?(?:man|woman|person|someone|him|her|child)",
                      r"\bhit(?:s|ting)? (?:a |an |the |another )?(?:man|woman|person|someone|him|her|child|boy|girl)",
                      r"\bkick(?:s|ing)? (?:a |an |the |another )?(?:man|woman|person|someone|him|her)",
                      r"אלימות", r"תוקף", r"תוקפים", r"קטטה", r"מתקוטט", r"מכה (?:אדם|גבר|אישה|ילד|אותו|אותה|את ה(?:גבר|אישה|ילד|אדם))",
                      r"מכים (?:אדם|גבר|אישה|אותו|אותה)", r"הכה (?:אדם|גבר|אישה|אותו|אותה)"]),
}
VERIFY_ORDER = ("weapon", "vehicle", "violence")

# Clear serious things go out red at once, even when a verify word is there too.
CLEAR = _any([
    r"\bfire\b", r"\bflames?\b", r"\bsmoke\b", r"\blying motionless", r"\bunconscious", r"\bnot moving on the ground",
    r"\bclimb(?:s|ed|ing)? (?:in|into|through)\b", r"\bclimb(?:s|ed|ing)? (?:over|up) (?:the |a )?(?:fence|wall)",
    r"\bforc(?:e|es|ed|ing) (?:open )?(?:the |a )?(?:front |back |house |main )?(?:door|window|entry)",
    r"\bforced entry", r"\b(?:break|breaks|breaking|broke) into (?:the )?(?:house|home|building)", r"\bburglar",
    r"\b(?:break|breaks|breaking|broke|smash(?:es|ed|ing)?) (?:the |a )?(?:front |back |house |main )(?:door|window)",
    r"שריפה", r"(?<![א-ת])[בוה]?ה?אש(?![א-ת])", r"עשן", r"להבות", r"שוכב ללא תנועה", r"מחוסר הכרה", r"מטפס פנימה", r"נכנס דרך החלון",
    r"פורץ לבית", r"פריצה לבית", r"פורץ את הדלת", r"שובר את הדלת", r"פורץ דלת", r"פורץ חלון",
])

VERIFY_QUESTIONS = {
    "weapon": "Is a person holding a gun or knife as a weapon? Long tools, poles, boards, ladders and brooms are NOT "
              "weapons.",
    "vehicle": "Is someone breaking into a vehicle (smashing a window, forcing a door)? Someone getting out of or into "
               "their own car normally is NOT.",
    "violence": "Is someone hitting or attacking another person?",
}


def clear_class(text: str) -> bool:
    """A clear serious thing (house break-in, climbing in, fire or smoke, a person lying motionless)."""
    return bool(CLEAR.search(str(text or "")))


def verify_classes(text: str) -> List[str]:
    """Every second-look class *text* points to, in ``VERIFY_ORDER``; [] when it needs none (no verify word, or a
    clear class that goes out at once)."""
    text = str(text or "")
    if clear_class(text):
        return []
    return [name for name in VERIFY_ORDER if VERIFY_PATTERNS[name].search(text)]


def verify_class(text: str) -> Optional[str]:
    """The main second look an escalation needs (``weapon``, ``vehicle`` or ``violence``), or None."""
    found = verify_classes(text)
    return found[0] if found else None


def verify_question(classes: Sequence[str]) -> str:
    """One question for the one verification call; several classes are asked together ("is any of these so?")."""
    questions = [VERIFY_QUESTIONS[c] for c in classes if c in VERIFY_QUESTIONS]
    if len(questions) <= 1:
        return questions[0] if questions else ""
    return "Answer true if ANY of these is so. " + " ".join(f"({i}) {q}" for i, q in enumerate(questions, 1))


def verify_prompt(question: str, frames: int) -> str:
    """The second look's question with its strict JSON answer."""
    return (f"You are double-checking an alarm from a home security camera. These are {frames} sequential frames, "
            f"numbered 1 to {frames} in order.\n"
            f"Question: {question}\n"
            "Answer only from what is clearly visible. When it is not clear, the answer is false.\n"
            'Reply with EXACTLY ONE strict JSON object and nothing else: {"confirmed": true | false, '
            '"what_it_is": "<a few words: what the object or action really is>", '
            '"evidence_frame": <the frame number that shows it best>}')
