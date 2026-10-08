"""What people did in an event, as a few fixed activity tags (task 2.9, the per-camera memory; owner 2026-10-08).

Owner: "knocking on the door is usual at the main door but not at the back door". The per-camera baseline
(baseline.py) counts WHICH activities each camera sees, not only how many events. The tags come from the Eye's
English summary (and the owner's Hebrew words) through a small keyword map, in code: no model, the same text always
gives the same tags, and a negated mention ("no animals", "אין אף אחד") is not a tag.

Tags: knock (knock / ring at the door), delivery (a courier, a package), walk_past, enter_leave (into or out of the
house), working (working / building / cleaning), vehicle_move (a car arriving / leaving / parking), kids (children
playing), animal, standing_talking, at_window, at_vehicle (by a car, opening one).
"""

from __future__ import annotations

import re
from typing import Dict, Iterable, List, Pattern, Sequence, Tuple

TAGS: Tuple[str, ...] = ("knock", "delivery", "walk_past", "enter_leave", "working", "vehicle_move", "kids",
                         "animal", "standing_talking", "at_window", "at_vehicle")

# What the owner reads for each tag (a noun phrase, so "X at the back door" reads well).
NAMES: Dict[str, Dict[str, str]] = {
    "knock": {"he": "דפיקה בדלת", "en": "a knock at the door"},
    "delivery": {"he": "שליח או חבילה", "en": "a delivery"},
    "walk_past": {"he": "מישהו שעובר", "en": "someone walking past"},
    "enter_leave": {"he": "כניסה או יציאה מהבית", "en": "going into or out of the house"},
    "working": {"he": "עבודה", "en": "work (building, cleaning)"},
    "vehicle_move": {"he": "רכב שמגיע, יוצא או חונה", "en": "a car arriving, leaving or parking"},
    "kids": {"he": "ילדים", "en": "children"},
    "animal": {"he": "בעל חיים", "en": "an animal"},
    "standing_talking": {"he": "אנשים שעומדים או מדברים", "en": "people standing or talking"},
    "at_window": {"he": "מישהו ליד חלון", "en": "someone at a window"},
    "at_vehicle": {"he": "מישהו ליד רכב", "en": "someone at a car"},
}

_HE = "א-ת"
_HE_PREFIX = "והבלמשכ"


def _he(*stems: str) -> str:
    """Hebrew stems with up to two prefix letters ("והשליח"), any ending."""
    return rf"(?<![{_HE}])[{_HE_PREFIX}]{{0,2}}(?:{'|'.join(stems)})"


_VEHICLE_EN = r"(?:car|vehicle|truck|van|pickup|motorcycle|motorbike|scooter|jeep|suv)s?"
_VEHICLE_HE = "רכב|מכונית|אוטו|טנדר|משאית|אופנוע|קטנוע|ג'יפ|גיפ"

_PATTERNS: Dict[str, Tuple[str, ...]] = {
    "knock": (r"\bknock\w*", r"\bring(?:s|ing|ed)?\s+(?:the\s+|a\s+)?(?:door)?bell", r"\bdoorbell", r"\bintercom",
              r"\bbuzz(?:es|ed|ing)?\s+(?:the\s+)?(?:door|gate|intercom)",
              _he("דופק", "דופקת", "דופקים", "דפק", "דפיקה", "דפיקות", "מצלצל", "צלצל", "צלצול", "פעמון",
                  "אינטרקום")),
    "delivery": (r"\bdeliver\w*", r"\bpackages?\b", r"\bparcels?\b", r"\bcouriers?\b", r"\bpostman\b",
                 r"\bmail\s*(?:man|carrier|box)?\b",
                 _he("שליח", "חבילה", "חבילות", "משלוח", "דואר", "דוור")),
    "walk_past": (r"\b(?:walk|pass|cross|stroll|jog|run)(?:s|es|ed|ing)?\s+(?:past|by|through|across|along|down)\b",
                  r"\bpass(?:es|ed|ing)\b", r"\bpasser-?by\b", r"\bpassers-?by\b",
                  r"\bwalk(?:s|ed|ing)?\s+(?:down|up|along|on)\s+the\s+(?:street|road|sidewalk|pavement)",
                  _he("עובר", "עוברת", "עוברים", "חולף", "חולפת", "חולפים", "חוצה", "חוצים")),
    "enter_leave": (r"\b(?:enter|leav|exit|go|went|walk|come|came|step|head)\w*\s+(?:in(?:to|side)?\s+|out\s+of\s+|"
                    r"from\s+)?(?:the\s+|their\s+|his\s+|her\s+)?(?:house|home|building|apartment|front door)\b",
                    r"\bgo(?:es|ing)?\s+(?:back\s+)?inside\b", r"\bcomes?\s+out\s+of\s+the\s+(?:house|door)",
                    r"\b(?:opens?|opening|closes?|closing|locks?|locking|unlocks?|unlocking)\s+the\s+(?:front\s+)?door",
                    _he("נכנס הביתה", "נכנסת הביתה", "נכנסים הביתה", "נכנס לבית", "נכנסת לבית", "נכנסים לבית",
                        "יוצא מהבית", "יוצאת מהבית", "יוצאים מהבית", "יצא מהבית", "יצאה מהבית", "יצאו מהבית",
                        "הביתה", "פותח את הדלת", "פותחת את הדלת", "סוגר את הדלת", "נועל")),
    "working": (r"\bwork(?:s|ing|ers?|man|men)\b", r"\bconstruct\w*", r"\bbuilds?\b",
                r"\bbuilding\s+(?:a|an|the|something|work)\b", r"\brenovat\w*", r"\brepair\w*", r"\binstall\w*",
                r"\bclean(?:s|ing|ed)?\b", r"\bsweep\w*", r"\bmop\w*", r"\bpaint(?:s|ing|ed)\b", r"\bdrill\w*",
                r"\bdigging\b", r"\bladders?\b", r"\btools?\b", r"\bwelding\b", r"\bsaw(?:ing)?\b",
                r"\bgarden(?:ing|er)\b", r"\bwatering\b",
                _he("עובד", "עובדת", "עובדים", "פועל", "פועלים", "בונה", "בונים", "בנייה", "בניה", "מנקה", "מנקים",
                    "ניקיון", "ניקוי", "שיפוץ", "משפצ", "מתקן", "מתקנים", "צובע", "צובעים", "סולם", "כלי עבודה",
                    "קודח", "גנן", "משקה")),
    "vehicle_move": (rf"\b{_VEHICLE_EN}\b(?:\s+(?:is|was|then|slowly|also|now|just|and\s+then))?\s+(?:arriv\w*|pulls?\s+(?:in|up|into|out|away)|pulling|"
                     r"parks\b|parking\b|is\s+parking|leav(?:es|ing)|left\b|drives?\b|driving|drove|revers\w*|"
                     r"backs?\s+(?:in|out|up)|enter(?:s|ed|ing)|exit(?:s|ed|ing)|depart\w*|moves?\b|moving)",
                     r"\b(?:drives?|driving|drove)\s+(?:in|into|away|off|out|through|up|past)\b",
                     r"\b(?:arrives?|arriving)\s+(?:by|in a)\s+car\b",
                     rf"(?<![{_HE}])[{_HE_PREFIX}]{{0,2}}(?:{_VEHICLE_HE})[^.;]{{0,25}}?"
                     rf"(?:נכנס|נכנסת|יוצא|יוצאת|יצא|יצאה|חנה|חנתה|נוסע|נוסעת|נסע|נסעה|הגיע|הגיעה|מגיע|מגיעה|עוזב|"
                     rf"עזב|נעצר|נעצרת|חונה עכשיו|נכנס לחניה)"),
    "kids": (r"\bchild(?:ren)?\b", r"\bkids?\b", r"\bboys?\b", r"\bgirls?\b", r"\btoddlers?\b", r"\bplay(?:s|ing)\b",
             r"\bball\b",
             _he("ילד", "ילדה", "ילדים", "ילדות", "משחק", "משחקת", "משחקים", "כדור")),
    "animal": (r"\bdogs?\b", r"\bcats?\b", r"\banimals?\b", r"\bpupp(?:y|ies)\b", r"\bkittens?\b", r"\bbirds?\b",
               _he("כלב", "כלבה", "כלבים", "חתול", "חתולה", "חתולים", "בעל חיים", "בעלי חיים", "ציפור", "ציפורים")),
    "standing_talking": (r"\bstand(?:s|ing)\b", r"\btalk(?:s|ing|ed)?\b", r"\bchat(?:s|ting)?\b", r"\bconvers\w*",
                         r"\bwait(?:s|ing)\b", r"\bsit(?:s|ting)\b", r"\bsmok(?:es|ing)\b", r"\bloiter\w*",
                         r"\bling(?:er|ers|ering)\b", r"\bon (?:the|a|his|her) phone\b",
                         _he("עומד", "עומדת", "עומדים", "מדבר", "מדברת", "מדברים", "משוחח", "משוחחים", "ממתין",
                             "מחכה", "מחכים", "יושב", "יושבת", "יושבים", "מעשן", "מסתובב", "מסתובבים")),
    "at_window": (r"\bwindows?\b", _he("חלון", "חלונות")),
    "at_vehicle": (rf"\b(?:next to|near|beside|by|at|around|into|inside|opens?|opening|checks?|checking|looks? into|"
                   rf"looking into|leans? on|leaning on|touch\w*|tries|trying|loads?|loading|unloads?|unloading|"
                   rf"gets? (?:in|into|out of)|getting (?:in|into|out of))\s+(?:the\s+|a\s+|his\s+|her\s+|their\s+|"
                   rf"parked\s+|white\s+|black\s+|red\s+|silver\s+|grey\s+|gray\s+|blue\s+)*{_VEHICLE_EN}\b",
                   rf"\b{_VEHICLE_EN}\s+door\b",
                   rf"(?<![{_HE}])(?:ליד|בתוך|פותח את|פותחת את|נכנס ל|נכנסת ל|יוצא מ|מעמיס על|פורק מ)\s*ה?"
                   rf"(?:דלת ה)?(?:{_VEHICLE_HE})"),
}

_COMPILED: Dict[str, Tuple[Pattern[str], ...]] = {
    tag: tuple(re.compile(p, re.IGNORECASE) for p in pats) for tag, pats in _PATTERNS.items()}
# A mention right after one of these words is negated ("no animals", "not knocking", "אין אף אחד", "בלי חבילה").
_NEGATION = re.compile(rf"(?:\b(?:no|not|without|nor|never|nobody|none|isn't|aren't|doesn't|don't)\b|"
                       rf"(?<![{_HE}])(?:אין|לא|ללא|בלי|אף)(?![{_HE}]))[^.;,]{{0,18}}$", re.IGNORECASE)


def _negated(text: str, start: int) -> bool:
    return bool(_NEGATION.search(text[max(0, start - 30):start]))


def tags_of(text: str, keep_negated: bool = False) -> List[str]:
    """The activity tags *text* names, in TAGS order. Deterministic; a negated mention does not count unless
    *keep_negated* (an owner's rule "אין שליחים בחצר" is ABOUT deliveries)."""
    text = " ".join(str(text or "").split())
    if not text:
        return []
    out: List[str] = []
    for tag in TAGS:
        for pattern in _COMPILED[tag]:
            if any(keep_negated or not _negated(text, m.start()) for m in pattern.finditer(text)):
                out.append(tag)
                break
    return out


def tags_of_all(texts: Iterable[str]) -> List[str]:
    found = set()
    for text in texts:
        found.update(tags_of(text))
    return [t for t in TAGS if t in found]


def name(tag: str, lang: str = "he") -> str:
    names = NAMES.get(tag) or {}
    return names.get("he" if str(lang).startswith("he") else "en") or tag.replace("_", " ")


def name_at(tag: str, place: str, lang: str = "he") -> str:
    """The tag's name before "at <place>": a knock at a camera named for a door is just "a knock" ("דפיקה בדלת
    האחורית", not "דפיקה בדלת בדלת האחורית")."""
    if tag == "knock" and any(w in str(place or "").casefold() for w in ("דלת", "door")):
        return "דפיקה" if str(lang).startswith("he") else "a knock"
    return name(tag, lang)


def names(tags: Sequence[str], lang: str = "he") -> str:
    joiner = " ו" if str(lang).startswith("he") else " and "
    items = [name(t, lang) for t in tags]
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + joiner + items[-1]
