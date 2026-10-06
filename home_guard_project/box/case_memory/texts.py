"""Owner-facing words in Hebrew (the owner reads Hebrew) and English. Templates only, no model."""
from __future__ import annotations

from datetime import datetime
from typing import Dict, Sequence, Tuple

from .models import ALL_DAYS, WORKWEEK, Case, Signature

WHO_TEXT: Dict[str, Tuple[str, str]] = {
    "neighbour": ("שכן", "a neighbour"),
    "family": ("בן משפחה", "family"),
    "worker": ("עובד / ספק", "a worker or supplier"),
    "courier": ("שליח", "a courier"),
    "other": ("אחר", "someone else"),
}

ZONE_HE = {"street": "רחוב", "entrance": "כניסה", "window": "חלון", "gate": "שער", "fence": "גדר", "yard": "חצר",
           "parking": "חניה", "car": "רכב", "roof": "גג", "door": "דלת", "other": "אחר"}

DAY_HE = ("שני", "שלישי", "רביעי", "חמישי", "שישי", "שבת", "ראשון")     # Python weekday order
DAY_EN = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")
_DAY_ORDER = (6, 0, 1, 2, 3, 4, 5)                                       # Sunday first, as in Israel


def pick(he: str, en: str, lang: str) -> str:
    return he if lang == "he" else en


def path_text(path: Sequence[str], lang: str) -> str:
    if lang == "he":
        return " ← ".join(ZONE_HE.get(z, z) for z in path)
    return " > ".join(path)


def path_sentence(path: Sequence[str], lang: str) -> str:
    """ "מהשער לרחוב" / "from the gate to the street"."""
    if not path:
        return ""
    if len(path) == 1:
        return pick(f"ב{ZONE_HE.get(path[0], path[0])}", f"at the {path[0]}", lang)
    a, b = path[0], path[-1]
    return pick(f"מה{ZONE_HE.get(a, a)} ל{ZONE_HE.get(b, b)}", f"from the {a} to the {b}", lang)


def days_text(days: Sequence[int], lang: str) -> str:
    wanted = set(days)
    if wanted == set(ALL_DAYS):
        return pick("כל יום", "every day", lang)
    if wanted == set(WORKWEEK):
        return pick("ימי חול (א׳–ה׳)", "weekdays (Sun-Thu)", lang)
    names = DAY_HE if lang == "he" else DAY_EN
    return ", ".join(names[d] for d in _DAY_ORDER if d in wanted)


def day_name(weekday: int, lang: str) -> str:
    return pick(f"יום {DAY_HE[weekday]}", DAY_EN[weekday], lang)


def hours_text(hours: Sequence[str]) -> str:
    return f"{hours[0]}–{hours[1]}"


def date_short(ts: float) -> str:
    """ "5.10": day.month, as the owner writes dates."""
    moment = datetime.fromtimestamp(ts)
    return f"{moment.day}.{moment.month}"


def case_title(case: Case, lang: str) -> str:
    if case.title:
        return case.title
    he, en = WHO_TEXT.get(case.who, WHO_TEXT["other"])
    return pick(he, en, lang)


def case_line(case: Case, lang: str) -> str:
    """One line describing what a case remembers (listing and summary card)."""
    s = case.scope
    bits = [case_title(case, lang), days_text(s.weekdays, lang), hours_text(s.hours)]
    where = path_sentence(s.path, lang)
    if where:
        bits.append(where)
    if case.recognise:
        bits.append(pick("מזהים לפי: ", "recognised by: ", lang) + ", ".join(case.recognise))
    return " · ".join(bits)


# -- what the memory adds to a delivery ----------------------------------------------------------------------------

def softened_text(case: Case, lang: str) -> str:
    when = date_short(case.created_at)
    return pick(f"רגיל (לפי ההסבר שלך מ-{when}): {case_title(case, 'he')}",
                f"Normal (per your explanation from {when}): {case_title(case, 'en')}", lang)


def digest_line(case: Case, sig: Signature, lang: str) -> str:
    clock = f"{sig.minute // 60:02d}:{sig.minute % 60:02d}"
    return f"{clock} {sig.camera}: {softened_text(case, lang)}"


def shadow_text(case: Case, lang: str) -> str:
    return pick(f"זיהיתי: {case_title(case, 'he')}. בפעם הבאה לא אתריע על זה, בסדר?",
                f"I recognised: {case_title(case, 'en')}. Next time I won't alert for this, OK?", lang)


def keep_alerting_text(case: Case, lang: str) -> str:
    return pick(f"זיהיתי: {case_title(case, 'he')} (ביקשת שאמשיך להתריע).",
                f"I recognised: {case_title(case, 'en')} (you asked me to keep alerting).", lang)


def context_text(case: Case, lang: str) -> str:
    """For a suspicious event that resembles a case: the case never softens it."""
    return pick(f"דומה ל{case_title(case, 'he')}, אבל ההתרעה בתוקף.",
                f"Similar to {case_title(case, 'en')}, but this alert stands.", lang)


def difference_text(field: str, sig: Signature, lang: str) -> str:
    clock = f"{sig.minute // 60:02d}:{sig.minute % 60:02d}"
    if field in ("path", "exit_edge", "entry_edge"):
        where = path_sentence(sig.path, lang) or pick("במקום אחר", "somewhere else", lang)
        return pick(f"הלך {where}", f"went {where}", lang)
    if field == "dwell":
        secs = f"{sig.dwell_s:.0f}" if sig.dwell_s is not None else "?"
        return pick(f"שהה {secs} שניות", f"stayed {secs} seconds", lang)
    if field == "people":
        return pick(f"היו {sig.people} אנשים", f"there were {sig.people} people", lang)
    if field == "vehicles":
        return pick(f"היו {sig.vehicles} רכבים", f"there were {sig.vehicles} vehicles", lang)
    if field == "hour":
        return pick(f"הגיע ב-{clock}", f"came at {clock}", lang)
    if field == "appearance":
        return pick("נראה אחרת", "looked different", lang)
    if field == "movement":
        return pick("התנהג אחרת", "moved differently", lang)
    if field == "weekday":
        return pick(f"ב{day_name(sig.weekday, 'he')}", f"on {day_name(sig.weekday, 'en')}", lang)
    return pick(f"{field} שונה", f"the {field} was different", lang)


def similar_text(case: Case, sig: Signature, mismatched: Sequence[str], lang: str) -> str:
    seen, diffs = set(), []
    for f in mismatched:
        text = difference_text(f, sig, lang)
        if text not in seen:
            seen.add(text)
            diffs.append(text)
    joined = pick(" ו", " and ", lang).join(diffs) if diffs else pick("משהו השתנה", "something changed", lang)
    return pick(f"נראה כמו {case_title(case, 'he')}, אבל הפעם {joined}",
                f"Looks like {case_title(case, 'en')}, but this time {joined}", lang)


BUTTONS: Dict[str, Tuple[str, str]] = {
    "confirm": ("כן", "Yes"),
    "not_them": ("לא הוא", "Not them"),
    "keep_alerting": ("לא, תמשיך להתריע", "No, keep alerting"),
    "keep": ("כן, עדיין רלוונטי", "Yes, still relevant"),
    "forget": ("לא, תשכח", "No, forget it"),
    "delete": ("מחק", "Delete"),
    "fine_too": ("זה בסדר גם ככה", "That's fine too"),   # similar-but-different: proposes a widening
}


def button(action: str, case_id: str = "", **extra: str) -> Dict[str, str]:
    he, en = BUTTONS[action]
    return {"action": action, "case_id": case_id, "he": he, "en": en, **extra}


def review_question(case: Case, lang: str) -> str:
    return pick(f"לא ראיתי את זה 30 יום: {case_line(case, 'he')}. עדיין רלוונטי?",
                f"I haven't seen this for 30 days: {case_line(case, 'en')}. Still relevant?", lang)
