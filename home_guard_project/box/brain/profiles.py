# home_guard_project/box/brain/profiles.py
"""What the model is told and may use, per mode (Guard / Assistant) and tier (fast / big).

Guard and Assistant share the honesty, language and asking rules (common.txt)
and differ in focus and in one tool each: Guard judges a saved clip
(assess_event), Assistant answers questions about one (describe_event). The
fast first responder also gets hand_off, to pass a message to the big model.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Dict, List, Optional, Sequence

log = logging.getLogger("box.brain.profiles")

_DIR = os.path.dirname(os.path.abspath(__file__))
TOOLS_PATH = os.path.join(_DIR, "agent_tools_v2.json")
PROMPTS_DIR = os.path.join(_DIR, "prompts")

COMMON_TOOLS = ("find_events", "summarize_period", "ask_vision", "recent_activity", "check_camera", "look_around", "record_clip",
                "send_media", "pause_alerts", "resume_alerts", "set_camera_active", "set_alias", "change_setting",
                "record_verdict", "mark_known", "ask_clarification", "get_alert_settings", "set_alert_types", "set_sensitivity", "house_status",
                "house_state", "house_expect", "house_cancel", "reply")
GUARD_TOOLS = COMMON_TOOLS[:2] + ("assess_event",) + COMMON_TOOLS[2:]
ASSISTANT_TOOLS = COMMON_TOOLS[:2] + ("describe_event",) + COMMON_TOOLS[2:]
STATE_TOOLS = ("pause_alerts", "resume_alerts", "set_camera_active", "set_alias", "change_setting", "record_verdict",
               "mark_known", "set_alert_types", "set_sensitivity", "house_state", "house_expect", "house_cancel")
PROMPT_VERSIONS = {"guard": "2026-10-03.guard.v1", "assistant": "2026-10-03.assistant.v1"}

# Messages that go straight to the big model, decided in code (the fast model would have to judge its own
# competence otherwise): anything that changes state, a reply to an alert, negation or an upset tone.
_APOS = "['\u2019]"
_BIG_WORDS_EN = re.compile(
    r"\b(?:stop|pause|mute|unmute|silence|quiet|resume|continue|turn(?:ed)?\s+(?:\w+\s+){0,3}(?:on|off)|"
    r"switch (?:on|off)|shut (?:off|down)|disarm|arm|snooze|disable|enable|change|"
    r"set|call (?:it|camera)|rename|wrong|false|mistake|not (?:me|us|true|right)|it" + _APOS + r"?s (?:me|us)|"
    r"that" + _APOS + r"?s me|(?:that |it )?was (?:me|us)|nobody|why|"
    r"angry|annoying|useless|stupid|broken|doesn" + _APOS + r"?t work|language|hebrew|english|suspicious|"
    r"alert me about|alerts for|cars too|also vehicles|animals|sensitivity|sensitive|remember|"
    r"vacation|holiday|asleep|going to (?:bed|sleep)|we left|we" + _APOS + r"?(?:re| are) (?:back|up|home|leaving|away)|"
    r"expecting|"
    r"(?:less|fewer|more) alerts|"
    # the owner says who is there (the Memory Keeper's mark_known: only the big model may save it)
    r"(?:it|that)" + _APOS + r"?s (?:fine|ok|okay)|"
    r"(?:those|these) are (?:my|our|the)|(?:they|those|these)" + _APOS + r"?re (?:my|our|the))\b",
    re.IGNORECASE)
_BIG_WORDS_HE = ("תכבה", "תדליק", "תשתיק", "תפסיק", "עצור", "תמשיך", "תחזיר", "תשנה", "שנה", "תקרא", "טעות",
                 "שגוי", "לא נכון", "זה אני", "זה אנחנו", "אין אף אחד", "למה", "מעצבן", "לא עובד", "שפה",
                 "עברית", "אנגלית", "חשוד",
                 "תפעיל", "הפעל", "תכבי", "כבה", "תדליקי", "הדלק", "השתק", "תשתיקי", "הפסק", "תפסיקי",
                 "תגדיר", "הגדר", "תחזירי", "תמשיכי",
                 "התראות על", "גם על רכבים", "רגישות", "פחות התראות", "יותר התראות",
                 "תזכור", "תזכרי", "זכור", "תרשום", "תרשמי", "תקראי",
                 # house state the code did not parse ("we're off to Eilat till the weekend"): only the big model
                 # may change it
                 "לישון", "ישנים", "יצאנו", "יוצאים", "חזרנו", "קמנו", "חופשה", "מצפים", "מחכים",
                 # "זה בסדר" (the owner says who is there is says_known below: questions about workers stay fast)
                 "זה בסדר")


def hebrew_words(phrases: Sequence[str]) -> "re.Pattern[str]":
    """Hebrew phrases as whole words (2026-10-05: "למה" matched inside "מצלמה", so every camera message skipped
    the fast model), each with up to two prefix letters (ולמה, שתכבה) and a plural or feminine ending on the last
    word (תפסיקו)."""
    return re.compile("|".join(r"(?<!\w)[ושהבלמכ]{0,2}" + r"\s+".join(map(re.escape, words.split())) + r"[וי]?(?!\w)"
                               for words in phrases))


_BIG_HE = hebrew_words(_BIG_WORDS_HE)


# "בטל" as a word, with up to two prefix letters (לבטל, ולבטל, תבטל) and one suffix (בטלו, תבטלי) - never inside
# another word such as בטלפון / לטלפון / בטלוויזיה.
_CANCEL_HE = re.compile(r"(?<!\w)[ולשהתמכ]{0,2}בטל[ויה]?(?!\w)")

def needs_big(text: str, threaded: bool = False) -> bool:
    """True when this message should skip the fast model."""
    if threaded:
        return True
    if text is not None and not isinstance(text, str):
        log.warning("Invalid message text; routing to the big model")
        return True
    return (bool(_BIG_WORDS_EN.search(text or "")) or bool(_CANCEL_HE.search(text or ""))
            or bool(_BIG_HE.search(text or "")) or says_known(text or ""))


# "Are there people near the pergola?" is about right now: a live look, not a search of the saved events
# (2026-10-05: the fast model searched history). A word of the past ("was", "היו", "yesterday") makes it history.
_NOW = re.compile(r"\b(?:now|right now|currently|at the moment)\b|(?<!\w)[וה]?(?:עכשיו|כרגע|כעת)(?!\w)",
                  re.IGNORECASE)
_PRESENCE = re.compile(
    r"\b(?:is|are)\s+there\s+(?:any(?:one|body)?|some(?:one|body)|people|a\s+\w+|\w+s)\b|"
    r"\b(?:is|are)\s+(?:any(?:one|body)|some(?:one|body)|people)\b|"
    r"\bany(?:one|body)\s+(?:there|around|outside|at|near|in)\b|"
    r"(?<!\w)[וה]?יש\s+(?:\S+\s+)?(?:אנשים|מישהו|משהו|אדם|רכב|רכבים|ילדים|עובדים|פועלים|חיות|כלב|חתול)(?!\w)|"
    r"(?<!\w)[וה]?מישהו(?!\w)|"
    # "anything outside?" (2026-10-06: answered from the 23:14 history instead of a live look)
    r"(?<!\w)[וה]?קורה\s+משהו(?!\w)|(?<!\w)מה\s+(?:יש|קורה|רואים)\s+[בל]?חוץ(?!\w)|(?<!\w)משהו\s+מעניין(?!\w)|"
    r"(?<!\w)הכל\s+(?:שקט|בסדר)(?!\w)|"
    r"\bany(?:thing|one)\s+(?:outside|out there|going on|happening|interesting|unusual|new)\b|"
    r"\bwhat" + _APOS + r"?s\s+(?:outside|out there|going on)\b|\ball\s+(?:quiet|good|ok|clear)\b", re.IGNORECASE)
_PAST = re.compile(
    r"\b(?:was|were|did|had|happened|came|yesterday|earlier|last night|today|this morning|ago|before)\b|"
    r"(?<!\w)[וש]?(?:היה|היו|היתה|הייתה|קרה|הגיע|הגיעו|אתמול|קודם|הבוקר|הלילה|היום|לפני|מאתמול)(?!\w)",
    re.IGNORECASE)


def asks_about_now(text: str) -> bool:
    """True when the message asks what is happening right now (people near X, anyone at the gate)."""
    if not isinstance(text, str) or not text.strip() or _PAST.search(text):
        return False
    return bool(_NOW.search(text) or _PRESENCE.search(text))


def load_schemas(path: str = TOOLS_PATH) -> Dict[str, Dict]:
    """Load usable definitions, skipping malformed entries with one warning per load.

    An unreadable or wrongly shaped file raises RuntimeError: a model without its tools must not run.
    """
    try:
        with open(path, encoding="utf-8") as f:
            items = json.load(f)
        if not isinstance(items, list):
            raise ValueError("tool schemas must be a list")
    except (OSError, ValueError, TypeError) as exc:
        raise RuntimeError(f"Cannot load tool schemas from {path}: {exc}") from exc
    schemas = {}
    skipped = 0
    for item in items:
        function = item.get("function") if isinstance(item, dict) else None
        name = function.get("name") if isinstance(function, dict) else None
        parameters = function.get("parameters") if isinstance(function, dict) else None
        if (not isinstance(name, str) or not name or item.get("type") != "function"
                or not isinstance(parameters, dict) or parameters.get("type") != "object"
                or not isinstance(parameters.get("properties", {}), dict)):
            skipped += 1
            continue
        schemas[name] = item
    if skipped:
        log.warning("Skipped %d malformed tool schemas", skipped)
    return schemas


_SCHEMAS: Optional[Dict[str, Dict]] = None    # loaded on first use so a broken file cannot break the import


def _schemas() -> Dict[str, Dict]:
    global _SCHEMAS
    if _SCHEMAS is None:
        _SCHEMAS = load_schemas(TOOLS_PATH)
    return _SCHEMAS


def tool_names(mode: str, tier: str = "big") -> List[str]:
    names = list(GUARD_TOOLS if mode == "guard" else ASSISTANT_TOOLS)
    if tier == "fast":
        names = [n for n in names if n not in STATE_TOOLS] + ["hand_off"]   # the fast model never changes state
    return names


def tools_for(mode: str, tier: str = "big") -> List[Dict]:
    names = tool_names(mode, tier)
    schemas = _schemas()
    missing = [name for name in names if name not in schemas]
    if missing:
        log.warning("Skipping unavailable tool schemas: %s", ", ".join(missing))
    return [schemas[name] for name in names if name in schemas]


def _read(name: str) -> str:
    with open(os.path.join(PROMPTS_DIR, name), encoding="utf-8", errors="replace") as f:
        return f.read().strip()


def system_prompt(mode: str, retention_days: float, tier: str = "big") -> str:
    try:
        if isinstance(retention_days, bool):
            raise ValueError("retention must be a number of days")
        days = int(retention_days)
    except (TypeError, ValueError, OverflowError):
        log.warning("Invalid retention days; using the existing 14-day default")
        days = 14
    try:
        parts = [_read("common.txt").replace("{retention_days}", str(days)),
                 _read("guard.txt" if mode == "guard" else "assistant.txt")]
        if tier == "fast":
            parts.append(_read("fast.txt"))
        if any("\ufffd" in part for part in parts):
            log.warning("Replaced invalid UTF-8 in system prompt")
        return "\n\n".join(parts)
    except (OSError, ValueError, TypeError) as exc:
        raise RuntimeError(f"Cannot read system prompt from {PROMPTS_DIR}: {exc}") from exc


# The owner says who the people are ("זה בסדר זה עובדים אצלי", "it's me", "זה הגנן"): the Memory Keeper's
# mark_known, not a verdict (2026-10-07: "רשמתי את זה כהתרעה צפויה" and 142 more alerts about the same workers).
_KNOWN_HE = re.compile(
    r"(?<!\w)(?:זה|זאת|זו|אלה|אלו|הם|הן)\s+(?:\S+\s+){0,2}?ה?(?:עובדים|פועלים|גנן|שכן|שכנה|שכנים|"
    r"מנקה|מנקים|ילדים|אשתי|בעלי|אמא|אבא|אני|אנחנו|משפחה|חברים|אורחים|שליח|מוכרים|מוכר)(?!\w)|"
    r"(?<!\w)(?:עובדים|פועלים)\s+(?:אצלי|אצלנו|שלי|שלנו)(?!\w)|(?<!\w)אני\s+(?:\(\S+\)\s+)?יוצא(?!\w)")
_KNOWN_EN = re.compile(
    r"\b(?:it|that)" + _APOS + r"?s\s+(?:just\s+)?(?:me|us|my|our|the\s+(?:gardener|neighbou?rs?|workers|cleaner|kids))\b|"
    r"\b(?:those|these|they)\s+(?:are|" + _APOS + r"re)\s+(?:just\s+)?(?:my|our|the)\s+"
    r"(?:workers|gardener|neighbou?rs?|cleaners?|kids|family|guests|builders)\b|\bmy\s+workers\b", re.IGNORECASE)


def says_known(text: str) -> bool:
    """True when the message says who the people at a camera are (for mark_known); a question ("זה הגנן?") is not."""
    return (isinstance(text, str) and "?" not in text
            and bool(_KNOWN_HE.search(text) or _KNOWN_EN.search(text)))


# "על איזה סרטון אתה מדבר?" (2026-10-07: answered twice with the same video and once as a verdict). The box answers
# with the event being discussed, in code.
_WHICH_EVENT = re.compile(
    r"(?<!\w)(?:איזה|איזו|אילו)\s+(?:\S+\s+)?(?:סרטון|סירטון|וידאו|התרעה|התראה|הודעה|תזכורת|אירוע|תמונה)(?!\w)|"
    r"(?<!\w)על\s+מה\s+(?:אתה\s+|את\s+)?(?:מדבר|מדברת|דיברת)(?!\w)|"
    r"\b(?:which|what)\s+(?:\w+\s+)?(?:video|clip|alert|event|message|reminder)\b|"
    r"\bwhat\s+are\s+you\s+talking\s+about\b", re.IGNORECASE)


def asks_which_event(text: str) -> bool:
    """True when the owner asks which video / alert the assistant means."""
    return isinstance(text, str) and bool(_WHICH_EVENT.search(text))
