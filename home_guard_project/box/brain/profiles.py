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
from typing import Dict, List, Optional

log = logging.getLogger("box.brain.profiles")

_DIR = os.path.dirname(os.path.abspath(__file__))
TOOLS_PATH = os.path.join(_DIR, "agent_tools_v2.json")
PROMPTS_DIR = os.path.join(_DIR, "prompts")

COMMON_TOOLS = ("find_events", "summarize_period", "check_camera", "record_clip", "send_media", "pause_alerts",
                "resume_alerts", "set_camera_active", "set_alias", "change_setting", "record_verdict",
                "ask_clarification", "get_alert_settings", "set_alert_types", "set_sensitivity", "reply")
GUARD_TOOLS = COMMON_TOOLS[:2] + ("assess_event",) + COMMON_TOOLS[2:]
ASSISTANT_TOOLS = COMMON_TOOLS[:2] + ("describe_event",) + COMMON_TOOLS[2:]
STATE_TOOLS = ("pause_alerts", "resume_alerts", "set_camera_active", "set_alias", "change_setting", "record_verdict",
               "set_alert_types", "set_sensitivity")
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
    r"(?:less|fewer|more) alerts)\b",
    re.IGNORECASE)
_BIG_WORDS_HE = ("תכבה", "תדליק", "תשתיק", "תפסיק", "עצור", "תמשיך", "תחזיר", "תשנה", "שנה", "תקרא", "טעות",
                 "שגוי", "לא נכון", "זה אני", "זה אנחנו", "אין אף אחד", "למה", "מעצבן", "לא עובד", "שפה",
                 "עברית", "אנגלית", "חשוד",
                 "תפעיל", "הפעל", "תכבי", "כבה", "תדליקי", "הדלק", "השתק", "תשתיקי", "הפסק", "תפסיקי",
                 "תגדיר", "הגדר", "תחזירי", "תמשיכי",
                 "התראות על", "גם על רכבים", "רגישות", "פחות התראות", "יותר התראות",
                 "תזכור", "תזכרי", "זכור", "תרשום", "תרשמי", "תקראי")
# Whole words only (2026-10-05: "למה" matched inside "מצלמה", so every camera message skipped the fast model),
# each with up to two Hebrew prefix letters (ולמה, שתכבה) and a plural or feminine ending on the last word (תפסיקו).
_BIG_HE = re.compile("|".join(
    r"(?<!\w)[ושהבלמכ]{0,2}" + r"\s+".join(map(re.escape, words.split())) + r"[וי]?(?!\w)" for words in _BIG_WORDS_HE))


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
            or bool(_BIG_HE.search(text or "")))


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
