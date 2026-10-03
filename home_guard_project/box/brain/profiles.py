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
from typing import Dict, List

log = logging.getLogger("box.brain.profiles")

_DIR = os.path.dirname(os.path.abspath(__file__))
TOOLS_PATH = os.path.join(_DIR, "agent_tools_v2.json")
PROMPTS_DIR = os.path.join(_DIR, "prompts")

COMMON_TOOLS = ("find_events", "summarize_period", "check_camera", "record_clip", "send_media", "pause_alerts",
                "resume_alerts", "set_camera_active", "set_alias", "change_setting", "record_verdict",
                "ask_clarification", "reply")
GUARD_TOOLS = COMMON_TOOLS[:2] + ("assess_event",) + COMMON_TOOLS[2:]
ASSISTANT_TOOLS = COMMON_TOOLS[:2] + ("describe_event",) + COMMON_TOOLS[2:]
STATE_TOOLS = ("pause_alerts", "resume_alerts", "set_camera_active", "set_alias", "change_setting", "record_verdict")
PROMPT_VERSIONS = {"guard": "2026-10-03.guard.v1", "assistant": "2026-10-03.assistant.v1"}

# Messages that go straight to the big model, decided in code (the fast model would have to judge its own
# competence otherwise): anything that changes state, a reply to an alert, negation or an upset tone.
_BIG_WORDS_EN = re.compile(
    r"\b(?:stop|pause|mute|silence|quiet|resume|continue|turn (?:on|off)|switch (?:on|off)|disable|enable|change|"
    r"set|call (?:it|camera)|rename|wrong|false|mistake|not (?:me|us|true|right)|it'?s (?:me|us)|nobody|why|"
    r"angry|annoying|useless|stupid|broken|doesn'?t work|language|hebrew|english|suspicious)\b", re.IGNORECASE)
_BIG_WORDS_HE = ("תכבה", "תדליק", "תשתיק", "תפסיק", "עצור", "תמשיך", "תחזיר", "תשנה", "שנה", "תקרא", "טעות",
                 "שגוי", "לא נכון", "זה אני", "זה אנחנו", "אין אף אחד", "למה", "מעצבן", "לא עובד", "שפה",
                 "עברית", "אנגלית", "חשוד")


def needs_big(text: str, threaded: bool = False) -> bool:
    """True when this message should skip the fast model."""
    if threaded:
        return True
    if text is not None and not isinstance(text, str):
        log.warning("Invalid message text; routing to the big model")
        return True
    return bool(_BIG_WORDS_EN.search(text or "")) or any(w in (text or "") for w in _BIG_WORDS_HE)


def load_schemas(path: str = TOOLS_PATH) -> Dict[str, Dict]:
    """Load usable definitions, skipping malformed entries with one warning per load."""
    try:
        with open(path, encoding="utf-8") as f:
            items = json.load(f)
        if not isinstance(items, list):
            raise ValueError("tool schemas must be a list")
    except (OSError, ValueError, TypeError) as exc:
        log.warning("Cannot load tool schemas: %s", exc)
        return {}
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


_SCHEMAS = load_schemas()


def tool_names(mode: str, tier: str = "big") -> List[str]:
    names = list(GUARD_TOOLS if mode == "guard" else ASSISTANT_TOOLS)
    if tier == "fast":
        names = [n for n in names if n not in STATE_TOOLS] + ["hand_off"]   # the fast model never changes state
    return names


def tools_for(mode: str, tier: str = "big") -> List[Dict]:
    names = tool_names(mode, tier)
    missing = [name for name in names if name not in _SCHEMAS]
    if missing:
        log.warning("Skipping unavailable tool schemas: %s", ", ".join(missing))
    return [_SCHEMAS[name] for name in names if name in _SCHEMAS]


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
        log.warning("Cannot read system prompt: %s", exc)
        return ""
