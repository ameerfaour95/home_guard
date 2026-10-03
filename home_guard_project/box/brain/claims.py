# home_guard_project/box/brain/claims.py
"""The safety net for the model's answer text: action words with no receipt behind them.

The prompt forbids describing actions; this catches the cases where the model
does it anyway ("here are the two videos", "המצלמה כובתה") although no tool
did it. The agent then asks for a rewrite once, and otherwise sends only the
code-written confirmation lines.
"""

from __future__ import annotations

import logging
import re
from typing import Dict, List, Sequence

from .receipts import DONE, FAILED, REQUESTED, Receipt

log = logging.getLogger("box.brain.claims")

def _fp(verbs: str) -> str:
    """English first-person action claim: "I sent", "I've just paused", "I have muted"."""
    return r"\bI(?:['\u2019]ve| have| just)?\s+(?:just\s+)?(?:" + verbs + r")\b"


_HERE_MEDIA = r"\bhere(?:['\u2019]s| is| are)\b[^.?!\n]{0,40}\b(?:videos?|photos?|pictures?|clips?|recordings?)\b"
_BEEN = r"\bhas been (?:%s) (?:to you|for you|now)\b"
_NOW_AR = r"تم (?:%s)(?:\s+\S+){0,3}\s+الآن"

# Only claims that the assistant itself did something (first person), shows something right now, or states a
# present state tied to now / a deadline. Bare participles ("was sent", "is enabled") are plain facts about
# the house's history and are NOT claims. Hebrew and Arabic entries are regexes too.
CLAIMS: Dict[str, Dict[str, object]] = {
    "send": {
        "tools": {"send_media", "check_camera", "record_clip"},
        "en": [_fp("sent"), _HERE_MEDIA, _BEEN % "sent"],
        "he": ["שלחתי", "הנה הסרטון", "הנה התמונה", "הנה שני הסרטונים", "הנה הסרטונים", "מצורף"],
        "ar": ["أرسلت", "إليك الفيديو", "إليك الصورة", "مرفق", _NOW_AR % "إرسال"],
    },
    "off": {
        "tools": {"set_camera_active"},
        "en": [r"\bI(?:['\u2019]ve| have| just)?\s+(?:just\s+)?(?:turned|switched)\b[^.?!\n]{0,30}\b(?:off|on)\b",
               _fp("disabled|enabled"),
               r"\b(?:is|are)\s+now\s+(?:off|on|disabled|enabled)\b",
               _BEEN % "turned (?:off|on)|disabled|enabled"],
        "he": ["כיביתי", "הדלקתי", "כובתה עד", "כובה עד", "כבויה עכשיו", "כבויה עד"],
        "ar": ["أطفأت", "أوقفت الكاميرا", "شغلت الكاميرا"],
    },
    "pause": {
        "tools": {"pause_alerts"},
        "en": [_fp("paused|muted|silenced"), r"\b(?:is|are)\s+now\s+(?:paused|muted)\b",
               r"\bnow\s+(?:paused|muted)\b", _BEEN % "paused|muted"],
        "he": ["השתקתי", "מושתקות עד", "הושתקו עד", "מושתקת עד"],
        "ar": ["كتמت", "أوقفت التنبيهات", _NOW_AR % "إيقاف التنبيهات"],
    },
    "resume": {
        "tools": {"resume_alerts"},
        "en": [_fp("resumed"), r"\b(?:is|are)\s+now\s+back on\b"],
        "he": ["החזרתי"],
        "ar": ["أعدت تشغيل التنبيهات", "عادت التنبيهات"],
    },
    "save": {
        "tools": {"record_verdict", "set_alias"},
        "en": [_fp("marked|saved|noted|changed|updated|set"), _BEEN % "saved|marked|changed|updated"],
        "he": ["סימנתי", "שמרתי", "רשמתי", "שיניתי", "עדכנתי", "הגדרתי"],
        "ar": ["سجلت(?! (?:لك )?(?:فيديو|مقطع))", "حفظت", "غيرت", "حدثت", _NOW_AR % "التسجيل|تغيير"],
    },
    "record": {
        "tools": {"record_clip"},
        "en": [_fp("recorded")],
        "he": ["הקלטתי"],
        "ar": ["سجلت (?:لك )?(?:فيديو|مقطع)"],
    },
}

_NEGATIONS = {
    "en": {"not", "no", "nothing", "never", "didn't", "haven't", "wasn't", "weren't", "couldn't", "can't"},
    "he": {"לא", "אין", "טרם"},
    "ar": {"لم", "لا", "لن", "ما"},
}
_ALL_NEGATIONS = set().union(*_NEGATIONS.values())


def _negated(text: str, start: int) -> bool:
    """True if a negation word is among the 3 words right before *start*."""
    words = re.findall(r"[\w'\u2019]+", text[:start].lower().replace("\u2019", "'"))[-3:]
    return any(w in _ALL_NEGATIONS for w in words)


def _claimed(patterns: Sequence[str], text: str, flags: int = 0) -> bool:
    return any(not _negated(text, m.start()) for p in patterns for m in re.finditer(p, text, flags))


def unbacked_claims(answer: str, receipts: Sequence[Receipt]) -> List[str]:
    """Claim kinds in *answer* that no ``done`` receipt of this turn backs, in CLAIMS order. A ``requested``
    receipt (a camera change waiting for the restart) backs nothing: "turned off" would be premature.
    Malformed inputs are skipped, with at most one warning per call."""
    malformed = not isinstance(answer, str)
    text = answer if isinstance(answer, str) else ""
    if not isinstance(receipts, Sequence) or isinstance(receipts, (str, bytes, bytearray)):
        receipts = ()
        malformed = True
    backed = set()
    for receipt in receipts:
        if (not isinstance(receipt, Receipt) or not isinstance(receipt.tool, str)
                or not isinstance(receipt.status, str) or receipt.status not in (DONE, FAILED, REQUESTED)):
            malformed = True
            continue
        if receipt.status == DONE:
            backed.add(receipt.tool)
    if malformed:
        log.warning("Skipped malformed input while checking action claims")
    out = []
    for name, spec in CLAIMS.items():
        if set(spec["tools"]) & backed:  # type: ignore[arg-type]
            continue
        hit = _claimed(spec["en"], text, re.IGNORECASE)  # type: ignore[arg-type]
        hit = hit or _claimed(list(spec["he"]) + list(spec["ar"]), text)  # type: ignore[arg-type]
        if hit:
            out.append(name)
    return out
