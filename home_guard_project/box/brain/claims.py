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
_OPEN = r"(?:^|[.!?\n]\s{0,3})(?:done\s{0,3}[,.!:\-\u2013\u2014]\s{0,3})?"
_BEEN = r"\bhas been (?:%s) (?:to you|for you|now)\b"
_NOW_AR = r"تم (?:%s)(?:\s+\S+){0,3}\s+الآن"

# Only claims that the assistant itself did something (first person), shows something right now, or states a
# present state tied to now / a deadline. Bare participles ("was sent", "is enabled") are plain facts about
# the house's history and are NOT claims. Hebrew and Arabic entries are regexes too.
CLAIMS: Dict[str, Dict[str, object]] = {
    "send": {
        "tools": {"send_media", "check_camera", "record_clip"},
        "en": [_fp("sent"), _HERE_MEDIA, _BEEN % "sent", _OPEN + r"sent\b"],
        "he": ["שלחתי", "הנה הסרטון", "הנה התמונה", "הנה שני הסרטונים", "הנה הסרטונים", "מצורף"],
        "ar": ["أرسلت", "إليك الفيديو", "إليك الصورة", "مرفق", _NOW_AR % "إرسال"],
    },
    "off": {
        "tools": {"set_camera_active"},
        "en": [r"\bI(?:['\u2019]ve| have| just)?\s+(?:just\s+)?(?:turned|switched)\b[^.?!\n]{0,30}\b(?:off|on)\b",
               _fp("disabled|enabled"),
               r"\b(?:is|are)\s+now\s+(?:off|on|disabled|enabled)\b",
               _BEEN % "turned (?:off|on)|disabled|enabled",
               r"\b(?:off|disabled)\s+(?:until|till)\b",
               r"\b(?:turned|switched)\s+(?:it\s+|the\s+\w+(?:\s+\w+)?\s+)?(?:off|on)\s+now\b",
               _OPEN + r"(?:(?:the\s+)?\w+\s+(?:camera\s+)?)?(?:turned|switched)\s+(?:off|on)\b",
               _OPEN + r"(?:turned|switched)\s+(?:off|on)\b"],
        "he": ["כיביתי", "הדלקתי", "כובתה עד", "כובה עד", "כבויה עכשיו", "כבויה עד"],
        "ar": ["أطفأت", "أوقفت الكاميرا", "شغلت الكاميرا"],
    },
    "pause": {
        "tools": {"pause_alerts"},
        "en": [_fp("paused|muted|silenced"), r"\b(?:is|are)\s+now\s+(?:paused|muted)\b",
               r"\bnow\s+(?:paused|muted)\b", _BEEN % "paused|muted",
               r"\b(?:paused|muted|silenced)\s+(?:until|till|for)\b",
               r"\b(?:is|are)\s+(?:paused|muted)\s+(?:until|till|for)\b",
               _OPEN + r"(?:paused|muted)\b"],
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
        "en": [_fp("marked|saved|changed|updated"), _BEEN % "saved|marked|changed|updated",
               _OPEN + r"(?:saved|marked)\b"],
        "he": ["סימנתי", "שמרתי", "רשמתי", "שיניתי", "עדכנתי", "הגדרתי"],
        "ar": ["سجلت(?! (?:لك )?(?:فيديو|مقطع))", "حفظت", "غيرت", "حدثت", _NOW_AR % "التسجيل|تغيير"],
    },
    "record": {
        "tools": {"record_clip"},
        "en": [_fp("recorded")],
        "he": ["הקלטתי"],
        "ar": ["سجلت (?:لك )?(?:فيديو|مقطع)"],
    },
    "setting": {
        "tools": {"change_setting", "set_alert_types", "set_sensitivity"},
        "en": [r"\bI(?:'ve| have| just)?\s+(?:just\s+)?(?:set|changed|updated|switched)\b", r"\b(?:is|are)\s+now\s+set\s+to\b"],
        "he": ["שיניתי", "עדכנתי", "הגדרתי"],
        "ar": ["غيرت", "حدثت"],
    },
}

_NEGATIONS = {
    "en": {"not", "no", "nothing", "never", "didn't", "haven't", "wasn't", "weren't", "couldn't", "can't"},
    "he": {"לא", "אין", "טרם"},
    "ar": {"لم", "لا", "لن", "ما"},
}
_ALL_NEGATIONS = set().union(*_NEGATIONS.values())


_CLAUSE_END = re.compile(r"[,;:.!?\u2014\u2013-]")
_NOT_NEGATIONS = re.compile(r"\b(?:no problem|not sure)\b")
_AFTER_NEGATIONS = {"no", "nothing", "none"}


def _words(segment: str) -> List[str]:
    return re.findall(r"[\w'\u2019]+", segment.lower().replace("\u2019", "'"))


def _negated(text: str, start: int, end: int = -1) -> bool:
    """True if a negation word is among the 3 words right before *start* (in the same clause), or "no"/
    "nothing"/"none" is among the 2 words right after the match (*end*, same clause)."""
    before = text[max(0, start - 200):start]
    cut = list(_CLAUSE_END.finditer(before))
    if cut:
        before = before[cut[-1].end():]
    before = _NOT_NEGATIONS.sub(" ", before.lower())
    if any(w in _ALL_NEGATIONS for w in _words(before)[-3:]):
        return True
    if end >= 0:
        after = text[end:end + 60]
        m = _CLAUSE_END.search(after)
        if m:
            after = after[:m.start()]
        if any(w in _AFTER_NEGATIONS for w in _words(after)[:2]):
            return True
    return False


def _claimed(patterns: Sequence[str], text: str, flags: int = 0) -> bool:
    return any(not _negated(text, m.start(), m.end()) for p in patterns for m in re.finditer(p, text, flags))


_SETTING_BODY = (
    r"(?:(?:the\s+)?(?:alert hours?|hours|time between alerts|cooldown|sensitivity|language|quiet log|settings?)\b"
    r"|(?:\u05d0\u05ea\s+)?\u05d4?(?:\u05e9\u05e2\u05d5\u05ea|\u05e8\u05d2\u05d9\u05e9\u05d5\u05ea|\u05e9\u05e4\u05d4|\u05d4\u05d2\u05d3\u05e8\u05d5\u05ea"
    r"|\u05d6\u05de\u05df \u05d1\u05d9\u05df|\u05ea\u05d9\u05e2\u05d5\u05d3 \u05e9\u05e7\u05d8)\b"
    r"|(?:\u0627\u0644)?(?:\u0633\u0627\u0639\u0627\u062a|\u062d\u0633\u0627\u0633\u064a\u0629|\u0644\u063a\u0629|\u0625\u0639\u062f\u0627\u062f(?:\u0627\u062a)?)\b)"
)
_SETTING_SUBJECT = re.compile(r"^\s*" + _SETTING_BODY, re.IGNORECASE)
_SETTING_ANYWHERE = re.compile(r"(?<![\w])" + _SETTING_BODY, re.IGNORECASE)


def _setting_claimed(patterns: Sequence[str], text: str, flags: int = 0) -> bool:
    """A settings claim needs a settings subject in the verb's own clause: after the verb ("I changed the alert
    hours") or, for "is now set to", before it ("The hours are now set to")."""
    for pattern in patterns:
        for match in re.finditer(pattern, text, flags):
            if _negated(text, match.start(), match.end()):
                continue
            tail = re.split(r"[,.!?;\n]|\band\s+I\b", text[match.end():], maxsplit=1, flags=re.IGNORECASE)[0]
            head = re.split(r"[,.!?;\n]", text[:match.start()])[-1]
            if _SETTING_SUBJECT.match(tail) or _SETTING_ANYWHERE.search(head):
                return True
    return False


def _save_claimed(patterns: Sequence[str], text: str, flags: int = 0) -> bool:
    for pattern in patterns:
        for match in re.finditer(pattern, text, flags):
            if _negated(text, match.start(), match.end()):
                continue
            # Only overlapping first-person setting verbs need disambiguation. The span following
            # this match stops at the next clause, so an independent saved alias still counts.
            tail = re.split(r"[,.!?;\n]|\band\s+I\b", text[match.end():], maxsplit=1, flags=re.IGNORECASE)[0]
            setting_patterns = list(CLAIMS["setting"]["en"]) + list(CLAIMS["setting"]["he"]) + list(CLAIMS["setting"]["ar"])
            overlaps = any(re.fullmatch(p, match.group(), re.IGNORECASE) for p in setting_patterns)
            if overlaps and _SETTING_SUBJECT.match(tail):
                continue
            return True
    return False


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
        claimed = {"save": _save_claimed, "setting": _setting_claimed}.get(name, _claimed)
        hit = claimed(spec["en"], text, re.IGNORECASE)  # type: ignore[arg-type]
        hit = hit or claimed(list(spec["he"]) + list(spec["ar"]), text)  # type: ignore[arg-type]
        if hit:
            out.append(name)
    return out
