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

CLAIMS: Dict[str, Dict[str, object]] = {
    "send": {
        "tools": {"send_media", "check_camera", "record_clip"},
        "en": [r"\bsent\b", r"\bsending\b", r"\bhere (?:is|are) (?:the |your )?(?:\w+ )?(?:videos?|photos?|pictures?|clips?)\b",
               r"\battached\b"],
        "he": ["שלחתי", "נשלח", "מצורף", "הנה הסרטון", "הנה התמונה", "הנה שני הסרטונים", "הנה הסרטונים"],
        "ar": ["أرسلت", "تم إرسال", "مرفق", "إليك الفيديو", "إليك الصورة"],
    },
    "off": {
        "tools": {"set_camera_active"},
        "en": [r"\bturned (?:it |the camera )?(?:off|on)\b", r"\bswitched (?:off|on)\b", r"\bdisabled\b",
               r"\benabled\b"],
        "he": ["כיביתי", "כובתה", "כובה", "הדלקתי", "הופעלה מחדש"],
        "ar": ["أطفأت", "أوقفت الكاميرا", "شغلت الكاميرا"],
    },
    "pause": {
        "tools": {"pause_alerts"},
        "en": [r"\bpaused\b", r"\bmuted\b", r"\bsilenced\b"],
        "he": ["השתקתי", "הושתקו", "מושתקות"],
        "ar": ["كتمت", "أوقفت التنبيهات", "تم إيقاف التنبيهات"],
    },
    "resume": {
        "tools": {"resume_alerts"},
        "en": [r"\bback on\b", r"\bresumed\b"],
        "he": ["חזרו לפעול", "החזרתי את ההתראות"],
        "ar": ["عادت التنبيهات", "أعدت تشغيل التنبيهات"],
    },
    "save": {
        "tools": {"record_verdict", "set_alias"},
        "en": [r"\bmarked\b", r"\bsaved\b", r"\bnoted\b"],
        "he": ["סימנתי", "שמרתי", "נרשם", "רשמתי"],
        "ar": ["سجلت", "حفظت", "تم التسجيل"],
    },
    "record": {
        "tools": {"record_clip"},
        "en": [r"\brecorded (?:a|the|you)\b"],
        "he": ["הקלטתי"],
        "ar": ["سجلت فيديو"],
    },
}


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
    low = text.lower()
    out = []
    for name, spec in CLAIMS.items():
        if set(spec["tools"]) & backed:  # type: ignore[arg-type]
            continue
        hit = any(re.search(p, low) for p in spec["en"])  # type: ignore[union-attr]
        hit = hit or any(w in text for w in list(spec["he"]) + list(spec["ar"]))  # type: ignore[arg-type]
        if hit:
            out.append(name)
    return out
