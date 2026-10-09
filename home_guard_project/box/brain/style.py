# home_guard_project/box/brain/style.py
"""The last pass over every text the assistant sends: no closing boilerplate, no camera ids.

Owner decisions 2026-10-08, after two days of the real chat (13 of 25 replies ended with "אם יש משהו נוסף…
אני כאן", one complaint was "תפסיק עם ההודעה המטומטת הזאת של אם יש עוד משהו… זה מעצבן"):

- never a closing offer ("If there's anything else…", "Let me know if…", "אני כאן") and never "תודה על
  ההבהרה" - the model writes them, so code removes them, whatever the prompt says;
- never a camera id ("ameer_week_0_1_ch6"): the family's name for it, else "מצלמה 6" (``camera_names``).

``clean_outgoing`` is applied in the send path (``telegram_agent.TelegramInbox._say`` and ``Deliverer.text``),
for v1 and v2 alike. It never raises: a failure returns the text unchanged.
"""

from __future__ import annotations

import logging
import re
from typing import Iterable, List, Optional

log = logging.getLogger("box.brain.style")

# A clause that only offers more help or thanks for nothing. Searched inside one sentence; the sentence (or,
# when real words come first, the part from the clause on) is dropped.
_BOILERPLATE = [re.compile(p, re.IGNORECASE) for p in (
    # Hebrew
    r"(?<!\w)ו?אם\s+(?:יש|יהיה|תרצ[הי]|תצטרכ[וי]?|צריך|תהיה)\s+(?:לך\s+|לכם\s+)?(?:עוד\s+)?"
    r"(?:משהו|דבר|שאלה|שאלות|בקשה)",
    r"(?<!\w)ו?אם\s+(?:תרצ[הי]|תצטרכ[וי]?)\s+(?:עוד|משהו|לדעת|לשאול|לשנות)",
    r"(?<!\w)אני\s+כאן(?!\w)",
    r"(?<!\w)אני\s+זמי(?:ן|נה)(?!\w)",
    r"(?<!\w)תודה\s+על\s+ה(?:הבהרה|עדכון|מידע|שיתוף|פירוט|תשובה)",
    r"(?<!\w)(?:אשמח|שמח(?:ה)?)\s+לעזור",
    r"(?<!\w)אל\s+תהסס[וי]?",
    # 2026-10-09 replay: "אם יש צורך בתיקון נוסף, אנא עדכן אותי", "אני מבין את התסכול שלך ואשתדל לשפר"
    r"(?<!\w)ו?אם\s+(?:יש|יהיה)\s+צורך\s+ב",
    r"(?<!\w)אנא\s+(?:עדכן|עדכני|תעדכן)\s+אותי",
    r"(?<!\w)(?:ו?אני\s+)?מבי(?:ן|נה)\s+את\s+ה?(?:תסכול|כעס)",
    r"(?<!\w)ו?(?:אני\s+)?(?:אשתדל|אשתפר)(?!\w)",
    r"\bI\s+understand\s+your\s+frustration\b",
    r"\bI(?:['’]ll| will)\s+(?:try|do\s+(?:my\s+best|better))\b",
    r"(?<!\w)רק\s+(?:תגיד|תכתוב|תודיע)",
    # English
    r"\bif\s+(?:there(?:['’]s| is)|you\s+(?:need|have|want|would like))\b[^.!?\n]*?"
    r"\b(?:anything|something|else|questions?|more)\b",
    r"\blet\s+me\s+know\b",
    r"\bI(?:['’]m| am)\s+(?:always\s+)?here\b",
    r"\bfeel\s+free\b",
    r"\b(?:happy|glad)\s+to\s+help\b",
    r"\bthanks?(?:\s+you)?\s+for\s+(?:the\s+|your\s+)?(?:clarif\w*|update|information|info|letting me know)\b",
    r"\banything\s+else\b",
    r"\bdon['’]?t\s+hesitate\b",
)]

_SENTENCE = re.compile(r"(?<=[.!?…])\s+")
# What may be left dangling before a dropped clause: commas, dashes, a lone "and" / "ו".
_TAIL = re.compile(r"[\s,;:\-–—]*(?:(?<!\w)(?:and|but|so|ו|אבל)\s*)?[\s,;:\-–—]*$", re.IGNORECASE)


def _clean_sentence(sentence: str) -> str:
    start = min((m.start() for p in _BOILERPLATE for m in [p.search(sentence)] if m), default=-1)
    if start < 0:
        return sentence
    head = sentence[:start]
    if not re.search(r"\w", head):
        return ""
    head = _TAIL.sub("", head).rstrip()
    if not re.search(r"\w", head):
        return ""
    return head if head[-1:] in ".!?…" else head + "."


# An opening that only empathises before the real answer ("אני מבין אותך. ההתראה הייתה בכניסה"): dropped when real
# words follow (2026-10-09); a reply that is nothing else is the agent's to rewrite (claims.empty_reply).
_LEADING_EMPATHY = re.compile(
    r"^\s*(?:ו?אני\s+)?(?:מבין|מבינה)\s+(?:אותך|את\s+ה?(?:תסכול|כעס)(?:\s+שלך)?)\s*[.!,]\s*|"
    r"^\s*I\s+(?:completely\s+)?understand(?:\s+(?:you|your\s+frustration|how\s+you\s+feel))?\s*[.!,]\s*",
    re.IGNORECASE)


def strip_boilerplate(text: str) -> str:
    """*text* without closing offers and empty thanks; may be "" when that is all it was."""
    if not isinstance(text, str) or not text.strip():
        return text if isinstance(text, str) else ""
    rest = _LEADING_EMPATHY.sub("", text, count=1)
    if rest != text and re.search(r"\w{2,}", rest):
        text = rest
    lines: List[str] = []
    for line in text.split("\n"):
        if not line.strip():
            if lines and lines[-1] != "":
                lines.append("")
            continue
        kept = [s for s in (_clean_sentence(part) for part in _SENTENCE.split(line.strip())) if s.strip()]
        if kept:
            lines.append(" ".join(kept))
    while lines and lines[-1] == "":
        lines.pop()
    return "\n".join(lines)


# A camera id of the box's own kind: <site>_chN ("ameer_week_0_1_ch6", "ameer_tes2_ch6"). A Hebrew prefix letter
# may be glued to it ("בameer_tes2_ch6").
_CAMERA_ID = re.compile(r"(?<![A-Za-z0-9_])[A-Za-z][A-Za-z0-9]*(?:_[A-Za-z0-9]+)*_ch\d+(?![A-Za-z0-9_])")


def _lang_of(text: str, lang: Optional[str]) -> str:
    if lang:
        return str(lang)
    from .i18n import detect_language  # noqa: PLC0415

    return detect_language(text) or "en"


def replace_camera_ids(text: str, cameras: Iterable[str] = (), lang: Optional[str] = None) -> str:
    """*text* with every camera id (the box's cameras and any ``<site>_chN`` id) as the owner's name for it."""
    from ..camera_names import display_name, replace_ids  # noqa: PLC0415

    lang = _lang_of(text, lang)
    out = replace_ids(text, [str(c) for c in cameras or () if c], lang)
    out = _CAMERA_ID.sub(lambda m: display_name(m.group(0), lang), out)
    # "במצלמה ameer_x_ch6" became "במצלמה מצלמה 6": one "מצלמה" is enough.
    out = _DOUBLE_HE.sub(r"\1מצלמה", out)
    return _DOUBLE_EN.sub(r"\1amera", out)


_DOUBLE_HE = re.compile(r"(?<!\w)([ובלמה]{0,2})מצלמה\s+מצלמה(?=\s+\d)")
_DOUBLE_EN = re.compile(r"\b([Cc])amera\s+Camera(?=\s+\d)")


def clean_outgoing(text: str, cameras: Iterable[str] = (), lang: Optional[str] = None) -> str:
    """The text as it may reach the owner. Never raises; never returns "" for a non-empty text."""
    if not isinstance(text, str) or not text.strip():
        return text if isinstance(text, str) else ""
    try:
        out = replace_camera_ids(strip_boilerplate(text), cameras, lang)
        return out if out.strip() else "👍"
    except Exception as exc:  # noqa: BLE001 - a filter must never stop an answer
        log.warning("Outgoing text not cleaned: %s", exc)
        return text
