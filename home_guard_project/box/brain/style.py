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
from typing import Dict, Iterable, List, Optional

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
    r"(?<!\w)ו?אם\s+(?:יש|יהיה)\s+צורך(?!\w)",
    r"(?<!\w)אני\s+יכול(?:ה)?\s+לעזור(?!\w)",
    r"(?<!\w)ה?אם\s+יש\s+(?:עוד\s+)?(?:משהו|דבר)\s+(?:נוסף|אחר)",
    r"(?<!\w)ו?אם\s+יש\s+לך\s+(?:עוד\s+)?(?:מידע|שאלות|משהו)",
    r"(?<!\w)אנא\s+(?:עדכן|עדכני|תעדכן)\s+אותי",
    r"(?<!\w)(?:ו?אני\s+)?מבי(?:ן|נה)\s+את\s+ה?(?:תסכול|כעס)",
    r"(?<!\w)ו?(?:אני\s+)?(?:אשתדל|אשתפר)(?!\w)",
    r"(?<!\w)ו?(?:אני\s+)?אקח\s+(?:את\s+)?(?:זה\s+)?(?:\S+\s+)?בחשבון",
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
    cams = [str(c) for c in cameras or () if c]
    out = replace_ids(text, cams, lang)
    out = _CAMERA_ID.sub(lambda m: display_name(m.group(0), lang), out)
    # "במצלמה ameer_x_ch6" became "במצלמה מצלמה 6": one "מצלמה" is enough.
    out = _DOUBLE_HE.sub(r"\1מצלמה", out)
    out = _DOUBLE_EN.sub(r"\1amera", out)
    return name_numbered_cameras(out, cams)


_DOUBLE_HE = re.compile(r"(?<!\w)([ובלמה]{0,2})מצלמה\s+מצלמה(?=\s+\d)")
_DOUBLE_EN = re.compile(r"\b([Cc])amera\s+Camera(?=\s+\d)")
# "מצלמה 3" / "במצלמה 3" / "Camera 3": what the box says for a camera WITHOUT a name. 2026-10-09 12:47 the assistant
# wrote "במצלמה 3 ... במצלמה 6" for the pergola and the main entrance, which have names.
_NUMBERED_HE = re.compile(r"(?<![\w])([ובלמהכש]{0,3})מצלמה\s+(\d+)(?!\d)")
_NUMBERED_EN = re.compile(r"\b[Cc]amera\s+(\d+)(?!\d)")


def channel_names(cameras: Iterable[str] = ()) -> Dict[str, str]:
    """Channel number -> the family's name, for every camera (*cameras*, and the ids that have names) that has one.
    A channel two cameras name differently (two old sites) is left out: no guessing."""
    from ..camera_names import _load, channel_of, family_names  # noqa: PLC0415

    aliases = _load(None)
    found: Dict[str, set] = {}
    for cam in {*(str(c) for c in cameras if c), *aliases}:
        ch = channel_of(cam)
        names = family_names(cam, aliases) if ch is not None else []
        if names and names[-1].strip():
            found.setdefault(ch, set()).add(names[-1].strip())
    return {ch: next(iter(names)) for ch, names in found.items() if len(names) == 1}


def _glue(prefix: str, name: str) -> str:
    """Hebrew prefix letters + a name: "ב" + "הפרגולה" is "בפרגולה", "ה" + "הפרגולה" is "הפרגולה"."""
    if prefix and name.startswith("ה") and prefix[-1] in "בלכה":
        name = name[1:] if prefix[-1] != "ה" else name
        prefix = prefix[:-1] if prefix[-1] == "ה" else prefix
    return prefix + name


def name_numbered_cameras(text: str, cameras: Iterable[str] = ()) -> str:
    """*text* with "מצלמה N" / "Camera N" as the family's name for camera N when it has one (the last guard on any
    outgoing text). A camera without a name keeps its number. Never raises."""
    if not isinstance(text, str) or not text or not re.search(r"מצלמה|[Cc]amera", text):
        return text
    try:
        names = channel_names(cameras)
    except Exception as exc:  # noqa: BLE001 - the text goes out as it is
        log.debug("camera names not read: %s", exc)
        return text
    if not names:
        return text
    out = _NUMBERED_HE.sub(lambda m: _glue(m.group(1), names[m.group(2)]) if m.group(2) in names else m.group(0), text)
    return _NUMBERED_EN.sub(lambda m: names.get(m.group(1), m.group(0)), out)


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
