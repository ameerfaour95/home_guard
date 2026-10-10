"""One last look at every reply before it goes out (2026-10-09).

13:54-13:55 the owner got the same memory status four times ("העובדים של הפרגולה כבר מסומנים בפרגולה ובכניסה
הראשית עד 18:00."), twice word for word ("כן, אני קורא את ההיסטוריה ובודק את הזיכרון. ..." at 13:55:18 and
13:55:31): the no-repeat check ran on the model's answers only, and those came from a code path. "סבבה" was
answered with the status; "אלה אותם אנשים?" with "כן, אני בטוח" and no evidence.

- :func:`final_reply`: every outgoing reply. The memory-status sentence goes out once per STATUS_GAP_SEC unless
  the owner asked about the memory; a reply equal or nearly equal to one of the bot's last LAST_REPLIES is
  rewritten once by the model, told what was already said; still the same: a short different line.
- :func:`is_ack` (activity_chat): "סבבה / בסדר / הבנתי / תודה / 👍" gets a short human acknowledgement, never a
  status.
- :func:`asks_same_people` / :func:`same_people_lines` / :func:`overclaims_same`: "are these the same people?"
  is answered from evidence only.
"""

from __future__ import annotations

import datetime as dt
import difflib
import logging
import re
from typing import Any, Callable, List, Optional, Sequence

from ...prompts import load
from .activity_chat import is_ack  # noqa: F401 - re-exported
from .style import strip_boilerplate

log = logging.getLogger("box.brain.reply_guard")

LAST_REPLIES = 5
STATUS_GAP_SEC = 1800.0
NEAR = 0.9

_SENTENCE = re.compile(r"(?<=[.!?\n])\s+")
_STATUS_WORDS = re.compile(r"(?<![א-ת])[ושה]?(?:מסומנים|מסומן|מסומנות|סימנתי|שמור|שמורים|רשום|רשומים|זוכר|בזיכרון|"
                           r"בזכרון)(?![א-ת])|\b(?:marked|remembered|saved|in memory)\b", re.IGNORECASE)


def _norm(text: Any) -> str:
    return " ".join(strip_boilerplate(str(text or "")).split())


def near(a: str, b: str, ratio: float = NEAR) -> bool:
    """Equal, or nearly (a reply of 12 characters or more): a short "👍" may come again."""
    a, b = _norm(a), _norm(b)
    if len(a) < 12 or len(b) < 12:
        return False
    return a == b or difflib.SequenceMatcher(None, a, b).ratio() >= ratio


def recent_replies(state: Any, last: int = LAST_REPLIES) -> List[str]:
    return [str(turn.get("reply")) for turn in (getattr(state, "turns", None) or [])
            if isinstance(turn, dict) and turn.get("kind") not in ("alert",) and turn.get("reply")][-last:]


def repeats_recent(state: Any, reply: str, last: int = LAST_REPLIES) -> bool:
    return any(near(reply, old) or _norm(old) in _norm(reply) and len(_norm(old)) >= 30
               for old in recent_replies(state, last))


def status_sentences(reply: str, marks: Sequence[Any]) -> List[str]:
    """The sentences of *reply* that restate the memory: they name the people of a live mark and say they are
    marked / saved / remembered."""
    whos = [str((m or {}).get("text") or "").strip() for m in marks if isinstance(m, dict)]
    whos = [w for w in whos if w]
    if not whos:
        return []
    out = []
    for sentence in _SENTENCE.split(str(reply or "")):
        words = set(re.findall(r"[א-ת]+|[A-Za-z]+", sentence))
        named = any(len(set(re.findall(r"[א-ת]+|[A-Za-z]+", w)) & words) >= max(1, len(re.findall(r"[א-ת]+|[A-Za-z]+", w)) - 1)
                    for w in whos)
        if named and _STATUS_WORDS.search(sentence):
            out.append(sentence)
    return out


def status_said_since(state: Any, marks: Sequence[Any], since: float) -> bool:
    for turn in getattr(state, "turns", None) or []:
        if not isinstance(turn, dict) or turn.get("kind") == "alert":
            continue
        try:
            ts = float(turn.get("ts") or 0)
        except (TypeError, ValueError):
            continue
        if ts >= since and status_sentences(str(turn.get("reply") or ""), marks):
            return True
    return False


def keep_private(text: str, unrelated: Sequence[Any]) -> str:
    """*text* without what it says about marks this message is not about: their status or their people
    (2026-10-10: "why even mention the workers!!!")."""
    from . import human  # noqa: PLC0415

    out = str(text or "")
    hidden = status_sentences(out, unrelated)
    if hidden:
        log.warning("reply guard: the memory status of marks this message is not about; dropped")
        out = " ".join(s.strip() for s in _SENTENCE.split(out) if s.strip() and s not in hidden)
    gone = human.drop_unrelated(out, unrelated)
    if gone != out:
        log.warning("reply guard: a saved mark this message is not about was brought up; dropped")
    return gone


def final_reply(reply: str, state: Any, now: float, marks: Sequence[Any], asked_memory: bool, lang: str,
                rewrite: Optional[Callable[[str, List[str]], str]] = None, unrelated: Sequence[Any] = (),
                fallback: Optional[Callable[[], str]] = None) -> str:
    """*reply* as it may go out. *marks* are the live marks this message is about; *unrelated* the others, which a
    reply never brings up (2026-10-10: "why even mention the workers!!!"). Robotic sentences ("מה לתקן?", "הבנתי
    אותך.") never go out; a reply left empty becomes *fallback()* (one plain line about the thing at hand). Never
    raises (on a failure the reply goes out as it was)."""
    from .i18n import t  # noqa: PLC0415
    from . import human  # noqa: PLC0415

    def plain() -> str:
        try:
            return str(fallback() or "") if fallback is not None else ""
        except Exception as exc:  # noqa: BLE001
            log.warning("Fallback line failed: %s", exc)
            return ""

    try:
        out = str(reply or "")
        if not asked_memory:
            out = keep_private(out, unrelated)
            said = status_sentences(out, marks)
            if said and status_said_since(state, marks, now - STATUS_GAP_SEC):
                kept = [s for s in _SENTENCE.split(out) if s not in said]
                log.warning("reply guard: the memory status was said in the last %d min; dropped", STATUS_GAP_SEC // 60)
                out = " ".join(s.strip() for s in kept if s.strip())
        if human.generic_sentences(out):
            log.warning("reply guard: a robotic sentence dropped: %s", human.generic_sentences(out))
            out = human.drop_generic(out)
            if not out.strip():
                out = plain()
        if out.strip() and repeats_recent(state, out):
            log.warning("reply guard: a repeat of a recent reply; one rewrite")
            again = ""
            if rewrite is not None:
                try:
                    again = human.drop_generic(str(rewrite(out, recent_replies(state)) or "").strip())
                    if not asked_memory:
                        again = human.drop_unrelated(again, unrelated)
                except Exception as exc:  # noqa: BLE001
                    log.warning("Rewrite failed: %s", exc)
            if again and not repeats_recent(state, again) and not (
                    not asked_memory and status_sentences(again, marks)
                    and status_said_since(state, marks, now - STATUS_GAP_SEC)):
                out = again
            else:
                used = {_norm(r) for r in recent_replies(state)}
                out = next((x for x in (t("ack_other", lang), t("ack_short", lang)) if _norm(x) not in used),
                           t("ack_short", lang))
        if not out.strip():
            out = plain() or t("ack_short", lang)
        return out
    except Exception as exc:  # noqa: BLE001
        log.warning("Reply guard failed: %s", exc)
        return reply


# ---------- "are these the same people?" ----------
_SAME = re.compile(r"(?<![א-ת])(?:אותם|אותו|אותה|אותן)\s+(?:אנשים|אדם|האנשים|עובדים|הפועלים|הבחור|האיש|האנשים)|"
                   r"(?<![א-ת])(?:התחלפו|מתחלפים|אחרים|חדשים)\??\s*$|\bsame (?:people|person|guys?|men|workers?)\b",
                   re.IGNORECASE)
_SURE = re.compile(r"(?<![א-ת])(?:בטוח|בוודאות|בטוחה|ודאי)(?![א-ת])|\b(?:sure|certain|definitely)\b", re.IGNORECASE)
_HEDGE = re.compile(r"(?<![א-ת])(?:לא בטוח|לא בטוחה|נראה|כנראה|ייתכן|אולי|לפי הבגדים)(?![א-ת])|"
                    r"\b(?:not sure|looks like|seems|probably|maybe|might)\b", re.IGNORECASE)


def asks_same_people(text: str) -> bool:
    return "?" in str(text or "") and bool(_SAME.search(str(text or "")))


def same_people_lines(events: Any, snapshot: Any, lang: str, now: float) -> List[str]:
    """[SAME PEOPLE EVIDENCE]: today's events with their entities, and any appearance link the box made (shadow
    too). The model answers from these only."""
    from .registry import display  # noqa: PLC0415

    rows = []
    try:
        start = dt.datetime.fromtimestamp(now).replace(hour=0, minute=0, second=0).timestamp()
        for s in (events.recent(start) if events is not None else [])[-12:]:
            when = dt.datetime.fromtimestamp(float(s.get("opened") or 0)).strftime("%H:%M")
            line = f"{when} {display(snapshot, str(s.get('camera') or ''), lang)}: {s.get('summary_line') or ''}"
            link = s.get("incident_from") or s.get("cross_seen") or {}
            if link:
                line += f" (appearance link from {link.get('camera')} score {link.get('score')}, mode {link.get('mode', 'on')})"
            rows.append(line)
    except Exception as exc:  # noqa: BLE001
        log.warning("Same-people evidence not read: %s", exc)
    head = load("brain_context_same_people.prompt")
    return [head + (" Today's events: " + "; ".join(rows) if rows else " No events today.")]


def overclaims_same(answer: str, evidence: Sequence[str]) -> bool:
    """A sure "yes, the same people" with no appearance link in the evidence."""
    if not _SURE.search(str(answer or "")) or _HEDGE.search(str(answer or "")):
        return False
    return not any("appearance link" in e for e in evidence)
