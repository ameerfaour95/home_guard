"""What every v3 model call sees (report §8.5a): rebuilt each turn, small, stable parts first.

1. the house: cameras by the family's names (with a short key ``cam3`` for the understanding step), on/off;
2. the DAY LEDGER: every alert, live photo and clip of the last 24 h as ONE line with its handle (E7), written once
   from what was seen first (the Eye's summary, the live look's caption), plus later answers ("asked ...: ...") and
   the owner's outcome. Pictures are never re-sent; ``look_again(handle, question)`` reaches deeper;
3. the box's own sessions of the day not sent to the chat (head counts per camera and hour), for "same people?";
4. the conversation: the LIVE WINDOW (last 12 turns or 2 h) with the owner's words verbatim and each bot reply as a
   one-line summary (verbatim replies feed the repetition loop: Xu et al. 2022), older turns of the day as the
   owner's words only (or the rolling summary when there is one), and the open question;
5. the memory relevant to THIS message (memory_view.relevant, at most 3) and the owner's standing preferences.

Typical size 2-4k tokens (v2: 13k+ with 24 h verbatim, all marks and 32 tool schemas).
"""
from __future__ import annotations

import datetime as dt
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..brain.registry import display, normalize

WINDOW_TURNS = 12
WINDOW_SEC = 2 * 3600.0
DAY_SEC = 24 * 3600.0
LEDGER_MAX = 40
ONE_LINE = 170
LEVEL = {"escalation": "🔴 אדום", "suspicious": "🟡 חשוד", "normal": "⚪ רגיל"}


def clip(text: Any, limit: int = ONE_LINE) -> str:
    text = " ".join(str(text or "").split())
    return text if len(text) <= limit else text[:limit - 1].rstrip() + "…"


def hhmm(ts: Any) -> str:
    try:
        return dt.datetime.fromtimestamp(float(ts)).strftime("%H:%M")
    except (TypeError, ValueError, OverflowError, OSError):
        return ""


def day_and_time(ts: float, now: float) -> str:
    when = dt.datetime.fromtimestamp(ts)
    if when.date() == dt.datetime.fromtimestamp(now).date():
        return when.strftime("%H:%M")
    return when.strftime("%d.%m %H:%M")


@dataclass
class Cameras:
    """The house's cameras: short keys for the understanding step, the family's names for everything else."""

    ids: List[str] = field(default_factory=list)
    names: Dict[str, str] = field(default_factory=dict)       # id -> display name
    keys: Dict[str, str] = field(default_factory=dict)        # "cam3" / a name / an alias (casefolded) -> id
    key_of: Dict[str, str] = field(default_factory=dict)      # id -> "cam3"
    enabled: Dict[str, bool] = field(default_factory=dict)

    @classmethod
    def of(cls, snapshot: Any, lang: str) -> "Cameras":
        out = cls()
        for i, cam in enumerate(getattr(snapshot, "cameras", ()) or ()):
            m = re.search(r"(\d+)$", cam.name)
            key = f"cam{m.group(1) if m else i + 1}"
            if key in out.keys:
                key = f"cam{i + 1}x"
            out.ids.append(cam.name)
            out.names[cam.name] = display(snapshot, cam.name, lang)
            out.key_of[cam.name] = key
            out.enabled[cam.name] = bool(cam.enabled)
            for word in [key, cam.name, out.names[cam.name], *(cam.aliases or ())]:
                out.keys[str(word).casefold()] = cam.name
                out.keys[normalize(str(word))] = cam.name
        return out

    def table(self) -> str:
        return "\n".join(f"{self.key_of[c]} = {self.names[c]}{'' if self.enabled[c] else ' (כבויה)'}" for c in self.ids)

    def name(self, camera: str) -> str:
        return self.names.get(camera, camera)


def ledger(state: Any, cams: Cameras, now: float, include_answers: bool = True) -> List[str]:
    """One line per alert, photo and clip of the last 24 hours, oldest first (grouped when there are many)."""
    rows: List[Tuple[float, str]] = []
    for handle, e in (state.handles or {}).items():
        if not isinstance(e, dict) or e.get("kind") not in ("event", "photo", "clip"):
            continue
        ts = float(e.get("ts") or 0)
        if now - ts > DAY_SEC or ts > now + 60:
            continue
        where = cams.name(str(e.get("camera") or ""))
        if e["kind"] == "event":
            level = LEVEL.get(str(e.get("label") or ""), "התראה")
            what = clip(e.get("observation") or e.get("summary"), 260)
            line = f"{handle} {day_and_time(ts, now)} · {where} · התראה {level} · {what}"
        elif e["kind"] == "photo":
            what = clip(e.get("observation"), 220) or "(בלי תיאור)"
            line = f"{handle} {day_and_time(ts, now)} · {where} · תמונה חיה שנשלחה · {what}"
        else:
            line = f"{handle} {day_and_time(ts, now)} · {where} · סרטון שנשלח"
        if include_answers:
            for a in (e.get("answers") or [])[-2:]:
                if isinstance(a, dict):
                    line += f" · נבדק שוב \"{clip(a.get('q'), 60)}\": {clip(a.get('a'), 160)}"
        if e.get("v3_outcome"):
            line += f" · {e['v3_outcome']}"
        rows.append((ts, line))
    rows.sort()
    lines = [r[1] for r in rows]
    if len(lines) > LEDGER_MAX:
        lines = [f"(+{len(lines) - LEDGER_MAX} פריטים מוקדמים יותר היום)"] + lines[-LEDGER_MAX:]
    return lines


def sessions(book: Any, cams: Cameras, now: float, limit: int = 12) -> List[str]:
    """The box's own events of the last 24 hours per camera (sent to the chat or not): when, how many people, what."""
    if book is None:
        return []
    try:
        rows = book.recent(now - DAY_SEC)
    except Exception:  # noqa: BLE001
        return []
    out = []
    for s in rows[-limit:]:
        start, end = float(s.get("opened") or 0), float(s.get("closed") or s.get("last_active") or 0)
        obs = [o for o in s.get("observations") or [] if isinstance(o, dict)]
        last = clip((obs[-1] or {}).get("summary"), 120) if obs else ""
        out.append(f"{cams.name(str(s.get('camera') or ''))} {hhmm(start)}–{hhmm(end)} · עד "
                   f"{int(s.get('people_max') or 0)} אנשים · {len(obs)} תצפיות" + (f" · אחרונה: {last}" if last else ""))
    return out


def _bot_line(turn: Dict[str, Any]) -> str:
    reply = clip(re.sub(r"\[[^\]]*\]", "", str(turn.get("reply") or "")), ONE_LINE)
    media = [h for h in turn.get("handles") or []]
    extra = f" (+ {', '.join(media)})" if media else ""
    return (reply or "(בלי טקסט)") + extra


def window(state: Any, now: float) -> Tuple[List[str], List[str], List[Dict[str, Any]]]:
    """``(live window lines, older owner lines of the day, the live window's turns)``."""
    turns = [t for t in state.turns or [] if isinstance(t, dict) and now - float(t.get("ts") or 0) <= DAY_SEC
             and float(t.get("ts") or 0) <= now + 1]
    live = [t for t in turns if now - float(t.get("ts") or 0) <= WINDOW_SEC][-WINDOW_TURNS:]
    older = [t for t in turns if t not in live and t.get("kind") != "alert"]
    lines = []
    for t in live:
        when = hhmm(t.get("ts"))
        if t.get("kind") == "alert":
            lines.append(f"[{when}] התראה נשלחה: {', '.join(t.get('handles') or [])}")
            continue
        if t.get("kind") == "tag":
            lines.append(f"[{when}] בעל הבית (אחרי כפתור 🏷️ תיוג, על {', '.join(t.get('handles') or [])}): "
                         f"{clip(str(t.get('text') or '').lstrip('🏷️ '), 300)}")
            continue
        about = f" (תגובה ל-{t['handles'][0]})" if t.get("handles") and str(t.get("text") or "") else ""
        lines.append(f"[{when}] בעל הבית{about}: {clip(t.get('text'), 400)}")
        lines.append(f"[{when}] אתה: {_bot_line(t)}")
    old = [f"[{hhmm(t.get('ts'))}] {clip(t.get('text'), 140)}" for t in older[-8:] if str(t.get("text") or "").strip()]
    return lines, old, live


_REMEMBER = re.compile(r"(?<![א-ת])(?:ש|ו)?(?:תזכור|לזכור|זיכרון|זכרון|זהרון|שמור|תשמור)(?![א-ת])|remember",
                       re.IGNORECASE)


def earlier_owner_words(state: Any, now: float, message: str = "") -> List[str]:
    """The owner's messages of the last two hours (an ``earlier`` act may quote them). Words he typed after the 🏷️
    tag button are the clip's TAG and stay out, unless THIS message asks to remember them (owner, 2026-10-09)."""
    tags_too = bool(_REMEMBER.search(message or ""))
    return [str(t.get("text") or "") for t in state.turns or []
            if isinstance(t, dict) and t.get("kind") != "alert" and (tags_too or t.get("kind") != "tag")
            and now - float(t.get("ts") or 0) <= WINDOW_SEC]


def last_replies(state: Any, n: int = 5) -> List[str]:
    return [str(t.get("reply") or "") for t in (state.turns or []) if isinstance(t, dict) and t.get("kind") != "alert"
            and str(t.get("reply") or "").strip()][-n:]


def block(snapshot: Any, cams: Cameras, state: Any, book: Any, now: float, *, memory_lines: Sequence[str] = (),
          prefs: Sequence[str] = (), message: str = "", reply_to: str = "", for_understanding: bool = False,
          open_question: str = "") -> str:
    """The turn's context as one text block."""
    parts: List[str] = []
    when = dt.datetime.fromtimestamp(now)
    days = ("שני", "שלישי", "רביעי", "חמישי", "שישי", "שבת", "ראשון")
    parts.append(f"עכשיו: יום {days[when.weekday()]} {when.strftime('%d.%m %H:%M')}")
    parts.append("המצלמות:\n" + (cams.table() if for_understanding else
                                 "\n".join(f"- {cams.name(c)}{'' if cams.enabled[c] else ' (כבויה)'}" for c in cams.ids)))
    led = ledger(state, cams, now)
    parts.append("יומן היום (התראות, תמונות וסרטונים; E# = מזהה פנימי, לעולם לא לכתוב אותו לבעל הבית):\n"
                 + ("\n".join(led) if led else "(אין)"))
    ses = sessions(book, cams, now)
    if ses:
        parts.append("אירועים שהקופסה ראתה היום (גם כאלה שלא נשלחו):\n" + "\n".join(ses))
    lines, old, _ = window(state, now)
    summary = str((state.prefs or {}).get("v3_summary") or "")
    if summary:
        parts.append("סיכום השיחה מוקדם יותר היום:\n" + summary)
    elif old:
        parts.append("מה בעל הבית כתב מוקדם יותר היום:\n" + "\n".join(old))
    parts.append("השיחה האחרונה:\n" + ("\n".join(lines) if lines else "(אין)"))
    if open_question:
        parts.append(f"שאלה פתוחה ששאלת: {open_question}")
    if memory_lines:
        parts.append("זיכרון רלוונטי להודעה הזאת (להשתמש בשקט, לא לצטט):\n" + "\n".join(memory_lines))
    if prefs:
        parts.append("העדפות קבועות של בעל הבית:\n" + "\n".join(f"- {p}" for p in prefs))
    if message:
        parts.append(f"ההודעה החדשה של בעל הבית{f' (תגובה ל-{reply_to})' if reply_to else ''}:\n{message}")
    return "\n\n".join(parts)
