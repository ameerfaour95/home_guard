""""Are these the same people?" and "are you sure?" answered in code, from evidence only (2026-10-09 13:05).

13:05:16 "אלה אותם אנשים שעבדו מהבוקר או שהתחלפו?" got "האנשים שראיתם הם אותם עובדים מהבוקר", and 13:05:29 "אתה
בטוח?" got "כן, אני בטוח" - nothing linked those people. The box keeps three kinds of evidence, and the answer says
which one it rests on, with confidence words that match it:

- **continuity**: the camera's open event (and the events it rolled over from) with no break since a time; the
  tracker kept the same P1, P2 all along ("ראיתי אותם ברצף מאז 08:10");
- **clothes**: the same-day clothing ReID (reid.py; shadow mode too) linked a person to an earlier one
  (``entities.appearance_said``), or said the clothes differ from the one geometry joined;
- **the owner's mark**: "העובדים של הפרגולה" until 18:00 - what the owner said, never something the box saw.

Without continuity or a clothes link the answer is "לא בטוח, לא ראיתי אותם ברצף". "אתה בטוח?" never becomes
"כן, אני בטוח": strong evidence is "די בטוח" with what it is based on.
"""

from __future__ import annotations

import datetime as dt
import logging
import re
from typing import Any, Dict, List, Optional

from .. import entities as ent
from .i18n import t
from .registry import display
from .reply_guard import asks_same_people

log = logging.getLogger("box.brain.same_people")

PHOTO_WINDOW_SEC = 900.0       # the photos the owner is looking at: sent these 15 minutes
SURE_FOLLOW_SEC = 600.0        # "אתה בטוח?" follows a same-people question asked this recently
CONTINUOUS_MIN_SEC = 1800.0    # an unbroken event this long is "seen them all along"
PRESENT_SEC = 300.0            # an entity seen this recently is one of "these people"
MAX_IDS = 4

_SURE = re.compile(r"^\s*(?:ו?(?:אתה|את)\s+)?(?:בטוח|בטוחה)(?:\s+(?:בזה|בכך))?\s*[?!.]*\s*$|"
                   r"^\s*(?:are\s+you\s+)?(?:sure|certain)(?:\s+about\s+(?:it|that))?\s*[?!.]*\s*$", re.IGNORECASE)


def asks_sure(text: str) -> bool:
    return bool(_SURE.match(str(text or "")))


def follows_same_question(state: Any, now: float) -> bool:
    """The owner's last message (these SURE_FOLLOW_SEC) asked whether these are the same people."""
    for turn in reversed(getattr(state, "turns", None) or []):
        if not isinstance(turn, dict) or turn.get("kind") == "alert":
            continue
        try:
            fresh = now - float(turn.get("ts") or 0) <= SURE_FOLLOW_SEC
        except (TypeError, ValueError):
            return False
        return fresh and asks_same_people(str(turn.get("text") or ""))
    return False


def camera_in_question(state: Any, alert: Optional[Dict[str, Any]], now: float) -> str:
    """The camera "these people" are at: the latest photo these 15 minutes that showed people (the most people
    first), else the alert replied to, else the camera being talked about. "" when none."""
    best = None
    for entry in (getattr(state, "handles", None) or {}).values():
        if not isinstance(entry, dict) or entry.get("kind") != "photo" or not entry.get("camera"):
            continue
        try:
            ts = float(entry.get("ts") or 0)
        except (TypeError, ValueError):
            continue
        if now - ts > PHOTO_WINDOW_SEC:
            continue
        people = entry.get("people")
        people = people if isinstance(people, int) else 1        # an older photo without the count: maybe people
        if people <= 0:
            continue
        key = (round(ts / 60.0), people, ts)                       # the latest look, then the most people
        if best is None or key > best[0]:
            best = (key, str(entry["camera"]))
    if best is not None:
        return best[1]
    if alert and alert.get("camera"):
        return str(alert["camera"])
    topic = state.topic_camera(now) if hasattr(state, "topic_camera") else None
    return topic[0] if topic else ""


def _hm(ts: Any) -> str:
    return dt.datetime.fromtimestamp(float(ts or 0)).strftime("%H:%M")


def evidence(events: Any, camera: str, now: float) -> Dict[str, Any]:
    """What the box itself saw at *camera* today: ``level`` "continuous" | "mixed" | "clothes" | "none", the ids,
    since when, today's separate visits, and the clothes' links and splits."""
    start = dt.datetime.fromtimestamp(now).replace(hour=0, minute=0, second=0, microsecond=0).timestamp()
    sessions = [s for s in (events.recent(start, camera) if events is not None else [])
                if float(s.get("opened") or 0) <= now]
    by_id = {s.get("id"): s for s in sessions}
    current = next((s for s in reversed(sessions) if not s.get("closed")
                    and now - float(s.get("last_active") or 0) <= PRESENT_SEC), None)
    chain, since = [], 0.0
    s = current
    while s is not None and s not in chain:
        chain.append(s)
        since = float(s.get("opened") or 0)
        s = by_id.get(s.get("parent")) if s.get("parent") else None
    roots = [s for s in sessions if not (s.get("parent") and s.get("parent") in by_id)]
    present = []
    if current is not None:
        present = [e for e in current.get("entities") or () if isinstance(e, dict) and e.get("kind") == "person"
                   and (e.get("state") == ent.ACTIVE or now - float(e.get("last_seen") or 0) <= PRESENT_SEC)]
    ids = ent.order_ids(present, [e["id"] for e in present])[:MAX_IDS]
    # Only what the clothes said about THESE people (the open event and the ones it rolled over from): a person who
    # came back to another event at 13:14 says nothing about the 13:05 crew.
    said = [dict(x, at=float(s.get("opened") or 0)) for s in chain for x in ent.appearance_said(s.get("entities"))]
    links = [x for x in said if x["what"] == "link"]
    for s in chain:
        link = s.get("incident_from") or s.get("cross_seen") or {}
        if isinstance(link, dict) and link.get("score") is not None and float(link.get("score") or 0) >= 0.7:
            links.append({"entity": str(link.get("to_entity") or ""), "what": "link", "other": str(link.get("entity") or ""),
                          "score": float(link["score"]), "acted": False, "at": float(s.get("opened") or 0)})
    chain_ids = {s.get("id") for s in chain}
    splits = [x for x in said if x["what"] == "split"]
    if current is not None and now - since >= CONTINUOUS_MIN_SEC:
        level = "mixed" if splits else "continuous"
    elif links:
        level = "clothes"
    else:
        level = "none"
    return {"camera": camera, "level": level, "ids": ids, "since": since, "visits": len(roots),
            "first": float(roots[0].get("opened") or 0) if roots else 0.0,
            "last": float(roots[-1].get("opened") or 0) if roots else 0.0,
            "links": links, "splits": splits, "open": current is not None, "chain": sorted(chain_ids)}


def mark_for(events: Any, camera: str, now: float) -> Optional[Dict[str, Any]]:
    """The owner's live mark for *camera* (or for the whole house)."""
    try:
        marks = [k for k in (events.list_known(now) if events is not None else []) if isinstance(k, dict)]
    except Exception as exc:  # noqa: BLE001
        log.warning("Marks not read: %s", exc)
        return None
    mine = [k for k in marks if k.get("camera") in (camera, "", None)]
    return mine[-1] if mine else None


def _place(snapshot: Any, camera: str, lang: str) -> str:
    """"בפרגולה" / "בכניסה הראשית" / "במצלמה 1" / "at the pergola"."""
    name = display(snapshot, camera, lang)
    if not str(lang).startswith("he"):
        return f"at {name}"
    from .activity_chat import definite_he  # noqa: PLC0415

    sure = definite_he(name)
    return f"ב{sure[1:]}" if sure.startswith("ה") else f"ב{sure}"


def _pairs(links: List[Dict[str, Any]]) -> str:
    out: List[str] = []
    for x in links[-2:]:
        pair = f"{x['entity']}={x['other']}" if x["entity"] != x["other"] else x["entity"]
        if pair and pair not in out:
            out.append(pair)
    return ", ".join(out)


def answer(events: Any, snapshot: Any, camera: str, now: float, lang: str, sure: bool = False) -> str:
    """One or two lines: the verdict and what it rests on; then the owner's mark, said as his words."""
    ev = evidence(events, camera, now)
    place = _place(snapshot, camera, lang)
    ids = ", ".join(ev["ids"]) or "-"
    level = ev["level"]
    if level == "continuous":
        basis = t("same_basis_continuous", lang, place=place, since=_hm(ev["since"]), ids=ids)
    elif level == "mixed":
        who = ", ".join(sorted({x["entity"] for x in ev["splits"]})[:MAX_IDS])
        basis = t("same_basis_mixed", lang, place=place, since=_hm(ev["since"]), ids=who)
    elif level == "clothes":
        basis = t("same_basis_clothes", lang, pairs=_pairs(ev["links"]), at=_hm(ev["links"][-1]["at"]))
    elif ev["visits"]:
        basis = t(f"{'sure' if sure else 'same'}_basis_visits", lang, place=place, n=ev["visits"],
                  first=_hm(ev["first"]), last=_hm(ev["last"]))
    else:
        basis = t(f"{'sure' if sure else 'same'}_basis_nothing", lang, place=place)
    head = t(f"{'sure' if sure else 'same'}_verdict_{'none' if level == 'mixed' else level}", lang)
    lines = [f"{head} {basis}"]
    mark = mark_for(events, camera, now)
    if mark is not None:
        until = str(mark.get("daily_to") or "") or _hm(mark.get("until"))
        lines.append(t("sure_mark_line" if sure else "same_mark_line", lang, text=str(mark.get("text") or "").strip(),
                       until=until))
    log.info("same people at %s: %s (visits %d, links %d, splits %d)", camera, level, ev["visits"], len(ev["links"]),
             len(ev["splits"]))
    return "\n".join(lines)
