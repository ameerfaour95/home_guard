"""ONE typed view over the house's memory (report §8.3 [3]), on top of the stores the guard loop already reads.

The stores stay where they are (the guard loop, the undo buttons and the Admin Center read them): the owner's marks
of people (``EventBook`` known.json), what an action means (``ActivityBook`` activity.json), facts and places per
camera (``camera_profiles``) and the chat's preferences (``ChatState.prefs``). This module gives the assistant one
record shape over all of them, the per-TYPE validity policy, and relevance retrieval:

==============  =====================  ==================================================================
type            validity               asked?
==============  =====================  ==================================================================
place_fact      timeless               never ("זה הבית של השכן" never gets "עד מתי?")
camera_fact     timeless               never (routines: "מדי פעם אני יוצא החוצה")
person_mark     daily window + end     one question with buttons, only when the owner gave no hour
activity_rule   inherits a mark        no, when a live mark covers the camera; else "today" said plainly
preference      timeless               never
==============  =====================  ==================================================================

Retrieval is structured first (camera / event scope and validity at the time), then by shared words, at most three
records. Memory is used silently: the writer sees only the records retrieved for THIS message, so it cannot recite
the workers while the owner talks about the neighbour's house.
"""
from __future__ import annotations

import datetime as dt
import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from .. import activity_memory as am
from ..brain import known_memory as km
from ..brain.receipts import DONE, FAILED
from ..brain.registry import display
from ..brain.tools import _issue, profiles_for

log = logging.getLogger("box.assistant_v3.memory")

PREF_PREFIX = "v3_pref_"
MAX_PREFS = 12


@dataclass
class Record:
    id: str
    type: str                  # place_fact | camera_fact | person_mark | activity_rule | preference
    text: str                  # the owner's words / the subject
    cameras: List[str] = field(default_factory=list)   # [] = the whole house
    daily: str = ""            # "07:00-18:00"
    until: float = 0.0         # 0 = timeless
    extra: Dict[str, Any] = field(default_factory=dict)

    def line(self, snapshot: Any, lang: str, now: float) -> str:
        where = ", ".join(display(snapshot, c, lang) for c in self.cameras) if self.cameras else "כל הבית"
        when = ""
        if self.daily:
            when = f" · כל יום {self.daily}"
        if self.until:
            same_day = dt.datetime.fromtimestamp(self.until).date() == dt.datetime.fromtimestamp(now).date()
            day = "" if same_day else km.day_text(self.until, lang) + " "
            when += f" · עד {day}{dt.datetime.fromtimestamp(self.until).strftime('%H:%M')}"
        kind = {"place_fact": "מקום", "camera_fact": "עובדה", "person_mark": "אנשים מוכרים",
                "activity_rule": "פעולה מוסברת", "preference": "העדפה"}.get(self.type, self.type)
        return f"{self.id} [{kind}] {self.text} · {where}{when}"


_STOP = frozenset("זה זאת זו הם הן הוא היא של על עם את יש גם כאן פה רק עד אצלי אצלנו אני אתה לא כן מה מי למה איך "
                  "כל עוד אבל או אם כי שזה וזה the a an is are of to in on at and".split())
_PREFIX = re.compile(r"^[ושהבלמכ]{1,2}(?=[א-ת]{3,})")


def words(text: str) -> set:
    out = set()
    for w in re.findall(r"[\w֐-׿']+", str(text or "").casefold()):
        if len(w) < 3 or w in _STOP or w.isdigit():
            continue
        out.add(_PREFIX.sub("", w)[:5])         # a crude Hebrew stem: prefix letters off, the first 5 letters
    return out


class MemoryView:
    def __init__(self, services: Any, state: Any, snapshot: Any, lang: str, now: float) -> None:
        self.services, self.state, self.snapshot, self.lang, self.now = services, state, snapshot, lang, now

    # -- read ------------------------------------------------------------------------------------------------------
    def records(self) -> List[Record]:
        out: List[Record] = []
        book = getattr(self.services, "events", None)
        for i, k in enumerate(km.live_marks(book, self.now) if book is not None else []):
            daily = f"{k.get('daily_from')}-{k.get('daily_to')}" if k.get("daily_from") else ""
            out.append(Record(f"M{i + 1}", "person_mark", str(k.get("text") or ""),
                              [str(k["camera"])] if k.get("camera") else [], daily, float(k.get("until") or 0),
                              {"known_id": str(k.get("id") or ""), "raw": k}))
        acts = getattr(self.services, "activities", None)
        try:
            live = acts.live(self.now) if acts is not None else []
        except Exception:  # noqa: BLE001
            live = []
        for i, f in enumerate(live):
            daily = f"{f.daily_from}-{f.daily_to}" if f.daily_from else ""
            text = f"{am.actions_text(f.actions, 'he', short=True)}{' ' + am.place_text(f, 'he') if f.place else ''} = {f.cause}"
            out.append(Record(f"A{i + 1}", "activity_rule", text, list(f.cameras), daily, float(f.until or 0),
                              {"fact_id": f.id}))
        store = profiles_for(self.services)
        try:
            facts = [("", f) for f in store.house_facts()] if store is not None else []
            if store is not None:
                for cam, rows in store.all_facts().items():
                    facts += [(cam, f) for f in rows]
        except Exception:  # noqa: BLE001
            facts = []
        for i, (cam, f) in enumerate(facts):
            kind = "place_fact" if f.get("kind") == "place" else "camera_fact"
            out.append(Record(f"F{i + 1}", kind, str(f.get("text") or ""), [cam] if cam else [], "", 0.0,
                              {"fact_id": str(f.get("id") or "")}))
        for key, value in sorted((self.state.prefs or {}).items()):
            if key.startswith(PREF_PREFIX) and isinstance(value, str):
                out.append(Record(f"P{key[len(PREF_PREFIX):]}", "preference", value))
        return out

    def relevant(self, text: str, cameras: Sequence[str], limit: int = 3,
                 include_types: Sequence[str] = ("person_mark", "activity_rule", "place_fact", "camera_fact")) -> List[Record]:
        """At most *limit* records that touch THIS message: same camera (or the whole house when a camera is in
        play) and/or shared words, ranked by both. Preferences are not here (they are always applied)."""
        mine = words(text)
        scored = []
        for r in self.records():
            if r.type not in include_types:
                continue
            score = 0.0
            if cameras and (set(r.cameras) & set(cameras)):
                score += 2.0
            elif cameras and not r.cameras:
                score += 0.5
            shared = mine & words(r.text)
            score += 1.0 * len(shared)
            if score >= 1.0:
                scored.append((score, r))
        scored.sort(key=lambda x: -x[0])
        return [r for _, r in scored[:limit]]

    def prefs(self) -> List[str]:
        return [r.text for r in self.records() if r.type == "preference"]

    # -- write (each by its type's policy; every write has a receipt, so Undo works as in v2) -------------------------
    def add_preference(self, text: str) -> str:
        prefs = self.state.prefs
        existing = sorted(k for k in prefs if k.startswith(PREF_PREFIX))
        for k in existing:
            if prefs.get(k) == text:
                return k
        while len(existing) >= MAX_PREFS:
            prefs.pop(existing.pop(0), None)
        key = f"{PREF_PREFIX}{int(self.now)}"
        prefs[key] = text[:200]
        return key

    def save_place(self, ctx: Any, camera: str, words_: str, owner: str, event: Optional[Dict[str, Any]],
                   whole_house: bool = False) -> Optional[Dict[str, Any]]:
        """A place fact: permanent, never asks until when. With the alert it was said about, where its people were
        (place_facts.alert_region) so the guard loop keeps quiet what stays inside it."""
        store = profiles_for(self.services)
        if store is None or not words_:
            return None
        from .. import place_facts  # noqa: PLC0415

        by = str((ctx.speaker or {}).get("name") or "owner")
        try:
            if camera and not whole_house:
                region, zone = (None, "")
                if event and event.get("alert_id"):
                    try:
                        roots = list(self.services.roots()) if self.services.roots else []
                        region, zone = place_facts.alert_region(roots, str(event.get("camera") or camera),
                                                                str(event.get("alert_id") or ""))
                    except Exception as exc:  # noqa: BLE001
                        log.debug("place region not read: %s", exc)
                fact = store.add_place(camera, words_, owner or "other", region, zone,
                                       str((event or {}).get("alert_id") or ""), by=by, now=self.now)
            else:
                fact = store.add_fact("", words_, by=by, now=self.now)
        except ValueError as exc:
            _issue(ctx, "camera_fact", FAILED, camera or "house", {"camera": camera, "fact": words_}, str(exc))
            return None
        _issue(ctx, "camera_fact", DONE, camera or "house",
               {"camera": "" if whole_house else camera, "fact": fact["text"], "fact_id": fact["id"], "removed": False,
                "place": fact.get("kind") == "place", "owner": owner, "already": bool(fact.get("already")),
                "whole_house": whole_house or not camera})
        self.state.prefs["last_place"] = {"words": fact["text"], "camera": camera, "ts": self.now,
                                          "id": fact["id"], "owner": owner or "",
                                          "alert_ts": float((event or {}).get("ts") or 0)}
        return fact

    def save_camera_fact(self, ctx: Any, camera: str, words_: str) -> Optional[Dict[str, Any]]:
        store = profiles_for(self.services)
        if store is None or not words_:
            return None
        by = str((ctx.speaker or {}).get("name") or "owner")
        try:
            fact = store.add_fact(camera or "", words_, by=by, now=self.now)
        except ValueError as exc:
            _issue(ctx, "camera_fact", FAILED, camera or "house", {"camera": camera, "fact": words_}, str(exc))
            return None
        _issue(ctx, "camera_fact", DONE, camera or "house",
               {"camera": camera, "fact": fact["text"], "fact_id": fact["id"], "removed": False,
                "whole_house": not camera, "rule": (fact.get("rule") or {}).get("kind", ""),
                "already": bool(fact.get("already"))})
        return fact

    def mark_people(self, ctx: Any, who: str, camera: str, until: float, daily_from: str = "", daily_to: str = "",
                    replaces: Sequence[Dict[str, Any]] = ()) -> Optional[Dict[str, Any]]:
        """A person mark (camera "" = the whole house). *replaces*: the live marks it corrects (UPDATE: they go,
        this one stays), so a corrected time never leaves the old one live."""
        book = getattr(self.services, "events", None)
        if book is None or not who or until <= self.now:
            return None
        by = str((ctx.speaker or {}).get("name") or "owner")
        old = [k for k in replaces if k.get("id")]
        same = [k for k in old if str(k.get("camera") or "") == camera and abs(float(k.get("until") or 0) - until) < 60
                and str(k.get("daily_from") or "") == daily_from and str(k.get("daily_to") or "") == daily_to]
        detail: Dict[str, Any] = {"camera": camera, "who": who, "until_ts": until, "at": self.now,
                                  "house": not camera, "daily_from": daily_from, "daily_to": daily_to}
        if same and len(old) == 1:
            detail.update(known_id=str(same[0].get("id") or ""), already=True)
            _issue(ctx, "mark_known", DONE, camera or "house", detail)
            return same[0]
        try:
            if old:
                saved = book.replace_known([str(k["id"]) for k in old], camera, who, by=by, until=until, now=self.now,
                                           daily_from=daily_from, daily_to=daily_to)
            else:
                saved = book.mark_known(camera, who, by=by, until=until, now=self.now, daily_from=daily_from,
                                        daily_to=daily_to)
        except ValueError as exc:
            log.warning("v3 mark refused: %s", exc)
            _issue(ctx, "mark_known", FAILED, camera or "house", detail, "error")
            return None
        detail.update(known_id=str(saved.get("id") or ""), people=int(saved.get("people") or 0))
        if old:
            detail["replaced"] = [{"id": str(k.get("id") or ""), "camera": str(k.get("camera") or ""),
                                   "until": float(k.get("until") or 0), "text": str(k.get("text") or ""),
                                   "daily_from": str(k.get("daily_from") or ""),
                                   "daily_to": str(k.get("daily_to") or "")} for k in old]
        _issue(ctx, "mark_known", DONE, camera or "house", detail)
        return saved

    def add_activity(self, ctx: Any, cameras: List[str], actions: List[str], cause: str, until: float,
                     place: str = "", daily_from: str = "", daily_to: str = "", who: str = "", known_id: str = "",
                     owner_words: str = "", alert_id: str = "") -> Optional[Any]:
        acts = getattr(self.services, "activities", None)
        if acts is None or not actions or until <= self.now:
            return None
        by = str((ctx.speaker or {}).get("name") or "owner")
        for fact in acts.live(self.now):                       # the same camera and an overlapping action: UPDATE
            if set(fact.cameras) & set(cameras) and set(fact.actions) & set(actions):
                merged = list(dict.fromkeys(fact.actions + actions))
                updated = acts.update(fact.id, self.now, actions=merged, cause=cause or fact.cause,
                                      place=place or fact.place)
                if updated is not None:
                    _issue(ctx, "activity", DONE, ",".join(cameras),
                           {"fact_id": fact.id, "cameras": cameras, "actions": merged, "updated": True})
                    return updated
        try:
            fact = acts.add(cameras, actions, cause, until, self.now, place=place, owner_words=owner_words[:300], by=by,
                            daily_from=daily_from, daily_to=daily_to, who=who, known_id=known_id, alert_id=alert_id)
        except ValueError as exc:
            log.warning("v3 activity refused: %s", exc)
            return None
        _issue(ctx, "activity", DONE, ",".join(cameras), {"fact_id": fact.id, "cameras": cameras, "actions": actions})
        return fact


def dumps(records: Sequence[Record], snapshot: Any, lang: str, now: float) -> str:
    return "\n".join(r.line(snapshot, lang, now) for r in records)


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)
