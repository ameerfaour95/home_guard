"""The event's story for the owner: an update continues what the owner was already told (stage 2a of the alert fix).

Owner (2026-10-08): "Qwen should report independently, but whoever sends me the message must understand the context:
msg1 'a man in a white shirt walks near the entrance with another man cleaning the floor', msg2 'both moved now
toward the pergola', msg3 'the one who was cleaning put the broom in the pickup'."

``story_line(session, lang)`` reads only what code measured and stored in the event session (events.Session:
entities from the tracker, what the Eye said each one did, what the owner was told) and writes one or a few short
sentences in Hebrew or English. It never invents continuity:

- "Both moved toward X" only when two (or more) entities the owner was told about are now in the same scene-map area
  X and were somewhere else when last told.
- "P2 (earlier: ...)" only for an entity that really is the same one (never for an ambiguous return, ``maybe_of``,
  which reads "maybe P1 or P2 back, not sure").
- Otherwise a plain current sentence ("In view now: P1, P2").

Inference puts it above the new observation in the event's thread (updates only; the first message of an event is
unchanged). No camera ids here: the graded alert line under it names the camera by its display name.
"""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any, Dict, List, Optional, Sequence

HINT_CHARS = 60

_T = {
    "he": {"both": "שניהם ({ids}) עברו ל{to_area}.", "group": "{ids} עברו ל{to_area}.",
           "moved": "{name} עבר ל{to_area}.", "new": "{name} חדש בתמונה.", "at": "{name} עכשיו ליד {area}.",
           "still": "{name} עדיין בתמונה.", "still_many": "עדיין בתמונה: {ids}.", "left": "{ids} יצא מהתמונה.",
           "left_many": "{ids} יצאו מהתמונה.", "now": "בתמונה עכשיו: {ids}.", "earlier": "קודם: {text}",
           "maybe": "אולי {ids} שחזר, לא בטוח", "and": " ו-", "or": " או "},
    "en": {"both": "Both ({ids}) moved toward {area}.", "group": "{ids} moved toward {area}.",
           "moved": "{name} moved toward {area}.", "new": "{name} is new in view.", "at": "{name} is now at {area}.",
           "still": "{name} is still in view.", "still_many": "Still in view: {ids}.", "left": "{ids} left the view.",
           "left_many": "{ids} left the view.", "now": "In view now: {ids}.", "earlier": "earlier: {text}",
           "maybe": "maybe {ids} back, not sure", "and": " and ", "or": " or "},
}



def _after_lamed(name: str) -> str:
    """*name* after the Hebrew "ל" ("ל" + "הפרגולה" is "לפרגולה"); English templates ignore it."""
    name = str(name or "").strip()
    return name[1:] if len(name) > 2 and name.startswith("ה") else name

def _lang(lang: str) -> str:
    return "he" if str(lang or "").startswith("he") else "en"


def _join(ids: Sequence[str], word: str) -> str:
    ids = list(ids)
    if len(ids) <= 1:
        return "".join(ids)
    return ", ".join(ids[:-1]) + word + ids[-1]


def _short(text: str) -> str:
    text = " ".join(str(text or "").split()).rstrip(". ")
    return text if len(text) <= HINT_CHARS else text[:HINT_CHARS - 1].rstrip() + "…"


def _as_dict(session: Any) -> Dict[str, Any]:
    return asdict(session) if is_dataclass(session) else dict(session or {})


def _area(e: Dict[str, Any]) -> str:
    return e["path"][-1] if e.get("mapped") and e.get("path") else ""


def story_line(session: Any, lang: str = "he", now: Optional[float] = None,
               in_view: Optional[Sequence[str]] = None, announced: Sequence[str] = ()) -> str:
    """The owner's update text that continues the event's story, or "" when the session has no entities (no tracker
    data): then the update reads as before. *now* and *in_view* default to the session's latest observation;
    *announced* are new ids the update's first line already names (``new_people_line``), not repeated as new."""
    s = _as_dict(session)
    entities = list(s.get("entities") or [])
    if not entities:
        return ""
    t = _T[_lang(lang)]
    last = (s.get("observations") or [{}])[-1]
    now = float(last.get("ts", 0.0)) if now is None else float(now)
    view = list(last.get("entities") or []) if in_view is None else list(in_view)
    index = {e["id"]: e for e in entities}
    view = [i for i in view if i in index]
    told = set(s.get("reported_entities") or [])
    sentences: List[str] = []

    def hint(e: Dict[str, Any]) -> str:
        if e.get("maybe_of"):
            return t["maybe"].format(ids=_join(e["maybe_of"], t["or"]))
        bits = []
        if e.get("owner_label"):
            bits.append(_short(e["owner_label"]))
        earlier = [n for n in e.get("notes") or () if float(n.get("ts", 0.0)) < now - 0.5]
        if earlier:
            bits.append(t["earlier"].format(text=_short(earlier[-1]["text"])))
        return "; ".join(bits)

    def name(e: Dict[str, Any]) -> str:
        h = hint(e)
        return f"{e['id']} ({h})" if h else e["id"]

    def action(e: Dict[str, Any]) -> str:
        """What the Eye said this one does in this very look (``per_entity``), "" otherwise."""
        now_notes = [n for n in e.get("notes") or () if n.get("source") == "eye" and abs(float(n.get("ts", 0.0)) - now) <= 0.5]
        return _short(now_notes[-1]["text"]) if now_notes else ""

    # Who moved since the owner was last told: told before, in a mapped area now, somewhere else then.
    movers: Dict[str, List[str]] = {}
    for i in view:
        e = index[i]
        area = _area(e)
        if i in told and not e.get("maybe_of") and area and e.get("told_area") and area != e["told_area"]:
            movers.setdefault(area, []).append(i)
    said = set()
    for area, ids in movers.items():
        if len(ids) >= 2:
            key = "both" if len(ids) == 2 and len(view) == 2 else "group"
            sentences.append(t[key].format(ids=_join(ids, t["and"]) if key == "group" else ", ".join(ids), area=area, to_area=_after_lamed(area)))
        else:
            sentences.append(t["moved"].format(name=ids[0], area=area, to_area=_after_lamed(area)))
        said.update(ids)
    still: List[str] = []
    for i in view:
        e = index[i]
        act = action(e)
        if act:
            sentences.append(f"{name(e)}: {act}.")
        elif i in said:
            continue
        elif i not in told:
            if i not in announced or hint(e):
                sentences.append(t["new"].format(name=name(e)))
        elif hint(e) and _area(e):
            sentences.append(t["at"].format(name=name(e), area=_area(e)))
        elif hint(e):
            sentences.append(t["still"].format(name=name(e)))
        else:
            still.append(i)
    left = [e["id"] for e in entities if e["id"] in told and e["id"] not in view and e.get("state") != "active"
            and float(e.get("told_at") or 0.0) > 0 and float(e["last_seen"]) >= float(e.get("told_at") or 0.0)
            and float(e.get("told_gone") or 0.0) < float(e["last_seen"])]
    if still and (sentences or left):
        sentences.append(t["still"].format(name=still[0]) if len(still) == 1
                         else t["still_many"].format(ids=_join(still, t["and"])))
    if left:
        sentences.append((t["left"] if len(left) == 1 else t["left_many"]).format(ids=_join(left, t["and"])))
    if not sentences and view:
        sentences.append(t["now"].format(ids=_join(view, t["and"])))
    return " ".join(sentences)


def new_people_line(fresh: Sequence[str], unmarked: bool, known_text: str, lang: str) -> str:
    """The first line of an update when the tracker saw someone new arrive: "P4 is new" style, and, when the owner had
    marked the others as known, that this one is not one of them."""
    fresh = list(fresh)
    if not fresh:
        return ""
    he = _lang(lang) == "he"
    ids = _join(fresh, " ו-" if he else " and ")
    if unmarked:
        words = f" ({_short(known_text)})" if known_text else ""
        if he:
            return (f"אדם חדש הגיע ({ids}), לא מאלה שסימנת{words}" if len(fresh) == 1
                    else f"{len(fresh)} אנשים חדשים הגיעו ({ids}), לא מאלה שסימנת{words}")
        return (f"A new person arrived ({ids}), not one of those you marked{words}" if len(fresh) == 1
                else f"{len(fresh)} new people arrived ({ids}), not among those you marked{words}")
    if he:
        return f"עוד אדם אחד הגיע ({ids})" if len(fresh) == 1 else f"עוד {len(fresh)} אנשים הגיעו ({ids})"
    return f"1 more person arrived ({ids})" if len(fresh) == 1 else f"{len(fresh)} more people arrived ({ids})"


def incident_line(entity_id: str, from_camera: str, to_camera: str, lang: str) -> str:
    """The first line of an alert that continues an incident from another camera (stage 3.3, events.py
    ``cross_camera: on``): "אותו אדם (P1) עבר מהשער לכניסה". *from_camera* / *to_camera* are DISPLAY names
    (camera_names), never ids; *entity_id* is the id the first camera's thread used."""
    who = f" ({entity_id})" if entity_id else ""
    if _lang(lang) == "he":
        return f"אותו אדם{who} עבר מ{from_camera} ל{_after_lamed(to_camera)}"
    return f"The same person{who} went from {from_camera} to {to_camera}"
