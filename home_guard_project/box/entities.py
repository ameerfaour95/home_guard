"""Who is who inside one event: P1, P2 for people and CAR1 for moving vehicles (stage 2a of the alert fix).

Owner (2026-10-08): "open a SESSION where I keep who is PERSON 1 and who is PERSON 2, give each a number; even if he
disappears from the frame for a second keep him as PERSON 1; if new then new; as long as we talked about PERSON 1
continue the story". The tracker (tracker.py) already follows every person between detector looks; this module maps
its tracks onto the event session's entities with short owner-facing ids.

- An entity is one person (``P1``) or one moving vehicle (``CAR1``). Parked vehicles never become entities.
- A track the tracker linked as a return (``prev_id``) keeps the entity of the lost track.
- A new track that starts within REATTACH_SEC of a lost entity's last sighting, within REATTACH_DISTANCE of its last
  foot point, re-attaches when it is the only such candidate. With two or more candidates it is a NEW entity with
  ``maybe_of`` naming them: ambiguous people are never merged, and the story never claims continuity for them.
- Ids restart at P1 in a new session; a rolled-over session (events.py ``parent``) keeps its entities and ids.

Entities are plain dicts stored in ``events.Session.entities`` (so a session still round-trips through JSON):
``id, kind, tracks`` (tracker keys ``"<id>@<first_seen>"``), ``track_ids, first_seen, last_seen, state``
(active / lost / gone), ``path`` (the owner's area names with a scene map, else the picture edges they entered and
left by), ``mapped``, ``owner_label`` and ``known_ids`` (from the owner's own words, events.mark_known), ``notes``
(what the Eye said this one did), ``maybe_of``, and what the owner was last told (``told_at``, ``told_area``,
``told_gone``). Pure data, no I/O, no locks: the event book holds its lock while calling these.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set

REATTACH_SEC = 120.0             # a lost entity is still "the same one" when a lone candidate starts within this
REATTACH_DISTANCE = 0.15         # picture widths from its last foot point (as tracker.RETURN_DISTANCE)
PARKED_DISTANCE = 0.03           # as scene_map.PARKED_DISTANCE: a vehicle that moved less is parked, never an entity
KNOWN_RECENT_SEC = 120.0         # the owner's "these are my workers" covers who was seen this recently
NOTES_KEPT = 12
NOTE_CHARS = 120
ACTIVE, LOST, GONE = "active", "lost", "gone"
PREFIX = {"person": "P", "vehicle": "CAR"}
ENTITIES_VERSION = "ent1"        # the Eye's prompt version gets "+ent1" when the roster line is shown
ROSTER_HEAD = "PEOPLE/VEHICLES IN VIEW (from the tracker): "
ROSTER_RULE = ("These ids are the box's own names for who it follows; use them only in per_entity to say what each "
               "one does. Count people from the frames, not from this list.")
ROSTER_LIMIT = 300


def track_key(track: Dict[str, Any]) -> str:
    """A tracker track's key: its id and first sighting (the tracker's ids restart when its clock jumps)."""
    return f"{int(track['id'])}@{float(track['first_seen']):.3f}"


def _num(entity_id: str, prefix: str) -> int:
    rest = str(entity_id)[len(prefix):]
    return int(rest) if str(entity_id).startswith(prefix) and rest.isdigit() else 0


def _next_id(entities: Sequence[Dict[str, Any]], kind: str) -> str:
    prefix = PREFIX[kind]
    return f"{prefix}{1 + max([_num(e['id'], prefix) for e in entities if e.get('kind') == kind] or [0])}"


def _dist(p: Sequence[float], q: Sequence[float]) -> float:
    return math.hypot(float(p[0]) - float(q[0]), float(p[1]) - float(q[1]))


def _collapse(words: Iterable[str]) -> List[str]:
    out: List[str] = []
    for word in words:
        if word and (not out or out[-1] != word):
            out.append(word)
    return out


def new_entity(entities: Sequence[Dict[str, Any]], kind: str, ts: float) -> Dict[str, Any]:
    return {"id": _next_id(entities, kind), "kind": kind, "tracks": [], "track_ids": [], "first_seen": ts,
            "last_seen": ts, "state": ACTIVE, "path": [], "mapped": False, "first_foot": None, "last_foot": None,
            "owner_label": "", "known_ids": [], "notes": [], "maybe_of": [], "told_at": 0.0, "told_area": "",
            "told_gone": 0.0, "areas": {}, "edges": {}}


def _candidates(entities: Sequence[Dict[str, Any]], track: Dict[str, Any], reattach_sec: float,
                gate: float) -> List[Dict[str, Any]]:
    """Lost entities of the same kind this new track could be: last seen before it started, within *reattach_sec*,
    within *gate* of where it starts."""
    start, first = track.get("first_foot"), float(track["first_seen"])
    out = []
    for e in entities:
        if e.get("kind") != track["kind"] or e.get("last_foot") is None or start is None:
            continue
        gap = first - float(e["last_seen"])
        if 0.0 <= gap <= reattach_sec and _dist(e["last_foot"], start) <= gate:
            out.append(e)
    return out


def _attach(entities: List[Dict[str, Any]], track: Dict[str, Any], reattach_sec: float,
            gate: float) -> Dict[str, Any]:
    """The entity a track we have not seen before belongs to (an existing one, or a new one appended)."""
    prev = track.get("prev_id")
    if prev is not None:
        linked = [e for e in entities if e.get("kind") == track["kind"] and int(prev) in e.get("track_ids", [])
                  and float(e["last_seen"]) <= float(track["first_seen"])]
        if linked:
            return max(linked, key=lambda e: float(e["last_seen"]))
    found = _candidates(entities, track, reattach_sec, gate)
    if len(found) == 1:
        return found[0]
    e = new_entity(entities, track["kind"], float(track["first_seen"]))
    e["first_foot"] = list(track.get("first_foot") or ()) or None
    if len(found) > 1:
        e["maybe_of"] = sorted((x["id"] for x in found), key=lambda i: (len(i), i))
    entities.append(e)
    return e


def _refresh_path(e: Dict[str, Any]) -> None:
    order = sorted(e["tracks"], key=lambda k: float(k.split("@", 1)[1]))
    areas = [a for k in order for a in e["areas"].get(k, [])]
    e["mapped"] = bool(areas)
    e["path"] = _collapse(areas) if areas else _collapse(x for k in order for x in e["edges"].get(k, []))


def ingest(entities: List[Dict[str, Any]], tracks: Iterable[Dict[str, Any]], now: float,
           since: Optional[float] = None, reattach_sec: float = REATTACH_SEC,
           gate: float = REATTACH_DISTANCE, not_before: Optional[float] = None) -> List[str]:
    """Map the tracker's *tracks* (``CameraTracker.snapshot``) onto *entities* (changed in place). Returns the ids of
    the entities seen in ``[since, now]`` (with *since* None: the ones still in view), people first, in id order.
    Tracks that ended before *not_before* (before the session opened) are left out."""
    by_key = {k: e for e in entities for k in e.get("tracks", [])}
    active_keys: Set[str] = set()
    seen: Set[str] = set()
    for t in sorted(tracks or (), key=lambda x: float(x["first_seen"])):
        kind = t.get("kind")
        if kind not in PREFIX:
            continue
        if not_before is not None and float(t["last_seen"]) < not_before:
            continue
        if kind == "vehicle" and float(t.get("moved", 0.0)) < PARKED_DISTANCE:
            continue
        key = track_key(t)
        e = by_key.get(key)
        if e is None:
            e = _attach(entities, t, reattach_sec, gate)
            e["tracks"].append(key)
            e["track_ids"].append(int(t["id"]))
            by_key[key] = e
        first, last = float(t["first_seen"]), float(t["last_seen"])
        e["first_seen"] = min(float(e["first_seen"]), first)
        if last >= float(e["last_seen"]) or e.get("last_foot") is None:
            e["last_seen"] = last
            if t.get("last_foot") is not None:
                e["last_foot"] = list(t["last_foot"])
        e["areas"][key] = list(t.get("path") or [])
        e["edges"][key] = [x for x in (t.get("entry_edge"), t.get("exit_edge")) if x]
        _refresh_path(e)
        if t.get("active"):
            active_keys.add(key)
        if (since is None and t.get("active")) or (since is not None and last >= since):
            seen.add(e["id"])
    for e in entities:
        if any(k in active_keys for k in e.get("tracks", [])):
            e["state"] = ACTIVE
        else:
            e["state"] = LOST if now - float(e["last_seen"]) <= reattach_sec else GONE
    return order_ids(entities, seen)


def order_ids(entities: Sequence[Dict[str, Any]], ids: Iterable[str]) -> List[str]:
    wanted = set(ids)
    rank = {"person": 0, "vehicle": 1}
    return [e["id"] for e in sorted(entities, key=lambda e: (rank.get(e.get("kind"), 2), _num(e["id"], PREFIX.get(e.get("kind"), ""))))
            if e["id"] in wanted]


def by_id(entities: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {e["id"]: e for e in entities}


def people(entities: Sequence[Dict[str, Any]], ids: Iterable[str]) -> List[Dict[str, Any]]:
    index = by_id(entities)
    return [index[i] for i in ids if i in index and index[i].get("kind") == "person"]


def label_present(entities: Sequence[Dict[str, Any]], now: float, text: str, known_id: str,
                  recent_sec: float = KNOWN_RECENT_SEC) -> List[str]:
    """The owner said who is there: every entity in view (or seen in the last *recent_sec*) gets their words."""
    ids = []
    for e in entities:
        if e.get("state") == ACTIVE or now - float(e.get("last_seen", 0.0)) <= recent_sec:
            e["owner_label"] = str(text)[:80]
            if known_id not in e["known_ids"]:
                e["known_ids"].append(known_id)
            ids.append(e["id"])
    return ids


def covered(e: Dict[str, Any], live_known: Set[str]) -> bool:
    return bool(set(e.get("known_ids") or ()) & live_known)


def fresh_people(entities: Sequence[Dict[str, Any]], in_view: Iterable[str], reported: Iterable[str],
                 live_known: Set[str]) -> List[str]:
    """People in view the owner was neither told about nor marked as known. An ambiguous return (``maybe_of``) whose
    candidates were all told or all marked is not new: it is one of them or someone in their exact place."""
    index = by_id(entities)
    told = set(reported)

    def settled(entity_id: str) -> bool:
        e = index.get(entity_id)
        return e is not None and (entity_id in told or covered(e, live_known))

    out = []
    for e in people(entities, in_view):
        if settled(e["id"]):
            continue
        if e.get("maybe_of") and all(settled(m) for m in e["maybe_of"]):
            continue
        out.append(e["id"])
    return out


def add_note(e: Dict[str, Any], ts: float, text: str, label: str = "", source: str = "summary") -> None:
    text = " ".join(str(text or "").split())[:NOTE_CHARS]
    if text:
        e["notes"] = (e.get("notes", []) + [{"ts": ts, "text": text, "label": label, "source": source}])[-NOTES_KEPT:]


def attribute(entities: Sequence[Dict[str, Any]], in_view: Sequence[str], ts: float, note: str, label: str,
              per_entity: Any = None) -> List[str]:
    """What the Eye said, given to the entities it is about. With *per_entity* (``[{"id": "P1", "action": ...}]``,
    box.yaml ``eye_entities: on``) each action goes to its own entity in view; otherwise one person in view (or, with
    no people, one vehicle) gets the whole *note*. Returns the ids that got a note."""
    index = by_id(entities)
    view = [i for i in in_view if i in index]
    got: List[str] = []
    if isinstance(per_entity, list):
        for item in per_entity:
            if not isinstance(item, dict):
                continue
            entity_id = str(item.get("id") or "").strip().upper()
            if entity_id in view and str(item.get("action") or "").strip() and entity_id not in got:
                add_note(index[entity_id], ts, str(item["action"]), label, "eye")
                got.append(entity_id)
        if got:
            return got
    persons = [i for i in view if index[i].get("kind") == "person"]
    vehicles = [i for i in view if index[i].get("kind") == "vehicle"]
    lone = persons if persons else vehicles
    if len(lone) == 1 and note:
        add_note(index[lone[0]], ts, note, label, "summary")
        return [lone[0]]
    return []


def told(entities: Sequence[Dict[str, Any]], in_view: Iterable[str], reported: Iterable[str], ts: float) -> None:
    """A message reached the owner at *ts* with *in_view* in it: remember where each one was, and that the ones told
    before who are no longer in view were said to have left."""
    view = set(in_view)
    before = set(reported)
    for e in entities:
        if e["id"] in view:
            e["told_at"] = ts
            e["told_area"] = e["path"][-1] if e.get("mapped") and e.get("path") else ""
        elif e["id"] in before and e.get("state") != ACTIVE and float(e.get("told_gone") or 0.0) < float(e["last_seen"]):
            e["told_gone"] = ts


def _duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    return f"{seconds // 60} min" if seconds >= 120 else f"{seconds} s"


def roster_line(entities: Sequence[Dict[str, Any]], in_view: Sequence[str], now: float,
                since: Optional[float] = None) -> str:
    """``PEOPLE/VEHICLES IN VIEW (from the tracker): P1 in view 3 min, gate>patio; P2 new`` for the Eye, or "" with
    nobody in view. Never the owner's words or labels: the Eye judges the frames, the roster only names who is who."""
    index = by_id(entities)
    parts = []
    for entity_id in in_view:
        e = index.get(entity_id)
        if e is None:
            continue
        if since is not None and float(e["first_seen"]) >= since and not e.get("maybe_of"):
            part = f"{entity_id} new"
        else:
            part = f"{entity_id} in view {_duration(now - float(e['first_seen']))}"
        if e.get("maybe_of"):
            part += f" (maybe {' or '.join(e['maybe_of'])} back, not sure)"
        if e.get("mapped") and len(e.get("path") or ()) > 1:
            part += ", " + ">".join(e["path"][-3:])
        parts.append(part)
    if not parts:
        return ""
    line = ROSTER_HEAD
    for i, part in enumerate(parts):
        piece = ("; " if i else "") + part
        if len(line) + len(piece) > ROSTER_LIMIT:
            break
        line += piece
    return line


def prompt_block(line: str) -> str:
    """The roster line and its rule for the Eye's prompt; "" without a line."""
    return f"{line}\n{ROSTER_RULE}" if line else ""
