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
- With an ``Appearance`` (stage 3.2, reid.py: clothing embeddings, same day only) a new person track with no geometric
  re-attach (beyond REATTACH_SEC, or ambiguous) re-links to a lost person of the session when the clothes match
  (cosine >= ``link``, better than the second candidate by ``margin``, time and place plausible): ``linked_by:
  "appearance"``. When geometry would re-attach but the clothes match less than ``veto`` it is a NEW entity
  (``not_of``): a weak match never merges two people. In ``shadow`` mode nothing changes; ``Appearance.said`` and
  the entity's ``reid_shadow`` say what it would have done.

Entities are plain dicts stored in ``events.Session.entities`` (so a session still round-trips through JSON):
``id, kind, tracks`` (tracker keys ``"<id>@<first_seen>"``), ``track_ids, first_seen, last_seen, state``
(active / lost / gone), ``path`` (the owner's area names with a scene map, else the picture edges they entered and
left by), ``mapped``, ``owner_label`` and ``known_ids`` (from the owner's own words, events.mark_known), ``notes``
(what the Eye said this one did), ``maybe_of``, and what the owner was last told (``told_at``, ``told_area``,
``told_gone``). Pure data, no I/O, no locks: the event book holds its lock while calling these.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from ..prompts import load

REATTACH_SEC = 120.0             # a lost entity is still "the same one" when a lone candidate starts within this
REATTACH_DISTANCE = 0.15         # picture widths from its last foot point (as tracker.RETURN_DISTANCE)
PARKED_DISTANCE = 0.03           # as scene_map.PARKED_DISTANCE: a vehicle that moved less is parked, never an entity
# (a "person" the tracker found to be a fixture - tracker.STATIC_PERSON_*, a wall lamp - is never one either)
KNOWN_RECENT_SEC = 120.0         # the owner's "these are my workers" covers who was seen this recently
NOTES_KEPT = 12
NOTE_CHARS = 120
ACTIVE, LOST, GONE = "active", "lost", "gone"
PREFIX = {"person": "P", "vehicle": "CAR"}
ENTITIES_VERSION = "ent1"        # the Eye's prompt version gets "+ent1" when the roster line is shown
ROSTER_HEAD = "PEOPLE/VEHICLES IN VIEW (from the tracker): "
ROSTER_RULE = load("eye_entities_roster_rule.prompt")
ROSTER_LIMIT = 300
# Stage 3.2, appearance (reid.py); box.yaml reid_link / reid_margin / reid_veto override the first three.
REID_LINK = 0.70                 # cosine to re-link a lost person by clothes ...
REID_MARGIN = 0.08               # ... better than the second candidate by this much
REID_VETO = 0.35                 # geometry would re-attach, the clothes match less: a new entity
REID_MAX_GAP_SEC = 3600.0        # time plausible: a lost person is a re-link candidate this long after last seen
REID_PLACE_GATE = 0.35           # place plausible: starts this close to where they were lost, or at a picture edge


@dataclass
class Appearance:
    """What the clothes say (reid.Reid.appearance): ``score(track, entity)`` is the cosine of a new track's look and
    an entity's look, None when either has no embedding yet. ``mode`` "on" acts; "shadow" only records, in ``said``
    (``{"what": "link" | "veto", "track", "entity", "to", "score", "acted"}``) and on the entity (``reid_shadow``)."""

    score: Callable[[Dict[str, Any], Dict[str, Any]], Optional[float]]
    mode: str = "shadow"
    link: float = REID_LINK
    margin: float = REID_MARGIN
    veto: float = REID_VETO
    max_gap_sec: float = REID_MAX_GAP_SEC
    said: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def acting(self) -> bool:
        return self.mode == "on"

    def lines(self) -> List[str]:
        """Log lines: ``reid: would link P3->P1 (0.78)``, ``reid: linked track 17->P1 (0.78)``, ``reid: would keep P1
        apart from P1 (0.21)``, ``reid: kept P3 apart from P1 (0.21)``."""
        out = []
        for item in self.said:
            acted = bool(item.get("acted"))
            if item["what"] == "link":
                verb = "linked" if acted else "would link"
                out.append(f"reid: {verb} {item['entity']}->{item['to']} ({item['score']:.2f})")
            else:
                verb = "kept" if acted else "would keep"
                out.append(f"reid: {verb} {item['entity']} apart from {item['to']} ({item['score']:.2f})")
        return out


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
            "told_gone": 0.0, "areas": {}, "edges": {}, "entry_edge": "", "exit_edge": ""}


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


def _score(appearance: Optional[Appearance], track: Dict[str, Any], e: Dict[str, Any]) -> Optional[float]:
    if appearance is None or track.get("kind") != "person" or e.get("kind") != "person":
        return None
    try:
        value = appearance.score(track, e)
    except Exception:  # noqa: BLE001 - no appearance: geometry decides as before
        return None
    return None if value is None else float(value)


def _plausible(e: Dict[str, Any], track: Dict[str, Any], appearance: Appearance) -> bool:
    """A lost person this new track could be by time and place: lost before it started, not too long ago, and it
    starts near where they were lost or comes in at a picture edge."""
    if e.get("kind") != "person" or e.get("last_foot") is None:
        return False
    gap = float(track["first_seen"]) - float(e["last_seen"])
    if not 0.0 <= gap <= appearance.max_gap_sec:
        return False
    start = track.get("first_foot")
    return bool(track.get("entry_edge")) or (start is not None and _dist(e["last_foot"], start) <= REID_PLACE_GATE)


def _vetoed(appearance: Optional[Appearance], track: Dict[str, Any], e: Dict[str, Any],
            vetoes: List[Tuple[Dict[str, Any], float]]) -> bool:
    """Geometry says *track* is *e*; do the clothes clearly say not? Kept in *vetoes* either way; True only when
    acting (in shadow geometry still decides)."""
    value = _score(appearance, track, e)
    if value is None or value >= appearance.veto:
        return False
    vetoes.append((e, value))
    return appearance.acting


def _by_appearance(entities: Sequence[Dict[str, Any]], track: Dict[str, Any], appearance: Appearance,
                   exclude: Sequence[Dict[str, Any]]) -> Optional[Tuple[Dict[str, Any], float]]:
    """The lost person the clothes clearly point to, ``(entity, score)``, or None: the best plausible candidate at
    ``link`` or more, ahead of the second by ``margin``."""
    scored = []
    for e in entities:
        if any(e is x for x in exclude) or not _plausible(e, track, appearance):
            continue
        value = _score(appearance, track, e)
        if value is not None:
            scored.append((value, e))
    if not scored:
        return None
    scored.sort(key=lambda x: -x[0])
    best, second = scored[0][0], (scored[1][0] if len(scored) > 1 else -1.0)
    if best >= appearance.link and best - second >= appearance.margin:
        return scored[0][1], best
    return None


def _attach(entities: List[Dict[str, Any]], track: Dict[str, Any], reattach_sec: float,
            gate: float, appearance: Optional[Appearance] = None) -> Dict[str, Any]:
    """The entity a track we have not seen before belongs to (an existing one, or a new one appended). With
    *appearance* (stage 3.2) the clothes may keep apart what geometry would re-attach and may re-link a lost person
    geometry cannot; in shadow mode they only say so."""
    vetoes: List[Tuple[Dict[str, Any], float]] = []
    chosen: Optional[Dict[str, Any]] = None
    found: List[Dict[str, Any]] = []
    prev = track.get("prev_id")
    if prev is not None:
        linked = [e for e in entities if e.get("kind") == track["kind"] and int(prev) in e.get("track_ids", [])
                  and float(e["last_seen"]) <= float(track["first_seen"])]
        if linked:
            e = max(linked, key=lambda e: float(e["last_seen"]))
            if not _vetoed(appearance, track, e, vetoes):
                chosen = e
    if chosen is None:
        found = [x for x in _candidates(entities, track, reattach_sec, gate) if not any(x is v for v, _ in vetoes)]
        if len(found) == 1:
            if not _vetoed(appearance, track, found[0], vetoes):
                chosen = found[0]
            found = []
    pick = None
    if chosen is None and appearance is not None and track.get("kind") == "person":
        pick = _by_appearance(entities, track, appearance, [v for v, _ in vetoes])
    key = track_key(track)
    if chosen is None and pick is not None and appearance.acting:
        chosen, value = pick
        chosen["linked_by"] = "appearance"
        chosen["link_score"] = round(value, 3)
        chosen.setdefault("links", []).append({"track": key, "by": "appearance", "score": round(value, 3),
                                               "ts": float(track["first_seen"])})
        appearance.said.append({"what": "link", "track": key, "entity": f"track {int(track['id'])}",
                                "to": chosen["id"], "score": round(value, 3), "acted": True})
        _say_vetoes(appearance, key, track, chosen, vetoes)
        return chosen
    if chosen is None:
        chosen = new_entity(entities, track["kind"], float(track["first_seen"]))
        chosen["first_foot"] = list(track.get("first_foot") or ()) or None
        if len(found) > 1:
            chosen["maybe_of"] = sorted((x["id"] for x in found), key=lambda i: (len(i), i))
        if vetoes:                       # only acting vetoes get here: the clothes kept them apart
            chosen["not_of"] = [v["id"] for v, _ in vetoes]
            chosen["veto_score"] = round(min(x for _, x in vetoes), 3)
        entities.append(chosen)
        if pick is not None:             # shadow: what the clothes would have done
            other, value = pick
            chosen["reid_shadow"] = {"would": "link", "to": other["id"], "score": round(value, 3)}
            appearance.said.append({"what": "link", "track": key, "entity": chosen["id"], "to": other["id"],
                                    "score": round(value, 3), "acted": False})
    _say_vetoes(appearance, key, track, chosen, vetoes)
    return chosen


def _say_vetoes(appearance: Optional[Appearance], key: str, track: Dict[str, Any], chosen: Dict[str, Any],
                vetoes: List[Tuple[Dict[str, Any], float]]) -> None:
    """The clothes' "not the same one" for the log, and in shadow on the entity geometry kept."""
    for other, value in vetoes:
        acted = appearance.acting
        appearance.said.append({"what": "veto", "track": key, "entity": chosen["id"] if acted else f"track {int(track['id'])}",
                                "to": other["id"], "score": round(value, 3), "acted": acted})
        if not acted:
            chosen["reid_shadow"] = {"would": "split", "from": other["id"], "score": round(value, 3)}


def _refresh_path(e: Dict[str, Any]) -> None:
    order = sorted(e["tracks"], key=lambda k: float(k.split("@", 1)[1]))
    areas = [a for k in order for a in e["areas"].get(k, [])]
    e["mapped"] = bool(areas)
    e["path"] = _collapse(areas) if areas else _collapse(x for k in order for x in e["edges"].get(k, []))


def ingest(entities: List[Dict[str, Any]], tracks: Iterable[Dict[str, Any]], now: float,
           since: Optional[float] = None, reattach_sec: float = REATTACH_SEC,
           gate: float = REATTACH_DISTANCE, not_before: Optional[float] = None,
           appearance: Optional[Appearance] = None) -> List[str]:
    """Map the tracker's *tracks* (``CameraTracker.snapshot``) onto *entities* (changed in place). Returns the ids of
    the entities seen in ``[since, now]`` (with *since* None: the ones still in view), people first, in id order.
    Tracks that ended before *not_before* (before the session opened) are left out. *appearance* (stage 3.2,
    reid.Reid.appearance) lets the clothes re-link or keep apart a new person; None: geometry only, as before."""
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
        if kind == "person" and t.get("fixture"):        # a wall lamp read as a person (tracker.STATIC_PERSON_*)
            continue
        key = track_key(t)
        e = by_key.get(key)
        if e is None:
            e = _attach(entities, t, reattach_sec, gate, appearance)
            e["tracks"].append(key)
            e["track_ids"].append(int(t["id"]))
            by_key[key] = e
        first, last = float(t["first_seen"]), float(t["last_seen"])
        e["first_seen"] = min(float(e["first_seen"]), first)
        if last >= float(e["last_seen"]) or e.get("last_foot") is None:
            e["last_seen"] = last
            if t.get("last_foot") is not None:
                e["last_foot"] = list(t["last_foot"])
            e["exit_edge"] = str(t.get("exit_edge") or "")      # where it was last seen leaving (stage 3.3)
        if first <= float(e["first_seen"]) or "entry_edge" not in e:
            e["entry_edge"] = str(t.get("entry_edge") or "")
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


def appearance_said(entities: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """What the clothes said about these entities (read only; stage 3.2), for the assistant's "are these the same
    people?": ``{"entity", "what": "link" | "split", "other", "score", "acted"}``. A link: the clothes match a lost
    person (``reid_shadow`` would-link in shadow, ``links`` by appearance when on). A split: geometry joined a track
    to *other* but the clothes clearly differ (shadow ``reid_shadow`` split, ``not_of`` when on)."""
    out: List[Dict[str, Any]] = []
    for e in entities or ():
        if not isinstance(e, dict) or e.get("kind") != "person":
            continue
        shadow = e.get("reid_shadow") if isinstance(e.get("reid_shadow"), dict) else {}
        try:
            if shadow.get("would") == "link" and shadow.get("to"):
                out.append({"entity": e["id"], "what": "link", "other": str(shadow["to"]),
                            "score": float(shadow.get("score") or 0.0), "acted": False})
            elif shadow.get("would") == "split" and shadow.get("from"):
                out.append({"entity": e["id"], "what": "split", "other": str(shadow["from"]),
                            "score": float(shadow.get("score") or 0.0), "acted": False})
            if e.get("linked_by") == "appearance":
                out.append({"entity": e["id"], "what": "link", "other": e["id"],
                            "score": float(e.get("link_score") or 0.0), "acted": True})
            for other in e.get("not_of") or ():
                out.append({"entity": e["id"], "what": "split", "other": str(other),
                            "score": float(e.get("veto_score") or 0.0), "acted": True})
        except (KeyError, TypeError, ValueError):
            continue
    return out


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
              per_entity: Any = None, people: Optional[int] = None) -> List[str]:
    """What the Eye said, given to the entities it is about. With *per_entity* (``[{"id": "P1", "action": ...}]``,
    box.yaml ``eye_entities: on``) each action goes to its own entity in view; otherwise one person in view (or, with
    no people, one vehicle) gets the whole *note*, unless the Eye itself counted more than one person (*people*):
    then the note is about several and stays the session's observation. Returns the ids that got a note."""
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
    if len(lone) == 1 and note and not (persons and people is not None and people > 1):
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
