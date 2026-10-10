"""A PLACE the owner names in an alert's picture is a permanent fact about the camera, never a timed mark (2026-10-10).

13:36 ch2 alerted "P1 · אדם בבגדים כהים וכובע לבן: הולך לאורך הקיר ומביט לתוך חלון." and the owner answered
"זה הבית של השכן". The bot read it as WHO the people are and asked "עד מתי לזכור את הבית של השכן?" - a house does not
expire. Owner: "THIS IS STUPID".

- :func:`place_statement`: the owner's words say whose place it is or what it is ("זה הבית של השכן", "זה השטח של
  השכן", "זו החצר של השכן", "זה הרחוב", "זה המחסן שלנו"): ``{"words", "owner"}`` with owner ``neighbour`` /
  ``public`` / ``mine``; None for anything else (a question, people, an action).
- :func:`alert_region`: where in the picture the alert's people were (their boxes in the clip's ``.tracks.json``),
  widened by REGION_MARGIN: the neighbour's region, ``[x1, y1, x2, y2]`` of 0..1.
- :func:`confirmation`: ONE natural line ("הבנתי, זה הבית של השכן. מה שקורה אצלו לא יגיע אליך, רק אם מישהו עובר
  לשטח שלך.").
- :func:`quiet_place`: in the guard loop, a suspicious whose people all stayed inside a neighbour's / public place
  the owner named at that camera (its region, or its scene-map area) is kept quiet - "neighbour's place (owner
  said)". Someone who leaves the place (onto the owner's ground) or an escalation still goes out.

Code only, never a model. Never raises from the guard-loop helpers.
"""

from __future__ import annotations

import glob
import json
import logging
import os
import re
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

log = logging.getLogger("box.place_facts")

NEIGHBOUR, PUBLIC, MINE = "neighbour", "public", "mine"
REGION_MARGIN = 0.12        # picture widths/heights around where the alert's people walked
INSIDE_SHARE = 0.8          # this much of a person's foot points inside the place is "stayed there"
QUIET_REASON = "neighbour's place (owner said)"
QUIET_REASON_PUBLIC = "public place (owner said)"

_HE = "א-ת"
_PLACES = (r"בית|הבית|שטח|השטח|חצר|החצר|רחוב|הרחוב|כביש|הכביש|מדרכה|המדרכה|מחסן|המחסן|גינה|הגינה|חניה|החניה|"
           r"חנייה|החנייה|מגרש|המגרש|בניין|הבניין|דירה|הדירה|מרפסת|המרפסת|גג|הגג|קיר|הקיר|גדר|הגדר|חלון|החלון|"
           r"צד|הצד|מגרש|שביל|השביל|סמטה|הסמטה|חלקה|החלקה|מבנה|המבנה")
_NEIGHBOUR = r"(?:של\s+)?ה?(?:שכנים|שכנה|שכן)(?:\s+ממול)?"
_MINE = r"שלנו|שלי|שלכם"
# "זה הבית של השכן" / "זו החצר של השכן" / "זה השטח של השכנים" / "זה הרחוב" / "זה המחסן שלנו" / "זה אצל השכן" /
# "זה בבית של השכן" / "לא, זה הבית של השכן".
_STATEMENT = re.compile(
    rf"^\s*(?:לא[\s,.!]+)?(?P<pron>זה|זו|זאת|זהו|זוהי|שם)\s+(?:כבר\s+)?(?P<prep>ב|אצל\s+|בתוך\s+)?"
    rf"(?P<place>(?:{_PLACES})(?![{_HE}])(?:\s+(?:{_NEIGHBOUR}|{_MINE}))?|(?:{_NEIGHBOUR}))"
    rf"(?P<rest>[^?]{{0,40}})$")
_EN = re.compile(r"^\s*(?:no[,.!\s]+)?(?P<pron>this|that|it)(?:'s| is)\s+(?P<prep>)(?:the\s+|our\s+|my\s+)?(?P<place>"
                 r"(?:neighbou?rs?'?s?\s+)?(?:house|home|yard|garden|land|property|street|road|sidewalk|shed|building|"
                 r"wall|fence|window)(?:\s+of\s+(?:the\s+)?neighbou?rs?)?)(?P<rest>[^?]{0,30})$", re.IGNORECASE)
_PERSON_ONLY = re.compile(rf"^(?:{_NEIGHBOUR})$")
_PUBLIC_WORDS = re.compile(rf"(?<![{_HE}])ה?(?:רחוב|כביש|מדרכה|סמטה|שביל)(?![{_HE}])|\b(?:street|road|sidewalk)\b",
                           re.IGNORECASE)
_NEIGHBOUR_WORDS = re.compile(rf"(?<![{_HE}])ה?(?:שכנים|שכנה|שכן)(?![{_HE}])|\bneighbou?r", re.IGNORECASE)
_MINE_WORDS = re.compile(rf"(?<![{_HE}])(?:שלנו|שלי|שלכם)(?![{_HE}])|\b(?:our|my)\b", re.IGNORECASE)
# What turns it into something else: people, a question, an action, a time.
_NOT_A_PLACE = re.compile(rf"\?|(?<![{_HE}])(?:מי|למה|מתי|איך|האם|עד|כל יום|היום|מחר|עובד|עובדים|פועל|"
                          rf"פועלים|גנן|אדם|איש|אישה|בחור|ילד|נכנס|יוצא|גונב|פורץ)(?![{_HE}])|"
                          r"\b(?:who|why|when|until|today|worker|man|woman|guy|steal|break)\w*\b", re.IGNORECASE)


def place_statement(text: str) -> Optional[Dict[str, str]]:
    """``{"words": "הבית של השכן", "owner": "neighbour"}`` when the owner's message says whose place / what place
    is in the picture; None otherwise. "זה השכן" (a person) is not a place; "זה אצל השכן" is."""
    text = " ".join(str(text or "").split()).strip(" .!")
    if not text or len(text) > 80 or _NOT_A_PLACE.search(text):
        return None
    m = _STATEMENT.match(text) or _EN.match(text)
    if m is None:
        return None
    place, prep, rest = " ".join(m.group("place").split()), (m.group("prep") or "").strip(), m.group("rest") or ""
    if _PERSON_ONLY.match(place):
        if prep != "אצל":
            return None                      # "זה השכן": who, not where
        place = f"אצל {place}"
    if prep == "ב" and not place.startswith("ה"):
        place = "ה" + place                  # "בבית של השכן" is "הבית של השכן"
    said = f"{place} {rest}"
    if _NEIGHBOUR_WORDS.search(said):
        owner = NEIGHBOUR
    elif _MINE_WORDS.search(said):
        owner = MINE
    elif _PUBLIC_WORDS.search(said):
        owner = PUBLIC
    else:
        owner = MINE
    pron = "זו" if m.group("pron") in ("זו", "זאת", "זוהי") else "זה"
    return {"words": place, "owner": owner, "pron": pron}


def confirmation(words: str, owner: str, lang: str = "he", pron: str = "זה") -> str:
    """ONE natural line for the saved place."""
    he = str(lang).startswith("he")
    pron = pron if pron in ("זה", "זו") else "זה"
    words = " ".join(str(words or "").split())
    if owner == NEIGHBOUR:
        plural = bool(re.search(r"שכנים|neighbou?rs\b", words, re.IGNORECASE))
        female = bool(re.search(r"שכנה(?![א-ת])", words))
        there = "אצלם" if plural else "אצלה" if female else "אצלו"
        return (f"הבנתי, {pron} {words}. מה שקורה {there} לא יגיע אליך, רק אם מישהו עובר לשטח שלך." if he else
                f"Got it, that's {words}. What happens there won't reach you, only someone crossing onto your ground.")
    if owner == PUBLIC:
        return (f"הבנתי, {pron} {words}. מה שקורה שם לא יגיע אליך, רק אם מישהו נכנס לשטח שלך." if he else
                f"Got it, that's {words}. What happens there won't reach you, only someone coming onto your ground.")
    return (f"הבנתי, {pron} {words}. מה שקורה שם ימשיך להגיע אליך." if he else
            f"Got it, that's {words}. I'll keep telling you what happens there.")


# ---------- where in the picture ----------
def _foot(box: Sequence[float]) -> Tuple[float, float]:
    x1, _y1, x2, y2 = (float(v) for v in box[:4])
    return (x1 + x2) / 2.0, y2


def tracks_file(roots: Iterable[str], camera: str, alert_id: str) -> str:
    """The alert clip's ``responses/<camera>/<day>/<alert_id>.tracks.json`` in any of *roots*; "" when none."""
    for root in dict.fromkeys(str(r) for r in roots if r):
        hits = glob.glob(os.path.join(glob.escape(root), "responses", glob.escape(camera), "*",
                                      glob.escape(alert_id) + ".tracks.json"))
        if hits:
            return sorted(hits)[-1]
    return ""


def people_boxes(doc: Dict[str, Any]) -> List[List[float]]:
    """Every person box of a ``.tracks.json`` document."""
    out = []
    for track in doc.get("tracks") or ():
        if not isinstance(track, dict) or track.get("kind") != "person":
            continue
        for b in track.get("boxes") or ():
            box = b.get("box") if isinstance(b, dict) else None
            if isinstance(box, (list, tuple)) and len(box) >= 4:
                out.append([float(v) for v in box[:4]])
    return out


def region_of(boxes: Sequence[Sequence[float]], margin: float = REGION_MARGIN) -> Optional[List[float]]:
    """The people's whole boxes, widened by *margin*: the place they were in. None without boxes."""
    if not boxes:
        return None
    x1 = min(b[0] for b in boxes) - margin
    y1 = min(b[1] for b in boxes) - margin
    x2 = max(b[2] for b in boxes) + margin
    y2 = max(b[3] for b in boxes) + margin
    return [round(max(0.0, x1), 4), round(max(0.0, y1), 4), round(min(1.0, x2), 4), round(min(1.0, y2), 4)]


def alert_region(roots: Iterable[str], camera: str, alert_id: str) -> Tuple[Optional[List[float]], str]:
    """``(region, scene-map area name)`` of the alert's people; ``(None, "")`` without the clip's tracks."""
    path = tracks_file(roots, camera, alert_id)
    if not path:
        return None, ""
    try:
        with open(path, encoding="utf-8") as f:
            doc = json.load(f)
    except (OSError, ValueError) as exc:
        log.warning("Tracks of %s not read: %s", alert_id, exc)
        return None, ""
    boxes = people_boxes(doc)
    return region_of(boxes), _zone_of(camera, [_foot(b) for b in boxes])


def _zone_of(camera: str, feet: Sequence[Tuple[float, float]]) -> str:
    """The scene map's area most of the people's foot points are in; "" without a map."""
    if not feet:
        return ""
    try:
        from . import scene_map  # noqa: PLC0415

        scene = scene_map.load_scene_map(camera)
        if not getattr(scene, "informative", False):
            return ""
        names = [a.name for a in (scene.area_at(p) for p in feet) if a is not None]
        return max(set(names), key=names.count) if names else ""
    except Exception as exc:  # noqa: BLE001
        log.warning("[%s] scene map not read for a place: %s", camera, exc)
        return ""


# ---------- the guard loop ----------
def _inside(point: Tuple[float, float], region: Sequence[float]) -> bool:
    x, y = point
    return region[0] <= x <= region[2] and region[1] <= y <= region[3]


def quiet_place(camera: str, tracks: Sequence[Any], profiles: Any, scene: Any = None) -> Optional[Dict[str, Any]]:
    """The neighbour's / public place the owner named at *camera* that holds every person of *tracks* (scene_map
    Tracks: ``kind``, ``points`` of ``(ts, x, y)`` foot points) for their whole visit, ending inside it; None when
    there is none, when nobody was tracked, or when someone left it. Never raises."""
    try:
        places = [p for p in profiles.places(camera) if p.get("owner") in (NEIGHBOUR, PUBLIC)
                  and (p.get("region") or p.get("zone"))]
        people = [t for t in tracks or () if getattr(t, "kind", "") == "person" and getattr(t, "points", None)]
        if not places or not people:
            return None
        for place in places:
            region, zone = place.get("region"), str(place.get("zone") or "")

            def held(p: Tuple[float, float]) -> bool:
                if region and _inside(p, region):
                    return True
                if zone and scene is not None and getattr(scene, "informative", False):
                    area = scene.area_at(p)
                    return area is not None and area.name == zone
                return False

            if all(_stays(sorted(t.points), held) for t in people):
                return place
        return None
    except Exception as exc:  # noqa: BLE001 - the alert goes out as before
        log.warning("[%s] owner's places not checked: %s", camera, exc)
        return None


def _stays(points: Sequence[Tuple[float, float, float]], held: Any) -> bool:
    inside = [held((x, y)) for _, x, y in points]
    return bool(inside) and inside[-1] and sum(inside) >= INSIDE_SHARE * len(inside)


def reason_of(place: Dict[str, Any]) -> str:
    return QUIET_REASON_PUBLIC if place.get("owner") == PUBLIC else QUIET_REASON
