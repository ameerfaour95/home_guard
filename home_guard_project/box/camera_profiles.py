"""Two layers of what the box knows about the house: the house memory and one personality per camera (task 2.9).

Owner (2026-10-08): "the memory should be general and also per camera. Knocking on the door is usual at the main
door but not at the back door. One general memory, plus a personality and its own memory for each camera."

1. **House memory** (:func:`house_profile`): what holds at every camera. It only READS the stores that already
   exist (house_state.py: home / asleep / away, what the family expects; events.EventBook's known.json: who the
   owner said is there, ``camera=""`` meaning everywhere) plus the owner's household facts kept here under
   ``house`` ("יש לנו כלב", "שלושה ילדים"), from which code reads a dog, a cat, kids and a head-count.
2. **Camera personality** (:class:`CameraProfiles`, ``<state>/events/camera_profiles.json``), one per camera:
   - ``role``: entrance / private / street / parking / work_area. box.yaml ``camera_roles`` (setup) wins, then the
     owner's own confirmed role, then the scene map, then a role proposed from the camera's statistics
     (baseline.py), which the owner can confirm.
   - ``facts``: what the owner taught about this camera ("בדלת האחורית משתמשים רק אנחנו", "שליחים מגיעים רק
     לכניסה הראשית", "בחצר האחורית אין אף אחד בלילה"). They never expire; only the owner removes them (the
     assistant's camera_fact tool, from the owner's own words, with a receipt and Undo). Code reads a RULE from
     each one (:func:`parse_rule`): ``nobody`` (with a time of day or activities), ``family_only``, ``only_here``
     (these activities happen only at this camera), ``usual`` (an activity the owner says is routine here), else
     ``note``.
   - the learned activity statistics are baseline.py's (per camera x activity tag x hour).

A camera renamed by a site rename (``ameer_tes2_ch6`` -> ``ameer_week_0_1_ch6``) keeps its profile: a lookup falls
back to the one stored id with the same ``_chN`` ending, like camera_names.family_names.

None of this goes into the Eye's prompt; baseline.surprise combines the layers in code, and they can only RAISE.
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
import uuid
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from .activities import tags_of
from .camera_names import channel_of

log = logging.getLogger("box.camera_profiles")

PROFILES_NAME = "camera_profiles.json"
HOUSE = ""                       # the key of the house-wide facts
ROLES: Tuple[str, ...] = ("entrance", "private", "street", "parking", "work_area")
ROLE_NAMES: Dict[str, Dict[str, str]] = {
    "entrance": {"he": "כניסה", "en": "an entrance"},
    "private": {"he": "חצר פרטית", "en": "a private yard"},
    "street": {"he": "פונה לרחוב", "en": "street-facing"},
    "parking": {"he": "חניה", "en": "parking"},
    "work_area": {"he": "אזור עבודה", "en": "a work area"},
}
MAX_FACTS = 30                   # per camera
MAX_FACT_CHARS = 200
PHASES = ("day", "evening", "night", "late_night")

_HE = "א-ת"
_NIGHT = re.compile(rf"(?<![{_HE}])[בל]?(?:ה)?לילה|(?<![{_HE}])בלילות|\bnights?\b|\bat night\b|\bovernight\b",
                    re.IGNORECASE)
_EVENING = re.compile(rf"(?<![{_HE}])[בל]?(?:ה)?ערב|\bevenings?\b", re.IGNORECASE)
_DAY = re.compile(rf"(?<![{_HE}])[בל]?(?:ה)?(?:בוקר|צהריים|צהרים)|(?<![{_HE}])ביום(?![{_HE}])|\bmornings?\b|"
                  rf"\bby day\b|\bduring the day\b|\bdaytime\b|\bafternoons?\b", re.IGNORECASE)
_NOBODY = re.compile(rf"(?<![{_HE}])(?:אין|לא|אף|ללא|בלי)(?![{_HE}])|\b(?:nobody|no one|no-one|never|no)\b",
                     re.IGNORECASE)
_NOBODY_PEOPLE = re.compile(rf"אף אחד|אף אחת|אף פעם|(?<![{_HE}])אנשים|\b(?:nobody|no one|no-one|never|anyone)\b",
                            re.IGNORECASE)
_FAMILY = re.compile(rf"(?<![{_HE}])רק\s+(?:אנחנו|אנו|אני|המשפחה|משפחה|בני הבית|אנשי הבית|הילדים|שלנו|לנו|אנחנו)|"
                     rf"\bonly\s+(?:us|we|me|the family|family|our family)\b|\bfamily only\b|\bjust us\b",
                     re.IGNORECASE)
_ONLY = re.compile(rf"(?<![{_HE}])רק(?![{_HE}])|\bonly\b", re.IGNORECASE)
_USUAL = re.compile(rf"בדרך כלל|תמיד|כל יום|כל בוקר|כל ערב|זה רגיל|רגיל ש|נורמלי|\b(?:usually|always|every day|"
                    rf"every morning|every evening|often|normal here|is normal)\b", re.IGNORECASE)

# Household words in the owner's house facts.
_DOG = re.compile(rf"כלב|\bdogs?\b|\bpupp(?:y|ies)\b", re.IGNORECASE)
_CAT = re.compile(rf"חתול|\bcats?\b", re.IGNORECASE)
_KIDS = re.compile(rf"(?<![{_HE}])[ו]?(?:ה)?(?:ילד|ילדה|ילדים|ילדות|תינוק|נכד|נכדים)|\b(?:kids?|child(?:ren)?|"
                   rf"sons?|daughters?|baby|toddlers?)\b", re.IGNORECASE)
_HE_COUNT = {"שניים": 2, "שנים": 2, "שתיים": 2, "שלושה": 3, "שלוש": 3, "ארבעה": 4, "ארבע": 4, "חמישה": 5, "חמש": 5,
             "שישה": 6, "שש": 6, "שבעה": 7, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7}
_RESIDENTS = re.compile(rf"(?:גרים|גרות|גר|אנחנו|we are|there are|family of|lives?|live)\D{{0,12}}?"
                        rf"(\d+|{'|'.join(_HE_COUNT)})", re.IGNORECASE)


PLACE_KEYS = ("kind", "owner", "region", "zone", "alert_id")     # a place fact's own fields (add_place)


def _box(region: Any) -> Optional[List[float]]:
    """``[x1, y1, x2, y2]`` inside 0..1 with x1 < x2 and y1 < y2, else None."""
    try:
        x1, y1, x2, y2 = (min(1.0, max(0.0, float(v))) for v in list(region)[:4])
    except (TypeError, ValueError):
        return None
    return [round(x1, 4), round(y1, 4), round(x2, 4), round(y2, 4)] if x1 < x2 and y1 < y2 else None


def default_path() -> str:
    from . import paths  # noqa: PLC0415

    return os.path.join(paths.state_dir(), "events", PROFILES_NAME)


def phases_of(text: str) -> List[str]:
    out: List[str] = []
    if _DAY.search(text):
        out.append("day")
    if _EVENING.search(text):
        out.append("evening")
    if _NIGHT.search(text):
        out += ["night", "late_night"]
    return out


def parse_rule(text: str) -> Dict[str, Any]:
    """What code reads from one of the owner's facts: ``{"kind", "tags", "phases"}``. Never a model."""
    text = " ".join(str(text or "").split())
    tags = tags_of(text, keep_negated=True)
    phases = phases_of(text)
    if _FAMILY.search(text):
        kind = "family_only"
    elif _NOBODY.search(text) and (tags or _NOBODY_PEOPLE.search(text)):
        kind = "nobody"
    elif _ONLY.search(text) and tags:
        kind = "only_here"
    elif _USUAL.search(text) and tags:
        kind = "usual"
    else:
        kind = "note"
    return {"kind": kind, "tags": tags, "phases": phases}


def household_of(facts: Sequence[str]) -> Dict[str, Any]:
    """A dog, a cat, kids and how many live there, from the owner's house facts (code, never a model)."""
    text = " ".join(str(f) for f in facts)
    residents: Optional[int] = None
    m = _RESIDENTS.search(text)
    if m:
        word = m.group(1)
        residents = int(word) if word.isdigit() else _HE_COUNT.get(word.lower())
    return {"dog": bool(_DOG.search(text)), "cat": bool(_CAT.search(text)), "kids": bool(_KIDS.search(text)),
            "residents": residents}


def resolve_key(camera: str, keys: Sequence[str]) -> Optional[str]:
    """The stored key for *camera*: itself, else the one stored id with the same channel (a site rename)."""
    camera = str(camera or "")
    if camera in keys:
        return camera
    ch = channel_of(camera)
    if ch is None:
        return None
    same = [k for k in keys if k and channel_of(k) == ch]
    return same[0] if len(same) == 1 else None


class CameraProfiles:
    """The camera personalities and the house facts in one JSON file. Thread-safe; every write is atomic."""

    def __init__(self, path: Optional[str] = None, clock: Callable[[], float] = time.time) -> None:
        self.path = path or default_path()
        self.clock = clock
        self._lock = threading.RLock()

    # ---------- file ----------
    def _read(self) -> Dict[str, Any]:
        try:
            with open(self.path, encoding="utf-8") as f:
                data = json.load(f)
        except FileNotFoundError:
            data = {}
        except (OSError, ValueError) as exc:
            log.warning("camera profiles not read (%s); starting empty", exc)
            data = {}
        if not isinstance(data, dict):
            data = {}
        data.setdefault("version", 1)
        if not isinstance(data.get("house"), dict):
            data["house"] = {"facts": []}
        if not isinstance(data.get("cameras"), dict):
            data["cameras"] = {}
        return data

    def _write(self, data: Dict[str, Any]) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
        tmp = f"{self.path}.{uuid.uuid4().hex[:6]}.tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=1)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, self.path)

    def data(self) -> Dict[str, Any]:
        with self._lock:
            return self._read()

    def _entry(self, data: Dict[str, Any], camera: str, create: bool = False) -> Optional[Dict[str, Any]]:
        if camera == HOUSE:
            return data["house"]
        key = resolve_key(camera, list(data["cameras"]))
        if key is None:
            if not create:
                return None
            key = camera
            data["cameras"][key] = {}
        entry = data["cameras"][key]
        if not isinstance(entry, dict):
            entry = data["cameras"][key] = {}
        return entry

    # ---------- reading ----------
    def profile(self, camera: str) -> Dict[str, Any]:
        """``{"role", "role_by", "facts"}`` of *camera* (empty values when nothing is stored)."""
        with self._lock:
            entry = self._entry(self._read(), str(camera or "")) or {}
        facts = [dict(f, rule=parse_rule(f.get("text", ""))) for f in entry.get("facts") or []
                 if isinstance(f, dict) and f.get("text")]
        return {"role": str(entry.get("role") or ""), "role_by": str(entry.get("role_by") or ""), "facts": facts}

    def facts(self, camera: str) -> List[Dict[str, Any]]:
        return self.profile(camera)["facts"]

    def house_facts(self) -> List[Dict[str, Any]]:
        return self.profile(HOUSE)["facts"]

    def all_facts(self) -> Dict[str, List[Dict[str, Any]]]:
        """Every camera's facts by its stored id (another camera's "only here" rule matters to this one)."""
        with self._lock:
            data = self._read()
        out: Dict[str, List[Dict[str, Any]]] = {}
        for cam, entry in data["cameras"].items():
            if isinstance(entry, dict):
                rows = [dict(f, rule=parse_rule(f.get("text", ""))) for f in entry.get("facts") or []
                        if isinstance(f, dict) and f.get("text")]
                if rows:
                    out[cam] = rows
        return out

    def find_fact(self, fact_id: str) -> Optional[Tuple[str, Dict[str, Any]]]:
        with self._lock:
            data = self._read()
        for cam, entry in [(HOUSE, data["house"])] + list(data["cameras"].items()):
            for f in (entry or {}).get("facts") or []:
                if isinstance(f, dict) and f.get("id") == fact_id:
                    return cam, dict(f)
        return None

    # ---------- writing (the owner's own words only: brain/tools.camera_fact) ----------
    def add_fact(self, camera: str, text: str, by: str = "owner", now: Optional[float] = None) -> Dict[str, Any]:
        """Keep one fact the owner taught about *camera* ("" = the whole house). Returns it; ValueError when empty.
        The same words again are not stored twice (``already``)."""
        text = " ".join(str(text or "").split())[:MAX_FACT_CHARS]
        if not text:
            raise ValueError("nothing to remember")
        now = float(self.clock()) if now is None else float(now)
        with self._lock:
            data = self._read()
            entry = self._entry(data, str(camera or ""), create=True)
            facts = [f for f in entry.get("facts") or [] if isinstance(f, dict)]
            same = next((f for f in facts if " ".join(str(f.get("text") or "").split()).casefold() == text.casefold()),
                        None)
            if same is not None:
                return dict(same, already=True, rule=parse_rule(text))
            if len(facts) >= MAX_FACTS:
                raise ValueError(f"at most {MAX_FACTS} facts per camera; remove one first")
            fact = {"id": "CF" + uuid.uuid4().hex[:8], "text": text, "by": str(by or "owner"), "at": now}
            entry["facts"] = facts + [fact]
            self._write(data)
            log.info("camera fact saved for %s: %s", camera or "the house", text)
            return dict(fact, rule=parse_rule(text))

    def add_place(self, camera: str, text: str, owner: str, region: Optional[Sequence[float]] = None,
                  zone: str = "", alert_id: str = "", by: str = "owner", now: Optional[float] = None) -> Dict[str, Any]:
        """Keep a PLACE the owner named in an alert's picture ("זה הבית של השכן", 2026-10-10): whose it is (*owner*:
        neighbour / public / mine), where in the picture (*region*, ``[x1, y1, x2, y2]`` of 0..1, from the alert's
        people) and the scene map's area (*zone*) when there is a map. Permanent, like every camera fact; the same
        words at the same camera again widen the region instead of adding a second fact (``already``)."""
        text = " ".join(str(text or "").split())[:MAX_FACT_CHARS]
        if not text or not str(camera or ""):
            raise ValueError("nothing to remember")
        box = _box(region)
        now = float(self.clock()) if now is None else float(now)
        with self._lock:
            data = self._read()
            entry = self._entry(data, str(camera), create=True)
            facts = [f for f in entry.get("facts") or [] if isinstance(f, dict)]
            same = next((f for f in facts if f.get("kind") == "place"
                         and " ".join(str(f.get("text") or "").split()).casefold() == text.casefold()), None)
            if same is not None:
                old = _box(same.get("region"))
                if box and old:
                    same["region"] = [min(old[0], box[0]), min(old[1], box[1]), max(old[2], box[2]),
                                      max(old[3], box[3])]
                elif box:
                    same["region"] = box
                if zone and not same.get("zone"):
                    same["zone"] = str(zone)
                entry["facts"] = facts
                self._write(data)
                return dict(same, already=True, rule=parse_rule(text))
            if len(facts) >= MAX_FACTS:
                raise ValueError(f"at most {MAX_FACTS} facts per camera; remove one first")
            fact = {"id": "CF" + uuid.uuid4().hex[:8], "text": text, "by": str(by or "owner"), "at": now,
                    "kind": "place", "owner": str(owner or ""), "region": box, "zone": str(zone or ""),
                    "alert_id": str(alert_id or "")}
            entry["facts"] = facts + [fact]
            self._write(data)
            log.info("place saved for %s: %s (%s, region %s, zone %s)", camera, text, owner, box, zone or "-")
            return dict(fact, rule=parse_rule(text))

    def places(self, camera: str) -> List[Dict[str, Any]]:
        """The places the owner named at *camera* (add_place), oldest first."""
        return [f for f in self.facts(camera) if f.get("kind") == "place"]

    def remove_fact(self, fact_id: str) -> Optional[Tuple[str, Dict[str, Any]]]:
        """Remove a fact by its id; ``(camera key, the fact)`` or None when it is not there."""
        with self._lock:
            data = self._read()
            for cam, entry in [(HOUSE, data["house"])] + list(data["cameras"].items()):
                facts = (entry or {}).get("facts") or []
                hit = next((f for f in facts if isinstance(f, dict) and f.get("id") == fact_id), None)
                if hit is not None:
                    entry["facts"] = [f for f in facts if f is not hit]
                    self._write(data)
                    log.info("camera fact removed from %s: %s", cam or "the house", hit.get("text"))
                    return cam, dict(hit)
        return None

    def restore_fact(self, camera: str, fact: Dict[str, Any]) -> bool:
        """Put a removed fact back with its own id (Undo of a removal). False when it is already there."""
        if not isinstance(fact, dict) or not fact.get("id") or not fact.get("text"):
            raise ValueError("invalid fact")
        with self._lock:
            data = self._read()
            entry = self._entry(data, str(camera or ""), create=True)
            facts = [f for f in entry.get("facts") or [] if isinstance(f, dict)]
            if any(f.get("id") == fact["id"] for f in facts):
                return False
            entry["facts"] = facts + [{k: fact[k] for k in ("id", "text", "by", "at") + PLACE_KEYS if k in fact}]
            self._write(data)
            return True

    def set_role(self, camera: str, role: str, by: str = "owner", now: Optional[float] = None) -> str:
        """The owner confirmed the camera's role; returns the role it had before ("" when none)."""
        role = str(role or "").strip().lower()
        if role and role not in ROLES:
            raise ValueError(f"role must be one of {', '.join(ROLES)}")
        with self._lock:
            data = self._read()
            entry = self._entry(data, str(camera or ""), create=True)
            old = str(entry.get("role") or "")
            if role:
                entry.update(role=role, role_by=str(by or "owner"),
                             role_at=float(self.clock()) if now is None else float(now))
            else:
                for key in ("role", "role_by", "role_at"):
                    entry.pop(key, None)
            self._write(data)
            return old


def role_of(camera: str, settings: Optional[Dict[str, Any]] = None, profiles: Optional[CameraProfiles] = None,
            proposed: str = "", scene_map: Any = None) -> Tuple[str, str]:
    """``(role, source)``: box.yaml camera_roles ("setup"), the owner's confirmed one ("owner"), the scene map's
    ("map"), the statistics' proposal ("proposed"), else ("", "")."""
    roles = (settings or {}).get("camera_roles") if isinstance(settings, dict) else None
    if isinstance(roles, dict):
        key = resolve_key(camera, list(roles))
        chosen = str(roles.get(key) or "").strip().lower() if key else ""
        if chosen in ROLES:
            return chosen, "setup"
    if profiles is not None:
        try:
            mine = profiles.profile(camera).get("role", "")
        except Exception:  # noqa: BLE001
            mine = ""
        if mine in ROLES:
            return mine, "owner"
    if scene_map is not None:
        try:
            from_map = str(scene_map.camera_role() or "")
        except Exception:  # noqa: BLE001
            from_map = ""
        if from_map in ROLES:
            return from_map, "map"
    if proposed in ROLES:
        return proposed, "proposed"
    return "", ""


def propose_role(tag_days: Dict[str, int], days: int) -> str:
    """A role from what the camera sees most (owner-confirmable): knocks / deliveries -> entrance, cars moving ->
    parking, work -> work_area, people passing -> street, else private. "" with under a week of data."""
    if days < 7:
        return ""
    rate = {t: n / max(1, days) for t, n in tag_days.items()}
    if rate.get("knock", 0) + rate.get("delivery", 0) >= 0.15:
        return "entrance"
    if rate.get("working", 0) >= 0.3:
        return "work_area"
    if rate.get("vehicle_move", 0) >= 0.3:
        return "parking"
    if rate.get("walk_past", 0) >= 0.5:
        return "street"
    return "private"


# ---------- the house memory ----------
def _known_rows(events_dir: str, now: float) -> List[Dict[str, Any]]:
    """Who the owner said is there (events.EventBook's known.json), live ones only. Read, never written here."""
    try:
        with open(os.path.join(events_dir, "known.json"), encoding="utf-8") as f:
            rows = json.load(f)
    except (OSError, ValueError):
        return []
    out = []
    for r in rows if isinstance(rows, list) else []:
        if isinstance(r, dict) and float(r.get("at") or 0) <= now < float(r.get("until") or 0):
            out.append({"who": str(r.get("text") or ""), "camera": str(r.get("camera") or ""),
                        "until": float(r.get("until") or 0), "people": int(r.get("people") or 0)})
    return out


def house_profile(now: Optional[float] = None, events_dir: Optional[str] = None,
                  profiles: Optional[CameraProfiles] = None, house_path: Optional[str] = None,
                  house_now: Any = None) -> Dict[str, Any]:
    """The house memory: what holds at every camera. Reads house_state (state, expecting), the event book's known
    people and the owner's house facts. Never raises: a store that cannot be read is left out."""
    now = time.time() if now is None else float(now)
    out: Dict[str, Any] = {"state": "", "expecting": [], "known": [], "facts": [], "household": household_of([])}
    try:
        if house_now is None:
            from . import house_state  # noqa: PLC0415

            house_now = house_state.current(now=now, path=house_path) if house_path else house_state.current(now=now)
        out["state"] = str(getattr(house_now, "state", "") or "")
        out["expecting"] = [{"text": str(e.get("text") or ""), "camera": str(e.get("camera") or "")}
                            for e in getattr(house_now, "expecting", []) or [] if isinstance(e, dict)]
    except Exception as exc:  # noqa: BLE001
        log.debug("house state not read: %s", exc)
    if events_dir is None:
        try:
            from . import paths  # noqa: PLC0415

            events_dir = os.path.join(paths.state_dir(), "events")
        except Exception:  # noqa: BLE001
            events_dir = ""
    if events_dir:
        out["known"] = _known_rows(events_dir, now)
    try:
        store = profiles or CameraProfiles(os.path.join(events_dir, PROFILES_NAME) if events_dir else None)
        facts = [str(f.get("text") or "") for f in store.house_facts()]
        out["facts"] = facts
        out["household"] = household_of(facts)
    except Exception as exc:  # noqa: BLE001
        log.debug("house facts not read: %s", exc)
    return out
