""""זה הרכב של החשמלאי מנסור, תזכור את המספר שלו": a car named by its plate, in code (owner, 2026-10-15/19).

2026-10-15 09:18 the owner answered a live video of the pergola with that sentence; the assistant replied "אני אזכור
לזהות אותו כרכב של החשמלאי" but saved only his sentence as a camera fact, on another camera, and read no number (the
box had no plate reader). On 10-19 the same car was "suspicious" three times. Now, before the act handlers:

- **When**: he names a car ("זה הרכב של X", "הרכב הזה של X") or asks to remember its number ("תזכור את המספר שלו"),
  or types a plate with a name ("המספר 16-098-75 זה של החשמלאי"). A question ("של מי הרכב?") is not this.
- **Which car**: a typed plate is taken as he wrote it. Otherwise the picture he means: the alert he replied to, else
  the last thing the assistant sent him (an alert, a live photo, a live video) within ``RECENT_SEC``. The plate is the
  one the box read there (car_plates.py: an alert's reads, or a live look read now, waited for at most
  ``READ_WAIT_SEC``).
- **The reply** is one short Hebrew line written here (``plan.final``), never by the writer: "רשמתי: הרכב הלבן
  ••••875 = הרכב של החשמלאי מנסור." with a memory receipt (``known_vehicle``) and Undo. No plate read: he is told so in
  one line and asked to type it ("the number is not readable in this picture"); nothing is claimed and nothing is
  saved. Several different plates: he is asked to type it. The plate is always shown masked.
- The memory acts the understanding made of the same words (a camera fact "תזכור את המספר שלו") are dropped: that
  path is what answered "אני אזכור" without any number.
"""
from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from .. import known_vehicles as kv
from .. import plates
from ..brain.receipts import DONE, FAILED
from ..brain.tools import _issue

log = logging.getLogger("box.assistant_v3.vehicles")

RECENT_SEC = 30 * 60.0          # "the last thing I sent him" is at most this old
ASK_SEC = 15 * 60.0             # a question about the number is answered within this
READ_WAIT_SEC = 20.0
PREF = "v3_plate_ask"

_CAR = r"(?:רכב|אוטו|אוטו׳|מכונית|טנדר|משאית|ג'יפ|ג׳יפ|ואן|מסחרית|אופנוע)"
_CAR_WORD = re.compile(rf"(?<![א-ת])(?:[ובכלמש]{{0,2}})?(ה?)({_CAR})(?![א-ת])")
_NUMBER = re.compile(r"(?<![א-ת])(?:[ובלהאת]{0,2})?(?:מספר|מס['׳]|לוחית|לוחות|נומרטור)(?![א-ת])")
_REMEMBER = re.compile(r"(?<![א-ת])(?:ו)?(?:תזכור|תזכרי|זכור|תזכרו|תשמור|שמור|תרשום|רשום|תרשמי|תשמרי)(?![א-ת])")
_OWNED = re.compile(rf"(?<![א-ת])ה?{_CAR}(?:\s+ה(?:זה|זאת|לבן|לבנה|שחור|שחורה|כסוף|כסופה|אפור|אפורה|אדום|אדומה|כחול|כחולה))?"
                    r"\s+(?:הוא\s+|זה\s+)?(של|שלי|שלנו|שלו|שלה)(?![א-ת])")
_OF = re.compile(r"(?<![א-ת])(?:זה|הוא|זאת)?\s*(של)\s+(?=[א-ת])")
_QUESTION = re.compile(r"\?|^\s*(?:מה|מי|איפה|למה|האם|איזה|איזו|כמה|של מי|מתי)(?![א-ת])")
_PLATE = re.compile(r"(?<!\d)(\d{2,3}[-‐–\s.]?\d{2,3}[-‐–\s.]?\d{2,3})(?!\d)")
_STOP = frozenset("זה זאת זו הוא היא לא כן תזכור תזכרי זכור תשמור שמור תרשום רשום את המספר מספר עם אבל רק גם "
                  "ותזכור ותשמור ותרשום ולא כי שזה וזה".split())
_MINE = {"שלי": "שלך", "שלנו": "שלכם", "אשתי": "אשתך", "בעלי": "בעלך", "אמא": "אמא", "אבא": "אבא"}
_COLOURS = {"לבן": "לבן", "לבנה": "לבן", "שחור": "שחור", "שחורה": "שחור", "כסוף": "כסוף", "כסופה": "כסוף",
            "אפור": "אפור", "אפורה": "אפור", "אדום": "אדום", "אדומה": "אדום", "כחול": "כחול", "כחולה": "כחול",
            "white": "לבן", "silver": "כסוף", "black": "שחור", "grey": "אפור", "gray": "אפור", "red": "אדום",
            "blue": "כחול"}
_COLOUR_WORD = re.compile(r"(?<![א-תA-Za-z])(?:ה)?(לבן|לבנה|שחור|שחורה|כסוף|כסופה|אפור|אפורה|אדום|אדומה|כחול|כחולה)"
                          r"(?![א-ת])|\b(white|silver|black|grey|gray|red|blue)\s+(?:\w+\s+)?(?:car|sedan|van|pickup|"
                          r"truck|suv|vehicle)\b", re.IGNORECASE)
MEMORY_ACTS = ("camera_fact", "place_fact", "person_mark", "activity_explain")


# ---------------------------------------------------------------------------------------------------------------
# His words
# ---------------------------------------------------------------------------------------------------------------
def typed_plate(text: str) -> str:
    """The plate he typed ("16-098-75", "1609875"), digits only; "" when none (or not 7-8 digits)."""
    for m in _PLATE.finditer(str(text or "")):
        digits = plates.digits_only(m.group(1))
        if plates.valid_plate(digits):
            return digits
    return ""


def is_question(text: str) -> bool:
    return bool(_QUESTION.search(str(text or "").strip()))


def asks_number(text: str) -> bool:
    """"תזכור את המספר שלו" / "תרשום את הלוחית": remember a car's number."""
    text = str(text or "")
    return bool(_REMEMBER.search(text) and _NUMBER.search(text))


_NOT = re.compile(r"(?<![א-ת])לא\s*$")
_THIS_BEFORE = re.compile(r"(?<![א-ת])(?:זה|זאת|זהו|זוהי|הוא)\s*$")
_THIS_INSIDE = re.compile(r"(?<![א-ת])(?:הזה|הזאת|הוא|זה)(?![א-ת])")


def _owned(text: str, pointing: bool = False) -> Optional["re.Match[str]"]:
    """The first "הרכב של X" that is not denied ("זה לא הרכב של X" names nobody); with *pointing*, only one that
    points at a car ("זה הרכב של X", "הרכב הזה של X", "הרכב הלבן הוא של X"), not "הרכב של השכן חוסם לי"."""
    for m in _OWNED.finditer(text):
        before = text[max(0, m.start() - 8):m.start()]
        if _NOT.search(before):
            continue
        if pointing and not (_THIS_BEFORE.search(before) or _THIS_INSIDE.search(m.group(0))):
            continue
        return m
    return None


def names_car(text: str) -> bool:
    """He points at a car and says whose it is: "זה הרכב של X" / "הרכב הלבן הוא של X" / "זה האוטו שלי"; never
    "זה לא הרכב של X", and not a sentence that only mentions someone's car."""
    return _owned(str(text or ""), pointing=True) is not None


def _owner_words(rest: str) -> str:
    words: List[str] = []
    for w in re.split(r"\s+", re.split(r"[,.!?;:\n()\-–—]|\d", rest, maxsplit=1)[0].strip()):
        if not w or w in _STOP or w.startswith(("ותז", "ותש", "ותר")):
            break
        words.append(_MINE.get(w, w))
        if len(words) >= 4:
            break
    return " ".join(words)


def label_of(text: str) -> str:
    """"הרכב של החשמלאי מנסור" from his words ("זה הרכב של החשמלאי מנסור זה לא של הפרגולה..."), the car word his;
    "הרכב שלך" for "הרכב שלי". "" when no owner is named."""
    text = str(text or "")
    m = _owned(text)
    car = "הרכב"
    if m:
        cw = _CAR_WORD.search(m.group(0))
        if cw:
            car = "ה" + cw.group(2)
        if m.group(1) != "של":
            return f"{car} {_MINE.get(m.group(1), m.group(1))}"
        who = _owner_words(text[m.end():])
        return f"{car} של {who}" if who else ""
    of = _OF.search(text)
    if of and not _OWNED.search(text):
        cw = _CAR_WORD.search(text)
        if cw:
            car = "ה" + cw.group(2)
        who = _owner_words(text[of.end():])
        return f"{car} של {who}" if who else ""
    return ""


def colour_of(*texts: str) -> str:
    """The car's colour in Hebrew from his words or a picture's description ("a white car" -> "לבן")."""
    for text in texts:
        m = _COLOUR_WORD.search(str(text or ""))
        if m:
            return _COLOURS.get((m.group(1) or m.group(2) or "").lower(), "")
    return ""


# ---------------------------------------------------------------------------------------------------------------
# The picture he means
# ---------------------------------------------------------------------------------------------------------------
def _recent_shown(t: Any) -> List[Tuple[str, Dict[str, Any]]]:
    """The newest things the assistant sent him (the latest turn that showed any), newest first: alert / photo /
    clip handles of that one turn, within RECENT_SEC."""
    state = t.state
    for turn in reversed(state.turns or []):
        if not isinstance(turn, dict):
            continue
        if t.now - float(turn.get("ts") or 0) > RECENT_SEC:
            break
        found = []
        for h in reversed(turn.get("handles") or []):
            entry = state.resolve(h)
            if isinstance(entry, dict) and entry.get("kind") in ("event", "photo", "clip"):
                found.append((h, entry))
        if found:
            return found
    return []


def _source(t: Any) -> List[Tuple[str, Dict[str, Any]]]:
    asked = _pending(t)
    if asked.get("ref"):              # "של מי הרכב?" was about this picture
        return [("", {"kind": asked.get("kind") or "event", "ref": asked["ref"], "camera": asked.get("camera") or ""})]
    if t.ctx.alert_handle:
        entry = t.state.resolve(t.ctx.alert_handle)
        if isinstance(entry, dict):
            return [(t.ctx.alert_handle, entry)]
    return _recent_shown(t)


def _no_vehicle(entry: Dict[str, Any]) -> bool:
    """A live photo whose detector objects (kept on its handle) show no vehicle: not worth reading."""
    objects = entry.get("objects")
    return entry.get("kind") == "photo" and isinstance(objects, list) and bool(objects) and not any(
        isinstance(o, dict) and (o.get("kind") == "vehicle" or o.get("label") in ("car", "truck", "bus", "motorcycle"))
        for o in objects)


def plates_in(t: Any) -> Tuple[List[Dict[str, Any]], str, Dict[str, Any], bool]:
    """``(plates read, camera, entry, reader available)`` of the picture he means: each plate once (reads within one
    digit are one car), with the alert / media it came from. Reading stops after READ_WAIT_SEC in all."""
    import time  # noqa: PLC0415

    from .. import car_plates  # noqa: PLC0415

    svc = car_plates.service()
    seen: List[Dict[str, Any]] = []
    camera, first = "", {}
    deadline = time.monotonic() + READ_WAIT_SEC
    for handle, entry in _source(t):
        first = first or entry
        camera = camera or t.camera_now(str(entry.get("camera") or ""))
        left = deadline - time.monotonic()
        if svc is None or left <= 0 or _no_vehicle(entry):
            continue
        ref = str(entry.get("ref") or "")
        try:
            result = (svc.result(ref, wait=min(5.0, left)) if entry.get("kind") == "event"
                      else svc.read_media(ref, str(entry.get("camera") or ""), wait=left))
        except Exception as exc:  # noqa: BLE001
            log.warning("v3 plates: %s not read: %s", handle, exc)
            result = None
        if result is None:
            continue
        for key, v in (result.get("vehicles") or {}).items():
            digits = str(v.get("plate") or "")
            if plates.valid_plate(digits) and not any(plates.levenshtein(digits, s["plate"]) <= 1 for s in seen):
                seen.append({"plate": digits, "source": str(entry.get("ref") or ""), "vehicle": key,
                             "kind": str(entry.get("kind") or ""), "box": v.get("box"),
                             "camera": t.camera_now(str(entry.get("camera") or "")),
                             "observation": str(entry.get("observation") or entry.get("summary") or "")})
    return seen, camera, first, svc is not None


# ---------------------------------------------------------------------------------------------------------------
# A live look: which of the cars in it are ones he named (2026-10-19 15:53 "מה יש בחוץ")
# ---------------------------------------------------------------------------------------------------------------
LOOK_BUDGET_SEC = 3.0           # plate reads a live look may add to the answer, in all


def _vehicle_boxes(entry: Dict[str, Any]) -> List[List[float]]:
    return [list(o["box"]) for o in entry.get("objects") or () if isinstance(o, dict)
            and (o.get("kind") == "vehicle" or o.get("label") in ("car", "truck", "bus", "motorcycle"))
            and isinstance(o.get("box"), (list, tuple)) and len(o["box"]) == 4]


def known_cars(ctx: Any, handles: List[str], now: float, budget: float = LOOK_BUDGET_SEC) -> List[Tuple[str, str]]:
    """``[(camera, "הרכב שלך (טסלה לבנה)")]`` for the cars he named that are in these live photos: by the plate
    reader on the photo's main-stream frame (plate cameras, by day, within *budget* s in all), else by a spot a plate
    confirmed earlier (stayed_vehicles.py, any hour). Plates only from the reader; an unknown plate names nothing."""
    import time as _time  # noqa: PLC0415

    from .. import car_plates  # noqa: PLC0415

    svc = car_plates.service()
    if svc is None:
        return []
    store = kv.for_services(ctx.services)
    stayed = getattr(svc, "stayed", None)
    deadline = _time.monotonic() + budget
    out: List[Tuple[str, str]] = []
    for handle in handles:
        entry = ctx.state.resolve(handle)
        if not isinstance(entry, dict) or entry.get("kind") != "photo":
            continue
        camera = str(entry.get("camera") or "")
        boxes = _vehicle_boxes(entry)
        if entry.get("objects") is not None and not boxes:
            continue                                   # the look's detector saw no vehicle here
        names: List[str] = []
        left = deadline - _time.monotonic()
        if svc.settings.wants(camera) and car_plates.is_day(now) and left > 0.2:
            try:
                result = svc.read_media(str(entry.get("ref") or ""), camera, wait=left, boxes=boxes or None)
            except Exception as exc:  # noqa: BLE001
                log.warning("v3: live look plates not read: %s", exc)
                result = None
            for v in ((result or {}).get("vehicles") or {}).values():
                rec = store.get(str(v.get("vehicle_id"))) if store is not None and v.get("vehicle_id") else None
                if rec and rec.get("label") and not rec.get("removed_at"):
                    names.append(_known_words(rec))
        if stayed is not None:
            for box in boxes:
                spot = stayed.at(camera, box)
                rec = store.get(str(spot.get("vehicle_id"))) if spot and store is not None else None
                words = _known_words(rec) if rec else (str(spot.get("label")) if spot else "")
                if words and words not in names:
                    names.append(words)
        out += [(camera, n) for n in names]
    return out


def _known_words(rec: Dict[str, Any]) -> str:
    d = " ".join(str(rec.get("description") or "").split())
    return f"{rec.get('label')} ({d})" if d else str(rec.get("label") or "")


# ---------------------------------------------------------------------------------------------------------------
# The turn
# ---------------------------------------------------------------------------------------------------------------
def _pending(t: Any) -> Dict[str, Any]:
    ask = (t.state.prefs or {}).get(PREF)
    if isinstance(ask, dict) and 0 <= t.now - float(ask.get("ts") or 0) <= ASK_SEC:
        return ask
    return {}


def _drop_memory_acts(t: Any) -> None:
    """The understanding's memory acts of these words go: they would save the sentence and claim "I'll remember"."""
    keep = [a for a in t.und.acts if a.act not in MEMORY_ACTS]
    if len(keep) != len(t.und.acts):
        t.plan.trace.append("car naming: memory acts dropped")
    from .acts import Act  # noqa: PLC0415

    t.und.acts = keep or [Act(act="ack")]


def _ask_typed(t: Any, label: str, camera: str, line: str, picture: Optional[Dict[str, Any]] = None) -> None:
    """Ask him for what is missing (the number, or whose car), remembered for ASK_SEC: his answer completes it."""
    t.state.prefs[PREF] = {"label": label, "camera": camera, "ts": t.now, **(picture or {})}
    t.plan.final = line


def bind(t: Any) -> bool:
    """Handle a car named or "remember its number" in this message (see the module). True when it answered."""
    text = t.text
    typed = typed_plate(text)
    pending = _pending(t)
    naming, number = names_car(text), asks_number(text)
    if is_question(text) and not typed:
        return False
    answers = bool(pending.get("ref") and not pending.get("label") and _OF.search(text))   # "של מנסור"
    if not (naming or number or answers or (typed and (pending or _OF.search(text) or _CAR_WORD.search(text)))):
        return False
    label = label_of(text) or str(pending.get("label") or "")
    if not label:
        for older in t.mem_owner_words()[:2]:
            if names_car(older) or asks_number(older):
                label = label_of(older)
                break
    _drop_memory_acts(t)
    store = kv.for_services(t.ctx.services)
    if store is None:
        t.plan.final = "אין לי כרגע איפה לשמור רכבים, לא שמרתי את המספר."
        return True
    if typed:
        if not label:
            t.plan.final = (f"של מי הרכב {kv.owner_mask(typed)}? תכתוב לי למשל: \"המספר {_dashed(typed)} זה של "
                            "החשמלאי\".")
            return True
        _save(t, store, typed, label, colour_of(text), str(pending.get("camera") or t.camera_of(None) or ""), "typed")
        return True
    found, camera, entry, available = plates_in(t)
    if not label:
        _ask_typed(t, "", camera, "של מי הרכב? תכתוב לי למשל: \"זה הרכב של החשמלאי\".",
                   {"kind": found[0]["kind"], "ref": found[0]["source"]} if len(found) == 1 else None)
        return True
    if not entry:
        _ask_typed(t, label, camera, "לא שלחתי לך לאחרונה תמונה של רכב; תכתוב לי את המספר שלו ואשמור אותו כ"
                   f"{label}.")
        return True
    if len(found) == 1:
        f = found[0]
        rec = _save(t, store, f["plate"], label, colour_of(text, f["observation"]), f["camera"] or camera, f["source"])
        _stays(rec, f["camera"] or camera, f.get("box"), f["plate"], t.now)
        if entry.get("kind") == "event":
            t.outcome(entry, f"בעל הבית: {label}")
        return True
    if len(found) > 1:
        _ask_typed(t, label, camera, "יש בתמונה יותר מרכב אחד עם מספר, ואני לא בטוח איזה מהם; תכתוב לי את המספר "
                   "שלו ואשמור אותו.")
        return True
    if not available:
        _ask_typed(t, label, camera, "אני לא יכול לקרוא מספרי רכב כרגע; תכתוב לי את המספר ואשמור אותו כ"
                   f"{label}.")
    else:
        _ask_typed(t, label, camera, "המספר לא קריא בתמונה הזו; תכתוב לי אותו (למשל 12-345-67) ואשמור אותו כ"
                   f"{label}.")
    return True


def _dashed(digits: str) -> str:
    return f"{digits[:2]}-{digits[2:5]}-{digits[5:]}" if len(digits) == 7 else f"{digits[:3]}-{digits[3:5]}-{digits[5:]}"


def _stays(rec: Optional[Dict[str, Any]], camera: str, box: Any, plate: str, now: float) -> None:
    """The car he just named, read in that picture on *box*: it keeps its name on that spot from now (stayed)."""
    if not rec or not camera or not isinstance(box, (list, tuple)) or len(box) != 4:
        return
    from .. import car_plates  # noqa: PLC0415

    svc = car_plates.service()
    stayed = getattr(svc, "stayed", None) if svc is not None else None
    if stayed is not None:
        try:
            stayed.plate_read(camera, box, plate, rec, now)
        except Exception as exc:  # noqa: BLE001
            log.warning("v3: the named car's spot not kept: %s", exc)


def _save(t: Any, store: Any, plate: str, label: str, colour: str, camera: str,
          source: str) -> Optional[Dict[str, Any]]:
    by = str((t.ctx.speaker or {}).get("name") or "owner")
    try:
        rec = store.add(plate, label, colour, camera, by=by, source=source, now=t.now)
    except (OSError, ValueError) as exc:
        _issue(t.ctx, "known_vehicle", FAILED, camera or "house", {"plate": plates.mask(plate), "label": label},
               "error")
        t.plan.final = "לא הצלחתי לשמור את הרכב, נסה שוב בעוד רגע."
        log.warning("v3: known vehicle not saved: %s", exc)
        return None
    replaced = rec.get("replaced") or {}
    _issue(t.ctx, "known_vehicle", DONE, camera or "house",
           {"vehicle_id": rec["id"], "plate": plates.mask(plate), "label": rec["label"],
            "description": rec.get("description", ""), "camera": camera, "already": bool(rec.get("already")),
            "replaced_id": str(replaced.get("id") or ""), "replaced_label": str(replaced.get("label") or "")})
    t.state.prefs.pop(PREF, None)
    line = kv.masked_line(rec)
    if rec.get("already"):
        t.plan.final = f"זה כבר שמור אצלי: {line}."
    elif replaced:
        t.plan.final = f"עדכנתי: {line} (קודם: {replaced.get('label')})."
    else:
        t.plan.final = f"רשמתי: {line}."
    t.did(t.plan.final)
    return rec
