"""Who is who in an alert the owner is about to get: the alert message v2 (owner, 2026-10-09).

Owner: "if it's not suspicious why send me? and I didn't understand what is this person doing? The message needs to
be professional, explaining outside, what the person is doing and how he looks, where you include the P1 P2." The
alert went out as one block of translated summary plus "למה". Now::

    🟡 חשוד · כניסה ראשית · 09:43
    מה קורה: שני אנשים ליד רכב לבן עם דלת פתוחה בחניה.
    P1 · גבר, כובע שחור, חולצה לבנה: עומד ליד העציץ ומחזיק חפץ.
    P2 · גבר, חולצה כהה: רוכן לתוך הרכב.
    למה הודעתי: אדם עם פנים מוסתרות ליד רכב פתוח.

Where the per-person lines come from. The stage-3 benchmark (home_guard_data/eval/tag_bench/REPORT.md) found that
drawing P1/P2 into the ALERT call costs real alerts, so the alert call (the Eye's prompt, labels and frames) is not
touched. A SEPARATE call, the describer, runs only for clips that are SENT (a few a day), after the decision:

- the same frames the Eye was sent, at most MAX_FRAMES of them, with the tracker's people and vehicles drawn as the
  benchmark drew them (marks.py: outline + id chip, in each frame's own crop pixels; P1/CAR1 as the event's entities
  number them, entities.py);
- a short strict-JSON prompt: one scene sentence, per id {appearance, action}, the reason;
- qwen/qwen3.5-9b via OpenRouter within DEFAULT_TIMEOUT_SEC; its English goes through the translator (messenger.py,
  with the glossary).

The benchmark measured per-person attribution for people at about 50%, so the describer must not invent: the prompt
asks for an empty action when unsure, and :func:`guard` drops any action that names an object, a weapon or a serious
act the alert's own summary did not (a weapon word anywhere only when the Eye's summary or why has one); such a line
then gives the appearance only ("P2 · גבר בחולצה כהה"). If the describer fails, times out, or nothing usable is left,
the alert goes out exactly as before: it is never delayed past the video hold (telegram_agent.VIDEO_WAIT_SEC) and
never lost. Its input and output are kept in the clip's meta (``describer``) for the Studio and later training.
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import re
import threading
import time
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from . import marks, usage_ledger

log = logging.getLogger("box.describer")

DESCRIBER_VERSION = "describer-1"
DEFAULT_PROVIDER = "openrouter"
DEFAULT_MODEL = "qwen/qwen3.5-9b"
DEFAULT_TIMEOUT_SEC = 15.0  # 2026-10-09: 9 of 10 answered in 2.9-10.3 s, one timed out at 12 s (21:34)
TRANSLATE_TIMEOUT_SEC = 10.0   # 2026-10-09 replay: Gemini Flash Lite took over 6 s on 3 of 8 field sets
MAX_FRAMES = 8           # of the Eye's frames, spread over those with marks: enough to see who does what, inside the budget
MAX_SIDE = 768           # the long side of each drawn copy (the Eye's own frames are untouched)
MAX_GAP_SEC = 1.0        # a track is drawn on a frame at most this far from its nearest look ...
MAX_SPAN_SEC = 4.0       # ... or between two looks at most this far apart (the live tracker looks 1-3 times a second)
MAX_VEHICLES = 2         # vehicles drawn besides the people (the closest to people first)
MAX_LINES = 4            # people / vehicle lines in the message, then "ועוד N"
WORDS_APPEARANCE = 10
WORDS_ACTION = 10
WORDS_SCENE = 22
WORDS_REASON = 14
REASON_HARD_WORDS = 28   # a longer reason is kept whole up to here (cut only at a full stop): its end is often the why

# Words the describer may not bring in on its own. A weapon word only when the Eye's summary or why has one.
WEAPONS = ("weapon", "gun", "pistol", "rifle", "shotgun", "firearm", "knife", "knives", "blade", "machete", "crowbar",
           "axe", "bat", "club", "taser", "sword")
# Serious acts: an action may say one only when the alert's own words do.
SERIOUS = ("break", "breaking", "broke", "steal", "stealing", "stole", "theft", "thief", "rob", "robbing", "burglar",
           "force", "forcing", "forced", "pry", "prying", "smash", "smashing", "climb", "climbing", "attack", "attacking",
           "hit", "hitting", "fight", "fighting", "threaten", "threatening", "vandal", "fire", "tamper", "tampering")
# Objects: an action may name one only when the alert's own words (or that id's own appearance) do.
OBJECTS = ("bag", "backpack", "box", "package", "parcel", "ladder", "tool", "phone", "bicycle", "bike", "scooter",
           "can", "rope", "flashlight", "torch", "camera", "sack", "bottle", "cup", "broom", "shovel", "hose", "bucket",
           "cart", "trolley", "suitcase", "plate", "license", "key", "keys", "wire", "cable", "pipe", "stick", "pole",
           "drill", "saw", "hammer", "wrench", "screwdriver", "gloves", "envelope", "letter", "food", "dog", "cat")


# ---------------------------------------------------------------------------------------------------------------
# Where each tracked one is on the sent frames
# ---------------------------------------------------------------------------------------------------------------
def sent_frame_times(frame_indices: Sequence[int], clock: Mapping[str, Any]) -> List[Optional[float]]:
    """The wall time of each sent frame: its clip frame index spread over the clip's ``[start, end]`` (*clock*:
    ``{"start", "end", "frames"}``, as the clip's frames are, clip_tracks.frame_index). None without a clock."""
    try:
        start, end, n = float(clock["start"]), float(clock["end"]), int(clock["frames"])
    except (KeyError, TypeError, ValueError):
        return [None] * len(frame_indices)
    if n <= 1 or end <= start:
        return [start] * len(frame_indices)
    return [start + int(i) * (end - start) / (n - 1) for i in frame_indices]


def box_at(boxes: Sequence[Mapping[str, Any]], ts: Optional[float],
           max_gap: float = MAX_GAP_SEC, max_span: float = MAX_SPAN_SEC) -> Optional[Tuple[float, ...]]:
    """The track's normalised box at *ts*: between two looks at most *max_span* apart it is interpolated, else the
    nearest look within *max_gap*; None when the track was not there then."""
    if ts is None:
        return None
    looks = sorted(((float(b["ts"]), tuple(float(v) for v in b["box"])) for b in boxes or ()
                    if b.get("box") is not None and len(b["box"]) == 4), key=lambda x: x[0])
    if not looks:
        return None
    before = [x for x in looks if x[0] <= ts]
    after = [x for x in looks if x[0] >= ts]
    if before and after:
        (t0, b0), (t1, b1) = before[-1], after[0]
        if t1 - t0 <= 1e-6:
            return b0
        if t1 - t0 <= max_span:
            w = (ts - t0) / (t1 - t0)
            return tuple(a + (b - a) * w for a, b in zip(b0, b1))
    near = min(looks, key=lambda x: abs(x[0] - ts))
    return near[1] if abs(near[0] - ts) <= max_gap else None


def _key(track: Mapping[str, Any]) -> str:
    from .entities import track_key  # noqa: PLC0415

    return track_key(dict(track))


def _num(entity_id: str, prefix: str) -> int:
    rest = entity_id[len(prefix):]
    return int(rest) if entity_id.startswith(prefix) and rest.isdigit() else 0


def assign_ids(tracks: Sequence[Mapping[str, Any]], mapped: Mapping[str, str]) -> Dict[str, str]:
    """Tracker key -> id: the event's own (P1, CAR1, from events / clip_tracks.entity_ids) where it has one; every
    other person / vehicle track the next free number, in order of first sighting (parked vehicles never become
    event entities, but the owner may still need "CAR1 · טנדר לבן: חונה, דלת פתוחה")."""
    out: Dict[str, str] = {}
    used = set(mapped.values())
    nxt = {"P": max([_num(i, "P") for i in used] + [0]), "CAR": max([_num(i, "CAR") for i in used if i.startswith("CAR")]
                                                                       + [0])}
    for t in sorted(tracks, key=lambda t: (float(t.get("first_seen") or 0.0), int(t.get("id") or 0))):
        key = _key(t)
        if key in mapped:
            out[key] = str(mapped[key])
            continue
        prefix = "P" if t.get("kind") == "person" else "CAR" if t.get("kind") == "vehicle" else ""
        if not prefix:
            continue
        nxt[prefix] += 1
        out[key] = f"{prefix}{nxt[prefix]}"
    return out


def place_marks(tracks: Sequence[Mapping[str, Any]], ids: Mapping[str, str], frame_ts: Sequence[Optional[float]],
                crops: Sequence[Any], sizes: Sequence[Tuple[int, int]],
                source_size: Optional[Tuple[int, int]]) -> List[List[Tuple[str, str, Tuple[int, int, int, int]]]]:
    """For each sent frame, ``[(id, kind, box in that frame's pixels)]``. A frame cut from the clip (its crop box, in
    the clip's *source_size* pixels) gets the box shifted and scaled into the crop (marks.to_frame_coords); a whole
    frame scales the normalised box to its own size."""
    out: List[List[Tuple[str, str, Tuple[int, int, int, int]]]] = []
    for k, ts in enumerate(frame_ts):
        w, h = sizes[k]
        crop = crops[k] if k < len(crops) else None
        here: List[Tuple[str, str, Tuple[int, int, int, int]]] = []
        for t in tracks:
            entity_id = ids.get(_key(t))
            nb = box_at(t.get("boxes") or (), ts) if entity_id else None
            if nb is None:
                continue
            if crop is not None and source_size:
                sw, sh = source_size
                fb = marks.to_frame_coords((nb[0] * sw, nb[1] * sh, nb[2] * sw, nb[3] * sh), tuple(crop), (w, h))
            else:
                fb = marks.to_frame_coords((nb[0] * w, nb[1] * h, nb[2] * w, nb[3] * h), None, (w, h))
            if fb is not None:
                here.append((str(entity_id), str(t.get("kind") or ""), fb))
        out.append(here)
    return out


def _order(entity_id: str) -> Tuple[int, int]:
    return (0, _num(entity_id, "P")) if entity_id.startswith("P") else (1, _num(entity_id, "CAR"))


def keep_relevant(per_frame: List[List[Tuple[str, str, Tuple[int, int, int, int]]]],
                  max_vehicles: int = MAX_VEHICLES) -> List[List[Tuple[str, str, Tuple[int, int, int, int]]]]:
    """Every person, and at most *max_vehicles* vehicles: those that come closest to a person (a parked car across
    the street is noise; the pickup someone leans into is the story)."""
    best: Dict[str, float] = {}
    for here in per_frame:
        people = [b for _, kind, b in here if kind == "person"]
        for entity_id, kind, b in here:
            if kind == "person":
                continue
            cx, cy = (b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0
            d = min((abs(cx - (p[0] + p[2]) / 2.0) + abs(cy - (p[1] + p[3]) / 2.0) for p in people), default=1e9)
            best[entity_id] = min(best.get(entity_id, 1e18), d)
    vehicles = set(sorted(best, key=lambda i: (best[i], _order(i)))[:max(0, max_vehicles)])
    return [[m for m in here if m[1] == "person" or m[0] in vehicles] for here in per_frame]


def pick_frames(per_frame: Sequence[Sequence[Any]], max_frames: int = MAX_FRAMES) -> List[int]:
    """Indices of at most *max_frames* sent frames, spread evenly over the frames that have marks (all frames when
    none has), first and last included."""
    pool = [k for k, here in enumerate(per_frame) if here] or list(range(len(per_frame)))
    if len(pool) <= max_frames:
        return pool
    step = (len(pool) - 1) / float(max_frames - 1)
    return sorted({pool[int(round(i * step))] for i in range(max_frames)})


def draw_frames(frames: Sequence[Any], per_frame: Sequence[Sequence[Tuple[str, str, Tuple[int, int, int, int]]]],
                picked: Sequence[int], times: Sequence[Optional[float]], max_side: int = MAX_SIDE) -> List[Any]:
    """Copies of the picked frames with the marks drawn (one colour per id across all frames), shrunk to *max_side*."""
    import cv2  # noqa: PLC0415

    order = sorted({m[0] for here in per_frame for m in here}, key=_order)
    out = []
    for n, k in enumerate(picked, start=1):
        here = sorted(per_frame[k], key=lambda m: _order(m[0]))
        img = marks.draw(frames[k], [(i, b, marks.color_of(i, order)) for i, _, b in here], number=n,
                         seconds=times[k] if k < len(times) else None)
        h, w = img.shape[:2]
        if max_side and max(w, h) > max_side:
            s = max_side / float(max(w, h))
            img = cv2.resize(img, (max(1, round(w * s)), max(1, round(h * s))), interpolation=cv2.INTER_AREA)
        out.append(img)
    return out


# ---------------------------------------------------------------------------------------------------------------
# The question and the answer
# ---------------------------------------------------------------------------------------------------------------
def build_prompt(ids: Mapping[str, str], summary: str, why: str, places: Sequence[str] = ()) -> str:
    """The describer's instructions. *ids*: id -> kind of everything drawn."""
    roster = ", ".join(f"{i} ({'person' if k == 'person' else 'vehicle'})" for i, k in sorted(ids.items(),
                                                                                             key=lambda x: _order(x[0])))
    place = (f'\nPlaces the tracker saw them in (the owner\'s own names, write them exactly so): '
             f'{", ".join(chr(34) + p + chr(34) for p in places)}.' if places else "")
    return f"""
You write the facts for a home-security alert the homeowner is about to get. The pictures are frames of one short
clip, in time order (#1, #2 ... bottom-left). The box's tracker drew a coloured box with an id chip on each one it
follows: {roster or "(nobody was tracked)"}.

The alert check already looked at this clip and said: "{summary.strip()}"
Why the owner is told: "{why.strip()}"{place}

Answer with ONLY this JSON:
{{"scene": "...", "entities": [{{"id": "P1", "appearance": "...", "action": "..."}}], "reason": "..."}}

- scene: ONE sentence, at most {WORDS_SCENE - 4} words: who is there, what is happening and where (the yard, the
  driveway, the street, by the gate, at the door...).
- entities: one item per id above that you can see, people first. A vehicle only when it matters (someone uses it,
  it moves, a door or trunk is open).
  - appearance: only what is visible, at most {WORDS_APPEARANCE - 2} words: man or woman only if clear, clothes and
    colours, hat, hood, mask, what they carry; say "person" when not sure which. A vehicle: its type and colour.
    If a person id's box is on something that is not a person (a lamp, a shadow, a plant), write "not a person".
  - action: what THIS id does in the frames, at most {WORDS_ACTION - 2} words, present tense. If you are not sure
    which one did it, write "" (empty). Never guess and never move an action from one id to another.
- reason: why the owner is told, at most {WORDS_REASON - 4} words, from the alert check's words above; no new facts.
- Say only what is visible. Name no object the alert check did not name unless you clearly see it carried; never
  say weapon, knife or gun unless the alert check did.
- Plain English, no ids other than those above.
""".strip()


def parse(raw: str, ids: Mapping[str, str]) -> Dict[str, Any]:
    """``{"scene", "entities": [{"id", "appearance", "action"}], "reason"}`` from the model's raw answer, trimmed;
    ValueError when it is not that JSON. Unknown or repeated ids are dropped."""
    text = (raw or "").strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    try:
        data = json.loads(text)
    except ValueError:
        start, end = text.find("{"), text.rfind("}")
        if not 0 <= start < end:
            raise ValueError("the answer is not JSON") from None
        data = json.loads(text[start:end + 1])
    if not isinstance(data, dict):
        raise ValueError("the answer is not a JSON object")

    def clean(value: Any, words: int, whole: bool = False) -> str:
        s = " ".join(str(value or "").split()).strip().strip('"').rstrip(".").strip()
        return cut_words(s, words, whole)

    seen, items = set(), []
    for item in data.get("entities") or []:
        if not isinstance(item, dict):
            continue
        entity_id = str(item.get("id") or "").strip().upper()
        if entity_id not in ids or entity_id in seen:
            continue
        seen.add(entity_id)
        items.append({"id": entity_id, "appearance": clean(item.get("appearance"), WORDS_APPEARANCE),
                      "action": clean(item.get("action"), WORDS_ACTION)})
    items.sort(key=lambda x: _order(x["id"]))
    return {"scene": clean(data.get("scene"), WORDS_SCENE), "entities": items,
            "reason": clean(data.get("reason"), REASON_HARD_WORDS, whole=True)}


# Words a cut text must not end on ("... next to a stone", "... bag. The").
_DANGLING = set("""a an the of to and or with without near next by at in on from into onto while as his her their its
is are was were be appears appear seems that this which who""".split())


def cut_words(text: str, words: int, whole: bool = False) -> str:
    """*text* in at most *words* words, never ending mid-thought. Cut at the last full stop inside the limit when there
    is one, else after the last word that is not a dangling "the / a / next to". *whole*: a text that does not fit
    and has no full stop inside the limit is "" (the caller has its own fallback). 2026-10-09 18:16 / 18:22: the
    reason, cut at 14 words, ended "... bag. The" ("... ה." in Hebrew) and "... next to a stone" ("ליד אבן")."""
    s = " ".join(str(text or "").split()).strip()
    parts = s.split(" ") if s else []
    if len(parts) <= words:
        return s
    kept = " ".join(parts[:words])
    stop = max(kept.rfind(". "), kept.rfind("! "), kept.rfind("? "), kept.rfind(".") if kept.endswith(".") else -1)
    if stop > 0:
        return kept[:stop].strip()
    if whole:
        return ""
    head = parts[:words]
    while head and head[-1].lower().strip(",;:") in _DANGLING:
        head.pop()
    return " ".join(head).rstrip(",;:")


# A dangling piece after the last full stop: one or two letters ("... גדול. ה.", "... bag. Th").
_TAIL_FRAGMENT = re.compile(r"(?<=[.!?])\s+[^\s.!?]{1,2}[.!?]?\s*$")


def tidy_translation(text: str) -> str:
    """The translator's line without a dangling 1-2 letter fragment after its last full stop."""
    s = " ".join(str(text or "").split()).strip()
    return _TAIL_FRAGMENT.sub("", s).strip()


def translated_reason_ok(source: str, told: str) -> bool:
    """Is the translated reason whole? False when it ends mid-word (a lone letter, a dangling Hebrew prefix) or is
    far shorter than its source (fewer than 40% of the words of a source of 6+ words; good Hebrew runs 50-100% of the
    English words, measured on 2026-10-09's messages, so the 60% first asked for would drop good lines)."""
    told_words = str(told or "").rstrip(".!? ").split()
    if not told_words:
        return False
    last = told_words[-1]
    if len(last) == 1 and last.isalpha():
        return False
    source_words = str(source or "").split()
    return not (len(source_words) >= 6 and len(told_words) < 0.4 * len(source_words))


def _words(text: str) -> List[str]:
    return re.findall(r"[a-z]+", str(text or "").lower())


def _has(words: Sequence[str], vocabulary: Sequence[str]) -> List[str]:
    """The vocabulary words in *words* (a plural "bags" counts as "bag")."""
    vocab = set(vocabulary)
    found = []
    for w in words:
        for form in (w, w[:-1] if w.endswith("s") else w, w[:-2] if w.endswith("es") else w):
            if form in vocab:
                found.append(form)
                break
    return found


# A person line must say it is a person. 2026-10-09 17:44 ch1: the tracker followed the wall lamp as three people and
# the owner read "P1, P2, P4 · מנורה שחורה על הקיר: נשארת על הקיר".
PERSON_WORDS = re.compile(
    r"\b(?:man|men|woman|women|person|persons|people|child|children|kid|kids|boy|boys|girl|girls|worker|workers|"
    r"guy|guys|adult|adults|teen|teens|teenager|teenagers|individual|individuals|figure|figures|someone|somebody|"
    r"male|female|pedestrian|courier|gardener|cyclist|officer|lady|gentleman)\b"
    r"|גבר|אישה|אשה|אדם|אנשים|ילד|ילדה|ילדים|נער|נערה|עובד|עובדת|עובדים|פועל|פועלים|מישהו|דמות|בחור|בחורה",
    re.IGNORECASE)
NOT_A_PERSON = re.compile(r"\bnot an? (?:person|human|people)\b|\bno (?:person|one)\b|לא אדם|אין אדם", re.IGNORECASE)


def _person_id(entity_id: str) -> bool:
    return entity_id[:1] == "P" and entity_id[1:].isdigit()


def not_a_person(item: Mapping[str, Any]) -> bool:
    """A person id (P1) whose appearance names no person (a lamp, a shadow, a plant), or says "not a person". An
    empty appearance is not evidence: the line stays."""
    appearance = str(item.get("appearance") or "").strip()
    if not _person_id(str(item.get("id") or "")) or not appearance:
        return False
    return bool(NOT_A_PERSON.search(appearance)) or not PERSON_WORDS.search(appearance)


def guard(answer: Dict[str, Any], summary: str, why: str) -> Tuple[Dict[str, Any], List[str]]:
    """The answer with anything the alert's own words do not back taken out, and what was dropped.

    - a weapon word anywhere (scene, appearance, action, reason) only when the Eye's summary or why has one;
    - an action naming a serious act (break, steal, climb...) the summary / why do not, or an object neither they nor
      that id's own appearance name, loses its action (the line keeps the appearance);
    - a person id whose appearance names no person (:func:`not_a_person`) is dropped whole and listed in the
      answer's ``not_people``: it is left out of the message and of the people counted in it."""
    told = set(_has(_words(f"{summary} {why}"), WEAPONS + SERIOUS + OBJECTS))
    answer = dict(answer)
    not_people = [str(e.get("id")) for e in answer.get("entities") or [] if not_a_person(e)]
    answer["entities"] = [e for e in answer.get("entities") or [] if str(e.get("id")) not in not_people]
    armed = bool(_has(_words(f"{summary} {why}"), WEAPONS))
    dropped: List[str] = [f"{i}: not a person" for i in not_people]
    out = {"scene": answer.get("scene", ""), "reason": answer.get("reason", ""), "entities": [],
           "not_people": not_people}
    for field in ("scene", "reason"):
        if out[field] and not armed and _has(_words(out[field]), WEAPONS):
            dropped.append(f"{field}: weapon word")
            out[field] = ""
    for item in answer.get("entities") or []:
        item = dict(item)
        if item.get("appearance") and not armed and _has(_words(item["appearance"]), WEAPONS):
            dropped.append(f"{item['id']}.appearance: weapon word")
            item["appearance"] = ""
        action = item.get("action") or ""
        if action:
            words = _words(action)
            own = set(_has(_words(item.get("appearance", "")), OBJECTS))
            bad = [w for w in _has(words, WEAPONS) if not armed]
            bad += [w for w in _has(words, SERIOUS) if w not in told]
            bad += [w for w in _has(words, OBJECTS) if w not in told and w not in own]
            if bad:
                dropped.append(f"{item['id']}.action: {', '.join(sorted(set(bad)))}")
                item["action"] = ""
        if item.get("appearance") or item.get("action"):
            out["entities"].append(item)
    return out, dropped


# ---------------------------------------------------------------------------------------------------------------
# The owner's message
# ---------------------------------------------------------------------------------------------------------------
def entity_line(entity_id: str, appearance: str, action: str) -> str:
    """"P1 · גבר, כובע שחור: עומד ליד השער"; without an action the appearance alone, without an appearance the
    action alone."""
    appearance, action = appearance.strip().rstrip("."), action.strip().rstrip(".")
    if appearance and action:
        return f"{entity_id} · {appearance}: {action}."
    if appearance:
        return f"{entity_id} · {appearance}"
    return f"{entity_id}: {action}."


def compose(label: str, camera: str, clock: str, scene: str, entities: Sequence[Mapping[str, str]], reason: str,
            lang: str, max_lines: int = MAX_LINES) -> str:
    """The alert text: the level, the camera's name and the time; "מה קורה"; one line per id (at most *max_lines*,
    then "ועוד N"); "למה הודעתי" (only when there is a reason)."""
    from .brain.i18n import t  # noqa: PLC0415

    key = {"normal": "alert_normal", "suspicious": "alert_suspicious",
           "escalation": "alert_escalation"}.get(label, "alert_unclassified")
    head = t(key, lang, camera=f"{camera} · {clock}" if clock else camera)
    lines = [head]
    if scene.strip():
        lines.append(t("msg_scene", lang, text=_sentence(scene)))
    # Ids the describer said exactly the same of share one line ("P1, P2 · אדם בבגדים כהים: הולך לאורך הקיר.").
    groups: List[Tuple[List[str], str, str]] = []
    for e in entities:
        appearance, action = str(e.get("appearance") or "").strip(), str(e.get("action") or "").strip()
        same = next((g for g in groups if g[1] == appearance and g[2] == action), None)
        if same is not None:
            same[0].append(str(e["id"]))
        else:
            groups.append(([str(e["id"])], appearance, action))
    shown = groups[:max_lines]
    for ids, appearance, action in shown:
        lines.append(entity_line(", ".join(ids), appearance, action))
    hidden = sum(len(g[0]) for g in groups[len(shown):])
    if hidden:
        lines.append(t("msg_more", lang, n=hidden))
    if reason.strip():
        lines.append(t("msg_reason", lang, text=_sentence(reason)))
    return "\n".join(lines)


def _sentence(text: str) -> str:
    text = " ".join(str(text or "").split()).strip()
    return text if text.endswith((".", "!", "?")) else f"{text}."


def to_owner_language(answer: Mapping[str, Any], lang: str, messenger: Any, keep: Sequence[str] = (),
                      timeout: float = TRANSLATE_TIMEOUT_SEC) -> Optional[Dict[str, Any]]:
    """The guarded answer in *lang* through the translator (one call, every field together, the glossary), or None
    when it cannot be translated (then the old message goes out). English passes through."""
    if lang == "en":
        return {"scene": answer.get("scene", ""), "reason": answer.get("reason", ""),
                "entities": [dict(e) for e in answer.get("entities") or []]}
    fields: Dict[str, str] = {}
    if answer.get("scene"):
        fields["scene"] = str(answer["scene"])
    if answer.get("reason"):
        fields["reason"] = str(answer["reason"])
    for e in answer.get("entities") or []:
        for part in ("appearance", "action"):
            if e.get(part):
                fields[f"{e['id']}.{part}"] = str(e[part])
    if not fields:
        return None
    told = messenger.translate(fields, lang, keep=tuple(keep), timeout=timeout) if messenger is not None else None
    if not told:
        return None
    told = {k: tidy_translation(v) for k, v in told.items()}
    reason = told.get("reason", "")
    if reason and not translated_reason_ok(fields.get("reason", ""), reason):
        reason = ""                                      # the caller's own why goes out instead
    return {"scene": told.get("scene", ""), "reason": reason,
            "entities": [{"id": e["id"], "appearance": told.get(f"{e['id']}.appearance", ""),
                          "action": told.get(f"{e['id']}.action", "")} for e in answer.get("entities") or []]}


# ---------------------------------------------------------------------------------------------------------------
# The call
# ---------------------------------------------------------------------------------------------------------------
def frames_sha(jpegs: Sequence[bytes]) -> str:
    h = hashlib.sha1()
    for data in jpegs:
        h.update(data)
    return h.hexdigest()


class Describer:
    """One OpenAI-compatible chat client (*client*, None: every call fails fast) and its model. Thread-safe."""

    def __init__(self, client: Any, model: str = DEFAULT_MODEL, timeout: float = DEFAULT_TIMEOUT_SEC,
                 extra_body: Optional[Dict[str, Any]] = None, unavailable: str = "") -> None:
        self._client, self.model, self.timeout = client, model, float(timeout)
        self._extra_body = dict(extra_body) if extra_body else None
        self.unavailable = unavailable

    def ask(self, prompt: str, jpegs: Sequence[bytes]) -> Tuple[str, Dict[str, int], str]:
        """``(raw answer, usage, model that answered)``; TimeoutError after the budget, or the client's error."""
        if self._client is None:
            raise RuntimeError(self.unavailable or "no describer client")
        content: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]
        for data in jpegs:
            content.append({"type": "image_url",
                            "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(data).decode("ascii")}})
        kwargs: Dict[str, Any] = dict(model=self.model, temperature=0, max_tokens=500,
                                      response_format={"type": "json_object"},
                                      messages=[{"role": "user", "content": content}])
        if self._extra_body:
            kwargs["extra_body"] = self._extra_body
        box: Dict[str, Any] = {}

        def call() -> None:
            try:
                box["resp"] = usage_ledger.call("describer", lambda: self._client.chat.completions.create(**kwargs),
                                                client=self._client, model=self.model, images=len(jpegs))
            except BaseException as exc:  # noqa: BLE001 - handed to the caller
                box["error"] = exc

        worker = threading.Thread(target=usage_ledger.carry(call), name="describer", daemon=True)
        worker.start()
        worker.join(self.timeout)
        if worker.is_alive():
            raise TimeoutError(f"no answer within {self.timeout:g} s")
        if "error" in box:
            raise box["error"]
        resp = box["resp"]
        usage = getattr(resp, "usage", None)
        used = {"prompt_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
                "completion_tokens": int(getattr(usage, "completion_tokens", 0) or 0)}
        answered = getattr(resp, "model", None)
        return resp.choices[0].message.content or "", used, answered if isinstance(answered, str) else self.model


def describe(describer: Describer, frames: Sequence[Any], frame_indices: Sequence[int], crops: Sequence[Any],
             clock: Mapping[str, Any], tracks: Sequence[Mapping[str, Any]], mapped: Mapping[str, str],
             summary: str, why: str, places: Sequence[str] = (), source_size: Optional[Tuple[int, int]] = None,
             model_times: Sequence[Optional[float]] = (), max_frames: int = MAX_FRAMES,
             max_side: int = MAX_SIDE) -> Dict[str, Any]:
    """Draw, ask and check. Returns the meta record ``describer``: what was drawn and asked, the raw and the guarded
    answer, timing; ``ok`` False with ``error`` on any failure. Never raises."""
    from ..data_collection.model_input import encode_jpeg  # noqa: PLC0415

    started = time.monotonic()
    record: Dict[str, Any] = {"version": DESCRIBER_VERSION, "model": describer.model, "ok": False}
    try:
        sizes = [(int(f.shape[1]), int(f.shape[0])) for f in frames]
        ts = sent_frame_times(frame_indices, clock)
        # A "person" the tracker found to be a fixture (tracker.STATIC_PERSON_*: a wall lamp) is not drawn; parked
        # vehicles still are (the owner may need "CAR1 · טנדר לבן: חונה").
        ids = assign_ids([t for t in tracks if t.get("kind") == "vehicle"
                          or (t.get("kind") == "person" and t.get("shown", True))], mapped)
        per_frame = keep_relevant(place_marks(tracks, ids, ts, crops, sizes, source_size))
        picked = pick_frames(per_frame, max_frames)
        drawn = draw_frames(frames, per_frame, picked, list(model_times) or [None] * len(frames), max_side)
        jpegs = [d for d in (encode_jpeg(img) for img in drawn) if d]
        kinds = {i: k for here in per_frame for i, k, _ in here}
        prompt = build_prompt(kinds, summary, why, places)
        record.update(frames_sha=frames_sha(jpegs), frames=len(jpegs), picked=list(picked),
                      ids=dict(sorted(kinds.items(), key=lambda x: _order(x[0]))),
                      marks=[[{"id": i, "box": list(b)} for i, _, b in per_frame[k]] for k in picked],
                      # what a replay needs to redraw them exactly (the clip meta keeps only the union crop)
                      crops=[list(crops[k]) if k < len(crops) and crops[k] is not None else None for k in picked],
                      clock=dict(clock), source_size=list(source_size) if source_size else None,
                      prompt=prompt)
        if not jpegs:
            raise ValueError("no frame to send")
        raw, usage, answered = describer.ask(prompt, jpegs)
        record.update(raw=raw, usage=usage, model=answered or describer.model)
        parsed = parse(raw, kinds)
        checked, dropped = guard(parsed, summary, why)
        record.update(answer=checked, dropped=dropped, ok=bool(checked["scene"] or checked["entities"]))
        if not record["ok"]:
            record["error"] = "nothing usable in the answer"
    except Exception as exc:  # noqa: BLE001 - the alert goes out as before
        record["error"] = f"{type(exc).__name__}: {exc}"[:300]
    record["seconds"] = round(time.monotonic() - started, 2)
    return record


# ---------------------------------------------------------------------------------------------------------------
# box.yaml
# ---------------------------------------------------------------------------------------------------------------
def settings_of(box_settings: Mapping[str, Any]) -> Tuple[bool, str, str, float]:
    """``(on, provider, model, timeout)``. box.yaml ``alert_describer: off`` turns it off (default on);
    ``describer_provider`` / ``describer_model`` / ``describer_timeout_sec`` (3-30 s) override the defaults."""
    g = box_settings.get
    on = str(g("alert_describer", "on")).strip().lower() not in ("off", "false", "0", "no", "none")
    provider = str(g("describer_provider") or DEFAULT_PROVIDER).strip().lower()
    model = str(g("describer_model") or DEFAULT_MODEL).strip()
    try:
        timeout = float(g("describer_timeout_sec") or DEFAULT_TIMEOUT_SEC)
    except (TypeError, ValueError):
        timeout = DEFAULT_TIMEOUT_SEC
    return on, provider, model, min(max(timeout, 3.0), 30.0)


_DESCRIBERS: Dict[Tuple[str, str, float], Describer] = {}
_DESCRIBERS_LOCK = threading.Lock()


def describer_for(box_settings: Mapping[str, Any], env: Mapping[str, str]) -> Describer:
    """The house's describer, built once per (provider, model, timeout). One that cannot be built still answers:
    every call then fails at once and the alert goes out as before."""
    _, provider, model, timeout = settings_of(box_settings)
    key = (provider, model, timeout)
    with _DESCRIBERS_LOCK:
        found = _DESCRIBERS.get(key)
        if found is None:
            try:
                from .messenger import build_client  # noqa: PLC0415

                client, extra = build_client(provider, env, timeout, model)
                found = Describer(client, model, timeout, extra)
            except Exception as exc:  # noqa: BLE001
                log.warning("Describer %s (%s) cannot be used: %s; alerts keep the plain text.", model, provider, exc)
                found = Describer(None, model, timeout, unavailable=f"{type(exc).__name__}: {exc}")
            _DESCRIBERS[key] = found
        return found
