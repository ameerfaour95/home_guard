"""What the Eye can say a scene is, and what that means in each situation.

One fixed list of categories (N1-N10 normal, S1-S9 suspicious, E1-E8 escalation, plus ``other``) is shared
by the vision prompt (``eye_prompt.py``), the labelling studio, the eval and the training data. The ids never
change; wording may be refined, and ``TAXONOMY_VERSION`` is bumped when it is.

The second half is the priors table (plan page section 4, spec
``docs/superpowers/specs/2026-10-05-situation-aware-eye-and-investigator-design.md``): what the same category
means by day, at night or while the family sleeps, and while nobody is home. What a scene *is* does not
depend on the hour; what it *means* does. ``contextual_label`` turns the Eye's context-free observation
into the label the box acts on. It never lowers an escalation and never goes below the Eye's own raw label.

Pure code, no model, no I/O.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

TAXONOMY_VERSION = "2026-10-06.v2"

LABELS = ("normal", "suspicious", "escalation")      # same order and words as inference.LABELS
GROUP_LABEL = {"N": "normal", "S": "suspicious", "E": "escalation"}
OTHER = "other"


@dataclass(frozen=True)
class Category:
    id: str            # "N3"
    name: str          # short English name, used in prompts and reports
    definition: str    # one line for the prompt: what the camera shows
    he: str            # the owner's/tagger's Hebrew name

    @property
    def group(self) -> str:
        return self.id[0]

    @property
    def label(self) -> str:
        return GROUP_LABEL[self.group]


CATEGORIES: Tuple[Category, ...] = (
    Category("N1", "passing by", "someone walks, cycles or drives past without stopping or entering", "עובר ברחוב"),
    Category("N2", "coming home or leaving", "a person enters or leaves the house the normal way", "חוזר הביתה או יוצא"),
    Category("N3", "delivery or service", "a courier, postman or food delivery drops something off or collects it; "
             "a courier helmet that covers the face is normal", "שליח או שירות"),
    Category("N4", "visitor at the door", "someone comes to the door, knocks or rings, and waits", "אורח בדלת"),
    Category("N5", "work", "a gardener, cleaner, technician or builder working", "עבודה: גנן, ניקיון, טכנאי"),
    Category("N6", "household life", "family life: playing, talking, sitting, cleaning, carrying bags, smoking",
             "חיי הבית"),
    Category("N7", "vehicle routine", "someone uses their own car the normal way: parks, unlocks it and gets in, "
             "unloads it, drives off", "שגרת רכב"),
    Category("N8", "animals", "only animals move (cats, dogs, birds)", "בעלי חיים"),
    Category("N9", "soldier or guard", "a soldier or security guard with a weapon slung on the body, not in hand",
             "חייל או מאבטח עם נשק תלוי"),
    Category("N10", "nothing", "nobody is there and nothing moves (light changes, plants, parked cars)", "שום דבר"),
    Category("S1", "testing access", "tries door handles, windows, gates or car doors without simply opening "
             "and using them, or tries more than one", "בודק גישה: ידיות, חלונות"),
    Category("S2", "looking in", "peers into windows or cars", "מציץ פנימה"),
    Category("S3", "surveying", "moves between entry points or walks along the fence, studying the property",
             "סוקר את הבית"),
    Category("S4", "lingering", "stays without a visible purpose (how long is measured by the box, not "
             "guessed from frames)", "שוהה בלי מטרה"),
    Category("S5", "hiding the face while approaching", "covers or hides the face while coming toward the house",
             "מסתיר פנים תוך כדי התקרבות"),
    Category("S6", "in a private area", "is in the yard, side passage or roof with no visible reason",
             "באזור פרטי בלי סיבה"),
    Category("S7", "vehicle watching", "a vehicle waits or circles with no clear purpose", "רכב צופה"),
    Category("S8", "hiding", "crouches or hides behind cover", "מסתתר"),
    Category("S9", "camera tampering", "covers, turns or touches the camera", "פגיעה במצלמה"),
    Category("E1", "forced entry", "breaks or forces a door, window or lock", "פריצה"),
    Category("E2", "climbing in", "climbs a fence, wall or window into the property", "טיפוס פנימה"),
    Category("E3", "theft", "picks up something that is not theirs and leaves with it", "גניבה"),
    Category("E4", "car break-in", "breaks into or forces a car", "פריצת רכב"),
    Category("E5", "violence", "a fight or an attack", "אלימות"),
    Category("E6", "weapon in use", "a weapon held in the hand or aimed", "נשק ביד או מכוון"),
    Category("E7", "fire, smoke or crash", "fire, smoke, or a vehicle crash", "אש, עשן, תאונה"),
    Category("E8", "person down", "a person lies on the ground and does not move", "אדם שוכב בלי תנועה"),
)

BY_ID: Dict[str, Category] = {c.id: c for c in CATEGORIES}
CATEGORY_IDS: Tuple[str, ...] = tuple(c.id for c in CATEGORIES) + (OTHER,)

# Situation vocabulary (built in code by situation.py, never guessed by a model).
PHASES = ("day", "evening", "late_night", "dawn")
HOUSE_STATES = ("home_awake", "home_asleep", "away")
INTENTS = ("alert_triage", "snapshot", "event_question", "follow_up")
CAMERA_ROLES = ("street", "entrance", "private", "parking")

# Observation vocabulary (what the Eye fills in, context-free).
ZONES = ("street", "entrance", "window", "gate", "fence", "yard", "parking", "car", "roof", "other")
MOVEMENTS = ("passing", "approaching", "leaving", "staying", "moving_around", "none")
FLAGS = ("face_covered", "touching_handle", "item_carried_away", "tool_in_hand", "weapon_visible", "flashlight",
         "crouching", "running", "uniform_or_helmet", "key_or_door_opened_from_inside")
VISIBILITY = ("clear", "partial")

# What a category means in the current situation.
EXPECTED, UNUSUAL, SERIOUS, ESCALATION = "expected", "unusual", "serious", "escalation"
EXPECTATIONS = (EXPECTED, UNUSUAL, SERIOUS, ESCALATION)

# The three columns of the priors table.
DAY, NIGHT, AWAY = "day", "night", "away"

_SERIOUS_ANYTIME = frozenset({"S1", "S2", "S6", "S8", "S9"})   # serious even by day
_SERIOUS_AT_NIGHT = frozenset({"S3", "S4", "S5", "S7"})        # suspicious by day, serious at night or away


def get(category_id: str) -> Optional[Category]:
    return BY_ID.get(str(category_id or "").strip().upper())


def normalize_id(value: Any) -> str:
    """``"n3"`` -> ``"N3"``; anything unknown -> ``"other"``."""
    text = str(value or "").strip()
    cat = get(text)
    return cat.id if cat else OTHER


def group_of(category_id: str) -> str:
    """``"N"``, ``"S"``, ``"E"`` or ``""`` for other/unknown."""
    cat = get(category_id)
    return cat.group if cat else ""


def column(phase: str, house_state: str) -> str:
    """Which priors column applies: away wins; then late night or a sleeping house; else day.
    Evening and dawn with the family awake read as day (darkness is handled by ``dark``)."""
    if house_state == "away":
        return AWAY
    if phase == "late_night" or house_state == "home_asleep":
        return NIGHT
    return DAY


@dataclass(frozen=True)
class Context:
    """What the priors table needs to know beyond the category. All fields come from code."""
    phase: str = "day"
    house_state: str = "home_awake"
    dark: bool = False
    camera_role: str = ""
    expecting: bool = False      # an owner "expecting" note covers this camera now
    fact_covers: bool = False    # a live house note (facts_for) covers this camera and hour
    movement: str = ""
    zone: str = ""
    flags: Tuple[str, ...] = ()


def expectation(category_id: str, ctx: Context) -> str:
    """The priors table: ``expected`` / ``unusual`` / ``serious`` / ``escalation``.
    ``other`` and unknown ids are ``unusual``: a case looks at them."""
    cat = get(category_id)
    if cat is None:
        return UNUSUAL
    col = column(ctx.phase, ctx.house_state)
    covered = ctx.expecting or ctx.fact_covers
    if cat.group == "E":
        return ESCALATION
    if cat.group == "S":
        if cat.id in _SERIOUS_ANYTIME or col != DAY:
            return SERIOUS
        return UNUSUAL
    cid = cat.id
    if cid in ("N1", "N8", "N10"):
        return EXPECTED
    if cid == "N2":
        if col == DAY:
            return EXPECTED
        if col == NIGHT:
            return EXPECTED if "key_or_door_opened_from_inside" in ctx.flags else UNUSUAL
        return EXPECTED if covered else UNUSUAL
    if cid == "N3":
        if col == NIGHT and ctx.phase == "late_night":
            return UNUSUAL          # food deliveries end at midnight
        return EXPECTED
    if cid == "N4":
        return UNUSUAL if col == NIGHT else EXPECTED
    if cid == "N5":
        if col == DAY:
            return EXPECTED if (not ctx.dark or covered) else UNUSUAL
        if col == NIGHT:
            return UNUSUAL
        return EXPECTED if covered else UNUSUAL
    if cid == "N6":
        if col == DAY:
            return EXPECTED
        if col == NIGHT:
            return EXPECTED if ctx.camera_role == "private" else UNUSUAL
        return EXPECTED if covered else UNUSUAL
    if cid == "N7":
        if col == DAY:
            return EXPECTED
        if col == NIGHT:
            # Passing or leaving is routine; a vehicle that stops or arrives at night is not.
            return EXPECTED if ctx.movement in ("passing", "leaving") else UNUSUAL
        return EXPECTED if covered else UNUSUAL
    if cid == "N9":
        if col == DAY:
            return EXPECTED
        return EXPECTED if ctx.movement == "passing" else UNUSUAL
    return UNUSUAL


@dataclass(frozen=True)
class Judgement:
    label: str                 # what the box acts on: normal / suspicious / escalation
    expectation: str           # expected / unusual / serious / escalation
    escalation_candidate: bool  # serious S at night or away: the Investigator (and judge) may raise it
    open_case: bool            # worth the Investigator's look
    reasons: Tuple[str, ...] = field(default_factory=tuple)


def _higher(a: str, b: str) -> str:
    valid = [x for x in (a, b) if x in LABELS]
    return max(valid, key=LABELS.index) if valid else "normal"


def contextual_label(category_id: str, raw_label: str, ctx: Context, visibility: str = "clear") -> Judgement:
    """The label the box acts on, from the Eye's context-free category + raw label and the situation.

    - E-class is always escalation. Nothing here lowers anything below the Eye's own ``raw_label``.
    - A serious S at night or away stays ``suspicious``: the Eye alone never turns an S into a call;
      it is marked ``escalation_candidate`` for the Investigator.
    - An unusual N becomes ``suspicious`` (worth the owner's look) and opens a case.
    - A raw label that disagrees with the category's group, partial visibility on a serious category,
      and ``other`` all open a case.
    """
    cid = normalize_id(category_id)
    raw = str(raw_label or "").strip().lower()
    exp = expectation(cid, ctx)
    reasons: List[str] = []
    derived = {EXPECTED: "normal", UNUSUAL: "suspicious", SERIOUS: "suspicious",
               ESCALATION: "escalation"}[exp]
    if cid == OTHER:
        reasons.append("category other")
    elif raw in LABELS and raw != BY_ID[cid].label:
        reasons.append(f"raw_label {raw} disagrees with {cid}")
    if exp == UNUSUAL and cid != OTHER:
        reasons.append(f"{cid} is unusual {column(ctx.phase, ctx.house_state)}")
    if exp in (SERIOUS, ESCALATION):
        reasons.append(f"{cid} is {exp}")
    if visibility == "partial" and exp in (SERIOUS, ESCALATION):
        reasons.append("partial visibility on a serious category")
    candidate = exp == SERIOUS and column(ctx.phase, ctx.house_state) != DAY
    label = _higher(derived, raw)
    return Judgement(label=label, expectation=exp, escalation_candidate=candidate,
                     open_case=bool(reasons), reasons=tuple(reasons))


def expectation_changes(ctx: Context, categories: Iterable[str] = ()) -> List[Tuple[str, str, str]]:
    """``(id, now, by day)`` for the categories whose meaning differs from a plain day, in this situation.
    ``categories`` limits the list (default: every N and S)."""
    wanted = list(categories) or [c.id for c in CATEGORIES if c.group in ("N", "S")]
    day = Context(phase="day", house_state="home_awake")
    out = []
    for cid in wanted:
        now, usual = expectation(cid, ctx), expectation(cid, day)
        if now != usual:
            out.append((cid, now, usual))
    return out


def expectation_lines(ctx: Context, categories: Iterable[str] = ()) -> List[str]:
    """For the prompt's expectations block: categories whose meaning differs from a plain day, in this situation.
    ``categories`` limits the list (default: every N and S)."""
    return [f"{cid} {BY_ID[cid].name}: {now} now (by day: {usual})"
            for cid, now, usual in expectation_changes(ctx, categories)]


def prompt_list(order: Sequence[str] = ("N", "S", "E")) -> str:
    """The taxonomy as prompt text, one category per line, groups in *order*."""
    titles = {"N": "Normal", "S": "Suspicious", "E": "Escalation"}
    out = []
    for group, title in ((g, titles[g]) for g in order):
        out.append(f"{title}:")
        out.extend(f"- {c.id} {c.name}: {c.definition}" for c in CATEGORIES if c.group == group)
    out.append(f"- {OTHER}: none of the above; describe it in other_text")
    return "\n".join(out)


def as_table() -> Dict[str, Any]:
    """JSON-ready taxonomy for the studio and reports."""
    return {
        "version": TAXONOMY_VERSION,
        "labels": list(LABELS),
        "categories": [{"id": c.id, "group": c.group, "label": c.label, "name": c.name,
                        "definition": c.definition, "he": c.he} for c in CATEGORIES]
        + [{"id": OTHER, "group": "", "label": "", "name": "other", "definition": "none of the above",
            "he": "אחר"}],
        "zones": list(ZONES), "movements": list(MOVEMENTS), "flags": list(FLAGS),
        "visibility": list(VISIBILITY), "phases": list(PHASES), "house_states": list(HOUSE_STATES),
        "camera_roles": list(CAMERA_ROLES), "intents": list(INTENTS),
    }


def priors_matrix(categories: Sequence[str] = ()) -> Dict[str, Dict[str, str]]:
    """Category -> column -> expectation, at the defaults of each column (for docs and the studio)."""
    cols = {DAY: Context(), NIGHT: Context(phase="late_night", house_state="home_asleep", dark=True),
            AWAY: Context(house_state="away")}
    ids = list(categories) or [c.id for c in CATEGORIES]
    return {cid: {name: expectation(cid, ctx) for name, ctx in cols.items()} for cid in ids}
