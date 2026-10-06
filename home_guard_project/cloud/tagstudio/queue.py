"""The work order: contradictions first, alerts and S/E disagreements before the rest, then untagged.

Two opinions about a clip *contradict* when one calls it an alert (suspicious, escalation, or an
old ``[alert]``) and the other calls it quiet (normal or empty), or when one says suspicious and
the other escalation: that is a **major** contradiction. Empty against normal, two different
categories of the same group, or an owner who says the AI was wrong without saying what it was,
is **minor** (unless one side is a model that was never offered "empty": its "normal" may mean nothing).

Tiers, in order:

0. a major contradiction
1. a minor contradiction, or our own tag marked needs-check
2. untagged: no old tag and no studio tag (a customer's answer alone is not a tag: it has no category)
3. done: our tag, or a migrated old tag nothing contradicts

Once we tag a clip, only an opinion given *after* our tag (a late customer answer, a new teacher run)
can reopen it. Within a tier: clips anyone calls an alert first, then the most contradictions, then
the newest.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple

from ...fleet_contract import taxonomy
from .items import AI, OLD, OWNER, STUDIO, ClipItem, Opinion, alertish, iso_ts, present
from .fields import Tag

MAJOR, MINOR, UNTAGGED, DONE = 0, 1, 2, 3
TIER_NAMES = {MAJOR: "contradiction", MINOR: "check", UNTAGGED: "untagged", DONE: "done"}


def conflict(a: Opinion, b: Opinion) -> int:
    """2 major, 1 minor, 0 none."""
    la, lb = a.effective_label(), b.effective_label()
    level = 0
    if la and lb:
        if alertish(la) != alertish(lb):
            return 2
        if {la, lb} == {"suspicious", "escalation"}:
            return 2
        if la != lb and {la, lb} <= {"normal", "empty"} and a.knows_empty and b.knows_empty:
            level = 1
    if a.category and b.category and a.category != b.category:
        level = max(level, 2 if taxonomy.group_of(a.category) != taxonomy.group_of(b.category) else 1)
    if (a.disputes_ai and b.who == AI) or (b.disputes_ai and a.who == AI):
        level = max(level, 1)
    return level


@dataclass
class Assessment:
    tier: int
    reasons: List[str] = field(default_factory=list)
    alert: bool = False
    conflicts: List[Tuple[str, str, int]] = field(default_factory=list)

    @property
    def tier_name(self) -> str:
        return TIER_NAMES[self.tier]

    def as_dict(self) -> Dict[str, object]:
        return {"tier": self.tier, "tier_name": self.tier_name, "reasons": self.reasons, "alert": self.alert,
                "conflicts": [{"a": a, "b": b, "level": "major" if lvl == 2 else "minor"}
                              for a, b, lvl in self.conflicts]}


def _pairs(ops: List[Opinion]) -> List[Tuple[str, str, int]]:
    out = []
    for i, a in enumerate(ops):
        for b in ops[i + 1:]:
            level = conflict(a, b)
            if level:
                out.append((a.who, b.who, level))
    return out


def _describe(pairs: List[Tuple[str, str, int]], ops: Dict[str, Opinion]) -> List[str]:
    reasons = []
    for a, b, _ in pairs:
        said = lambda o: o.effective_label() or ("says the AI was wrong" if o.disputes_ai else "?")  # noqa: E731
        la, lb = said(ops[a]), said(ops[b])
        if ops[a].category and ops[b].category and ops[a].category != ops[b].category:
            la, lb = f"{la} {ops[a].category}", f"{lb} {ops[b].category}"
        reasons.append(f"{a} {la} vs {b} {lb}")
    return reasons


def assess(item: ClipItem, tag: Optional[Tag] = None) -> Assessment:
    opinions = dict(item.opinions)
    if tag is not None:
        opinions[STUDIO] = tag.opinion()
    ops = present(opinions)
    is_alert = any(alertish(o.effective_label()) for o in ops)

    if tag is not None:
        if tag.fields.get("delete"):
            return Assessment(DONE, ["marked delete"], is_alert)
        studio = opinions[STUDIO]
        tagged_at = iso_ts(tag.at)
        newer = [o for o in ops if o.who != STUDIO and tagged_at is not None
                 and (iso_ts(o.at) or 0.0) > tagged_at]
        pairs = [(STUDIO, o.who, conflict(studio, o)) for o in newer]
        pairs = [p for p in pairs if p[2]]
        if pairs:
            tier = MAJOR if any(p[2] == 2 for p in pairs) else MINOR
            return Assessment(tier, ["new since our tag: " + r for r in _describe(pairs, opinions)], is_alert, pairs)
        if tag.fields.get("needs_check"):
            return Assessment(MINOR, ["marked needs-check"], is_alert)
        return Assessment(DONE, ["tagged"], is_alert)

    pairs = _pairs(ops)
    if pairs:
        tier = MAJOR if any(p[2] == 2 for p in pairs) else MINOR
        return Assessment(tier, _describe(pairs, opinions), is_alert, pairs)
    old = opinions.get(OLD)
    if old is not None and old.detail.get("delete"):
        return Assessment(DONE, ["migrated old tag: delete"], is_alert)
    if old is not None and old.effective_label():
        return Assessment(DONE, ["migrated old tag" + ("" if old.category else " (no category yet)")], is_alert)
    reasons = ["untagged"]
    if OWNER in opinions:
        reasons.append("customer answered")
    return Assessment(UNTAGGED, reasons, is_alert)


def order_key(item: ClipItem, a: Assessment) -> Tuple:
    # Among clips of the same priority, one whose video opens comes first: the tagger starts on a playable clip.
    return (a.tier, not a.alert, -len(a.conflicts), not item.info.get("has_media", True), -item.sort_ts, item.key)


def build(items: Iterable[ClipItem], tags: Dict[str, Tag]) -> List[Tuple[ClipItem, Assessment]]:
    """Every clip with its assessment, in work order."""
    rows = [(item, assess(item, tags.get(item.key))) for item in items]
    rows.sort(key=lambda row: order_key(*row))
    return rows
