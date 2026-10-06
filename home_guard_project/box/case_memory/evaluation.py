"""Evaluating the case memory on "case + new event" pairs.

The headline number is the **false-silence rate** (WSR): of the events that must still alert, how many memory
would have softened, reported with a one-sided 95% upper bound (exact Clopper-Pearson), because zero mistakes in a
small sample is not zero risk: 0 of 100 still allows about 3%, and only 0 of about 300 shows it is under 1%.
The second number is how many repeat alerts the memory saves while false silence stays at or under 1%.

Pairs are built from a case and a base event at difficulty levels, each testing one part of the pipeline:

| level | change | must |
|---|---|---|
| repeat | the routine again, a few minutes off, another day | may soften (the saving) |
| wrong_hour | the same event at 02:00 (night, family asleep) and in mid-afternoon | alert (hour gate) |
| wrong_path | gate>entrance instead of gate>street; the path reversed | alert (path gate) |
| staged_suspicious | same path and hour, plus touching a handle / S1 / staying 45 s | alert (veto, dwell gate) |
| different_count | one more person; a vehicle too | alert (count gates) |
| similar_clothing | another person, similar clothes, same innocent path | allowed to match (a situation, not a person) |

``similar_clothing`` is reported on its own and kept out of the false-silence rate: memory recognises situations,
not people, and that residual risk is declared, not hidden. Real pairs (owner-confirmed repeats and UCF-Crime
segments run through the same pipeline) use the same ``evaluate``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .models import ALERT, Case, Signature
from .policy import CaseEvent, CaseMemory
from .signature import template

MUST_ALERT, MAY_SOFTEN, ALLOWED = "must_alert", "may_soften", "allowed"
LEVELS = ("repeat", "wrong_hour", "wrong_path", "staged_suspicious", "different_count", "similar_clothing")


# -- the statistics ------------------------------------------------------------------------------------------------

def _binom_cdf(k: int, n: int, p: float) -> float:
    if p <= 0:
        return 1.0
    if p >= 1:
        return 1.0 if k >= n else 0.0
    total, log_p, log_q = 0.0, math.log(p), math.log1p(-p)
    for i in range(0, k + 1):
        total += math.exp(math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1) + i * log_p + (n - i) * log_q)
    return min(1.0, total)


def upper_bound(k: int, n: int, confidence: float = 0.95) -> float:
    """One-sided exact (Clopper-Pearson) upper bound on a rate after *k* events in *n* trials.
    0 of 100 -> 0.0295 (about the rule of three, 3/n)."""
    if n <= 0:
        return 1.0
    if not 0 <= k <= n:
        raise ValueError("k must be between 0 and n")
    if k == n:
        return 1.0
    alpha = 1.0 - confidence
    if k == 0:
        return 1.0 - alpha ** (1.0 / n)
    lo, hi = k / n, 1.0
    for _ in range(100):
        mid = (lo + hi) / 2
        if _binom_cdf(k, n, mid) > alpha:
            lo = mid
        else:
            hi = mid
    return hi


def negatives_needed(target: float = 0.01, confidence: float = 0.95) -> int:
    """How many must-alert pairs with zero mistakes show the false-silence rate is under *target* (299 for 1%)."""
    return math.ceil(math.log(1.0 - confidence) / math.log(1.0 - target))


@dataclass(frozen=True)
class OperatingPoint:
    threshold: Optional[float]
    saved_rate: float                  # repeats softened / repeats
    false_silence: float               # must-alert softened / must-alert
    false_silence_upper: float
    bound_ok: bool                     # the upper bound itself is at or under the target
    positives: int
    negatives: int


def saved_at_false_silence(scored: Sequence[Tuple[Optional[float], bool]], target: float = 0.01,
                           confidence: float = 0.95) -> OperatingPoint:
    """Over ``(score, is_repeat)`` pairs (score ``None``: gated out, never softened), the lowest score threshold
    whose false-silence rate is at or under *target*, and the share of repeats it saves."""
    pos = [s for s, positive in scored if positive]
    neg = [s for s, positive in scored if not positive]
    candidates = sorted({s for s, _ in scored if s is not None})
    best = OperatingPoint(None, 0.0, 0.0, upper_bound(0, len(neg), confidence), False, len(pos), len(neg))
    for t in candidates:                                  # ascending: the first that passes saves the most
        k = sum(1 for s in neg if s is not None and s >= t)
        fs = k / len(neg) if neg else 0.0
        if fs <= target:
            saved = sum(1 for s in pos if s is not None and s >= t) / len(pos) if pos else 0.0
            ub = upper_bound(k, len(neg), confidence)
            return OperatingPoint(t, saved, fs, ub, ub <= target, len(pos), len(neg))
    return best


# -- the pairs -----------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Pair:
    case_id: str
    event: CaseEvent
    level: str
    variant: str
    expect: str                        # must_alert / may_soften / allowed
    decision: Optional[Mapping[str, Any]] = None

    def decision_or_default(self) -> Mapping[str, Any]:
        return self.decision or {"final_label": "normal", "alert_command": "[send_message]"}


def variant(sig: Signature, **changes: Any) -> Signature:
    """*sig* with *changes*; a new ``ts`` also moves minute and weekday; the template is rebuilt."""
    if "ts" in changes:
        moment = datetime.fromtimestamp(changes["ts"])
        changes.setdefault("minute", moment.hour * 60 + moment.minute)
        changes.setdefault("weekday", moment.weekday())
    out = replace(sig, **changes)
    return replace(out, template=template(out))


def _same_day_at(ts: float, hour: int, minute: int = 0) -> float:
    m = datetime.fromtimestamp(ts)
    return datetime(m.year, m.month, m.day, hour, minute).timestamp()


def make_pairs(case: Case, base: Optional[Signature] = None, event_id: str = "eval") -> List[Pair]:
    """Synthetic pairs for *case* around *base* (default: its founding example) at every difficulty level."""
    sig = base or case.examples[0].signature
    week = (datetime.fromtimestamp(sig.ts) + timedelta(days=7)).timestamp()   # same weekday, a week later
    nxt = variant(sig, ts=week + 5 * 60)
    out: List[Tuple[str, str, str, Signature]] = [
        ("repeat", "a week later, 5 minutes off", MAY_SOFTEN, nxt),
        ("repeat", "a week later, same minute", MAY_SOFTEN, variant(sig, ts=week)),
        ("wrong_hour", "02:00 asleep", MUST_ALERT,
         variant(nxt, ts=_same_day_at(week, 2), phase="late_night", house_state="home_asleep")),
        ("wrong_hour", "mid-afternoon", MUST_ALERT, variant(nxt, ts=_same_day_at(week, 15, 30))),
        ("wrong_path", "to the door", MUST_ALERT,
         variant(nxt, path=sig.path[:-1] + ("entrance",) if sig.path else ("gate", "entrance"), exit_edge="entrance")),
        ("wrong_path", "reversed", MUST_ALERT,
         variant(nxt, path=tuple(reversed(sig.path)) or ("street", "gate"),
                 entry_edge=sig.exit_edge or "street", exit_edge=sig.entry_edge or "gate")),
        ("staged_suspicious", "touches a handle", MUST_ALERT, variant(nxt, flags=("touching_handle",))),
        ("staged_suspicious", "S1 testing access", MUST_ALERT, variant(nxt, category="S1")),
        ("staged_suspicious", "stays 45 s", MUST_ALERT, variant(nxt, dwell_s=45.0)),
        ("staged_suspicious", "looks in, crouching", MUST_ALERT, variant(nxt, category="S2", flags=("crouching",))),
        ("different_count", "two people", MUST_ALERT, variant(nxt, people=sig.people + 1)),
        ("different_count", "with a vehicle", MUST_ALERT, variant(nxt, vehicles=sig.vehicles + 1)),
        ("similar_clothing", "another person, similar clothes", ALLOWED,
         variant(nxt, appearance=tuple(sig.appearance[:1]) + ("cap",))),
    ]
    pairs = []
    for i, (level, name, expect, s) in enumerate(out):
        pairs.append(Pair(case.id, CaseEvent(f"{event_id}-{case.id}-{i}", s), level, name, expect))
    return pairs


# -- running -------------------------------------------------------------------------------------------------------

def softened(memory: CaseMemory, pair: Pair) -> Tuple[bool, Optional[float]]:
    """Would memory soften this event (shadow counts: it *would have* silenced), and the pair's case score."""
    result = memory.assess(pair.event, pair.decision_or_default())
    score = dict(result.ranked).get(pair.case_id)
    if result.level != ALERT:
        return True, score
    note = result.note
    return bool(note and note.kind == "shadow"), score


def evaluate(memory: CaseMemory, pairs: Iterable[Pair], target: float = 0.01,
             confidence: float = 0.95) -> Dict[str, Any]:
    """The report: false-silence rate with its upper bound, per level, saved repeats, the allowed-match rate,
    and the operating point at *target* false silence."""
    memory.log_matches = False
    per_level: Dict[str, Dict[str, int]] = {}
    must = soft_must = rep = soft_rep = allowed = soft_allowed = 0
    scored: List[Tuple[Optional[float], bool]] = []
    failures: List[Dict[str, Any]] = []
    for pair in pairs:
        did, score = softened(memory, pair)
        row = per_level.setdefault(pair.level, {"pairs": 0, "softened": 0})
        row["pairs"] += 1
        row["softened"] += int(did)
        if pair.expect == MUST_ALERT:
            must += 1
            soft_must += int(did)
            scored.append((score, False))
            if did:
                failures.append({"case_id": pair.case_id, "level": pair.level, "variant": pair.variant})
        elif pair.expect == MAY_SOFTEN:
            rep += 1
            soft_rep += int(did)
            scored.append((score, True))
        else:
            allowed += 1
            soft_allowed += int(did)
    point = saved_at_false_silence(scored, target, confidence)
    return {
        "false_silence": {"softened": soft_must, "pairs": must, "rate": soft_must / must if must else 0.0,
                          "upper_95": upper_bound(soft_must, must, confidence),
                          "needed_for_target": negatives_needed(target, confidence)},
        "saved_repeats": {"softened": soft_rep, "pairs": rep, "rate": soft_rep / rep if rep else 0.0},
        "allowed_matches": {"softened": soft_allowed, "pairs": allowed},
        "per_level": per_level,
        "operating_point": point.__dict__,
        "failures": failures,
    }


__all__: Sequence[str] = ("upper_bound", "negatives_needed", "saved_at_false_silence", "make_pairs", "evaluate",
                          "variant", "Pair", "OperatingPoint")
