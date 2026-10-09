"""The weighted score of a new event against each case's closest example.

S = 0.35 path + 0.25 description + 0.20 hour + 0.10 time in view + 0.10 appearance (starting guesses, to be fitted
on labelled pairs). Each component is 0..1. A component that cannot be computed (no tracker path, no embedding,
the owner said "the clothes change") is left out and the remaining weights are renormalised; ``available`` says
which ones counted, so the bands can refuse a high score that rests on too little.

The comparison is against the *closest* example, not an average, so a routine with natural variation (with a bag,
without) is still covered.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from ..embeddings import cosine
from .gates import circular_distance, in_window, path_distance
from .models import Case, Example, Signature, window_minutes

# "actions" plays the path's part for a case of an explained action (activity_memory): computed only for those.
DEFAULT_WEIGHTS: Mapping[str, float] = {"path": 0.35, "text": 0.25, "time": 0.20, "dwell": 0.10, "appearance": 0.10,
                                        "actions": 0.35}
TIME_SIGMA_MIN = 20.0


@dataclass(frozen=True)
class ScoreDetail:
    score: float
    components: Dict[str, float] = field(default_factory=dict)
    example_id: str = ""

    @property
    def available(self) -> Tuple[str, ...]:
        return tuple(sorted(self.components))


def time_score(a: int, b: int, sigma: float = TIME_SIGMA_MIN) -> float:
    d = circular_distance(a, b)
    return math.exp(-(d * d) / (2 * sigma * sigma))


def dwell_score(a: Optional[float], b: Optional[float]) -> Optional[float]:
    if a is None or b is None:
        return None
    hi = max(a, b)
    return 1.0 if hi <= 0 else min(a, b) / hi


def jaccard(a: Sequence[str], b: Sequence[str]) -> Optional[float]:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return None
    return len(sa & sb) / len(sa | sb)


def combine(components: Mapping[str, Optional[float]], weights: Mapping[str, float] = DEFAULT_WEIGHTS) -> float:
    used = {k: v for k, v in components.items() if v is not None and weights.get(k, 0) > 0}
    total = sum(weights[k] for k in used)
    if total <= 0:
        return 0.0
    return sum(weights[k] * max(0.0, min(1.0, v)) for k, v in used.items()) / total


def actions_score(case: Case, sig: Signature) -> Optional[float]:
    """For a case of an explained action: the share of the actions the event names that the owner explained."""
    if not case.scope.actions or not sig.actions:
        return None
    return len(set(sig.actions) & set(case.scope.actions)) / len(set(sig.actions))


def _time(case: Case, sig: Signature, ex_minute: int, sigma: float) -> float:
    """Closeness to the example's minute; for an explained action, a work window: anywhere inside it is as good."""
    if case.scope.actions and in_window(sig.minute, case.scope.hours):
        return 1.0
    return time_score(sig.minute, ex_minute, sigma)


def score_example(case: Case, example: Example, sig: Signature, embedding: Optional[Sequence[float]],
                  weights: Mapping[str, float] = DEFAULT_WEIGHTS, sigma: float = TIME_SIGMA_MIN) -> ScoreDetail:
    ex = example.signature
    parts: Dict[str, Optional[float]] = {
        "path": 1.0 - path_distance(sig.path, ex.path) if sig.path and ex.path else None,
        "text": cosine(embedding, example.embedding) if embedding and example.embedding else None,
        "time": _time(case, sig, ex.minute, sigma),
        "actions": actions_score(case, sig),
        "dwell": dwell_score(sig.dwell_s, ex.dwell_s),
        # The owner's confirmed words win; "the clothes change" (no words) leaves appearance out.
        "appearance": jaccard(sig.appearance, case.recognise) if case.recognise else None,
    }
    if parts["text"] is not None:
        parts["text"] = max(0.0, parts["text"])
    kept = {k: v for k, v in parts.items() if v is not None and weights.get(k, 0) > 0}
    return ScoreDetail(score=combine(kept, weights), components=kept, example_id=example.event_id)


def score_scope(case: Case, sig: Signature, weights: Mapping[str, float] = DEFAULT_WEIGHTS,
                sigma: float = TIME_SIGMA_MIN) -> ScoreDetail:
    """For a case with no examples (should not happen): the window's centre and the scope's path."""
    start, length = window_minutes(case.scope.hours)
    parts = {"time": _time(case, sig, (start + length // 2) % 1440, sigma)}
    if actions_score(case, sig) is not None:
        parts["actions"] = actions_score(case, sig)
    if sig.path and case.scope.path:
        parts["path"] = 1.0 - path_distance(sig.path, case.scope.path)
    return ScoreDetail(score=combine(parts, weights), components=parts)


def score_case(case: Case, sig: Signature, embedding: Optional[Sequence[float]],
               weights: Mapping[str, float] = DEFAULT_WEIGHTS, sigma: float = TIME_SIGMA_MIN) -> ScoreDetail:
    """The case's score: its closest example's."""
    if not case.examples:
        return score_scope(case, sig, weights, sigma)
    return max((score_example(case, e, sig, embedding, weights, sigma) for e in case.examples),
               key=lambda d: d.score)


def rank(cases: Sequence[Case], sig: Signature, embedding: Optional[Sequence[float]],
         weights: Mapping[str, float] = DEFAULT_WEIGHTS, sigma: float = TIME_SIGMA_MIN) -> List[Tuple[Case, ScoreDetail]]:
    scored = [(c, score_case(c, sig, embedding, weights, sigma)) for c in cases]
    return sorted(scored, key=lambda item: item[1].score, reverse=True)
