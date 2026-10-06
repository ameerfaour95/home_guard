"""The trust ladder: how far a matched case may soften, earned only by the owner's confirmations.

shadow  (fewer than ``quiet_after`` confirmations): decide and log "would have silenced", still alert, and ask.
quiet   (``quiet_after`` or more): the alert goes out as a quiet message.
digest  (``digest_after`` or more, and never a "not them"): a line in the daily digest only.

The owner's chosen effect is a ceiling (a case set to "quiet" never reaches digest). A "not them" steps the case
back one stage. Automatic matches never move it. The numbers are starting points, to be tuned.
"""
from __future__ import annotations

from dataclasses import dataclass

from .models import ALERT, DIGEST, EFFECTS, QUIET

SHADOW_STAGE = "shadow"


@dataclass(frozen=True)
class TrustLadder:
    quiet_after: int = 3
    digest_after: int = 10

    def stage(self, streak: int, contradictions: int) -> str:
        """``shadow`` / ``quiet`` / ``digest`` from the confirmations streak."""
        if streak >= self.digest_after and contradictions == 0:
            return DIGEST
        if streak >= self.quiet_after:
            return QUIET
        return SHADOW_STAGE

    def step_back(self, streak: int, contradictions_before: int) -> int:
        """The streak after one "not them": one stage lower (digest -> quiet, quiet -> shadow)."""
        stage = self.stage(streak, contradictions_before)
        if stage == DIGEST:
            return self.quiet_after
        return 0

    def level(self, streak: int, contradictions: int, effect: str) -> str:
        """What a confirmed match does now: the earned stage, capped by the owner's effect. Shadow -> alert."""
        stage = self.stage(streak, contradictions)
        if stage == SHADOW_STAGE or effect == ALERT or effect not in EFFECTS:
            return ALERT
        if stage == DIGEST and effect == DIGEST:
            return DIGEST
        return QUIET
