"""Shared fixtures for the case memory tests: the neighbour leaving through the gate on weekday mornings."""
from __future__ import annotations

import hashlib
import math
from datetime import datetime
from typing import Any, Dict, List, Optional

from home_guard_project.box.case_memory import (CaseEvent, CaseMemory, CaseMemoryConfig, CaseStore, MemoryBackend)
from home_guard_project.box.case_memory import interviewer as iv

NORMAL = {"final_label": "normal", "alert_command": "[send_message]", "serious_behaviour": False}
SUSPICIOUS = {"final_label": "suspicious", "alert_command": "[send_message]", "serious_behaviour": False}
OWNER = "owner:111"


def at(day: int, hour: int, minute: int = 0, month: int = 10) -> float:
    """Local time in October 2026 (4th = Sunday, 5th = Monday, 10th = Saturday)."""
    return datetime(2026, month, day, hour, minute).timestamp()


class Clock:
    def __init__(self, ts: float) -> None:
        self.ts = ts

    def __call__(self) -> float:
        return self.ts


def fake_embed(text: str, dim: int = 64) -> List[float]:
    """Bag of template tokens hashed into *dim* buckets: close templates get close vectors. No network."""
    vec = [0.0] * dim
    for part in text.replace(",", ";").split(";"):
        token = part.strip()
        if token:
            vec[int(hashlib.sha1(token.encode()).hexdigest(), 16) % dim] += 1.0
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


def observation(**overrides: Any) -> Dict[str, Any]:
    obs = {"category": "N2", "zone": "gate", "movement": "leaving", "flags": [], "people": 1, "vehicle_moving": False,
           "animals": 0, "visibility": "clear", "raw_label": "normal", "label": "normal", "serious_behaviour": False,
           "appearance": ["dark jacket", "backpack"]}
    obs.update(overrides)
    return obs


def tracker(**overrides: Any) -> Dict[str, Any]:
    trk = {"time_in_view_s": 9.0, "path": ["gate", "street"]}
    trk.update(overrides)
    return trk


def event(ts: Optional[float] = None, event_id: str = "gate_ev", obs: Optional[Dict[str, Any]] = None,
          trk: Optional[Dict[str, Any]] = None, situation: Optional[Dict[str, Any]] = None,
          camera: str = "gate", label: str = "normal", cameras_in_incident: int = 1) -> CaseEvent:
    return CaseEvent.build(event_id, camera, at(4, 7, 40) if ts is None else ts, obs or observation(),
                           tracker() if trk is None else trk, situation or {"phase": "day", "house_state": "home_awake"},
                           label=label, cameras_in_incident=cameras_in_incident)


def interview_case(ev: Optional[CaseEvent] = None, who: str = "who:neighbour", when: str = "when:workweek",
                   third: str = "rec:varies"):
    """Run the interview the plan page shows and return its outcome."""
    ev = ev or event()
    interview, _ = iv.start(ev.event_id, ev.signature)
    interview.answer(who)
    interview.answer(when)
    interview.answer(third)
    interview.answer("card:save")
    return interview.outcome()


def memory(clock: Optional[Clock] = None, judge=None, config: CaseMemoryConfig = CaseMemoryConfig(),
           embed=fake_embed) -> CaseMemory:
    store = CaseStore(MemoryBackend(), now=clock or Clock(at(4, 8, 0)), ladder=config.ladder)
    return CaseMemory(store, embed=embed, judge=judge, config=config)


def saved_case(mem: CaseMemory, **kwargs: Any):
    from home_guard_project.box.case_memory import save_interview

    result = save_interview(mem.store, interview_case(**kwargs), OWNER, embed=mem.embed)
    assert result.kind == "saved", result
    return result.case


def confirm_times(mem: CaseMemory, case_id: str, n: int) -> None:
    for i in range(n):
        mem.store.confirm(case_id, OWNER, f"confirm-{i}")
