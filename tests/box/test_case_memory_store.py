from __future__ import annotations

import json
import os
import tempfile
import unittest
from dataclasses import replace

from cm_helpers import OWNER, Clock, at, event, interview_case, memory, saved_case

from home_guard_project.box.case_memory import (ACTIVE, INVALID, PAUSED, SHADOW, Case, CaseStore, Example,
                                                MemoryBackend, Scope, TrustLadder)
from home_guard_project.box.case_memory.store import JsonlBackend, end_of_day, narrows, window_inside


class ModelsTest(unittest.TestCase):
    def test_case_round_trips_through_json(self) -> None:
        case = interview_case().case
        case = replace(case, id="C7", examples=[replace(case.examples[0], embedding=(0.5, 0.5))])
        back = Case.from_dict(json.loads(json.dumps(case.to_dict())))
        self.assertEqual(back.to_dict(), case.to_dict())
        self.assertEqual(back.scope.path, ("gate", "street"))
        self.assertEqual(back.examples[0].embedding, (0.5, 0.5))


class LadderTest(unittest.TestCase):
    def test_stages_and_step_back(self) -> None:
        ladder = TrustLadder()
        self.assertEqual(ladder.stage(2, 0), "shadow")
        self.assertEqual(ladder.stage(3, 0), "quiet")
        self.assertEqual(ladder.stage(10, 0), "digest")
        self.assertEqual(ladder.stage(10, 1), "quiet", "digest needs no 'not them' ever")
        self.assertEqual(ladder.step_back(12, 0), 3)
        self.assertEqual(ladder.step_back(5, 0), 0)

    def test_level_is_capped_by_the_owner_effect(self) -> None:
        ladder = TrustLadder()
        self.assertEqual(ladder.level(12, 0, "digest"), "digest")
        self.assertEqual(ladder.level(12, 0, "quiet"), "quiet")
        self.assertEqual(ladder.level(12, 0, "alert"), "alert")
        self.assertEqual(ladder.level(1, 0, "digest"), "alert", "shadow still alerts")


class StoreTest(unittest.TestCase):
    def setUp(self) -> None:
        self.mem = memory()
        self.store = self.mem.store
        self.case = saved_case(self.mem)

    def test_new_case_starts_in_shadow_with_no_confirmations(self) -> None:
        self.assertEqual(self.case.id, "C1")
        self.assertEqual(self.case.status, SHADOW)
        self.assertEqual(self.case.confirmations, 0)
        self.assertEqual(self.case.created_by, OWNER)

    def test_only_the_owner_creates_or_confirms(self) -> None:
        with self.assertRaises(ValueError):
            self.store.add(self.case, by="")
        with self.assertRaises(ValueError):
            self.store.confirm(self.case.id, by="")

    def test_three_confirmations_make_it_active(self) -> None:
        for i in range(3):
            got = self.store.confirm(self.case.id, OWNER, f"e{i}")
        self.assertEqual(got.status, ACTIVE)
        self.assertEqual(got.confirmations, 3)

    def test_automatic_matches_never_strengthen(self) -> None:
        for i in range(20):
            self.store.log_match(self.case.id, f"m{i}", "quiet", "high", 0.95, shadow=True)
        got = self.store.get(self.case.id)
        self.assertEqual(got.confirmations, 0)
        self.assertEqual(got.status, SHADOW)
        self.assertIsNotNone(got.last_seen_at)
        self.assertEqual(len(self.store.matches(self.case.id)), 20)

    def test_not_them_steps_back_and_keeps_a_negative(self) -> None:
        for i in range(10):
            self.store.confirm(self.case.id, OWNER, f"e{i}")
        got = self.store.contradict(self.case.id, OWNER, "bad1")
        self.assertEqual(got.streak, 3)
        self.assertEqual(got.status, ACTIVE)
        self.assertEqual(got.negatives, ["bad1"])
        got = self.store.contradict(self.case.id, OWNER, "bad2")
        self.assertEqual(got.status, SHADOW)

    def test_examples_keep_the_founder_and_the_newest(self) -> None:
        base = self.case.examples[0]
        for i in range(8):
            self.store.confirm(self.case.id, OWNER, f"e{i}", replace(base, event_id=f"e{i}"))
        got = self.store.get(self.case.id)
        self.assertEqual([e.event_id for e in got.examples], ["gate_ev", "e4", "e5", "e6", "e7"])

    def test_invalidate_keeps_history_and_restore_brings_it_back(self) -> None:
        self.store.invalidate(self.case.id, "the neighbour moved away", OWNER)
        self.assertEqual(self.store.cases(), [])
        self.assertEqual(self.store.live_cases("gate"), [])
        gone = self.store.cases(include_invalid=True)[0]
        self.assertEqual(gone.status, INVALID)
        self.assertEqual(gone.invalid_reason, "the neighbour moved away")
        self.assertEqual(self.store.restore(self.case.id, OWNER).status, SHADOW)

    def test_narrowing_applies_at_once_widening_waits(self) -> None:
        narrower = replace(self.case.scope, hours=("07:20", "08:00"))
        self.assertEqual(self.store.narrow(self.case.id, narrower, OWNER).scope.hours, ("07:20", "08:00"))
        wider = replace(narrower, weekdays=narrower.weekdays + (5,))
        with self.assertRaises(ValueError):
            self.store.narrow(self.case.id, wider, OWNER)
        proposal = self.store.propose_widen(self.case.id, wider, OWNER)
        self.assertIn("weekdays", proposal["diff"])
        self.assertEqual(self.store.get(self.case.id).scope, narrower, "nothing changes before the owner says yes")
        self.assertEqual(len(self.store.pending_widenings(self.case.id)), 1)
        self.assertEqual(self.store.answer_widen(proposal["wid"], True, OWNER).scope, wider)
        self.assertEqual(self.store.pending_widenings(), [])

    def test_rejected_widening_changes_nothing(self) -> None:
        wider = replace(self.case.scope, hours=("06:00", "09:00"))
        proposal = self.store.propose_widen(self.case.id, wider, OWNER)
        self.store.answer_widen(proposal["wid"], False, OWNER)
        self.assertEqual(self.store.get(self.case.id).scope, self.case.scope)

    def test_review_pauses_until_answered(self) -> None:
        self.assertEqual(self.store.ask_review(self.case.id).status, PAUSED)
        self.assertEqual(self.store.live_cases("gate"), [])
        self.assertEqual(self.store.answer_review(self.case.id, True, OWNER).status, SHADOW)
        self.assertEqual(self.store.answer_review(self.case.id, False, OWNER).status, INVALID)

    def test_merge_moves_examples_and_invalidates_the_source(self) -> None:
        other = self.store.add(interview_case(ev=event(ts=at(5, 8, 30), event_id="later"), when="when:every_day").case,
                               OWNER)
        merged_scope = replace(self.case.scope, hours=("07:10", "09:00"))
        target = self.store.merge(other.id, self.case.id, merged_scope, OWNER)
        self.assertEqual(target.scope.hours, ("07:10", "09:00"))
        self.assertEqual([e.event_id for e in target.examples], ["gate_ev", "later"])
        source = self.store.get(other.id)
        self.assertEqual(source.status, INVALID)
        self.assertEqual(source.merged_into, self.case.id)

    def test_expecting_note_ends_at_midnight(self) -> None:
        clock = Clock(at(4, 9, 0))
        store = CaseStore(MemoryBackend(), now=clock)
        note = store.add_expecting("gate", "the plumber", OWNER)
        self.assertEqual(note.until, end_of_day(at(4, 9, 0)))
        self.assertEqual(len(store.expecting_for("gate", at(4, 23, 59))), 1)
        self.assertEqual(store.expecting_for("gate", at(5, 0, 1)), [])
        self.assertEqual(store.expecting_for("yard", at(4, 10, 0)), [])
        store.cancel_expecting(note.id, OWNER)
        self.assertEqual(store.expecting_for("gate", at(4, 10, 0)), [])


class JsonlTest(unittest.TestCase):
    def test_file_backend_round_trip_and_damaged_lines(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, ".registry", "cases.jsonl")
            store = CaseStore.at(path, now=Clock(at(4, 8)))
            case = store.add(interview_case().case, OWNER)
            with open(path, "a", encoding="utf-8") as f:
                f.write("{not json\n")
                f.write(json.dumps({"event": "confirm", "ts": "x", "id": case.id}) + "\n")
            store.confirm(case.id, OWNER, "e1")
            with self.assertLogs("box.case_memory.store", "WARNING"):
                again = CaseStore.at(path).get(case.id)
            self.assertEqual(again.confirmations, 1)
            self.assertEqual(again.examples[0].signature.path, ("gate", "street"))


class ScopeMathTest(unittest.TestCase):
    def test_window_inside_across_midnight(self) -> None:
        self.assertTrue(window_inside(("23:30", "00:30"), ("23:00", "01:00")))
        self.assertFalse(window_inside(("22:30", "00:30"), ("23:00", "01:00")))
        self.assertTrue(window_inside(("05:00", "06:00"), ("00:00", "00:00")))

    def test_narrows(self) -> None:
        base = Scope("gate", ("07:00", "09:00"), weekdays=(6, 0, 1), path=("gate", "street"), exit_edge="street",
                     max_dwell_s=20.0, categories=("N2",))
        self.assertTrue(narrows(base, replace(base, weekdays=(6,))))
        self.assertTrue(narrows(base, replace(base, max_dwell_s=10.0)))
        self.assertFalse(narrows(base, replace(base, max_dwell_s=None)))
        self.assertFalse(narrows(base, replace(base, path=("gate", "entrance"))))
        self.assertFalse(narrows(base, replace(base, people=2)))
        self.assertFalse(narrows(base, replace(base, night=True)))
        self.assertFalse(narrows(base, replace(base, categories=())))


if __name__ == "__main__":
    unittest.main()
