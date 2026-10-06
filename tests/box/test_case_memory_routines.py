from __future__ import annotations

import unittest

from cm_helpers import OWNER, at, interview_case, memory, observation, saved_case, tracker

from home_guard_project.box.case_memory import routines
from home_guard_project.box.case_memory.models import SHADOW

NOW = at(14, 23)
DAYS = (5, 6, 7, 8, 11, 12)       # six weekdays inside the last 14 days


def rec(day: int, hour: int = 7, minute: int = 40, **kw) -> dict:
    out = {"event_id": f"gate_{day}_{hour}{minute}", "camera": "gate", "ts": at(day, hour, minute),
           "final_label": "normal", "alert_command": "[send_message]", "observation": observation(),
           "tracker": tracker(), "situation": {"phase": "day", "house_state": "home_awake"}}
    out.update(kw)
    return out


class ProposeTest(unittest.TestCase):
    def test_six_of_fourteen_days_within_half_an_hour_is_proposed(self) -> None:
        records = [rec(d, 7, 35 + i * 3) for i, d in enumerate(DAYS)] + [rec(9, 13, 0)]
        (p,) = routines.propose(records, now=NOW)
        self.assertEqual(p["camera"], "gate")
        self.assertEqual(p["path"], ["gate", "street"])
        self.assertEqual(p["days_seen"], 6)
        self.assertEqual(p["hours"], ["07:10", "08:20"])
        self.assertEqual(len(p["examples"]), 3)
        self.assertIn("מהשער לרחוב ב-6 מתוך 14 הימים האחרונים", p["text_he"])
        self.assertTrue(p["text_he"].endswith("זו שגרה?"))

    def test_five_days_or_spread_out_is_not(self) -> None:
        self.assertEqual(routines.propose([rec(d) for d in DAYS[:5]], now=NOW), [])
        spread = [rec(d, 7, 0) for d in DAYS[:3]] + [rec(d, 8, 0) for d in DAYS[3:]]
        self.assertEqual(routines.propose(spread, now=NOW), [])

    def test_older_than_fourteen_days_does_not_count(self) -> None:
        records = [rec(d) for d in DAYS[:5]] + [rec(28, month=9)]
        self.assertEqual(routines.propose(records, now=NOW), [])

    def test_night_suspicious_and_flagged_events_are_left_out(self) -> None:
        bad = [
            [rec(d, situation={"phase": "late_night", "house_state": "home_asleep"}) for d in DAYS],
            [rec(d, final_label="suspicious") for d in DAYS],
            [rec(d, observation=observation(flags=["crouching"])) for d in DAYS],
            [rec(d, observation=observation(category="S3")) for d in DAYS],
            [rec(d, alert_command="[call_owner]") for d in DAYS],
        ]
        for records in bad:
            self.assertEqual(routines.propose(records, now=NOW), [], records[0])

    def test_paths_are_separate_groups(self) -> None:
        mixed = [rec(d) for d in DAYS[:3]] + [rec(d, tracker=tracker(path=["gate", "entrance"])) for d in DAYS[3:]]
        self.assertEqual(routines.propose(mixed, now=NOW), [])

    def test_covered_or_answered_is_not_proposed_again(self) -> None:
        mem = memory()
        saved_case(mem)
        records = [rec(d) for d in DAYS]
        self.assertEqual(routines.propose(records, mem.store, now=NOW), [], "a live case covers it")
        mem2 = memory()
        (p,) = routines.propose(records, mem2.store, now=NOW)
        routines.answer(mem2.store, p, False, OWNER)
        self.assertEqual(routines.propose(records, mem2.store, now=NOW), [])
        shifted = [rec(d, 7, 55) for d in DAYS]
        self.assertEqual(routines.propose(shifted, mem2.store, now=NOW), [], "a near time is the same question")

    def test_yes_makes_a_shadow_case(self) -> None:
        mem = memory()
        (p,) = routines.propose([rec(d) for d in DAYS], mem.store, now=NOW)
        case = routines.answer(mem.store, p, True, OWNER)
        self.assertEqual((case.status, case.confirmations), (SHADOW, 0))
        self.assertEqual(case.scope.path, ("gate", "street"))
        self.assertEqual(case.scope.weekdays, (0, 1, 2, 3, 6), "seen Sun-Thu only: the work week")
        self.assertEqual(case.scope.max_dwell_s, 19.0)
        self.assertEqual(case.source["origin"], "routine")
        self.assertEqual(mem.store.routine_answers()[p["key"]]["case_id"], case.id)
        with self.assertRaises(ValueError):
            routines.answer(mem.store, p, True, "")

    def test_records_with_a_stored_signature(self) -> None:
        sig = interview_case().case.examples[0].signature
        self.assertEqual(routines.record_signature({"signature": sig.to_dict()}), sig)


if __name__ == "__main__":
    unittest.main()
