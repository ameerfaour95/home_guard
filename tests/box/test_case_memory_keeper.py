from __future__ import annotations

import unittest

from cm_helpers import OWNER, Clock, at, event, fake_embed, interview_case, memory, saved_case, tracker

from home_guard_project.box.case_memory import keeper
from home_guard_project.box.case_memory.models import INVALID, PAUSED


class SaveTest(unittest.TestCase):
    def setUp(self) -> None:
        self.mem = memory()
        self.store = self.mem.store

    def test_saved_with_embedded_example(self) -> None:
        result = keeper.save_interview(self.store, interview_case(), OWNER, embed=fake_embed)
        self.assertEqual(result.kind, "saved")
        self.assertEqual(len(result.case.examples[0].embedding), 64)

    def test_same_explanation_again_reinforces(self) -> None:
        first = saved_case(self.mem)
        again = keeper.save_interview(self.store, interview_case(ev=event(ts=at(5, 7, 40), event_id="e2")), OWNER)
        self.assertEqual((again.kind, again.target.id), ("reinforced", first.id))
        self.assertEqual(again.case.confirmations, 1)
        self.assertEqual(len(self.store.cases()), 1)

    def test_overlapping_routine_proposes_a_merge(self) -> None:
        first = saved_case(self.mem)
        draft = interview_case(ev=event(ts=at(5, 8, 20), event_id="late"), when="when:every_day")
        result = keeper.save_interview(self.store, draft, OWNER)
        self.assertEqual(result.kind, "merge_proposed")
        self.assertEqual(result.merged_scope.hours, ("07:10", "08:50"))
        self.assertIn("לאחד לזיכרון אחד", result.text_he)
        self.assertEqual(len(self.store.cases()), 1, "nothing saved before the owner answers")
        merged = keeper.confirm_merge(self.store, result, OWNER, combine=True)
        self.assertEqual(merged.id, first.id)
        self.assertEqual(len(merged.examples), 2)
        self.assertEqual(len(self.store.cases()), 1)
        self.assertEqual(len(self.store.cases(include_invalid=True)), 2, "the merged source stays as history")

    def test_keep_separate(self) -> None:
        saved_case(self.mem)
        draft = interview_case(ev=event(ts=at(5, 8, 20), event_id="late"), when="when:every_day")
        result = keeper.save_interview(self.store, draft, OWNER)
        keeper.confirm_merge(self.store, result, OWNER, combine=False)
        self.assertEqual(len(self.store.cases()), 2)

    def test_different_path_is_a_new_case(self) -> None:
        saved_case(self.mem)
        draft = interview_case(ev=event(event_id="door", trk=tracker(path=["gate", "entrance"])))
        self.assertEqual(keeper.save_interview(self.store, draft, OWNER).kind, "saved")

    def test_expecting_and_cancelled(self) -> None:
        self.assertEqual(keeper.save_interview(self.store, interview_case(when="when:today"), OWNER).kind,
                         "expecting")
        self.assertEqual(len(self.store.expecting_for("gate", at(4, 9))), 1)
        cancelled = interview_case()
        cancelled = type(cancelled)("cancelled")
        self.assertEqual(keeper.save_interview(self.store, cancelled, OWNER).kind, "cancelled")


class CorrectTest(unittest.TestCase):
    def setUp(self) -> None:
        self.mem = memory()
        self.store = self.mem.store
        self.case = saved_case(self.mem)

    def test_narrow_now_widen_later(self) -> None:
        kind, case = keeper.correct(self.store, self.case.id, OWNER, weekdays=[6, 0])
        self.assertEqual((kind, case.scope.weekdays), ("narrowed", (6, 0)))
        kind, proposal = keeper.correct(self.store, self.case.id, OWNER, weekdays=[6, 0, 5])
        self.assertEqual(kind, "proposed")
        self.assertIn("weekdays: 6,0 -> 6,0,5", keeper.describe_diff(proposal["diff"]))
        self.assertEqual(keeper.correct(self.store, self.case.id, OWNER, weekdays=[6, 0])[0], "unchanged")

    def test_fine_too_proposes_a_scope_that_covers_the_event(self) -> None:
        sig = event(ts=at(5, 8, 40), trk=tracker(time_in_view_s=30)).signature
        proposal = keeper.widen_for_event(self.store, self.case.id, sig, OWNER)
        self.assertEqual(set(proposal["diff"]), {"hours", "max_dwell_s"})
        self.assertEqual(proposal["diff"]["hours"][1], ["07:10", "08:41"])
        self.assertIsNone(keeper.widen_for_event(self.store, self.case.id, event(obs={"people": 2}).signature, OWNER))

    def test_window_math(self) -> None:
        self.assertEqual(keeper.union_window(("07:10", "08:10"), ("08:00", "09:00")), ("07:10", "09:00"))
        self.assertEqual(keeper.union_window(("23:00", "23:30"), ("00:10", "00:40")), ("23:00", "00:40"))
        self.assertEqual(keeper.window_gap(("07:00", "08:00"), ("09:30", "10:00")), 90)
        self.assertEqual(keeper.window_gap(("07:00", "08:00"), ("07:30", "10:00")), 0)


class ForgetAndListTest(unittest.TestCase):
    def setUp(self) -> None:
        self.clock = Clock(at(4, 8))
        self.mem = memory(clock=self.clock)
        self.store = self.mem.store
        self.case = saved_case(self.mem)

    def test_thirty_days_unseen_asks_once(self) -> None:
        self.assertEqual(keeper.due_reviews(self.store, at(20, 8)), [])
        due = keeper.due_reviews(self.store, at(4, 8, month=11) + 86400)
        self.assertEqual([d["case_id"] for d in due], [self.case.id])
        self.assertIn("עדיין רלוונטי", due[0]["text_he"])
        self.assertEqual(self.store.ask_review(self.case.id).status, PAUSED)
        self.assertEqual(keeper.due_reviews(self.store, at(4, 8, month=11) + 86400), [], "asked once")

    def test_a_match_counts_as_seen(self) -> None:
        self.clock.ts = at(30, 8)
        self.store.log_match(self.case.id, "m", "alert", "high", 0.9, True)
        self.assertEqual(keeper.due_reviews(self.store, at(4, 8, month=11) + 86400), [])

    def test_listing_with_delete(self) -> None:
        self.store.add_expecting("gate", "the plumber", OWNER, ts=at(4, 8))
        listing = keeper.remember_listing(self.store, "gate", now=at(4, 9))
        self.assertEqual(listing["text_he"], "מה שאני זוכר על מצלמת gate:")
        self.assertEqual(len(listing["items"]), 2)
        self.assertIn("שכן · ימי חול (א׳–ה׳) · 07:10–08:10 · מהשער לרחוב", listing["items"][0]["text_he"])
        self.assertEqual(listing["items"][0]["button"]["action"], "delete")
        keeper.on_button(self.store, "delete", self.case.id, OWNER)
        keeper.on_button(self.store, "delete", "", OWNER, expecting_id=listing["items"][1]["expecting_id"])
        self.assertEqual(keeper.remember_listing(self.store, "gate", now=at(4, 9))["items"], [])
        self.assertEqual(self.store.cases(include_invalid=True)[0].status, INVALID)

    def test_buttons(self) -> None:
        keeper.on_button(self.store, "confirm", self.case.id, OWNER, "e1")
        keeper.on_button(self.store, "not_them", self.case.id, OWNER, "e2")
        case = keeper.on_button(self.store, "keep_alerting", self.case.id, OWNER)
        self.assertEqual((case.confirmations, case.contradictions, case.effect), (1, 1, "alert"))
        with self.assertRaises(ValueError):
            keeper.on_button(self.store, "explode", self.case.id, OWNER)


if __name__ == "__main__":
    unittest.main()
