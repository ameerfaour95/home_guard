from __future__ import annotations

import json
import unittest

from cm_helpers import at, event, observation

from home_guard_project.box.case_memory import interviewer as iv
from home_guard_project.box.case_memory.models import ALL_DAYS, WORKWEEK


def codes(step) -> list:
    return [o.code for o in step.options]


class InterviewFlowTest(unittest.TestCase):
    def test_the_plan_page_conversation(self) -> None:
        ev = event()
        interview, step = iv.start(ev.event_id, ev.signature)
        self.assertEqual(step.slot, "who")
        self.assertEqual(step.text_he, 'סימנת "רגיל". מי זה?')
        self.assertEqual(codes(step)[:5], ["who:neighbour", "who:family", "who:worker", "who:courier", "who:other"])
        self.assertEqual(codes(step)[-2:], ["skip", "dont_know"])
        self.assertTrue(step.free_text)

        step = interview.answer("who:neighbour")
        self.assertEqual(step.slot, "when")
        self.assertIn("בסביבות 07:40, מהשער לרחוב", step.text_he)
        self.assertIn("או רק היום", step.text_he)
        self.assertEqual(codes(step)[:4], ["when:workweek", "when:every_day", "when:today", "when:other_hours"])

        step = interview.answer("when:workweek")
        self.assertEqual(step.slot, "recognise")
        self.assertEqual(step.text_he, "ראיתי backpack, dark jacket. ככה מזהים אותו?")

        card = interview.answer("rec:varies")
        self.assertIsInstance(card, iv.SummaryCard)
        self.assertEqual(interview.questions_asked, 3)
        self.assertIn("אזכור: שכן. מצלמת gate, ימי חול (א׳–ה׳) 07:10–08:10, מהשער לרחוב, 1 אדם.", card.text_he)
        self.assertIn("בפעמים הבאות לא אתריע, ואראה לך בסיכום היומי.", card.text_he)
        self.assertIn("לא חל על לילה", card.text_he)
        self.assertEqual(codes(card), ["card:save", "card:edit", "card:cancel"])

        outcome = interview.answer("card:save")
        self.assertEqual(outcome.kind, "case")
        case = outcome.case
        self.assertEqual(case.who, "neighbour")
        self.assertEqual(case.scope.weekdays, tuple(sorted(WORKWEEK)))
        self.assertEqual(case.scope.hours, ("07:10", "08:10"))
        self.assertEqual(case.scope.path, ("gate", "street"))
        self.assertEqual(case.scope.max_dwell_s, 19.0)
        self.assertEqual(case.scope.categories, ("N2",))
        self.assertEqual(case.recognise, (), "the clothes change")
        self.assertEqual(case.effect, "digest")
        self.assertEqual(case.examples[0].event_id, "gate_ev")

    def test_never_more_than_three_questions_and_next_time_when_no_clothes(self) -> None:
        ev = event(obs=observation(appearance=[]))
        interview, _ = iv.start(ev.event_id, ev.signature)
        interview.answer("who:family")
        step = interview.answer("when:every_day")
        self.assertEqual(step.slot, "next")
        card = interview.answer("next:quiet")
        self.assertIsInstance(card, iv.SummaryCard)
        self.assertIn("אשלח הודעה שקטה", card.text_he)
        case = interview.answer("card:save").case
        self.assertEqual((case.effect, case.scope.weekdays), ("quiet", ALL_DAYS))

    def test_only_today_is_an_expecting_note(self) -> None:
        ev = event()
        interview, _ = iv.start(ev.event_id, ev.signature)
        interview.answer("who:worker")
        card = interview.answer("when:today")
        self.assertIsInstance(card, iv.SummaryCard)
        self.assertIn("להיום בלבד", card.text_he)
        self.assertEqual(codes(card), ["card:save", "card:cancel"])
        outcome = interview.answer("card:save")
        self.assertEqual((outcome.kind, outcome.case, outcome.camera), ("expecting", None, "gate"))
        self.assertEqual(outcome.expecting, "עובד / ספק")

    def test_skips_give_the_narrow_default(self) -> None:
        ev = event()
        interview, _ = iv.start(ev.event_id, ev.signature)
        for _ in range(3):
            interview.answer("skip")
        case = interview.answer("card:save").case
        self.assertEqual(case.who, "other")
        self.assertEqual(case.scope.weekdays, (ev.signature.weekday,), "only the day it happened")
        self.assertEqual(case.scope.hours, ("07:10", "08:10"))

    def test_other_waits_for_typed_words(self) -> None:
        ev = event()
        interview, _ = iv.start(ev.event_id, ev.signature)
        step = interview.answer("who:other")
        self.assertEqual((step.slot, step.free_text), ("who", True))
        self.assertIn("כתוב", step.text_he)
        step = interview.answer("", text="  the  gardener ")
        self.assertEqual(step.slot, "when")
        interview.answer("when:other_hours")
        interview.answer("rec:yes")
        case = interview.answer("card:save").case
        self.assertEqual(case.title, "the gardener")
        self.assertEqual(case.scope.hours, ("06:10", "09:10"))
        self.assertEqual(case.recognise, ("backpack", "dark jacket"))

    def test_saturday_offers_every_saturday(self) -> None:
        ev = event(ts=at(10, 9, 5))
        interview, _ = iv.start(ev.event_id, ev.signature)
        step = interview.answer("who:family")
        self.assertEqual(codes(step)[0], "when:weekday")
        self.assertIn("שבת", step.options[0].he)
        interview.answer("when:weekday")
        interview.answer("rec:yes")
        self.assertEqual(interview.answer("card:save").case.scope.weekdays, (5,))

    def test_edit_reasks_one_slot_and_returns_to_the_card(self) -> None:
        ev = event()
        interview, _ = iv.start(ev.event_id, ev.signature)
        for code in ("who:neighbour", "when:workweek", "rec:varies"):
            interview.answer(code)
        step = interview.answer("card:edit")
        self.assertEqual(codes(step), ["edit:who", "edit:when", "edit:recognise", "edit:next"])
        step = interview.answer("edit:next")
        self.assertEqual(step.slot, "next")
        card = interview.answer("next:alert")
        self.assertIn("אמשיך להתריע", card.text_he)
        self.assertEqual(interview.answer("card:save").case.effect, "alert")

    def test_cancel(self) -> None:
        ev = event()
        interview, _ = iv.start(ev.event_id, ev.signature)
        for code in ("who:neighbour", "when:workweek", "rec:varies"):
            interview.answer(code)
        self.assertEqual(interview.answer("card:cancel").kind, "cancelled")

    def test_state_round_trips_between_messages(self) -> None:
        ev = event()
        interview, _ = iv.start(ev.event_id, ev.signature)
        interview.answer("who:courier")
        again = iv.Interview.from_dict(json.loads(json.dumps(interview.to_dict(), ensure_ascii=False)))
        self.assertEqual(again.step().slot, "when")
        again.answer("when:workweek")
        again.answer("rec:yes")
        self.assertEqual(again.answer("card:save").case.who, "courier")

    def test_learned_at_night_keeps_night_and_never_learns_away(self) -> None:
        ev = event(ts=at(4, 1, 30), situation={"phase": "late_night", "house_state": "away"})
        interview, _ = iv.start(ev.event_id, ev.signature)
        for code in ("who:family", "when:every_day", "rec:yes"):
            interview.answer(code)
        case = interview.answer("card:save").case
        self.assertTrue(case.scope.night)
        self.assertEqual(case.scope.house_states, ("home_awake",))


if __name__ == "__main__":
    unittest.main()
