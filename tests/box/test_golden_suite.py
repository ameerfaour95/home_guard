"""The golden conversation suite (tests/golden, home_guard_project/box/eval_chat.py), the fast part with no model:

- every case is well formed and its ideal reply passes its own deterministic checks;
- the checks catch the replies the owner complained about (2026-10-07 .. 2026-10-10);
- every case the box answers in code alone (tag answers after the 🏷️ button, acks) runs end to end through the real
  inbox tag path / brain with a model that must not be called, and passes its deterministic checks.

The full suite with the LLM judge: ``python -m home_guard_project.box.eval_chat run --suite tests/golden --runs 2``."""

from __future__ import annotations

import os
import unittest

from home_guard_project.box import eval_chat as ec
from home_guard_project.box.brain.models import ModelMessage

SUITE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "golden")

# Cases the current brain fails in code (no model involved), kept visible here rather than hidden: remove an id once
# the brain passes it.
KNOWN_OFFLINE_FAILURES = {
    "g09_10",   # 09:46 "you don't read the history?": answered in code with the memory list, no entrance mark
}


class NoModel:
    """A chat model that records that it was asked; the case then needs the full (model) run."""

    def __init__(self) -> None:
        self.calls = 0

    def chat(self, *args, **kwargs):
        self.calls += 1
        return ModelMessage(content=None, error="offline test: no model")


def _case(cases, cid):
    return next(c for c in cases if c["id"] == cid)


class GoldenSuiteShape(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cases = ec.load_suite(SUITE)

    def test_enough_cases_from_the_owners_chats(self):
        self.assertGreaterEqual(len(self.cases), 60)
        self.assertLessEqual(len(self.cases), 120)
        intents = {c["expect"]["intent"] for c in self.cases}
        for wanted in ("place_fact", "person_mark", "activity_explain", "tag_only", "question_live",
                       "question_history", "complaint", "ack", "preference", "command"):
            self.assertIn(wanted, intents)

    def test_every_case_is_well_formed_and_its_ideal_passes(self):
        self.assertEqual(ec.check_suite(self.cases), [])

    def test_every_case_names_its_source_turn(self):
        for c in self.cases:
            self.assertRegex(c["source"], r"^2026-10-\d\d \d\d:\d\d(:\d\d)?$", c["id"])

    def test_the_case_sees_only_what_came_before_it(self):
        c = _case(self.cases, "g09_06")             # 09:34:34: only the 23:59 mark exists yet
        view = ec.visible(c)
        self.assertEqual([m["until"] for m in view["memory"]["marks"]], ["23:59"])
        self.assertTrue(all(ec.ts_of(c, h["time"]) < ec.ts_of(c, c["time"]) for h in view["history"]))


class DeterministicChecks(unittest.TestCase):
    """The checks fail the real replies the owner complained about."""

    @classmethod
    def setUpClass(cls):
        cls.cases = ec.load_suite(SUITE)

    def _failed(self, cid, text, writes=(), tools=(), buttons=()):
        out = {"text": text, "writes": list(writes), "tools": list(tools), "buttons": list(buttons), "photos": [],
               "videos": [], "photo_cameras": [], "marks_after": []}
        return {c["name"] for c in ec.deterministic(_case(self.cases, cid), out) if not c["ok"]}

    def test_neighbours_house_until_when(self):
        failed = self._failed("g10_02", "עד מתי לזכור את הבית של השכן?")
        self.assertTrue({"must_not", "max_questions", "writes"} <= failed)

    def test_memory_dump_and_what_to_fix(self):
        failed = self._failed("g10_03", "שמור אצלי עכשיו: העובדים של הפרגולה, כל יום 07:00–18:00, עד יום ה׳ 15.10. "
                                         "מה לתקן?")
        self.assertTrue({"no_memory_dump", "no_what_to_fix", "must_not"} <= failed)

    def test_english_failure_text(self):
        self.assertIn("hebrew", self._failed("g10_01", "I could not work on that right now, but your message was saved."))

    def test_empty_empathy(self):
        self.assertIn("no_empty_empathy", self._failed("g09_07", "אני מבין אותך."))

    def test_closing_offer_and_jargon(self):
        failed = self._failed("g07_14", "רשמתי את זה כהתרעה צפויה. אם יש משהו נוסף שתרצה, אני כאן!")
        self.assertTrue({"no_closing_offer", "no_expected_alert_jargon"} <= failed)

    def test_internal_handles_leak(self):
        failed = self._failed("g09_27", "במצלמת הפרגולה יש אדם אחד. [handles: E30=photo פרגולה | receipts: R1 "
                                        "check_camera פרגולה done]")
        self.assertIn("no_internal_handles", failed)

    def test_live_question_needs_a_look(self):
        self.assertIn("must_do_look", self._failed("g06_01", "בכל המצלמות אין אנשים, האזור נראה ריק ונקי."))
        self.assertNotIn("must_do_look", self._failed("g06_01", "אין אף אחד בחוץ.", tools=["look_around"]))

    def test_unbacked_certainty(self):
        self.assertIn("must_not", self._failed("g09_17", "כן, אני בטוח. אלה אותם עובדים מהבוקר."))

    def test_a_tag_is_not_memory(self):
        failed = self._failed("g09_01", "🏷️ נשמר.", writes=[{"type": "tag"}, {"type": "person_mark", "camera": "x"}])
        self.assertIn("forbid_writes", failed)

    def test_invented_end_time_stays_live(self):
        out = {"text": "תיקנתי: עד 18:00.", "writes": [{"type": "person_mark", "camera": "ameer_v2_ch3",
                                                          "until": ec.ts_of({"date": "2026-10-09"}, "18:00")}],
               "tools": [], "buttons": [], "marks_after": [{"until": ec.ts_of({"date": "2026-10-09"}, "23:59")}]}
        failed = {c["name"] for c in ec.deterministic(_case(self.cases, "g09_06"), out) if not c["ok"]}
        self.assertEqual(failed, {"no_live_mark_until"})

    def test_an_echo_before_the_question_is_one_question(self):
        self.assertEqual(ec.count_questions("אה, הם של הפרגולה? עד איזו שעה הם עובדים?"), 1)
        self.assertEqual(ec.count_questions("עד מתי לזכור את הבית של השכן? ובכל הבית או רק בפרגולה תרצה?"), 2)
        self.assertEqual(ec.count_questions("סגור.", buttons=("17:00", "18:00")), 1)
        self.assertEqual(ec.count_questions("הנה.", buttons=("📹 שלח את הסרטון",)), 0)


class OfflineCases(unittest.TestCase):
    """Cases answered in code alone, through the real tag path and brain, with no model."""

    def test_offline_cases_pass_their_checks(self):
        cases = ec.load_suite(SUITE)
        ran = 0
        for case in cases:
            model = NoModel()
            result = ec.run_case(case, model, None)
            if model.calls:
                continue                      # needs a model: covered by the full run, not here
            ran += 1
            failed = [c for c in ec.deterministic(case, result) if not c["ok"]]
            with self.subTest(case=case["id"]):
                if case["id"] in KNOWN_OFFLINE_FAILURES:
                    self.assertTrue(failed, f"{case['id']} passes now: remove it from KNOWN_OFFLINE_FAILURES")
                else:
                    self.assertEqual(failed, [], f"{case['id']}: {result.get('text')!r} {result.get('error')}")
        self.assertGreaterEqual(ran, 15)      # the tag answers at least


if __name__ == "__main__":
    unittest.main()
