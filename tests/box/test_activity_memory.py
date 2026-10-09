"""activity_memory (2026-10-09): what an ACTION means at a camera, from the owner's words.

13:54 the owner explained the main entrance's lying-down as electricians installing LEDs on the stairs; 14:03, 14:41
and 15:44 the same lying-down still went out red. Now a fact maps the actions to the cause: a suspicious naming
them is kept quiet, a red gets a second look with the owner's words (consistent: lowered), and a red that names a
weapon, violence, a break-in, harm or a car door is never lowered.
"""

import datetime as dt
import os
import tempfile
import time
import unittest
from unittest import mock

from home_guard_project.box import activity_memory as am
from home_guard_project.box import inference

CH6, CH3 = "ameer_week_0_1_ch6", "ameer_week_0_1_ch3"
# The real reds of 2026-10-09 (why + summary).
RED_1403 = "אדם שוכב על הקרקע A person lies on the ground while two others stand nearby. One person appears to be wearing a hat."
RED_1441 = ("התנהגות חשודה - אדם עומד ליד רכב ואדם אחר שוכב על הקרקע A man in dark clothing and a cap stands next to a "
            "white car while another person lies on the ground. The standing man appears to be interacting with the "
            "person on the ground.")
RED_1544 = "אדם שוכב על הקרקע ואדם אחר עובר עליו A man lies on the ground while another man walks past him carrying a bag and approaches a car."
RED_1438 = ("התקרבות לרכב, פתיחת דלת וניסיון לקחת חפצים A person wearing dark clothing and a hood approaches a white car, "
            "opens the door, and appears to be taking items from inside.")
RED_1325 = ("התנהגות חריגה: אדם נמצא על הקרקע ואדם אחר מחזיק מוט מתכתי ארוך לידו. A man lies on the ground while another "
            "person in dark clothing and a cap crawls nearby, holding a long metal bar.")


def at(hour, minute=0, day=9):
    return dt.datetime(2026, 10, day, hour, minute).timestamp()


class Backend:
    def __init__(self, answer=None, error=None, delay=0.0):
        self.answer, self.error, self.delay, self.calls = answer, error, delay, []

    def verify(self, frames, question, language="English", timeout=15.0):
        self.calls.append(question)
        if self.delay:
            time.sleep(self.delay)
        if self.error:
            raise self.error
        return self.answer


class Base(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.book = am.ActivityBook(os.path.join(self.dir, "activity.json"))

    def electricians(self, **kw):
        args = dict(cameras=[CH6], actions=["lying", "crawling", "bending"], cause="החשמלאים שמתקינים לדים במדרגות",
                    until=at(18, 0, day=15), now=at(13, 54), cause_en="electricians installing LED lights on the stairs",
                    place="stairs", place_words="במדרגות", daily_from="07:00", daily_to="18:00", by="owner")
        args.update(kw)
        return self.book.add(**args)


class ActionsTests(Base):
    def test_actions_of_the_real_reds(self):
        self.assertEqual(am.actions_in(RED_1325), ["lying", "crawling", "holding_tool"])
        for red in (RED_1403, RED_1441, RED_1544):
            self.assertIn("lying", am.actions_in(red))
        self.assertIn("car_door", am.actions_in(RED_1438))
        self.assertEqual(am.actions_in("אני מדבר על שני האנשים שאחד מהם התכופף זה רגיל"), ["bending"])
        self.assertEqual(am.places_in("שני אנשים שמרכיבים את הלדים במדרגות"), ["stairs"])

    def test_join_and_short_words(self):
        self.assertEqual(am.actions_text(["lying", "bending"], "he"), "השכיבה והכיפוף")
        self.assertEqual(am.actions_text(["lying", "crawling", "bending"], "he"), "השכיבה, הזחילה והכיפוף")
        self.assertEqual(am.actions_text(["lying", "bending"], "he", short=True), "שכיבה/כיפוף")


class BlockTests(Base):
    def test_the_lying_reds_may_be_asked(self):
        for red in (RED_1403, RED_1441, RED_1544, RED_1325):
            self.assertEqual(am.red_blocked(red), "", red)

    def test_weapon_violence_breakin_harm_and_car_door_are_never_lowered(self):
        cases = {
            "A man holds a knife over a person lying on the ground": "weapon",
            "A man hits a man lying on the ground": "violence",
            "אדם תוקף אדם ששוכב על הקרקע": "violence",
            "A man stabs a person lying on the ground": "harm",
            "a person lying on the ground, not moving": "harm",
            "אדם שוכב על הקרקע ללא תנועה": "harm",
            "A person lies on the ground and appears injured, bleeding": "harm",
            "A man breaks into the house while another lies on the ground": "clear",
            "A man is lying on the ground next to the door, prying the lock": "break-in",
            "אדם שוכב ליד החלון ופורץ אותו": "break-in",
            RED_1438: "break-in",
            "A man opens the car door and bends inside": "car door",
            "אדם פותח את דלת הרכב ומתכופף פנימה": "car door",
        }
        for text, why in cases.items():
            self.assertEqual(am.red_blocked(text), why, text)


class MatchTests(Base):
    def test_inside_the_window_at_the_camera(self):
        fact = self.electricians()
        self.assertEqual(self.book.match(CH6, at(14, 3), RED_1403).id, fact.id)
        self.assertEqual(self.book.match(CH6, at(15, 44), RED_1544, for_red=True).id, fact.id)
        self.assertEqual(self.book.match(CH6, at(9, 0, day=12), RED_1403).id, fact.id)      # a later work day

    def test_outside_the_window_another_camera_or_action(self):
        self.electricians()
        self.assertIsNone(self.book.match(CH6, at(19, 30), RED_1403))            # after the day's hours
        self.assertIsNone(self.book.match(CH6, at(9, 0, day=16), RED_1403))      # after the end
        self.assertIsNone(self.book.match(CH3, at(14, 3), RED_1403))              # the pergola
        self.assertIsNone(self.book.match(CH6, at(14, 3), "A man climbs the fence"))

    def test_another_way_in_is_not_covered(self):
        self.electricians()
        self.assertIsNone(self.book.match(CH6, at(14, 3), "A man lies down at the front door"))
        self.assertIsNotNone(self.book.match(CH6, at(14, 3), "A man lies down on the stairs"))

    def test_the_car_door_never_matches_a_red(self):
        self.electricians(actions=["car_door", "bending"])
        self.assertIsNone(self.book.match(CH6, at(14, 38), "A man opens the car door", for_red=True))
        self.assertIsNotNone(self.book.match(CH6, at(14, 38), "A man opens the car door"))

    def test_cancel_and_update(self):
        fact = self.electricians(actions=["lying"])
        self.assertIsNone(self.book.match(CH6, at(14, 0), "a man bends down"))
        self.book.update(fact.id, at(13, 55), actions=fact.actions + ["bending"])
        self.assertIsNotNone(self.book.match(CH6, at(14, 0), "a man bends down"))
        reread = am.ActivityBook(self.book.path)
        self.assertEqual(reread.get(fact.id).actions, ["lying", "bending"])
        self.assertTrue(self.book.cancel(fact.id, at(14, 0)))
        self.assertIsNone(self.book.match(CH6, at(14, 3), RED_1403))
        self.assertIsNone(am.ActivityBook(self.book.path).match(CH6, at(14, 3), RED_1403))

    def test_a_fact_needs_actions_cause_and_a_future_end(self):
        with self.assertRaises(ValueError):
            self.book.add([CH6], ["dancing"], "x", at(18), at(13))
        with self.assertRaises(ValueError):
            self.book.add([CH6], ["lying"], " ", at(18), at(13))
        with self.assertRaises(ValueError):
            self.book.add([CH6], ["lying"], "x", at(12), at(13))

    def test_a_damaged_file_is_an_empty_book(self):
        with open(self.book.path, "w", encoding="utf-8") as f:
            f.write("{not json")
        self.assertEqual(am.ActivityBook(self.book.path).live(at(13)), [])


class RedLookTests(Base):
    def test_consistent_lowers_with_the_owners_words_in_the_question(self):
        self.electricians()
        backend = Backend({"confirmed": True, "what_it_is": "worker lying on the stairs fixing lights"})
        look = am.red_look(backend, ["f"], CH6, at(14, 3), RED_1403, "he", activities=self.book)
        self.assertTrue(look["lowered"])
        self.assertIn("electricians installing LED lights", backend.calls[0])
        self.assertIn("Answer false if anyone looks hurt", backend.calls[0])

    def test_not_consistent_unsure_or_no_answer_keeps_the_red(self):
        self.electricians()
        for backend in (Backend({"confirmed": False, "what_it_is": "a person on the ground"}), Backend(None),
                        Backend(error=RuntimeError("429")), Backend({"confirmed": "yes"})):
            look = am.red_look(backend, ["f"], CH6, at(14, 3), RED_1403, "he", activities=self.book)
            self.assertFalse(look["lowered"])

    def test_a_late_answer_keeps_the_red(self):
        self.electricians()
        look = am.red_look(Backend({"confirmed": True}, delay=0.5), ["f"], CH6, at(14, 3), RED_1403, "he",
                           timeout=0.05, activities=self.book)
        self.assertFalse(look["lowered"])
        self.assertIn("no answer", look["reason"])

    def test_a_yes_that_names_harm_keeps_the_red(self):
        self.electricians()
        backend = Backend({"confirmed": True, "what_it_is": "a worker who fell and is bleeding"})
        self.assertFalse(am.red_look(backend, ["f"], CH6, at(14, 3), RED_1403, "he", activities=self.book)["lowered"])

    def test_weapon_or_violence_red_is_never_asked(self):
        self.electricians()
        for red in ("A man hits a man lying on the ground", "A man holds a knife; another lies on the ground"):
            backend = Backend({"confirmed": True})
            look = am.red_look(backend, ["f"], CH6, at(14, 3), red, "he", activities=self.book)
            self.assertFalse(look["lowered"])
            self.assertEqual(backend.calls, [])

    def test_the_1438_car_door_red_stays_red(self):
        self.electricians(actions=["lying", "bending", "car_door"])
        backend = Backend({"confirmed": True})
        self.assertIsNone(am.red_look(backend, ["f"], CH6, at(14, 38), RED_1438, "he", activities=self.book))
        self.assertEqual(backend.calls, [])

    def test_nothing_explained_no_look(self):
        backend = Backend({"confirmed": True})
        self.assertIsNone(am.red_look(backend, ["f"], CH6, at(14, 3), RED_1403, "he", activities=self.book))
        self.assertEqual(backend.calls, [])


class InferenceHookTests(Base):
    def test_explained_red_lowers_and_records(self):
        self.electricians()
        decision = {"label": "escalation"}
        with mock.patch.object(am, "_BOOK", self.book):
            label, cmd = inference._explained_red(Backend({"confirmed": True, "what_it_is": "work"}), ["f"], CH6,
                                                  at(14, 3), RED_1403, "he", decision, "escalation", "[call_owner]")
        self.assertEqual((label, cmd), ("suspicious", inference.LABEL_COMMANDS["suspicious"]))
        self.assertTrue(decision["activity_look"]["lowered"])
        self.assertEqual(decision["final_label"], "suspicious")

    def test_explained_red_keeps_on_no(self):
        self.electricians()
        decision = {}
        with mock.patch.object(am, "_BOOK", self.book):
            label, _ = inference._explained_red(Backend({"confirmed": False}), ["f"], CH6, at(14, 3), RED_1403, "he",
                                                decision, "escalation", "[call_owner]")
        self.assertEqual(label, "escalation")

    def test_explained_suspicious_is_kept_not_sent(self):
        from home_guard_project.box.events import Decision

        fact = self.electricians()
        decision = {}
        with mock.patch.object(am, "_BOOK", self.book):
            out = inference._explained_suspicious(Decision(True, "s1", "suspicious"), CH6, at(14, 3), RED_1403,
                                                  decision)
            other = inference._explained_suspicious(Decision(True, "s1", "suspicious"), CH3, at(14, 3), RED_1403, {})
        self.assertFalse(out.notify)
        self.assertEqual(out.reason, "owner explained: " + fact.cause)
        self.assertEqual(decision["activity_fact"], fact.id)
        self.assertTrue(other.notify)


if __name__ == "__main__":
    unittest.main()


import test_guard_events as ge  # noqa: E402


class WorkerOrderTest(ge.GuardCase):
    """In the guard loop the context look runs FIRST: on the real worker clips the plain person_down look answers
    "probably hurt" (2026-10-09 afternoon), so a consistent context answer must win and the plain look is skipped."""

    def setUp(self):
        super().setUp()
        self.activities = am.ActivityBook(os.path.join(self.dir, "activity.json"))
        day = dt.datetime.fromtimestamp(ge.T0)
        self.activities.add([ge.DOOR], ["lying", "bending"], "החשמלאים שמתקינים לדים", (day.replace(hour=18)).timestamp(),
                            ge.T0 - 600, cause_en="electricians installing LED lights", place="stairs",
                            daily_from="07:00", daily_to="18:00")
        patch = mock.patch.object(am, "_BOOK", self.activities)
        patch.start()
        self.addCleanup(patch.stop)

    def test_consistent_context_wins_and_the_red_is_kept_quiet(self):
        asked = []
        backend = ge.Backend(ge.SecondLookTest.DOWN, verify=lambda f, q, **k: asked.append(q) or {
            "confirmed": True, "what_it_is": "worker lying on the stairs fitting lights", "evidence_frame": 2})
        job = self.work(backend, ge.T0, camera=ge.DOOR)
        self.assertEqual(len(asked), 1)                                    # the context look only
        self.assertIn("electricians installing LED lights", asked[0])
        self.assertEqual(job.alert["label"], "suspicious")
        self.assertTrue(job.alert["activity_look"]["lowered"])
        self.assertNotIn("second_look", job.alert)
        self.assertEqual(self.assistant.sent, [])                          # kept, not sent
        self.assertIn("owner explained", job.alert["not_sent_reason"])
        self.assertEqual(self.assistant.reminders, [])

    def test_not_consistent_falls_back_to_the_plain_look(self):
        answers = iter([{"confirmed": False, "what_it_is": "a man on the ground", "evidence_frame": 1},
                        {"confirmed": True, "what_it_is": "a man collapsed", "evidence_frame": 1}])
        job = self.work(ge.Backend(ge.SecondLookTest.DOWN, verify=lambda f, q, **k: next(answers)), ge.T0,
                        camera=ge.DOOR)
        self.assertEqual(job.alert["label"], "escalation")
        self.assertFalse(job.alert["activity_look"]["lowered"])
        self.assertIn("second_look", job.alert)
        self.assertTrue(self.assistant.sent[0]["text"].startswith("🔴"))

    def test_another_camera_is_untouched(self):
        verify = mock.Mock(return_value={"confirmed": False, "what_it_is": "a worker laying pavers", "evidence_frame": 1})
        job = self.work(ge.Backend(ge.SecondLookTest.DOWN, verify=verify), ge.T0, camera=ge.CAM)
        self.assertNotIn("activity_look", job.alert)
