"""2026-10-09 11:12: neither vision model answered at the pergola while the owner's workers mark (ch3 and ch6, every
day 07:00-18:00) was live, and the owner still got "the AI check did not finish". A failed AI check under a live mark
that covers the detector's head-count is kept, not sent; outside the mark's hours or without a mark the honest
alert goes out as before (box/inference.py _event_decision, box/events.py EventBook.known_covers)."""
import shutil
import tempfile
import unittest
from unittest import mock
from datetime import datetime

import numpy as np

from home_guard_project.box import inference as inf
from home_guard_project.box.events import KNOWN_EXTRA_PEOPLE, EventBook

from test_guard_events import CAM, DOOR, T0, Backend, GuardCase, answer

DAY = datetime(2026, 10, 9, 0, 0).timestamp()
AT_1112 = DAY + 11 * 3600 + 12 * 60
AT_1900 = DAY + 19 * 3600


def mark(book, camera=CAM, people=3, now=DAY + 7 * 3600):
    return book.mark_known(camera, "העובדים בפרגולה", "Ameer", until=DAY + 6 * 86400, now=now, people=people,
                           daily_from="07:00", daily_to="18:00")


class FailedCheckUnderMarkTest(GuardCase):
    def setUp(self):
        super().setUp()
        patcher = mock.patch.object(inf, "AI_FAILED_NOTIFY", True)   # these test the "on" behaviour
        patcher.start()
        self.addCleanup(patcher.stop)

    def run_failed(self, ts, detector_people=2, looks=None):
        job = inf.AlertJob(camera=CAM, stem=f"{CAM}_{int(ts)}_alert", ts=ts, labels=["person"],
                           input_meta={"vlm_input": "crop"})
        job.tracker = {"people_together": detector_people} if detector_people is not None else {}
        job.scene_looks = list(looks or [])
        with self.assertLogs("box.inference", "INFO") as logs:
            inf._worker(Backend(None), {"alert_channel": "telegram"}, {}, inf.AlertSettings(), CAM,
                        [np.zeros((4, 4, 3), np.uint8)] * 4, self.assistant, job)
        self.assertTrue(job.ready.is_set())
        return job, logs.output

    def test_inside_the_marks_hours_it_is_kept_not_sent(self):
        mark(self.book)
        job, logs = self.run_failed(AT_1112, detector_people=2)
        self.assertEqual(self.assistant.sent, [])
        self.assertTrue(any(f"[{CAM}] not sent (AI check failed, but the owner said who is here)" in m
                            for m in logs), logs)
        self.assertIs(job.alert["sent"], False)
        self.assertTrue(job.alert["vlm_failed"])
        self.assertEqual(job.alert["not_sent_reason"], "AI check failed, but the owner said who is here")
        self.assertEqual(job.alert["event"]["known_text"], "העובדים בפרגולה")

    def test_up_to_known_plus_extra_people_is_covered(self):
        mark(self.book, people=3)
        self.run_failed(AT_1112, detector_people=3 + KNOWN_EXTRA_PEOPLE)
        self.assertEqual(self.assistant.sent, [])

    def test_more_people_than_the_mark_covers_is_the_honest_alert(self):
        mark(self.book, people=3)
        job, _ = self.run_failed(AT_1112, detector_people=3 + KNOWN_EXTRA_PEOPLE + 1)
        self.assertEqual(len(self.assistant.sent), 1)
        self.assertIn("הבדיקה של ה-AI לא הספיקה", self.assistant.sent[0]["text"])
        self.assertTrue(job.alert["vlm_failed"])

    def test_the_crops_own_looks_count_too(self):
        mark(self.book, people=1)
        person = (0, 0.9, 0.1, 0.1, 0.2, 0.4)
        self.run_failed(AT_1112, detector_people=None, looks=[(AT_1112, [person] * 4), (AT_1112 + 1, [person])])
        self.assertEqual(len(self.assistant.sent), 1)                # 4 > 1 + 2

    def test_a_mark_that_counted_nobody_covers_any_count(self):
        mark(self.book, people=0)
        self.run_failed(AT_1112, detector_people=9)
        self.assertEqual(self.assistant.sent, [])

    def test_outside_the_marks_hours_the_alert_goes_out(self):
        mark(self.book)
        job, _ = self.run_failed(AT_1900)
        self.assertEqual(len(self.assistant.sent), 1)
        self.assertTrue(self.assistant.sent[0]["text"].startswith("⚪"))
        self.assertIn("הבדיקה של ה-AI לא הספיקה", self.assistant.sent[0]["text"])

    def test_without_a_mark_the_alert_goes_out(self):
        self.run_failed(AT_1112)
        self.assertEqual(len(self.assistant.sent), 1)

    def test_a_mark_at_another_camera_does_not_cover_this_one(self):
        mark(self.book, camera=DOOR)
        self.run_failed(AT_1112)
        self.assertEqual(len(self.assistant.sent), 1)

    def test_a_model_answer_under_the_mark_is_judged_as_before(self):
        mark(self.book)
        job = self.work(Backend(answer("escalation", people=2, why="breaks into the house")), AT_1112)
        self.assertEqual(len(self.assistant.sent), 1)
        self.assertNotIn("vlm_failed", job.alert)


class FailedAgainTest(GuardCase):
    """2026-10-09 ch1: "the AI check did not finish" at 11:35 and again at 11:37, two messages. One is enough."""

    def setUp(self):
        super().setUp()
        patcher = mock.patch.object(inf, "AI_FAILED_NOTIFY", True)   # these test the "on" behaviour
        patcher.start()
        self.addCleanup(patcher.stop)


    def failed(self, ts, camera=CAM):
        with self.assertLogs("box.inference", "INFO") as logs:
            job = self.work(Backend(None), ts, camera=camera)
        return job, logs.output

    def test_two_failed_checks_150_s_apart_are_one_message(self):
        self.failed(AT_1112)
        job, logs = self.failed(AT_1112 + 150)        # the detector saw nobody in between: a new event
        self.assertEqual(len(self.assistant.sent), 1)
        self.assertIs(job.alert["sent"], False)
        self.assertTrue(job.alert["vlm_failed"])
        self.assertTrue(job.alert["not_sent_reason"].startswith("AI check failed again in the same event"))
        self.assertTrue(any(f"[{CAM}] not sent (AI check failed again in the same event" in m for m in logs), logs)
        # Kept in its event: the story has the look.
        self.assertEqual(self.book.session_of_alert(job.stem)["observations"][-1]["alert_id"], job.stem)

    def test_inside_one_open_event(self):
        self.failed(AT_1112)
        for k in range(1, 15):
            self.book.activity(CAM, AT_1112 + k * 10, people=1)
        job, logs = self.failed(AT_1112 + 150)
        self.assertEqual(len(self.assistant.sent), 1)
        self.assertEqual(job.alert["not_sent_reason"], "AI check failed again in the same event")
        self.assertEqual(self.book.session_of_alert(job.stem)["id"],
                         self.book.session_of_alert(f"{CAM}_{int(AT_1112)}_alert")["id"])

    def test_after_a_real_message_in_the_event(self):
        self.work(Backend(answer("suspicious", people=1, why="tries the door handle")), AT_1112)
        self.book.activity(CAM, AT_1112 + 30, people=1)
        job, _ = self.failed(AT_1112 + 60)
        self.assertEqual(len(self.assistant.sent), 1)
        self.assertEqual(job.alert["not_sent_reason"], "AI check failed again in the same event")

    def test_ten_minutes_later_it_is_news_again(self):
        self.failed(AT_1112)
        self.failed(AT_1112 + inf.AI_FAILED_REPEAT_SEC + 30)
        self.assertEqual(len(self.assistant.sent), 2)

    def test_another_camera_is_its_own(self):
        self.failed(AT_1112)
        self.failed(AT_1112 + 30, camera=DOOR)
        self.assertEqual(len(self.assistant.sent), 2)

    def test_an_answered_look_after_a_failed_one_is_judged_as_always(self):
        self.failed(AT_1112)
        self.work(Backend(answer("escalation", people=1, why="breaks into the house")), AT_1112 + 60)
        self.assertEqual(len(self.assistant.sent), 2)


class FailedCheckDefaultOffTest(GuardCase):
    """The owner, 2026-10-09 12:55: a message that says only "a person, the AI check did not finish" is not wanted.
    By default (ai_failed_notify off) a clip no model answered is kept and logged, never sent."""

    def test_nothing_is_sent_when_no_model_answered(self):
        job = inf.AlertJob(camera=CAM, stem=f"{CAM}_{int(T0)}_alert", ts=T0, labels=["person"],
                           input_meta={"vlm_input": "crop"})
        job.tracker = {"people_together": 1}
        with self.assertLogs("box.inference", "INFO") as logs:
            inf._worker(Backend(None), {"alert_channel": "telegram"}, {}, inf.AlertSettings(), CAM,
                        [np.zeros((4, 4, 3), np.uint8)] * 4, self.assistant, job)
        self.assertTrue(job.ready.is_set())
        self.assertEqual(self.assistant.sent, [])
        self.assertTrue(any("ai_failed_notify off" in line for line in logs.output), logs.output)


class KnownCoversTest(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.dir, ignore_errors=True)
        self.book = EventBook(self.dir)

    def test_live_inside_hours_and_within_the_count(self):
        k = mark(self.book, people=2)
        self.assertEqual(self.book.known_covers(CAM, AT_1112, 4).id, k["id"])
        self.assertIsNone(self.book.known_covers(CAM, AT_1112, 5))
        self.assertEqual(self.book.known_covers(CAM, AT_1112, 0).id, k["id"])      # not counted
        self.assertEqual(self.book.known_covers(CAM, AT_1112, None).id, k["id"])
        self.assertIsNone(self.book.known_covers(CAM, AT_1900, 1))                  # after 18:00
        self.assertIsNone(self.book.known_covers(CAM, DAY + 6 * 3600, 1))           # before 07:00
        self.assertIsNone(self.book.known_covers(CAM, DAY + 7 * 86400 + 11 * 3600, 1))   # the mark is over
        self.assertIsNone(self.book.known_covers(DOOR, AT_1112, 1))


class DetectorPeopleTest(unittest.TestCase):
    def test_the_larger_of_the_tracker_and_the_looks(self):
        job = inf.AlertJob(camera=CAM, stem="s", ts=0.0)
        self.assertEqual(inf._detector_people(job), 0)
        self.assertEqual(inf._detector_people(None), 0)
        job.tracker = {"people_together": 2}
        car, person = (2, 0.9, 0, 0, 1, 1), (0, 0.8, 0, 0, 1, 1)
        job.scene_looks = [(1.0, [person, car, person, person]), (2.0, "junk")]
        self.assertEqual(inf._detector_people(job), 3)


if __name__ == "__main__":
    unittest.main()
