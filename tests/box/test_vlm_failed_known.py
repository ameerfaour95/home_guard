"""2026-10-09 11:12: neither vision model answered at the pergola while the owner's workers mark (ch3 and ch6, every
day 07:00-18:00) was live, and the owner still got "the AI check did not finish". A failed AI check under a live mark
that covers the detector's head-count is kept, not sent; outside the mark's hours or without a mark the honest
alert goes out as before (box/inference.py _event_decision, box/events.py EventBook.known_covers)."""
import shutil
import tempfile
import unittest
from datetime import datetime

import numpy as np

from home_guard_project.box import inference as inf
from home_guard_project.box.events import KNOWN_EXTRA_PEOPLE, EventBook

from test_guard_events import CAM, DOOR, Backend, GuardCase, answer

DAY = datetime(2026, 10, 9, 0, 0).timestamp()
AT_1112 = DAY + 11 * 3600 + 12 * 60
AT_1900 = DAY + 19 * 3600


def mark(book, camera=CAM, people=3, now=DAY + 7 * 3600):
    return book.mark_known(camera, "העובדים בפרגולה", "Ameer", until=DAY + 6 * 86400, now=now, people=people,
                           daily_from="07:00", daily_to="18:00")


class FailedCheckUnderMarkTest(GuardCase):
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
