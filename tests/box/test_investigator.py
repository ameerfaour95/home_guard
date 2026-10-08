"""Stage 2b (2026-10-08): the investigator's wait-and-watch. A "suspicious" only for lingering (loitering, standing
a while, looking around) is a question of time, which the tracker measures: a short visit that is over is lowered to
normal ("short visit" in the clip's meta); a person still in view is watched up to 20 s (owner-approved), then judged.
Escalations, house notes and other actions are never touched. Fake tracker facts and a fake clock; no model."""

import os
import threading
import unittest
from types import SimpleNamespace
from unittest import mock

from home_guard_project.box import event_memory as em
from home_guard_project.box import inference as inf
from home_guard_project.box.alert_guards import about_lingering

from test_guard_events import CAM, T0, Backend, GuardCase, answer


class Clock:
    def __init__(self, start):
        self.t = start
        self.slept = []

    def now(self):
        return self.t

    def sleep(self, seconds):
        self.slept.append(seconds)
        self.t += seconds


class Trackers:
    """One person visible from *first* to *last* (None: still there until *leaves*, if given)."""

    def __init__(self, first, last=None, leaves=None, people=1):
        self.first, self.last, self.leaves, self.n = first, last, leaves, people
        self.reads = 0

    def facts(self, camera, t0, t1):
        self.reads += 1
        if self.first is None:
            return SimpleNamespace(people=())
        last = self.last if self.last is not None else (min(t1, self.leaves) if self.leaves else t1)
        person = SimpleNamespace(first_seen=self.first, last_seen=last, time_in_view_s=int(last - self.first))
        return SimpleNamespace(people=(person,) * self.n)


class LingeringWordsTest(unittest.TestCase):
    def test_only_lingering_and_no_other_action(self):
        for text in ("loitering by the gate", "A man has been standing by the door for a while",
                     "looking around the yard", "מסתובב ליד השער", "עומד ליד הדלת הרבה זמן", "מסתכל סביב",
                     "שוהה ליד השער", "wandering around the yard"):
            self.assertTrue(about_lingering(text), text)
        for text in ("He tries the door handle while loitering", "loitering at night", "walks to the door",
                     "looking into the car while loitering", "wearing a mask", "climbs the fence", ""):
            self.assertFalse(about_lingering(text), text)


class InvestigateTest(unittest.TestCase):
    def setUp(self):
        self.clock = Clock(T0 + 10)
        patches = [mock.patch.object(inf, "_now", self.clock.now), mock.patch.object(inf, "_sleep", self.clock.sleep)]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)

    def test_a_short_visit_that_is_over_is_lowered_at_once(self):
        found = inf.investigate_lingering(CAM, T0 - 5, trackers=Trackers(T0 - 3, last=T0 + 2))
        self.assertTrue(found["lowered"])
        self.assertEqual((found["verdict"], found["in_view_s"], found["waited_s"]), ("short visit", 5, 0.0))
        self.assertEqual(self.clock.slept, [])

    def test_a_long_stay_stays_suspicious(self):
        found = inf.investigate_lingering(CAM, T0 - 5, trackers=Trackers(T0 - 60, last=T0 + 8))
        self.assertFalse(found["lowered"])
        self.assertEqual(found["verdict"], "stayed 68 s")

    def test_still_there_it_watches_until_they_leave(self):
        found = inf.investigate_lingering(CAM, T0 - 5, trackers=Trackers(T0 - 3, leaves=T0 + 16))
        self.assertTrue(found["lowered"])
        self.assertEqual(found["verdict"], "short visit")
        self.assertEqual(found["in_view_s"], 19)
        self.assertLessEqual(found["waited_s"], 20)
        self.assertTrue(self.clock.slept)

    def test_still_there_it_watches_until_they_stayed_long_enough(self):
        found = inf.investigate_lingering(CAM, T0 - 5, loiter_min=25, trackers=Trackers(T0 - 3))
        self.assertFalse(found["lowered"])
        self.assertEqual(found["verdict"], "stayed 25 s")

    def test_it_waits_at_most_twenty_seconds(self):
        found = inf.investigate_lingering(CAM, T0 - 5, wait_sec=600, trackers=Trackers(T0 + 5))
        self.assertFalse(found["lowered"])
        self.assertEqual(found["verdict"], "still there after waiting")
        self.assertAlmostEqual(sum(self.clock.slept), 20.0)

    def test_no_tracker_nobody_seen_or_a_broken_tracker_changes_nothing(self):
        with mock.patch.object(inf, "TRACKERS", None):
            self.assertEqual(inf.investigate_lingering(CAM, T0, trackers=None)["verdict"], "no tracker")
        self.assertFalse(inf.investigate_lingering(CAM, T0, trackers=Trackers(None))["lowered"])

        class Broken:
            def facts(self, *a):
                raise RuntimeError("boom")

        found = inf.investigate_lingering(CAM, T0, trackers=Broken())
        self.assertFalse(found["lowered"])
        self.assertIn("tracker failed", found["verdict"])

    def test_bad_settings_fall_back_to_the_defaults(self):
        found = inf.investigate_lingering(CAM, T0 - 5, loiter_min=float("nan"), wait_sec="soon",
                                          trackers=Trackers(T0 - 3, last=T0 + 2))
        self.assertEqual(found["loiter_min_sec"], inf.LOITER_MIN_SEC)
        self.assertTrue(found["lowered"])

    def test_the_waiting_worker_frees_the_ai_slot(self):
        seen = {}

        def sleep(seconds):
            seen["busy"] = inf._busy(threading.current_thread())
            self.clock.t += seconds

        with mock.patch.object(inf, "_sleep", sleep):
            done = threading.Event()

            def run():
                inf.investigate_lingering(CAM, T0 - 5, trackers=Trackers(T0 - 3, leaves=T0 + 14))
                done.set()

            worker = threading.Thread(target=run)
            worker.start()
            worker.join(5)
        self.assertTrue(done.is_set())
        self.assertIs(seen["busy"], False)          # while it watches, other cameras' alerts may start
        self.assertFalse(getattr(worker, "investigating", True))


class WorkerTest(GuardCase):
    def setUp(self):
        super().setUp()
        self.clock = Clock(T0 + 10)
        for p in (mock.patch.object(inf, "_now", self.clock.now), mock.patch.object(inf, "_sleep", self.clock.sleep)):
            self.stack.enter_context(p)

    def with_trackers(self, trackers):
        self.stack.enter_context(mock.patch.object(inf, "TRACKERS", trackers))

    def test_a_short_loiter_is_lowered_and_not_sent(self):
        self.with_trackers(Trackers(T0 - 3, last=T0 + 4))
        job = self.work(Backend(answer("suspicious", people=1, why="loitering by the gate")), T0)
        self.assertEqual(self.assistant.sent, [])
        self.assertEqual(job.alert["label"], "normal")
        self.assertEqual(job.alert["investigator"], "short visit")
        self.assertEqual(job.alert["investigation"]["in_view_s"], 7)

    def test_a_long_loiter_goes_out(self):
        self.with_trackers(Trackers(T0 - 80, last=T0 + 9))
        job = self.work(Backend(answer("suspicious", people=1, why="loitering by the gate")), T0)
        self.assertEqual(len(self.assistant.sent), 1)
        self.assertEqual(job.alert["label"], "suspicious")
        self.assertNotIn("investigator", job.alert)
        self.assertEqual(job.alert["investigation"]["verdict"], "stayed 89 s")

    def test_box_yaml_sets_the_minimum(self):
        self.with_trackers(Trackers(T0 - 3, last=T0 + 4))
        box = dict(self.HE_BOX, loiter_min_sec=5)
        job = self.work(Backend(answer("suspicious", people=1, why="loitering by the gate")), T0, box=box)
        self.assertEqual(job.alert["label"], "suspicious")

    def test_other_actions_escalations_and_no_tracker_are_left_alone(self):
        trackers = Trackers(T0 - 3, last=T0 + 4)
        self.with_trackers(trackers)
        job = self.work(Backend(answer("suspicious", people=1, why="tries the door handle")), T0)
        self.assertEqual(job.alert["label"], "suspicious")
        job = self.work(Backend(answer("escalation", people=1, why="loitering by the gate")), T0 + 400, camera="other")
        self.assertEqual(job.alert["label"], "escalation")
        self.assertEqual(trackers.reads, 0)
        self.stack.enter_context(mock.patch.object(inf, "TRACKERS", None))
        job = self.work(Backend(answer("suspicious", people=1, why="loitering by the gate")), T0 + 800,
                        camera="third")
        self.assertEqual(job.alert["label"], "suspicious")
        self.assertNotIn("investigation", job.alert)

    def test_the_first_job_of_an_event_keeps_its_keyframe(self):
        job = self.work(Backend(answer("normal", people=2)), T0)
        session = self.book.session_of_alert(job.stem)
        path = em.keyframe_path(self.dir, session["id"])
        with open(path, "rb") as f:
            self.assertEqual(f.read(), b"jpg")
        os.utime(path, (1, 1))
        self.work(Backend(answer("normal", people=2)), T0 + 30)
        self.assertEqual(os.path.getmtime(path), 1)                     # the second job does not replace it

    HE_BOX = {"alert_channel": "telegram", "owner_translation": "model"}


class StartEventsTest(unittest.TestCase):
    def test_start_events_attaches_the_memory(self):
        import tempfile

        folder = tempfile.mkdtemp()
        with mock.patch.object(inf.paths, "state_dir", return_value=folder), \
                mock.patch("home_guard_project.box.events._BOOK", None), \
                mock.patch.object(inf, "EVENTS", None):
            book = inf.start_events({})
            self.assertEqual(len(book.archive_hooks), 1)
            book.decide(CAM, T0, "normal", 1, "A man waters the plants.")
            book.tick(T0 + 300)
            memory = em.memory_for(book.directory)
            self.assertEqual(memory.records()[0]["observations"][0]["summary"], "A man waters the plants.")


if __name__ == "__main__":
    unittest.main()
