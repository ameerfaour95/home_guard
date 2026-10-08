"""Task 2.9 in the guard loop: what is usual at the camera (baseline.py) may only RAISE.

Shadow (the default) decides and logs only; ``baseline_alerts: on`` turns a rare "normal" into ONE quiet message per
event with the rarity line; a suspicious or an escalation is never changed, it only gets the line when rare. Local
fakes only."""
import datetime as dt
import os
import unittest
from unittest import mock

from home_guard_project.box import baseline as bl
from home_guard_project.box import inference as inf
from home_guard_project.box.camera_profiles import CameraProfiles
from home_guard_project.box.events import EventBook
from test_guard_events import CAM, HE, T0, Backend, GuardCase, answer

ON = dict(HE, baseline_alerts="on")
OFF = dict(HE, baseline_alerts="off")
RARE = {"rarity": "rare", "raise": True, "said_by": "activity", "tags": ["knock"], "days_of_data": 20,
        "expected_per_hour": 0.02, "seen_in_last_30_days_same_bucket": 0, "explained_by": [],
        "text_he": "דפיקה בפרגולה - לא רגיל למצלמה הזו (בדרך כלל רק בכניסה הראשית)",
        "text_en": "a knock at the pergola - not usual for this camera (usually only at the main entrance)"}
COMMON = {**RARE, "rarity": "common", "raise": False, "said_by": "",
          "text_he": "זה קורה כאן כמעט כל יום בשעה הזו", "text_en": "This happens here almost every day at this hour"}


class Historian:
    def __init__(self, result):
        self.result, self.calls = result, []

    def surprise(self, camera, ts, tags=()):
        self.calls.append((camera, ts, list(tags)))
        return dict(self.result)


class BaselineGuardTest(GuardCase):
    def use(self, result):
        historian = Historian(result)
        self.stack.enter_context(mock.patch.object(inf, "HISTORIAN", historian))
        return historian

    def test_shadow_only_logs(self):
        historian = self.use(RARE)
        with self.assertLogs("box.inference", level="INFO") as logs:
            job = self.work(Backend(answer("normal", people=1, summary="A man knocks on the door")), T0)
        self.assertEqual(self.assistant.sent, [])
        self.assertTrue(any("baseline: would raise (rare: a knock at the pergola" in line for line in logs.output))
        self.assertEqual(historian.calls[0][2], ["knock"])          # the activity tags from the Eye's words
        self.assertEqual(job.alert["baseline"]["mode"], "shadow")
        self.assertIs(job.alert["baseline"]["would_raise"], True)
        self.assertIs(job.alert["baseline"]["raise"], False)
        self.assertEqual(job.input_meta["baseline"]["rarity"], "rare")
        self.assertNotIn("raised", job.alert)

    def test_on_a_rare_normal_is_one_quiet_message_per_event(self):
        self.use(RARE)
        job = self.work(Backend(answer("normal", people=1, summary="A man knocks on the door")), T0, box=ON)
        self.assertEqual(len(self.assistant.sent), 1)
        sent = self.assistant.sent[0]
        self.assertIs(sent["silent"], True)
        self.assertIn("לא רגיל למצלמה הזו (בדרך כלל רק בכניסה הראשית)", sent["text"])
        self.assertNotIn(CAM, sent["text"])
        self.assertEqual(job.alert["label"], "normal")
        self.assertEqual(job.alert["raised"], "rare here")
        self.assertIn("normal, but rare here", job.alert["event"]["reason"])
        again = self.work(Backend(answer("normal", people=1, summary="A man knocks on the door")), T0 + 30, box=ON)
        self.assertEqual(len(self.assistant.sent), 1)                  # once per event
        self.assertIs(again.alert["sent"], False)

    def test_on_but_common_or_explained_changes_nothing(self):
        self.use(COMMON)
        self.work(Backend(answer("normal", people=1)), T0, box=ON)
        self.assertEqual(self.assistant.sent, [])
        self.use({**RARE, "raise": False, "explained_by": ["known: הגנן"]})
        job = self.work(Backend(answer("normal", people=1)), T0 + 600, box=ON)
        self.assertEqual(self.assistant.sent, [])
        self.assertEqual(job.alert["baseline"]["explained_by"], ["known: הגנן"])

    def test_a_rare_suspicious_keeps_its_label_and_gets_the_line(self):
        self.use(RARE)
        job = self.work(Backend(answer("suspicious", people=1, why="a man tries the door")), T0, box=ON)
        self.assertEqual(len(self.assistant.sent), 1)
        self.assertIs(self.assistant.sent[0]["silent"], False)
        self.assertEqual(job.alert["label"], "suspicious")
        self.assertIn("בדרך כלל רק בכניסה הראשית", self.assistant.sent[0]["text"])

    def test_never_lowers(self):
        """A common activity never softens a suspicious or an escalation: the same message as without a baseline."""
        texts = {}
        for name, historian, box in (("none", None, HE), ("common", Historian(COMMON), ON)):
            for label, why in (("suspicious", "a man tries the door"), ("escalation", "a man breaks the window")):
                with mock.patch.object(inf, "HISTORIAN", historian):
                    self.book = EventBook(os.path.join(self.dir, f"{name}_{label}"))
                    with mock.patch.object(inf, "EVENTS", self.book):
                        before = len(self.assistant.sent)
                        job = self.work(Backend(answer(label, people=1, why=why)), T0, box=box)
                self.assertEqual(len(self.assistant.sent), before + 1, (name, label))
                self.assertEqual(job.alert["label"], label)
                texts[(name, label)] = self.assistant.sent[-1]["text"]
        self.assertEqual(texts[("none", "suspicious")], texts[("common", "suspicious")])
        self.assertEqual(texts[("none", "escalation")], texts[("common", "escalation")])

    def test_off_does_not_ask(self):
        historian = self.use(RARE)
        job = self.work(Backend(answer("normal", people=1)), T0, box=OFF)
        self.assertEqual(historian.calls, [])
        self.assertNotIn("baseline", job.alert)
        self.assertEqual(self.assistant.sent, [])

    def test_a_broken_historian_never_stops_an_alert(self):
        broken = mock.Mock()
        broken.surprise.side_effect = RuntimeError("disk")
        self.stack.enter_context(mock.patch.object(inf, "HISTORIAN", broken))
        with self.assertLogs("box.inference", level="WARNING"):
            job = self.work(Backend(answer("suspicious", people=1, why="a man tries the door")), T0, box=ON)
        self.assertEqual(len(self.assistant.sent), 1)
        self.assertNotIn("baseline", job.alert)

    def test_false_positive_is_not_asked(self):
        historian = self.use(RARE)
        self.work(Backend(answer("normal", people=0, summary="An empty yard.")), T0, box=ON)
        self.assertEqual(historian.calls, [])


class DecideNeverLowersTest(unittest.TestCase):
    def test_baseline_changes_only_a_normal(self):
        import tempfile, shutil  # noqa: E401
        root = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, root, True)
        for label in ("suspicious", "escalation"):
            plain = EventBook(os.path.join(root, f"a_{label}")).decide(CAM, T0, label, 1, "x", "a1")
            usual = EventBook(os.path.join(root, f"b_{label}")).decide(CAM, T0, label, 1, "x", "a1",
                                                                         baseline={"raise": True, "text_en": "rare"})
            self.assertEqual((plain.notify, plain.reason), (usual.notify, usual.reason))
        book = EventBook(os.path.join(root, "c"))
        self.assertFalse(book.decide(CAM, T0, "normal", 1, "x", "n1", baseline={"raise": False}).notify)
        self.assertTrue(book.decide(CAM, T0 + 5, "normal", 1, "x", "n2", baseline={"raise": True}).notify)


class RealHistorianGuardTest(GuardCase):
    """The whole chain with a real ledger: 20 days of a quiet night at the camera, then someone at 03:10."""

    def test_a_person_at_an_unusual_hour(self):
        night = dt.datetime(2026, 10, 7, 3, 10).timestamp()
        first = dt.datetime(2026, 9, 17, 12, 0)
        rows = [{"event_id": f"e{d}", "camera": CAM, "start": (first + dt.timedelta(days=d)).timestamp(),
                 "end": (first + dt.timedelta(days=d, minutes=5)).timestamp(), "people_max": 2, "outcome": "left",
                 "observations": [{"ts": (first + dt.timedelta(days=d)).timestamp(), "people": 2,
                                   "summary": "Workers clean the pergola"}]} for d in range(20)]
        ledger = bl.Baseline(os.path.join(self.dir, "baseline.json"))
        ledger.build_from_archive(rows, [CAM], now=dt.datetime(2026, 10, 7, 3, 0).timestamp())
        historian = bl.Historian(ledger, CameraProfiles(os.path.join(self.dir, "camera_profiles.json")),
                                 names=inf.camera_display, house=lambda ts: {})
        self.stack.enter_context(mock.patch.object(inf, "HISTORIAN", historian))
        job = self.work(Backend(answer("normal", people=1, summary="Workers clean the pergola.")), night, box=ON)
        self.assertEqual(job.alert["baseline"]["said_by"], "time")
        self.assertEqual(len(self.assistant.sent), 1)
        self.assertIn("לא רגיל למצלמה הזו בשעה הזו", self.assistant.sent[0]["text"])
        self.assertNotIn(CAM, self.assistant.sent[0]["text"])


class NightlyBuildTest(unittest.TestCase):
    def test_due_at_once_without_a_baseline_then_at_0330(self):
        import tempfile, shutil, threading  # noqa: E401
        root = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, root, True)
        built = threading.Event()
        now = dt.datetime(2026, 10, 8, 23, 0).timestamp()
        nightly = bl.NightlyBuild(root, [CAM], now=now, builder=lambda: built.set() or {"days_added": 1})
        self.assertTrue(nightly.tick(now))
        self.assertTrue(built.wait(5))
        self.assertAlmostEqual(nightly.next, dt.datetime(2026, 10, 9, 3, 30).timestamp(), delta=1)
        self.assertFalse(nightly.tick(now + 60))
        with open(os.path.join(root, bl.BASELINE_NAME), "w") as f:
            f.write("{}")
        fresh = bl.NightlyBuild(root, [CAM], now=now + 60)
        self.assertGreater(fresh.next, now + 3600)                    # a fresh baseline waits for the night

    def test_a_failed_build_is_logged_not_raised(self):
        nightly = bl.NightlyBuild(os.devnull, [], now=0.0, builder=mock.Mock(side_effect=OSError("disk")))
        with self.assertLogs("box.baseline", level="WARNING"):
            nightly.build()

    def test_start_baseline_in_the_state_folder(self):
        import tempfile, shutil  # noqa: E401
        root = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, root, True)
        with mock.patch.object(inf.paths, "state_dir", return_value=root), \
                mock.patch.object(inf, "HISTORIAN", None), mock.patch.object(inf, "BASELINE_BUILD", None):
            historian = inf.start_baseline({"baseline_alerts": "on"}, [CAM])
            self.assertIs(inf.HISTORIAN, historian)
            self.assertEqual(historian.baseline.path, os.path.join(root, "events", bl.BASELINE_NAME))
            self.assertEqual(inf.BASELINE_BUILD.cameras, [CAM])
            self.assertEqual(inf.baseline_look(CAM, T0, "normal", "A man walks past", {"baseline_alerts": "on"})
                             ["rarity"], "unknown")


if __name__ == "__main__":
    unittest.main()
