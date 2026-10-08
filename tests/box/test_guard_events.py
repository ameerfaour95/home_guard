"""Stage 1 of the alert fix, the guard side (2026-10-08): events in the guard loop, camera names, appearance is not
suspicious, a second look before a red. Everything is a local fake: no model, no Telegram, no disk outside a temp dir."""
import json
import os
import shutil
import tempfile
import threading
import time
import unittest
from contextlib import ExitStack
from datetime import datetime
from types import SimpleNamespace
from unittest import mock

import numpy as np

from home_guard_project.box import camera_names, messenger
from home_guard_project.box import inference as inf
from home_guard_project.box.events import EventBook

CAM = "ameer_week_0_1_ch3"
DOOR = "ameer_week_0_1_ch6"
ALIASES = {DOOR: ["כניסה ראשית"]}
T0 = datetime(2026, 10, 7, 9, 42).timestamp()
HE = {"alert_channel": "telegram", "owner_translation": "model"}


def answer(label="suspicious", people=3, why="", summary="Three men work on the pergola.", **extra):
    return dict(summary=summary, label=label, raw_label=label, applied_fact_id="", serious_behaviour=False,
                people=people, vehicle_moving=False, animals=0, why=why, summary_owner="", **extra)


class Backend:
    """Answers from a list (the last one repeats); ``verify`` is given only when asked for."""

    def __init__(self, *answers, verify=None):
        self.answers = list(answers)
        self.calls = 0
        if verify is not None:
            self.verify = verify

    def analyze(self, frames, camera_name, t_sec, start_hour, end_hour, owner_language="en", **kwargs):
        parsed = self.answers[min(self.calls, len(self.answers) - 1)]
        self.calls += 1
        if parsed is None:
            raise RuntimeError("Error code: 402 - box_daily_cap")
        return json.dumps(parsed), parsed


class Assistant:
    """The owner's assistant as the guard loop sees it; takes reply_to like the stage-1 assistant will."""

    def __init__(self, held=False):
        self.sent, self.reminders, self.held = [], [], held
        self.index = SimpleNamespace(messages=lambda alert_id: [("-5", self.ids[alert_id])] if alert_id in self.ids else [])
        self.ids = {}

    def is_muted(self, camera):
        return False

    def send_alert(self, alert, text, image=None, silent=False, lang="en", reply_to=None):
        message_id = 100 + len(self.sent)
        self.sent.append({"alert": alert, "text": text, "silent": silent, "reply_to": reply_to, "id": message_id})
        self.ids[alert["alert_id"]] = message_id
        if self.held:
            return {"sent": True, "held": "waiting for the video"}
        return {"sent": True, "results": [{"chat_id": "-5", "ok": True, "message_id": message_id}]}

    def remind_if_silent(self, alert, text, lang="en"):
        self.reminders.append(alert["alert_id"])


class GuardCase(unittest.TestCase):
    """Runs _worker with an event book in a temp folder, the owner language and the family's camera names."""

    lang = "he"
    book_on = True

    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.book = EventBook(self.dir) if self.book_on else None
        self.stack = ExitStack()
        for patch in (mock.patch.object(inf, "EVENTS", self.book),
                      mock.patch.object(inf, "KNOWN_CAMERAS", (CAM, DOOR)),
                      mock.patch.object(inf, "TRACKERS", None),     # a run() test may leave its registry behind
                      mock.patch.object(inf, "FACTS_PROVIDER", mock.Mock(return_value=[])),
                      mock.patch.object(inf, "owner_language", side_effect=lambda: self.lang),
                      mock.patch.object(inf, "frame_to_jpeg_bytes", return_value=b"jpg"),
                      mock.patch.object(camera_names, "_load", side_effect=lambda a: ALIASES if a is None else a)):
            self.stack.enter_context(patch)
        self.assistant = Assistant()

    def tearDown(self):
        self.stack.close()
        shutil.rmtree(self.dir, ignore_errors=True)

    def work(self, backend, ts, camera=CAM, box=HE, status=None):
        job = inf.AlertJob(camera=camera, stem=f"{camera}_{int(ts)}_alert", ts=ts, labels=["person"],
                           input_meta={"vlm_input": "crop"})
        inf._worker(backend, box, {}, inf.AlertSettings(), camera, [np.zeros((4, 4, 3), np.uint8)] * 4,
                    self.assistant, job, status)
        self.assertTrue(job.ready.is_set())
        return job


class EventsTest(GuardCase):
    def test_normal_is_never_a_message(self):
        status = mock.Mock()
        job = self.work(Backend(answer("normal")), T0, status=status)
        self.assertEqual(self.assistant.sent, [])
        self.assertIs(job.alert["sent"], False)
        self.assertEqual(job.alert["alert_command"], "[send_message]")       # as computed, just not sent
        self.assertIn("normal", job.alert["not_sent_reason"])
        self.assertEqual(job.alert["event"]["session_id"], self.book.session_of_alert(job.stem)["id"])
        self.assertIs(status.decision.call_args.kwargs["sent"], False)

    def test_notify_normal_sends_the_first_normal_of_an_event(self):
        self.book.notify_normal = True
        self.work(Backend(answer("normal")), T0)
        for k in range(1, 18):
            self.book.activity(CAM, T0 + k * 10, people=3)
        self.work(Backend(answer("normal")), T0 + 180)
        self.assertEqual(len(self.assistant.sent), 1)

    def test_suspicious_once_then_a_reply_when_more_people_come(self):
        first = self.work(Backend(answer("suspicious", people=1, why="loitering by the gate")), T0)
        self.book.activity(CAM, T0 + 60, people=1)
        again = self.work(Backend(answer("suspicious", people=1, why="loitering by the gate")), T0 + 120)
        self.book.activity(CAM, T0 + 180, people=3)
        more = self.work(Backend(answer("suspicious", people=3, why="loitering by the gate")), T0 + 240)
        self.assertEqual(len(self.assistant.sent), 2)
        self.assertTrue(first.alert["sent"] and not again.alert["sent"] and more.alert["sent"])
        update = self.assistant.sent[1]
        self.assertEqual(update["reply_to"]["message_id"], self.assistant.sent[0]["id"])
        self.assertEqual(update["text"].splitlines()[0], "עוד 2 אנשים הגיעו")
        session = self.book.session_of_alert(first.stem)
        self.assertEqual(session["messages"][0]["message_id"], self.assistant.sent[0]["id"])
        self.assertEqual(session["reported_people"], 3)

    def test_an_unanswered_look_goes_out_once_per_event(self):
        self.work(Backend(None), T0)
        self.book.activity(CAM, T0 + 60, people=1)
        self.work(Backend(None), T0 + 120)
        self.assertEqual(len(self.assistant.sent), 1)
        self.assertTrue(self.assistant.sent[0]["text"].startswith("⚪"))

    def test_escalation_always_goes_out(self):
        self.work(Backend(answer("suspicious", why="trying the door handle")), T0)
        self.work(Backend(answer("escalation", why="breaks into the house")), T0 + 60)
        self.assertEqual(len(self.assistant.sent), 2)
        self.assertEqual(self.assistant.reminders, [f"{CAM}_{int(T0 + 60)}_alert"])   # a clear class

    def test_case_memory_runs_only_when_the_event_sends(self):
        with mock.patch.object(inf, "_case_memory", return_value=("alert", None, None)) as memory:
            self.work(Backend(answer("normal")), T0)
            memory.assert_not_called()
            self.work(Backend(answer("suspicious", why="trying the door handle")), T0 + 60)
            memory.assert_called_once()

    def test_an_alert_held_for_its_video_files_its_message_later(self):
        self.assistant.held = True
        job = self.work(Backend(answer("suspicious", why="trying the door handle")), T0)
        session = self.book.session_of_alert(job.stem)
        self.assertEqual((session["reported_level"], session["messages"]), ("suspicious", []))
        inf._record_held_message(job, self.assistant)
        inf._record_held_message(job, self.assistant)                     # once only
        self.assertEqual([m["message_id"] for m in self.book.session_of_alert(job.stem)["messages"]], [100])

    def test_a_broken_book_never_loses_an_alert(self):
        broken = mock.Mock()
        broken.decide.side_effect = RuntimeError("disk")
        with mock.patch.object(inf, "EVENTS", broken):
            self.work(Backend(answer("normal")), T0)
        self.assertEqual(len(self.assistant.sent), 1)


class WorkersDayTest(GuardCase):
    def test_seven_hours_of_workers_on_the_pergola_is_at_most_one_message(self):
        """The owner's 2026-10-07: a job every ~3 minutes, 3 people, mostly normal, a few suspicious (masks,
        hoodies, standing around); the owner says "these are my workers" after 10 minutes."""
        whys = ["wearing masks and hoodies", "standing by the pergola for a long time", "faces covered"]
        ts, i = T0, 0
        while ts < T0 + 7 * 3600:
            if i == 4:   # 12 minutes in
                self.book.mark_known(CAM, "עובדים בפרגולה", "owner", until=T0 + 9 * 3600, now=ts)
            label = "suspicious" if i % 4 == 1 else "normal"
            self.work(Backend(answer(label, people=3, why=whys[i % 3] if label == "suspicious" else "")), ts)
            for k in range(1, 18):
                self.book.activity(CAM, ts + k * 10, people=3)
            ts += 180
            i += 1
        self.assertLessEqual(len(self.assistant.sent), 1)
        # The one: the first suspicious, 3 minutes in, before the owner said who they are.
        self.assertEqual([m["alert"]["ts"] for m in self.assistant.sent], [T0 + 180])


class NamesTest(GuardCase):
    def test_the_owner_reads_the_family_s_name_never_the_id(self):
        backend = Backend(answer("suspicious", people=1, why=f"trying the door handle at {DOOR}",
                                 summary=f"A man at {DOOR} tries the door."))
        job = self.work(backend, T0, camera=DOOR)
        (sent,) = self.assistant.sent
        self.assertTrue(sent["text"].startswith("🟡 חשוד · כניסה ראשית"))
        self.assertNotIn(DOOR, sent["text"])
        self.assertEqual(sent["alert"]["camera"], DOOR)                  # keys keep the id
        self.assertEqual(job.alert["summary"], f"A man at {DOOR} tries the door.")

    def test_a_camera_without_a_name_is_its_channel(self):
        self.work(Backend(answer("suspicious", people=1, why="trying the door handle")), T0, camera="site_ch4")
        self.assertTrue(self.assistant.sent[0]["text"].startswith("🟡 חשוד · מצלמה 4"))

    def test_the_translator_keeps_the_name_and_never_sees_the_id(self):
        told = {}

        class Translator:
            def to_owner(self, texts, lang, keep=()):
                told.update(texts=dict(texts), keep=keep)
                return {"summary": "גבר מנסה את הדלת.", "why": "מנסה את הידית"}

        with mock.patch.object(messenger, "messenger_for", return_value=Translator()):
            self.work(Backend(answer("suspicious", people=1, why="trying the door handle",
                                     summary=f"A man at {DOOR} tries the door.")), T0, camera=DOOR,
                      box={"alert_channel": "telegram", "owner_translation": "translator"})
        self.assertEqual(told["keep"], ("כניסה ראשית",))
        self.assertNotIn(DOOR, told["texts"]["summary"])

    def test_the_plain_dispatch_summary_uses_the_name(self):
        self.lang = "en"
        with mock.patch.object(inf, "dispatch_alert", return_value={"sent": True}) as dispatch:
            self.work(Backend(answer("suspicious", people=1, why="trying the door handle")), T0, camera=DOOR)
        self.assertTrue(dispatch.call_args.args[3].startswith("כניסה ראשית: Suspicious:"))
        self.assertNotIn(DOOR, dispatch.call_args.kwargs["graded"])


class AppearanceTest(GuardCase):
    book_on = False

    def test_a_mask_alone_is_normal(self):
        job = self.work(Backend(answer("suspicious", why="wearing a mask and a hoodie")), T0)
        self.assertEqual((job.alert["label"], job.alert["final_label"]), ("normal", "normal"))
        self.assertEqual(job.alert["downgraded"], "appearance only")
        self.assertTrue(self.assistant.sent[0]["text"].startswith("🟢"))

    def test_a_mask_with_an_action_stays_suspicious(self):
        job = self.work(Backend(answer("suspicious", why="a masked man tries the door handle")), T0)
        self.assertEqual(job.alert["label"], "suspicious")
        self.assertNotIn("downgraded", job.alert)

    def test_escalation_is_never_touched(self):
        job = self.work(Backend(answer("escalation", why="wearing a mask")), T0)
        self.assertEqual(job.alert["label"], "escalation")
        self.assertNotIn("downgraded", job.alert)

    def test_the_prompt_says_actions_not_appearance(self):
        prompt = inf.build_prompt("cam", 0, "12:00:00", 0, 0)
        for words in ("Appearance is never by itself a reason", "covering the face WHILE approaching an entrance",
                      "a gun or knife clearly held as a weapon", "ladders, brooms"):
            self.assertIn(words, prompt)
        self.assertNotIn("faces hidden by hoods, masks or clothing", prompt)
        self.assertTrue(inf.PROMPT_VERSION.startswith("2026-10-08"))


class SecondLookTest(GuardCase):
    book_on = False
    WEAPON = answer("escalation", people=1, why="holding a possible weapon", summary="A man holds a long object.")

    def test_not_confirmed_is_a_suspicious_with_one_line(self):
        asked = []

        def verify(frames, question, language="English", timeout=15.0):
            asked.append((len(frames), question, language))
            return {"confirmed": False, "what_it_is": "מוט ארוך", "evidence_frame": 3}

        job = self.work(Backend(self.WEAPON, verify=verify), T0)
        ((frames, question, language),) = asked                          # one call, on the alert's own frames
        self.assertEqual((frames, language), (4, "Hebrew"))
        self.assertIn("NOT weapons", question)
        self.assertEqual((job.alert["label"], job.alert["alert_command"]), ("suspicious", "[send_message]"))
        look = job.alert["second_look"]
        self.assertEqual((look["answered"], look["confirmed"], look["verified"]), (True, False, False))
        text = self.assistant.sent[0]["text"]
        self.assertTrue(text.startswith("🟡"))
        self.assertIn("בדקתי שוב: מוט ארוך (פריים 3), לא נשק", text)
        self.assertEqual(self.assistant.reminders, [])

    def test_confirmed_stays_red_with_its_reminder(self):
        job = self.work(Backend(self.WEAPON, verify=lambda f, q, **k: {"confirmed": True, "what_it_is": "gun",
                                                                          "evidence_frame": 2}), T0)
        self.assertEqual((job.alert["label"], job.alert["alert_command"]), ("escalation", "[call_owner]"))
        self.assertTrue(job.alert["second_look"]["verified"])
        self.assertTrue(self.assistant.sent[0]["text"].startswith("🔴"))
        self.assertEqual(self.assistant.reminders, [job.stem])

    def test_a_failed_look_keeps_the_red_without_a_reminder(self):
        def broken(frames, question, **kwargs):
            raise RuntimeError("503")

        job = self.work(Backend(self.WEAPON, verify=broken), T0)
        self.assertEqual(job.alert["label"], "escalation")
        self.assertEqual(job.alert["second_look"]["reason"], "RuntimeError: 503")
        self.assertFalse(job.alert["second_look"]["verified"])
        self.assertEqual(self.assistant.reminders, [])

    def test_a_slow_look_is_cut_at_the_timeout(self):
        release = threading.Event()

        def slow(frames, question, **kwargs):
            release.wait(5)
            return {"confirmed": False, "what_it_is": "pole", "evidence_frame": 1}

        started = time.monotonic()
        look = inf.second_look(Backend(verify=slow), [], ["weapon"], "en", timeout=0.2)
        release.set()
        self.assertLess(time.monotonic() - started, 2)
        self.assertEqual(look["reason"], "no answer within 0 s")
        self.assertFalse(look["answered"])

    def test_a_slow_look_in_the_worker_keeps_the_red(self):
        release = threading.Event()
        with mock.patch.object(inf, "VERIFY_TIMEOUT_SEC", 0.2):
            job = self.work(Backend(self.WEAPON, verify=lambda f, q, **k: release.wait(5)), T0)
        release.set()
        self.assertEqual(job.alert["label"], "escalation")
        self.assertIn("no answer within", job.alert["second_look"]["reason"])
        self.assertEqual(self.assistant.reminders, [])

    def test_a_clear_class_needs_no_second_look(self):
        verify = mock.Mock()
        job = self.work(Backend(answer("escalation", why="a man with a knife breaks into the house"), verify=verify), T0)
        verify.assert_not_called()
        self.assertNotIn("second_look", job.alert)
        self.assertEqual(self.assistant.reminders, [job.stem])

    def test_a_car_break_in_is_checked_with_its_own_question(self):
        asked = []
        backend = Backend(answer("escalation", why="smashed the car window", summary="A man parks and gets out."),
                          verify=lambda f, q, **k: asked.append(q) or {"confirmed": False,
                                                                       "what_it_is": "a man leaving his car",
                                                                       "evidence_frame": 0})
        self.lang = "en"
        job = self.work(backend, T0, box={"alert_channel": "telegram"})
        self.assertIn("breaking into a vehicle", asked[0])
        self.assertEqual(job.alert["label"], "suspicious")
        self.assertIn("Second look: a man leaving his car, not a car break-in", self.assistant.sent[0]["text"])

    def test_a_backend_without_a_second_look_keeps_the_red(self):
        job = self.work(Backend(self.WEAPON), T0)
        self.assertEqual(job.alert["label"], "escalation")
        self.assertIn("cannot take a second look", job.alert["second_look"]["reason"])
        self.assertEqual(self.assistant.reminders, [])


class VerifyBackendTest(unittest.TestCase):
    def test_gpt_backend_asks_for_strict_json_on_the_frames(self):
        backend = object.__new__(inf.GptBackend)
        backend._model, backend._extra_body = "m", None
        resp = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(
            content='{"confirmed": false, "what_it_is": "a broom", "evidence_frame": "2"}'))])
        with mock.patch.object(inf, "frame_to_jpeg_bytes", return_value=b"jpg"), \
                mock.patch.object(backend, "_complete", return_value=resp, create=True) as complete:
            got = backend.verify([object(), object()], "Is it a gun?", language="Hebrew", timeout=15.0)
        self.assertEqual(got, {"confirmed": False, "what_it_is": "a broom", "evidence_frame": 2})
        content, fmt = complete.call_args.args
        self.assertEqual(fmt, {"type": "json_object"})
        self.assertEqual(complete.call_args.kwargs["timeout"], 15.0)
        self.assertIn("Is it a gun?", content[0]["text"])
        self.assertIn("Hebrew", content[0]["text"])
        self.assertEqual(len(content), 3)

    def test_an_answer_without_a_real_yes_or_no_is_none(self):
        for parsed in (None, {}, {"confirmed": "no"}, {"what_it_is": "x"}):
            self.assertIsNone(inf.verify_answer(parsed), parsed)

    def test_null_backend_and_fallback(self):
        self.assertIsNone(inf.NullBackend().verify([], "q"))
        primary = mock.Mock()
        primary.verify.side_effect = RuntimeError("down")
        fallback = mock.Mock()
        fallback.verify.return_value = {"confirmed": True, "what_it_is": "gun", "evidence_frame": 1}
        self.assertTrue(inf.FallbackBackend(primary, fallback).verify([], "q", language="English")["confirmed"])
        fallback.verify.assert_called_once_with([], "q", language="English", timeout=15.0)


class StartEventsTest(unittest.TestCase):
    def test_the_book_lives_in_the_state_folder_and_reads_notify_normal(self):
        tmp = tempfile.mkdtemp()
        try:
            from home_guard_project.box import events

            with mock.patch.object(events, "_BOOK", None), mock.patch.object(inf, "EVENTS", None), \
                    mock.patch.object(inf.paths, "state_dir", return_value=tmp):
                book = inf.start_events({"notify_normal": "on"})
                self.assertIs(inf.EVENTS, book)
                self.assertTrue(book.notify_normal)
                self.assertEqual(book.directory, os.path.join(tmp, "events"))
            with mock.patch.object(events, "_BOOK", None), mock.patch.object(inf, "EVENTS", None), \
                    mock.patch.object(inf.paths, "state_dir", return_value=tmp):
                self.assertFalse(inf.start_events({}).notify_normal)       # default off
            with mock.patch.object(events, "book", side_effect=OSError("disk")), mock.patch.object(inf, "EVENTS", None):
                self.assertIsNone(inf.start_events({}))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


class RunLoopTest(unittest.TestCase):
    """run() feeds the book on every detector look (people, and vehicles only when they moved) and ticks it."""

    def run_loop(self, boxes):
        from home_guard_project.box import ai_status, boxconfig, camera_alerts, telegram_agent
        from home_guard_project.box import tracker as tr
        from home_guard_project.box.brain import mode
        from home_guard_project.data_collection import config
        import test_vlm_crop_parity as golden

        cfg = config.Config()
        cfg.CAMERAS = {"cam": "synthetic-sub"}
        sub = SimpleNamespace(get_clip_last_seconds=mock.Mock(return_value=([], 0., 0., 2.)))
        clock = [1000.]
        adapters = {"cam": SimpleNamespace(read=lambda: np.zeros((48, 64, 3), np.uint8), last_ts=1000., sub_cap=sub)}
        adapters["cam"].last_ts = 10_000.0
        detector = mock.Mock()
        detector.predict.return_value = [golden._Result([golden._Box(0, (16, 8, 32, 40))])]
        book = mock.Mock()

        class StopLoop(Exception):
            pass

        def sleep(_):
            clock[0] += .5
            if clock[0] > 1004:
                raise StopLoop

        with ExitStack() as stack:
            for patch in [
                mock.patch("dotenv.load_dotenv"),
                mock.patch.object(boxconfig, "load_box_settings", return_value={}),
                mock.patch.object(config, "load_config", return_value=cfg),
                mock.patch.object(inf, "make_backend", return_value=mock.Mock()),
                mock.patch.object(inf, "load_detector", return_value=(detector, None)),
                mock.patch.object(inf, "_camera_streams", return_value=(adapters, {"cam": None})),
                mock.patch.object(inf, "LiveSettings"),
                mock.patch.object(inf, "filter_by_thresholds", side_effect=lambda r, *a: r),
                mock.patch.object(inf, "detect_trigger", return_value=(False, False, [])),
                mock.patch.object(inf, "count_people", return_value=2),
                mock.patch.object(inf, "vehicle_boxes", return_value=boxes),
                mock.patch.object(inf, "start_case_memory", return_value=False),
                mock.patch.object(inf, "start_events", return_value=book),
                mock.patch.object(inf, "KNOWN_CAMERAS", ()),
                mock.patch.object(ai_status, "AiStatus", return_value=mock.Mock()),
                mock.patch.object(ai_status, "objects_from_result", return_value=[]),
                mock.patch.object(mode, "ModeWatch", return_value=mock.Mock(due=lambda now: False)),
                mock.patch.object(camera_alerts, "LiveCameraAlerts", return_value=SimpleNamespace(
                    check=lambda now: False, overrides={}, sensitivity={},
                    thresholds_for=lambda name, defaults: defaults, for_camera=lambda name, defaults: defaults)),
                mock.patch.object(telegram_agent, "start", return_value=None),
                mock.patch.object(tr, "TrackerRegistry", return_value=mock.Mock()),
                mock.patch.object(inf.time, "time", side_effect=lambda: clock[0]),
                mock.patch.object(inf.time, "sleep", side_effect=sleep),
            ]:
                stack.enter_context(patch)
            with self.assertRaises(StopLoop):
                inf.run()
            known = inf.KNOWN_CAMERAS
        return book, detector, known

    def test_every_look_feeds_the_book_and_it_ticks(self):
        book, detector, known = self.run_loop([])
        self.assertEqual(book.activity.call_count, detector.predict.call_count)
        self.assertGreaterEqual(book.activity.call_count, 5)
        camera, _ts = book.activity.call_args.args
        self.assertEqual((camera, book.activity.call_args.kwargs), ("cam", {"people": 2, "vehicles": 0}))
        self.assertGreaterEqual(book.tick.call_count, 4)
        self.assertEqual(known, ("cam",))

    def test_a_parked_car_keeps_no_event_alive(self):
        book, _, _ = self.run_loop([(0.1, 0.1, 0.4, 0.4)])
        vehicles = [c.kwargs["vehicles"] for c in book.activity.call_args_list]
        self.assertEqual(vehicles[0], 1)                    # the first look: it arrived
        self.assertEqual(set(vehicles[1:]), {0})            # then it stands still


if __name__ == "__main__":
    unittest.main()
