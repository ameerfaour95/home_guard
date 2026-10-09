# tests/box/test_brain_alerts.py
from __future__ import annotations

import datetime as dt
import json
import os
import tempfile
import unittest

from home_guard_project.box.ai_status import AiStatus
from home_guard_project.box.brain.mode import ModeWatch
from home_guard_project.box.feedback import FEEDBACK_QUESTION, Feedback, save_feedback
from home_guard_project.box.inference import VLM_SCHEMA, build_prompt, dispatch_alert, is_silent
from home_guard_project.box.telegram_agent import feedback_keyboard, owner_reacted, send_alert
from home_guard_project.box.telegram_notify import TelegramConfig, graded_alert_text
from home_guard_project.box.brain.i18n import t

NOW = dt.datetime(2026, 10, 3, 22, 0).timestamp()


class GradedAlertTest(unittest.TestCase):
    def test_schema_and_prompt(self) -> None:
        self.assertIn("why", VLM_SCHEMA["required"])
        self.assertIn("summary_owner", VLM_SCHEMA["required"])
        he = build_prompt("cam", 0, "22:00:00", 22, 6, owner_language="he")
        en = build_prompt("cam", 0, "22:00:00", 22, 6)
        self.assertIn("translated into Hebrew", he)
        self.assertIn('"summary_owner": "<an empty string>"', en)
        self.assertIn("Dark clothing alone never makes a scene suspicious", en)

    def test_only_normal_is_ever_silent(self) -> None:
        self.assertTrue(is_silent("normal"))
        for label in ("suspicious", "escalation", "", "unknown"):
            self.assertFalse(is_silent(label))

    def test_graded_text(self) -> None:
        self.assertEqual(graded_alert_text("normal", "gate", "A courier leaves a parcel.", "", "en"),
                         "🟢 Looks normal · gate\nA courier leaves a parcel.")
        self.assertEqual(graded_alert_text("escalation", "gate", "A man breaks the window.", "breaks in", "en"),
                         "🔴 ESCALATION · gate\nA man breaks the window.\nWhy: breaks in")
        self.assertTrue(graded_alert_text("suspicious", "gate", "גבר מסתובב ליד הגדר.", "מסתובב", "he")
                        .startswith("🟡 חשוד · gate"))
        self.assertTrue(graded_alert_text("", "gate", "x", "", "en").startswith("⚪ Activity · gate"))

    def test_send_alert_is_loud_unless_asked_and_speaks_the_box_language(self) -> None:
        posts = []
        cfg = TelegramConfig(bot_token="t", chat_ids=["-5"], dry_run=False)

        class Index:
            def remember(self, *a):
                pass

        post = lambda token, method, fields, timeout=15.0: posts.append(fields) or {"ok": True, "result": {"message_id": 1}}  # noqa: E731
        send_alert(cfg, Index(), {"alert_id": "a"}, "x", post=post)
        send_alert(cfg, Index(), {"alert_id": "a"}, "x", post=post, silent=True, lang="he")
        self.assertNotIn("disable_notification", posts[0])
        self.assertTrue(posts[0]["text"].endswith(FEEDBACK_QUESTION))
        self.assertEqual(posts[1]["disable_notification"], "true")
        self.assertTrue(posts[1]["text"].endswith(t("feedback_question", "he")))
        labels = [b["text"] for row in json.loads(feedback_keyboard("en"))["inline_keyboard"] for b in row]
        # Task 18b: the tag buttons replaced the four verdict buttons; plan 8: three of them, no pause.
        self.assertEqual(labels, ["🟡 Suspicious", "🟢 Normal", "🏷️ Other tag"])

    def test_dispatch_uses_the_graded_text_and_silence(self) -> None:
        seen = {}

        class Assistant:
            def send_alert(self, alert, text, image=None, silent=False, lang="en"):
                seen.update(text=text, silent=silent, lang=lang)
                return {"sent": True}

        dispatch_alert({"alert_channel": "telegram"}, {}, "[send_message]", "gate: x", "", assistant=Assistant(),
                       alert={"alert_id": "a"}, graded="🟢 Looks normal · gate\nx", silent=True, lang="he")
        self.assertEqual(seen, {"text": "🟢 Looks normal · gate\nx", "silent": True, "lang": "he"})

    def test_owner_reacted(self) -> None:
        d = tempfile.mkdtemp()
        self.assertFalse(owner_reacted(d, "gate_1_alert"))
        save_feedback(d, {"alert_id": "gate_1_alert", "camera": "gate", "ts": NOW}, Feedback(), "who?", {}, "-5", NOW)
        self.assertTrue(owner_reacted(d, "gate_1_alert"))


class ModeStatusTest(unittest.TestCase):
    def test_mode_watch(self) -> None:
        watch = ModeWatch(every=30)
        self.assertTrue(watch.due(NOW))
        self.assertIsNone(watch.update(NOW - 3600, 22, 6))         # 21:00 assistant: first look is not a change
        self.assertFalse(watch.due(NOW - 3600 + 10))
        self.assertEqual(watch.update(NOW, 22, 6), "guard")
        self.assertIsNone(watch.update(NOW + 60, 22, 6))

    def test_ai_status_mode_and_offline(self) -> None:
        path = os.path.join(tempfile.mkdtemp(), "ai_status.json")
        status = AiStatus(path, min_interval=0)
        status.detection("gate", [], now=NOW - 5)            # the detector looked, but at an old picture
        status.frame_seen("gate", NOW - 120)
        status.frame_seen("door", NOW - 5)
        status.mode("guard", "🛡️ Guarding until 06:00", now=NOW)
        self.assertEqual(status.offline(NOW, cameras=["gate", "door", "never"]), ["gate", "never"])
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        self.assertEqual((data["mode"], data["status_line"]), ("guard", "🛡️ Guarding until 06:00"))


class _Timer:
    """Stands in for threading.Timer: keeps the function so the test runs it when it wants."""

    made: list = []

    def __init__(self, delay, fn) -> None:
        self.delay, self.fn, self.daemon = delay, fn, False
        _Timer.made.append(self)

    def start(self) -> None:
        pass


class _Assistant:
    def __init__(self, sent: bool = True) -> None:
        self.ok = sent
        self.sent: list = []
        self.reminders: list = []

    def is_muted(self, camera):
        return False

    def send_alert(self, alert, text, image=None, silent=False, lang="en"):
        self.sent.append({"alert": alert, "text": text, "silent": silent, "lang": lang})
        return {"sent": self.ok}

    def remind_if_silent(self, alert, text, lang="en"):
        self.reminders.append((alert["alert_id"], text, lang))


class _Backend:
    def __init__(self, parsed: dict) -> None:
        self.parsed = parsed
        self.languages: list = []

    def analyze(self, frames, camera_name, t_sec, start_hour, end_hour, owner_language="en"):
        self.languages.append(owner_language)
        return json.dumps(self.parsed), dict(self.parsed)


class GradedAlertWiringTest(unittest.TestCase):
    """How the guard loop uses the grade: the text, the sound, the reminder, and the video."""

    def _work(self, parsed: dict, assistant: _Assistant, lang: str = "en"):
        from unittest import mock

        from home_guard_project.box import inference as inf

        backend = _Backend(parsed)
        job = inf.AlertJob(camera="gate", stem="gate_100_alert", ts=100.0, labels=["person"])
        with mock.patch.object(inf, "frame_to_jpeg_bytes", return_value=b"jpg"), \
                mock.patch.object(inf, "owner_language", return_value=lang):
            inf._worker(backend, {"alert_channel": "telegram"}, {}, inf.AlertSettings(), "gate", [object()],
                        assistant, job)
        return backend, job

    def test_an_escalation_is_loud_in_the_box_language_and_reminded(self) -> None:
        assistant = _Assistant()
        # A clear class (a break-in into the house): the reminder needs no second look (2026-10-09).
        backend, job = self._work({"summary": "A man breaks into the house.", "label": "escalation", "people": 1,
                                   "why": "breaks in", "summary_owner": "גבר פורץ לבית."}, assistant, "he")
        expected = graded_alert_text("escalation", "gate", "גבר פורץ לבית.", "breaks in", "he")
        self.assertEqual(backend.languages, ["he"])
        (sent,) = assistant.sent
        self.assertEqual((sent["text"], sent["silent"], sent["lang"]), (expected, False, "he"))
        self.assertEqual(assistant.reminders, [("gate_100_alert", expected, "he")])
        self.assertEqual((job.alert["why"], job.alert["summary_owner"], job.alert["silent"], job.alert["people"]),
                         ("breaks in", "גבר פורץ לבית.", False, 1))
        self.assertEqual(job.alert["alert_command"], "[call_owner]")

    def test_a_normal_scene_is_silent_and_never_reminded(self) -> None:
        assistant = _Assistant()
        _, job = self._work({"summary": "A courier leaves a parcel.", "label": "normal", "people": 1,
                             "why": "", "summary_owner": ""}, assistant)
        (sent,) = assistant.sent
        self.assertEqual((sent["text"], sent["silent"]), ("🟢 Looks normal · gate\nA courier leaves a parcel.", True))
        self.assertEqual(assistant.reminders, [])
        self.assertTrue(job.alert["silent"])

    def test_an_undelivered_escalation_sets_no_reminder(self) -> None:
        assistant = _Assistant(sent=False)
        self._work({"summary": "A man breaks the window.", "label": "escalation", "people": 1}, assistant)
        self.assertEqual(len(assistant.sent), 1)
        self.assertEqual(assistant.reminders, [])

    def test_a_count_that_is_not_a_number_does_not_lose_the_alert(self) -> None:
        assistant = _Assistant()
        _, job = self._work({"summary": "A man at the gate.", "label": "suspicious", "people": "several"}, assistant)
        self.assertEqual(len(assistant.sent), 1)
        self.assertIsNone(job.alert["people"])
        self.assertFalse(job.alert["silent"])

    def test_the_reminder_goes_out_once_only_while_nobody_answered(self) -> None:
        from unittest import mock

        from home_guard_project.box import telegram_agent as ta

        d = tempfile.mkdtemp()
        assistant = ta.OwnerAssistant(cfg=TelegramConfig(), index=None, mute=None, inbox=None, feedback_dir=d)
        sent: list = []
        assistant.send_alert = lambda alert, text, image=None, silent=False, lang="en": sent.append(  # type: ignore
            (text, silent, lang))
        alert = {"alert_id": "gate_1_alert", "camera": "gate", "ts": NOW}
        _Timer.made = []
        with mock.patch.object(ta.threading, "Timer", _Timer):
            assistant.remind_if_silent(alert, "🔴 אירוע חמור · gate\nx", "he")
            assistant.remind_if_silent(alert, "🔴 אירוע חמור · gate\nx", "he")
        first, second = _Timer.made
        self.assertEqual((first.delay, first.daemon), (ta.REMIND_SEC, True))
        first.fn()
        self.assertEqual(sent, [(f"{t('alert_reminder', 'he')}\n🔴 אירוע חמור · gate\nx", False, "he")])
        save_feedback(d, alert, Feedback(), "ok", {}, "-5", NOW)
        second.fn()
        self.assertEqual(len(sent), 1)

    def test_an_answer_to_another_alert_is_not_an_answer(self) -> None:
        d = tempfile.mkdtemp()
        save_feedback(d, {"alert_id": "gate_1_alert", "camera": "gate", "ts": NOW}, Feedback(), "x", {}, "-5", NOW)
        self.assertFalse(owner_reacted(d, "gate_1"))
        self.assertFalse(owner_reacted("", "gate_1_alert"))

    def test_the_video_of_a_silent_alert_makes_no_sound(self) -> None:
        from home_guard_project.box.feedback import AlertIndex
        from home_guard_project.box.telegram_agent import send_clip

        d = tempfile.mkdtemp()
        clip = os.path.join(d, "gate_1_alert.mp4")
        with open(clip, "wb") as f:
            f.write(b"mp4")
        index = AlertIndex(os.path.join(d, "index.json"))
        index.remember("-5", 7, {"alert_id": "gate_1_alert", "camera": "gate"})
        calls: list = []

        def post_multipart(token, method, fields, files, timeout=20.0):
            calls.append(fields)
            return {"ok": True}

        cfg = TelegramConfig(bot_token="t", chat_ids=["-5"])
        send_clip(cfg, index, "gate_1_alert", clip, post_multipart=post_multipart)
        send_clip(cfg, index, "gate_1_alert", clip, post_multipart=post_multipart, silent=True)
        self.assertNotIn("disable_notification", calls[0])
        self.assertEqual(calls[1]["disable_notification"], "true")

    def test_buttons_speak_hebrew(self) -> None:
        labels = [b["text"] for row in json.loads(feedback_keyboard("he"))["inline_keyboard"] for b in row]
        self.assertEqual([label.split(" ", 1)[1] for label in labels],
                         [t(k, "he") for k in ("btn_tag_suspicious", "btn_tag_normal", "btn_tag_other")])
        self.assertEqual(t("feedback_question", "en"), FEEDBACK_QUESTION)


class ReviewFixTest(unittest.TestCase):
    """Task 18 review: no false reminder, no false offline at start, no placeholder text, a live frozen-frame clock."""

    def _assistant(self, live: str, archive: str, inbox=None, mute=None):
        from home_guard_project.box import telegram_agent as ta

        assistant = ta.OwnerAssistant(cfg=TelegramConfig(), index=None, mute=mute, inbox=inbox, feedback_dir=live,
                                      archive_dir=archive)
        sent: list = []
        assistant.send_alert = lambda alert, text, image=None, silent=False, lang="en": sent.append(text)  # type: ignore
        return assistant, sent

    def _fire_reminder(self, assistant, alert) -> None:
        from unittest import mock

        from home_guard_project.box import telegram_agent as ta

        _Timer.made = []
        with mock.patch.object(ta.threading, "Timer", _Timer):
            assistant.remind_if_silent(alert, "🔴 ESCALATION · gate\nx", "en")
        (timer,) = _Timer.made
        timer.fn()

    def test_an_answer_already_moved_to_the_archive_counts(self) -> None:
        live, archive = tempfile.mkdtemp(), tempfile.mkdtemp()
        alert = {"alert_id": "gate_1_alert", "camera": "gate", "ts": NOW}
        # The 15-minute upload moved the owner's answer to the archive's folder for this site.
        save_feedback(os.path.join(archive, "house_a"), alert, Feedback(), "ok", {}, "-5", NOW)
        assistant, sent = self._assistant(live, archive)
        self._fire_reminder(assistant, alert)
        self.assertEqual(sent, [])

    def test_no_answer_anywhere_still_reminds(self) -> None:
        live, archive = tempfile.mkdtemp(), tempfile.mkdtemp()
        os.makedirs(os.path.join(archive, "house_a"))
        assistant, sent = self._assistant(live, archive)
        self._fire_reminder(assistant, {"alert_id": "gate_1_alert", "camera": "gate", "ts": NOW})
        self.assertEqual(len(sent), 1)

    def test_an_answer_the_inbox_remembers_counts(self) -> None:
        class Inbox:
            answered = {"gate_1_alert"}

        assistant, sent = self._assistant(tempfile.mkdtemp(), tempfile.mkdtemp(), inbox=Inbox())
        self._fire_reminder(assistant, {"alert_id": "gate_1_alert", "camera": "gate", "ts": NOW})
        self.assertEqual(sent, [])

    def test_the_inbox_remembers_a_tap_and_a_message(self) -> None:
        from home_guard_project.box.feedback import AlertIndex, MuteState
        from home_guard_project.box.telegram_agent import TelegramInbox

        d = tempfile.mkdtemp()
        cfg = TelegramConfig(bot_token="t", chat_ids=["-5"])
        index = AlertIndex(os.path.join(d, "index.json"))
        index.remember("-5", 7, {"alert_id": "gate_1_alert", "camera": "gate", "ts": NOW})
        index.remember("-5", 8, {"alert_id": "gate_2_alert", "camera": "gate", "ts": NOW})
        post = lambda *a, **k: {"ok": True, "result": {"message_id": 99}}  # noqa: E731
        inbox = TelegramInbox(cfg, None, index, MuteState(os.path.join(d, "mute.json")), d,
                              os.path.join(d, "offset.json"), post=post, now=lambda: NOW)
        inbox._on_button({"id": "q", "data": "fb:true", "from": {"id": 1},
                          "message": {"chat": {"id": -5}, "message_id": 7}})
        inbox._on_message({"chat": {"id": -5}, "from": {"id": 1}, "text": "who is it?", "message_id": 20,
                           "reply_to_message": {"message_id": 8}})
        self.assertEqual(inbox.answered, {"gate_1_alert", "gate_2_alert"})

    def test_no_reminder_while_the_camera_or_the_house_is_paused(self) -> None:
        import time

        from home_guard_project.box.feedback import MuteState

        d = tempfile.mkdtemp()
        for scope in ("gate", None):
            mute = MuteState(os.path.join(tempfile.mkdtemp(), "mute.json"))
            mute.apply(Feedback(action="mute", mute_until=time.time() + 3600, camera=scope), time.time())
            assistant, sent = self._assistant(d, d, mute=mute)
            self._fire_reminder(assistant, {"alert_id": "gate_1_alert", "camera": "gate", "ts": NOW})
            self.assertEqual(sent, [], scope)

    def test_another_camera_paused_still_reminds(self) -> None:
        import time

        from home_guard_project.box.feedback import MuteState

        d = tempfile.mkdtemp()
        mute = MuteState(os.path.join(d, "mute.json"))
        mute.apply(Feedback(action="mute", mute_until=time.time() + 3600, camera="door"), time.time())
        assistant, sent = self._assistant(d, d, mute=mute)
        self._fire_reminder(assistant, {"alert_id": "gate_1_alert", "camera": "gate", "ts": NOW})
        self.assertEqual(len(sent), 1)

    def test_nothing_is_offline_right_after_start(self) -> None:
        status = AiStatus(os.path.join(tempfile.mkdtemp(), "ai_status.json"), min_interval=0)
        started = NOW
        self.assertEqual(status.offline(started + 10, cameras=["gate", "door"], since=started), [])
        status.frame_seen("door", started + 50)
        self.assertEqual(status.offline(started + 61, cameras=["gate", "door"], since=started), ["gate"])
        self.assertEqual(status.offline(started + 200, cameras=["gate", "door"], since=started), ["door", "gate"])

    def test_an_english_placeholder_never_reaches_the_owner(self) -> None:
        expected = graded_alert_text("suspicious", "gate", "A man at the gate.", "lingers", "en")
        for lang in ("en", "he"):
            for placeholder in ("an empty string", "<an empty string>", "Empty String", "<empty>"):
                assistant = _Assistant()
                GradedAlertWiringTest._work(self, {"summary": "A man at the gate.", "label": "suspicious",
                                                   "people": 1, "why": "lingers", "summary_owner": placeholder},
                                            assistant, lang)
                self.assertIn("A man at the gate.", assistant.sent[0]["text"], (lang, placeholder))
                if lang == "en":
                    self.assertEqual(assistant.sent[0]["text"], expected, placeholder)

    def test_english_uses_the_summary_even_when_summary_owner_is_filled(self) -> None:
        assistant = _Assistant()
        GradedAlertWiringTest._work(self, {"summary": "A man at the gate.", "label": "suspicious", "people": 1,
                                           "summary_owner": "Un homme au portail."}, assistant, "en")
        self.assertIn("A man at the gate.", assistant.sent[0]["text"])

    def test_a_failing_clip_encode_still_moves_the_frame_clock(self) -> None:
        import threading
        from unittest import mock

        from home_guard_project.box import alert_clips
        from home_guard_project.box import inference as inf

        class Ring:
            def wants(self, now):
                return True

            def add(self, now, data):
                pass

        stream = inf._Stream.__new__(inf._Stream)     # no capture, no thread
        stream.name, stream._mask, stream._ring, stream._frame = "gate", None, Ring(), None
        stream._lock, stream.last_ts = threading.Lock(), 0.0
        with mock.patch.object(alert_clips, "encode_frame", side_effect=RuntimeError("encoder gone")):
            with self.assertRaises(RuntimeError):
                stream._ingest("frame", 123.0)
        self.assertEqual((stream.last_ts, stream._frame), (123.0, "frame"))

    def test_the_worker_is_loud_without_a_valid_label(self) -> None:
        from unittest import mock

        from home_guard_project.box import inference as inf

        for backend in (inf.NullBackend(), _Backend({"summary": "Someone at the gate.", "people": 1}),
                        _Backend({"summary": "Someone at the gate.", "people": 1, "label": "maybe"})):
            assistant = _Assistant()
            job = inf.AlertJob(camera="gate", stem="gate_100_alert", ts=100.0, labels=["person"])
            with mock.patch.object(inf, "frame_to_jpeg_bytes", return_value=b"jpg"), \
                    mock.patch.object(inf, "owner_language", return_value="en"):
                inf._worker(backend, {"alert_channel": "telegram"}, {}, inf.AlertSettings(), "gate", [object()],
                            assistant, job)
            (sent,) = assistant.sent
            self.assertFalse(sent["silent"], backend)
            self.assertTrue(sent["text"].startswith(t("alert_unclassified", "en", camera="gate")), sent["text"])
            self.assertFalse(job.alert["silent"])


if __name__ == "__main__":
    unittest.main()
