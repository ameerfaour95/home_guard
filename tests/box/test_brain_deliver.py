# tests/box/test_brain_deliver.py
from __future__ import annotations

import json
import os
import tempfile
import unittest
import urllib.error
from unittest.mock import Mock, patch

from home_guard_project.box.brain.deliver import Deliverer, choice_keyboard
from home_guard_project.box.feedback import AlertIndex, Feedback, MuteState
from home_guard_project.box.telegram_notify import TelegramConfig

NOW = 1_790_000_000.0
CFG = TelegramConfig(bot_token="t", chat_ids=["-5"], dry_run=False)


class Recorder:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def post(self, token, method, fields, timeout=15.0):
        self.calls.append((method, fields))
        return self._next()

    def multipart(self, token, method, fields, files, timeout=20.0):
        self.calls.append((method, fields))
        return self._next()

    def _next(self):
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


class DelivererTest(unittest.TestCase):
    def setUp(self) -> None:
        self.dir = tempfile.mkdtemp()
        self.file = os.path.join(self.dir, "clip.mp4")
        with open(self.file, "wb") as f:
            f.write(b"mp4")

    def test_text_with_buttons_and_silence(self) -> None:
        rec = Recorder([{"ok": True, "result": {"message_id": 7}}])
        out = Deliverer(CFG, rec.post, rec.multipart).text("-5", "Which camera?", buttons=["a", "b"], silent=True)
        self.assertEqual(out, {"ok": True, "message_id": 7, "error": ""})
        method, fields = rec.calls[0]
        self.assertEqual(method, "sendMessage")
        self.assertEqual(fields["disable_notification"], "true")
        self.assertEqual(json.loads(fields["reply_markup"])["inline_keyboard"][1][0]["callback_data"], "cl::1")

    def test_video_retries_once_then_reports_the_error(self) -> None:
        rec = Recorder([urllib.error.URLError("down"), {"ok": True, "result": {"message_id": 9}}])
        self.assertTrue(Deliverer(CFG, rec.post, rec.multipart).video("-5", self.file)["ok"])
        rec = Recorder([{"ok": False, "description": "Bad Request: file too big"}] * 2)
        out = Deliverer(CFG, rec.post, rec.multipart).video("-5", self.file)
        self.assertEqual((out["ok"], out["error"]), (False, "Bad Request: file too big"))
        self.assertEqual(len(rec.calls), 2)

    def test_missing_file_is_an_error_not_an_exception(self) -> None:
        rec = Recorder([])
        self.assertFalse(Deliverer(CFG, rec.post, rec.multipart).photo("-5", "nope.jpg")["ok"])

    def test_dry_run_sends_nothing(self) -> None:
        rec = Recorder([])
        cfg = TelegramConfig(bot_token="t", chat_ids=["-5"], dry_run=True)
        self.assertEqual(Deliverer(cfg, rec.post, rec.multipart).text("-5", "hi")["error"], "dry_run")
        self.assertEqual(rec.calls, [])

    def test_choice_keyboard(self) -> None:
        rows = json.loads(choice_keyboard(["main_entrance", "front_side"], "ab12"))["inline_keyboard"]
        self.assertEqual(rows, [[{"text": "main_entrance", "callback_data": "cl:ab12:0"}],
                                [{"text": "front_side", "callback_data": "cl:ab12:1"}]])


class FeedbackAdditionsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.dir = tempfile.mkdtemp()

    def test_resume_one_camera_while_all_are_paused(self) -> None:
        mute = MuteState(os.path.join(self.dir, "mute.json"))
        mute.apply(Feedback(action="mute", mute_until=NOW + 3600), NOW)
        mute.resume("a", NOW, cameras=["a", "b", "c"])
        self.assertFalse(mute.is_muted(NOW, "a"))
        self.assertTrue(mute.is_muted(NOW, "b") and mute.is_muted(NOW, "c"))
        saved = mute.snapshot()
        mute.resume(None, NOW)
        self.assertFalse(mute.is_muted(NOW, "b"))
        self.assertFalse(MuteState(os.path.join(self.dir, "mute.json")).is_muted(NOW, "c"))
        mute.restore(saved, NOW)
        self.assertTrue(mute.is_muted(NOW, "b") and not mute.is_muted(NOW, "a"))

    def test_recent_alerts(self) -> None:
        index = AlertIndex(os.path.join(self.dir, "index.json"))
        index.remember("-5", 1, {"alert_id": "x", "ts": NOW - 4000})
        index.remember("-5", 2, {"alert_id": "y", "ts": NOW - 600})
        index.remember("-5", 3, {"alert_id": "z", "ts": NOW - 60})
        index.remember("-6", 4, {"alert_id": "w", "ts": NOW - 60})
        self.assertEqual([a["alert_id"] for a in index.recent("-5", NOW)], ["z", "y"])


class DeliveryRobustnessTest(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.file = os.path.join(tmp.name, "photo.jpg")
        with open(self.file, "wb") as stream:
            stream.write(b"jpeg")

    def test_malformed_replies_fail_without_raising_or_claiming_success(self) -> None:
        replies = [[], None, {"ok": "false"}, {"ok": True, "result": []},
                   {"ok": True, "result": {"message_id": float("nan")}},
                   {"ok": True, "result": {"message_id": "bad"}}, ValueError("bad JSON")]
        for reply in replies:
            with self.subTest(reply=reply):
                rec = Recorder([reply])
                with self.assertLogs("box.brain.deliver", level="WARNING") as logs:
                    out = Deliverer(CFG, rec.post, rec.multipart).text("-5", "hello")
                self.assertFalse(out["ok"])
                self.assertIsNone(out["message_id"])
                self.assertTrue(out["error"])
                self.assertEqual(len(logs.output), 1)

    def test_media_retries_bad_json_and_logs_only_final_failure(self) -> None:
        rec = Recorder([ValueError("bad JSON")] * 2)
        with self.assertLogs("box.brain.deliver", level="WARNING") as logs:
            out = Deliverer(CFG, rec.post, rec.multipart).photo("-5", self.file)
        self.assertFalse(out["ok"])
        self.assertEqual(len(rec.calls), 2)
        self.assertEqual(len(logs.output), 1)

    def test_feed_failure_does_not_change_delivery_or_retry(self) -> None:
        for method in ("text", "photo", "video"):
            with self.subTest(method=method):
                feed = Mock()
                feed.add.side_effect = TypeError("not JSON serializable")
                rec = Recorder([{"ok": True, "result": {"message_id": 12}}])
                sender = Deliverer(CFG, rec.post, rec.multipart, feed=feed)
                with self.assertLogs("box.brain.deliver", level="WARNING"):
                    out = getattr(sender, method)("-5", "hello" if method == "text" else self.file)
                self.assertEqual(out, {"ok": True, "message_id": 12, "error": ""})
                self.assertEqual(len(rec.calls), 1)

    def test_bad_paths_do_not_raise(self) -> None:
        rec = Recorder([])
        sender = Deliverer(CFG, rec.post, rec.multipart)
        for method in (sender.photo, sender.video):
            for path in ([], None, "bad\0path"):
                with self.subTest(method=method, path=path):
                    self.assertFalse(method("-5", path)["ok"])
        self.assertEqual(rec.calls, [])

    def test_malformed_keyboard_is_empty_and_logs_once(self) -> None:
        for choices in (None, 42, {"a": "b"}):
            with self.subTest(choices=choices):
                with self.assertLogs("box.brain.deliver", level="WARNING") as logs:
                    self.assertEqual(json.loads(choice_keyboard(choices)), {"inline_keyboard": []})
                self.assertEqual(len(logs.output), 1)

    def test_invalid_retry_counts_use_one_retry(self) -> None:
        for retries in ("bad", float("nan"), float("inf"), []):
            with self.subTest(retries=retries):
                rec = Recorder([OSError("offline"), {"ok": True, "result": {"message_id": 4}}])
                sender = Deliverer(CFG, rec.post, rec.multipart, retries=retries)
                self.assertTrue(sender.video("-5", self.file)["ok"])
                self.assertEqual(len(rec.calls), 2)

    def test_typing_failure_is_logged_and_never_raises(self) -> None:
        for reply in (TypeError("bad input"), [], {"ok": "false"}):
            with self.subTest(reply=reply):
                rec = Recorder([reply])
                with self.assertLogs("box.brain.deliver", level="WARNING"):
                    self.assertIsNone(Deliverer(CFG, rec.post, rec.multipart).typing("-5"))

    def test_disabled_delivery_blocks_every_method(self) -> None:
        for cfg, reason in ((TelegramConfig(), "not_configured"),
                            (TelegramConfig(bot_token="t", dry_run=True), "dry_run")):
            rec = Recorder([])
            sender = Deliverer(cfg, rec.post, rec.multipart)
            for method in (sender.text, sender.photo, sender.video):
                self.assertEqual(method("-5", self.file), {"ok": False, "message_id": None, "error": reason})
            sender.typing("-5")
            self.assertEqual(rec.calls, [])

    def test_text_failure_is_not_retried_and_is_recorded(self) -> None:
        rec = Recorder([{"ok": False, "description": "Forbidden"}])
        feed = Mock()
        sender = Deliverer(CFG, rec.post, rec.multipart, feed=feed)
        self.assertEqual(sender.text("-5", "hello"), {"ok": False, "message_id": None, "error": "Forbidden"})
        self.assertEqual(len(rec.calls), 1)
        feed.add.assert_called_with("assistant", "answer", "hello", delivered=False, error="Forbidden")

    def test_media_fields_feed_and_reply_fields(self) -> None:
        rec = Recorder([{"ok": True, "result": {"message_id": 8}}] * 2)
        feed = Mock()
        sender = Deliverer(CFG, rec.post, rec.multipart, feed=feed)
        sender.text("-5", "hello", reply_to=3)
        self.assertEqual(rec.calls[0][1]["reply_to_message_id"], "3")
        self.assertEqual(rec.calls[0][1]["allow_sending_without_reply"], "true")
        feed.add.assert_called_with("assistant", "answer", "hello", delivered=True, error="")
        sender.video("-5", self.file, "c" * 1100)
        self.assertEqual(rec.calls[1][1]["caption"], "c" * 1024)
        self.assertEqual(rec.calls[1][1]["supports_streaming"], "true")
        feed.add.assert_called_with("assistant", "video", "photo.jpg", delivered=True, error="")


class FeedbackRobustnessTest(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.path = os.path.join(tmp.name, "state.json")

    def write_state(self, data) -> None:
        with open(self.path, "w", encoding="utf-8") as stream:
            json.dump(data, stream)

    def test_mute_load_and_snapshot_skip_bad_values(self) -> None:
        self.write_state({"all": "bad", "cameras": {"a": NOW + 60, "b": [], "c": "nan", "d": "inf"}})
        with self.assertLogs("box.feedback", level="WARNING") as logs:
            state = MuteState(self.path)
        self.assertEqual(len(logs.output), 1)
        saved = state.snapshot()
        self.assertEqual(saved, {"all": 0.0, "cameras": {"a": NOW + 60}})
        saved["cameras"]["a"] = 0
        self.assertTrue(state.is_muted(NOW, "a"))

    def test_restore_skips_malformed_and_expired_values(self) -> None:
        state = MuteState(self.path)
        for snapshot in ([], {"all": "nan", "cameras": []},
                         {"all": "inf", "cameras": {"a": NOW + 60, "b": "bad", "c": NOW - 1, "d": object()}}):
            with self.subTest(snapshot=snapshot):
                with self.assertLogs("box.feedback", level="WARNING"):
                    state.restore(snapshot, NOW)
                expected = {"a": NOW + 60} if isinstance(snapshot, dict) and isinstance(snapshot.get("cameras"), dict) else {}
                self.assertEqual(state.snapshot(), {"all": 0.0, "cameras": expected})
                self.assertEqual(MuteState(self.path).snapshot(), state.snapshot())

    def test_bad_resume_and_restore_arguments_leave_state_unchanged(self) -> None:
        state = MuteState(self.path)
        state.apply(Feedback(action="mute", mute_until=NOW + 60), NOW)
        saved = state.snapshot()
        for camera, now, cameras in (([], NOW, []), ("a", "bad", ["a"]),
                                     ("a", float("nan"), ["a"]), ("a", NOW, 42)):
            with self.subTest(camera=camera, now=now, cameras=cameras):
                with self.assertLogs("box.feedback", level="WARNING"):
                    state.resume(camera, now, cameras)
                self.assertEqual(state.snapshot(), saved)
        with self.assertLogs("box.feedback", level="WARNING"):
            state.restore({}, float("inf"))
        self.assertEqual(state.snapshot(), saved)

    def test_resume_preserves_longer_camera_pause_and_drops_expired(self) -> None:
        state = MuteState(self.path)
        state.restore({"all": NOW + 60, "cameras": {"b": NOW + 120, "old": NOW - 1}}, NOW)
        state.resume("a", NOW, ["a", "b", "c"])
        self.assertEqual(state.snapshot(), {"all": 0.0, "cameras": {"b": NOW + 120, "c": NOW + 60}})

    def test_recent_skips_malformed_entries_and_deduplicates(self) -> None:
        self.write_state({"alerts": [
            {"chat_id": "-5", "message_id": 1, "alert": {"alert_id": "a", "ts": NOW - 10}},
            {"chat_id": "-5", "message_id": 2, "alert": {"alert_id": "b", "ts": NOW - 5}},
            {"chat_id": "-5", "message_id": 3, "alert": {"alert_id": "b", "ts": NOW - 5}},
            [], {}, {"chat_id": "-5", "alert": []},
            *[{"chat_id": "-5", "alert": {"alert_id": "bad", "ts": ts}}
              for ts in ("bad", "nan", "inf", [], None)],
            {"chat_id": "-5", "alert": {"alert_id": [], "ts": NOW}},
        ]})
        with self.assertLogs("box.feedback", level="WARNING"):
            index = AlertIndex(self.path)
            self.assertEqual([a["alert_id"] for a in index.recent("-5", NOW)], ["b", "a"])

    def test_recent_bad_time_arguments_return_empty(self) -> None:
        index = AlertIndex(self.path)
        index.remember("-5", 1, {"alert_id": "a", "ts": NOW})
        for now, age in (("bad", 1800), (float("nan"), 1800), (NOW, "bad"), (NOW, float("inf"))):
            with self.subTest(now=now, age=age):
                with self.assertLogs("box.feedback", level="WARNING"):
                    self.assertEqual(index.recent("-5", now, age), [])

    def test_wrong_alert_collection_type_is_logged_and_empty(self) -> None:
        self.write_state({"alerts": {"wrong": "shape"}})
        with self.assertLogs("box.feedback", level="WARNING") as logs:
            self.assertEqual(AlertIndex(self.path).recent("-5", NOW), [])
        self.assertEqual(len(logs.output), 1)

    def test_invalid_utf8_and_wrong_root_do_not_raise(self) -> None:
        for raw in (b'{"all": "\xff"}', b'[]'):
            with self.subTest(raw=raw):
                with open(self.path, "wb") as stream:
                    stream.write(raw)
                with self.assertLogs("box.feedback", level="WARNING"):
                    self.assertEqual(MuteState(self.path).snapshot(), {"all": 0.0, "cameras": {}})
                self.assertEqual(AlertIndex(self.path).recent("-5", NOW), [])

    def test_persistence_errors_never_escape_resume_or_restore(self) -> None:
        state = MuteState(self.path)
        for error in (OSError("disk full"), TypeError("not serializable")):
            with self.subTest(error=error):
                with patch("home_guard_project.box.feedback._write_json", side_effect=error):
                    with self.assertLogs("box.feedback", level="WARNING"):
                        state.resume(None, NOW)
                    with self.assertLogs("box.feedback", level="WARNING"):
                        state.restore({"all": NOW + 60}, NOW)

    def test_nonserializable_index_write_keeps_last_saved_state(self) -> None:
        index = AlertIndex(self.path)
        index.remember("-5", 1, {"alert_id": "a", "ts": NOW})
        with self.assertLogs("box.feedback", level="WARNING"):
            index.remember("-5", 2, {"alert_id": "b", "ts": NOW, "bad": object()})
        self.assertEqual([a["alert_id"] for a in AlertIndex(self.path).recent("-5", NOW)], ["a"])


if __name__ == "__main__":
    unittest.main()
