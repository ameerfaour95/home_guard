from __future__ import annotations

import io
import json
import os
import tempfile
import unittest
import urllib.error
from unittest import mock

import numpy as np

from home_guard_project.box import inference as inf
from home_guard_project.box.alert_clips import encode_frame
from home_guard_project.box.feedback import AlertIndex
from home_guard_project.box.inference import AlertSettings
from home_guard_project.box.telegram_agent import send_alert, send_clip, telegram_error
from home_guard_project.box.telegram_notify import TelegramConfig

CHAT = "-100200"
ALERT = {"alert_id": "door_100_alert", "camera": "door", "summary": "a person at the door", "ts": 100.0}


def refusal(code: int, description: str) -> urllib.error.HTTPError:
    body = json.dumps({"ok": False, "error_code": code, "description": description}).encode()
    return urllib.error.HTTPError("https://api.telegram.org/x", code, "Forbidden", {}, io.BytesIO(body))


class TelegramRefusalTest(unittest.TestCase):
    """What the box says when Telegram refuses an alert: Telegram's own reason, not just '403'."""

    def test_the_reason_telegram_gives_is_kept_with_what_to_do(self) -> None:
        reason = telegram_error(refusal(403, "Forbidden: bot is not a member of the group chat"))
        self.assertIn("bot is not a member of the group chat", reason)
        self.assertIn("add the bot to the Telegram group again", reason)

    def test_other_failures_read_as_they_are(self) -> None:
        self.assertEqual(telegram_error(refusal(400, "Bad Request: chat not found")), "Bad Request: chat not found")
        self.assertIn("timed out", telegram_error(TimeoutError("timed out")))

    def test_a_refused_alert_reports_the_reason_and_is_not_remembered(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            index = AlertIndex(os.path.join(tmp, "alerts.json"))

            def refuse(*args, **kwargs):
                raise refusal(403, "Forbidden: bot was kicked from the group chat")

            res = send_alert(TelegramConfig(bot_token="T", chat_ids=[CHAT]), index, ALERT, "door: a person",
                             image=b"jpg", post=refuse, post_multipart=refuse)
            self.assertFalse(res["sent"])
            self.assertIn("bot was kicked from the group chat", res["results"][0]["error"])
            self.assertEqual(index.messages("door_100_alert"), [])
            self.assertEqual(inf.delivery({"channel": "telegram", "telegram": {"telegram": res}})[0], False)
            self.assertIn("kicked", inf.delivery({"channel": "telegram", "telegram": {"telegram": res}})[1])


class AlertVideoTest(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = tmp.name
        self.index = AlertIndex(os.path.join(self.tmp, "alerts.json"))
        self.cfg = TelegramConfig(bot_token="T", chat_ids=[CHAT])
        self.clip = os.path.join(self.tmp, "door_100_alert.mp4")
        with open(self.clip, "wb") as f:
            f.write(b"video-bytes")
        self.calls: list = []

    def _post(self, token, method, fields, files, timeout=20.0):
        self.calls.append({"method": method, "fields": fields, "files": files})
        return {"ok": True, "result": {"message_id": 99}}

    def test_the_video_is_a_reply_under_the_alert_it_belongs_to(self) -> None:
        self.index.remember(CHAT, 41, ALERT)
        self.index.remember(CHAT, 42, {**ALERT, "alert_id": "yard_200_alert"})

        res = send_clip(self.cfg, self.index, "door_100_alert", self.clip, post_multipart=self._post)

        self.assertTrue(res["sent"])
        self.assertEqual(len(self.calls), 1)
        call = self.calls[0]
        self.assertEqual(call["method"], "sendVideo")
        self.assertEqual((call["fields"]["chat_id"], call["fields"]["reply_to_message_id"]), (CHAT, "41"))
        self.assertEqual(call["files"]["video"], ("door_100_alert.mp4", b"video-bytes", "video/mp4"))

    def test_no_video_for_an_alert_that_never_arrived_or_in_dry_run(self) -> None:
        self.assertFalse(send_clip(self.cfg, self.index, "door_100_alert", self.clip, post_multipart=self._post)["sent"])
        self.index.remember(CHAT, 41, ALERT)
        dry = TelegramConfig(bot_token="T", chat_ids=[CHAT], dry_run=True)
        self.assertFalse(send_clip(dry, self.index, "door_100_alert", self.clip, post_multipart=self._post)["sent"])
        self.assertEqual(self.calls, [])

    def test_a_missing_clip_or_a_refusal_never_raises(self) -> None:
        self.index.remember(CHAT, 41, ALERT)
        missing = send_clip(self.cfg, self.index, "door_100_alert", self.clip + ".gone", post_multipart=self._post)
        self.assertFalse(missing["sent"])

        def refuse(*args, **kwargs):
            raise refusal(403, "Forbidden: bot is not a member of the group chat")

        refused = send_clip(self.cfg, self.index, "door_100_alert", self.clip, post_multipart=refuse)
        self.assertFalse(refused["sent"])
        self.assertIn("not a member", refused["results"][0]["error"])


class _Assistant:
    def __init__(self) -> None:
        self.clips: list = []

    def send_clip(self, alert_id, clip_path, silent=False):
        self.clips.append((alert_id, clip_path, os.path.isfile(clip_path)))
        self.silent = silent
        return {"sent": True}


class ClipFollowsAlertTest(unittest.TestCase):
    """The clip writer sends the video only for an alert that really went out."""

    def _save(self, alert: dict, false_positive: bool = False):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        production, training = os.path.join(tmp.name, "production_multi"), os.path.join(tmp.name, "dataset_multi")
        frames = [(100.0 + i * 0.2, encode_frame(np.zeros((48, 64, 3), dtype=np.uint8))) for i in range(5)]
        job = inf.AlertJob(camera="door", stem="door_100_alert", ts=100.0, labels=["person"],
                           false_positive=false_positive, alert=alert)
        job.ready.set()
        assistant = _Assistant()
        with mock.patch("home_guard_project.box.alert_clips._to_h264", return_value=False):
            inf._save_clip(job, frames, production, training, assistant)
        self.assistant = assistant
        return assistant.clips, production

    def test_a_delivered_alert_is_followed_by_its_video(self) -> None:
        delivered = {"channel": "telegram", "telegram": {"telegram": {"sent": True, "results": [{"ok": True}]}}}
        clips, production = self._save({"summary": "a person", "alert_command": "[send_message]",
                                        "labels": ["person"], "dispatch": delivered})
        self.assertEqual(len(clips), 1)
        alert_id, path, existed = clips[0]
        self.assertEqual(alert_id, "door_100_alert")
        self.assertTrue(existed)
        self.assertTrue(path.startswith(production) and path.endswith("door_100_alert.mp4"))
        self.assertFalse(self.assistant.silent)
        (meta_path,) = [os.path.join(d, n) for d, _, names in os.walk(production) for n in names
                        if n.endswith(".meta.json")]
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
        self.assertEqual((meta["trigger_ts"], meta["mode"]), (100.0, "guard"))

    def test_the_video_of_a_silent_alert_is_silent_too(self) -> None:
        delivered = {"channel": "telegram", "telegram": {"telegram": {"sent": True, "results": [{"ok": True}]}}}
        clips, _ = self._save({"summary": "a person", "alert_command": "[send_message]", "labels": ["person"],
                               "dispatch": delivered, "silent": True})
        self.assertEqual(len(clips), 1)
        self.assertTrue(self.assistant.silent)

    def test_no_video_when_the_alert_was_refused_paused_or_a_false_positive(self) -> None:
        refused = {"channel": "telegram", "telegram": {"telegram": {"sent": False, "results": [
            {"ok": False, "error": "Forbidden: bot is not a member of the group chat"}]}}}
        base = {"summary": "a person", "alert_command": "[send_message]", "labels": ["person"]}
        self.assertEqual(self._save({**base, "dispatch": refused})[0], [])
        self.assertEqual(self._save({**base, "muted": True,
                                     "dispatch": {"sent": False, "reason": "paused by the owner"}})[0], [])
        self.assertEqual(self._save({**base, "alert_command": "[none]"}, false_positive=True)[0], [])


class _Status:
    def __init__(self) -> None:
        self.decisions: list = []

    def decision(self, camera, labels, summary, command, sent, false_positive=False, muted=False, error="",
                 label=""):
        self.decisions.append({"camera": camera, "labels": labels, "summary": summary, "command": command,
                               "sent": sent, "false_positive": false_positive, "muted": muted, "error": error,
                               "label": label})


class _Backend:
    def __init__(self, parsed: dict) -> None:
        self.parsed = parsed

    def analyze(self, frames, camera_name, t_sec, start_hour, end_hour, owner_language="en"):
        return json.dumps(self.parsed), self.parsed


class WorkerReportsTest(unittest.TestCase):
    """What the window is told about each trigger: the AI's words, and what happened to the alert."""

    def _run(self, parsed: dict, dispatch_result: dict) -> dict:
        status = _Status()
        job = inf.AlertJob(camera="door", stem="door_100_alert", ts=100.0, labels=["person"])
        with mock.patch.object(inf, "frame_to_jpeg_bytes", return_value=b"jpg"), \
                mock.patch.object(inf, "dispatch_alert", return_value=dispatch_result):
            inf._worker(_Backend(parsed), {"alert_channel": "telegram"}, {}, AlertSettings(), "door",
                        [object()], None, job, status)
        self.assertEqual(len(status.decisions), 1)
        return status.decisions[0]

    def test_a_delivered_alert(self) -> None:
        got = self._run({"summary": "A person is walking in the driveway.", "people": 1},
                        {"channel": "telegram", "telegram": {"telegram": {"sent": True}}})
        self.assertEqual((got["summary"], got["command"], got["sent"], got["error"], got["labels"]),
                         ("A person is walking in the driveway.", "[send_message]", True, "", ["person"]))
        self.assertEqual(got["label"], "normal")                      # no label given: normal

    def test_the_models_label_decides_how_loud_the_alert_is(self) -> None:
        with mock.patch.object(inf, "frame_to_jpeg_bytes", return_value=b"jpg"):
            for label, command, prefix in (("normal", "[send_message]", "door: Someone"),
                                           ("suspicious", "[send_message]", "door: Suspicious: Someone"),
                                           ("escalation", "[call_owner]", "door: Escalation: Someone")):
                status = _Status()
                job = inf.AlertJob(camera="door", stem="door_100_alert", ts=100.0, labels=["person"])
                with mock.patch.object(inf, "dispatch_alert", return_value={"telegram": {"telegram": {"sent": True}}}) as d:
                    inf._worker(_Backend({"summary": "Someone is trying the gate.", "label": label, "people": 1}),
                                {"alert_channel": "telegram"}, {}, AlertSettings(), "door", [object()], None, job, status)
                self.assertEqual(d.call_args[0][2], command, label)
                self.assertTrue(d.call_args[0][3].startswith(prefix), (label, d.call_args[0][3]))
                # Telegram gets the graded text; only a normal scene arrives without a sound.
                self.assertEqual(d.call_args.kwargs["silent"], label == "normal", label)
                self.assertIn("door\nSomeone is trying the gate.", d.call_args.kwargs["graded"])
                self.assertEqual((status.decisions[0]["label"], job.alert["label"]), (label, label))

    def test_a_refused_alert_carries_the_reason(self) -> None:
        got = self._run({"summary": "A person is walking in the driveway.", "people": 1},
                        {"channel": "telegram", "telegram": {"telegram": {"sent": False, "results": [
                            {"ok": False, "error": "Forbidden: bot is not a member of the group chat"}]}}})
        self.assertFalse(got["sent"])
        self.assertIn("not a member", got["error"])

    def test_a_false_positive(self) -> None:
        got = self._run({"summary": "Several cars are parked.", "people": 0, "vehicle_moving": False}, {})
        self.assertEqual((got["command"], got["sent"], got["false_positive"]), ("[none]", False, True))


if __name__ == "__main__":
    unittest.main()
