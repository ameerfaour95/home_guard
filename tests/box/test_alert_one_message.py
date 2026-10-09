"""Every alert is one message: the full clip as a video, a short caption and three buttons."""

from __future__ import annotations

import datetime as dt
import json
import os
import tempfile
import unittest
import urllib.error
from typing import Any, Dict, List
from unittest import mock

import numpy as np

from home_guard_project.box import inference as inf
from home_guard_project.box import telegram_agent as ta
from home_guard_project.box.alert_clips import encode_frame
from home_guard_project.box.brain.i18n import t
from home_guard_project.box.feedback import AlertIndex
from home_guard_project.box.telegram_agent import OwnerAssistant, feedback_keyboard, send_alert
from home_guard_project.box.telegram_notify import TelegramConfig

CHAT = "-1001"
NOW = 1_800_000_000.0
ALERT = {"alert_id": "door_1800000000_alert", "camera": "door", "summary": "a person", "label": "suspicious",
         "ts": NOW}
TEXT = "🟡 Suspicious · door\nA person stands at the door."


class FakeTelegram:
    def __init__(self, refuse_video: bool = False) -> None:
        self.calls: List[Dict[str, Any]] = []
        self.refuse_video = refuse_video
        self._next = 900

    def post(self, token, method, fields, timeout=15.0):
        self._next += 1
        self.calls.append({"method": method, "fields": fields, "files": None})
        return {"ok": True, "result": {"message_id": self._next}}

    def post_multipart(self, token, method, fields, files, timeout=20.0):
        if method == "sendVideo" and self.refuse_video:
            raise urllib.error.URLError("video too big")
        self._next += 1
        self.calls.append({"method": method, "fields": fields, "files": files})
        return {"ok": True, "result": {"message_id": self._next}}

    def methods(self) -> List[str]:
        return [c["method"] for c in self.calls]


def codes(raw: str) -> List[List[str]]:
    return [[b["callback_data"] for b in row] for row in json.loads(raw)["inline_keyboard"]]


class _Timer:
    made: list = []

    def __init__(self, delay, fn, args=()) -> None:
        self.delay, self.fn, self.args, self.daemon = delay, fn, args, False
        _Timer.made.append(self)

    def start(self) -> None:
        pass

    def fire(self) -> None:
        self.fn(*self.args)


class KeyboardTest(unittest.TestCase):
    def test_three_buttons_suspicious_normal_other(self) -> None:
        self.assertEqual(codes(feedback_keyboard("en")), [["tag:suspicious", "tag:normal", "tag:other"]])
        texts = [b["text"] for row in json.loads(feedback_keyboard("en", ai_label="normal"))["inline_keyboard"]
                 for b in row]
        self.assertEqual(texts, ["🟡 Suspicious", "✓ 🟢 Normal", "🏷️ Other tag"])

    def test_an_alert_a_house_rule_raised_gets_the_rule_button(self) -> None:
        self.assertEqual(codes(feedback_keyboard("en", rule=True)),
                         [["tag:suspicious", "tag:normal", "tag:other"], ["tag:rule_mismatch"]])
        rule = json.loads(feedback_keyboard("he", rule=True))["inline_keyboard"][1][0]["text"]
        self.assertIn(t("btn_tag_rule_mismatch", "he"), rule)
        self.assertEqual(t("btn_tag_rule_mismatch", "en"), "Doesn't match the rule")


class VideoAlertTest(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = tmp.name
        self.index = AlertIndex(os.path.join(self.dir, "alert_index.json"))
        self.cfg = TelegramConfig(bot_token="T", chat_ids=[CHAT])
        self.clip = os.path.join(self.dir, "door_1800000000_alert.mp4")
        with open(self.clip, "wb") as f:
            f.write(b"video-bytes")

    def test_one_video_with_a_short_caption_and_the_buttons(self) -> None:
        tg = FakeTelegram()
        res = send_alert(self.cfg, self.index, ALERT, TEXT, image=b"jpg", post=tg.post,
                         post_multipart=tg.post_multipart, video=self.clip, silent=True)
        self.assertTrue(res["sent"])
        self.assertEqual(tg.methods(), ["sendVideo"])                     # no photo, no second message
        (call,) = tg.calls
        clock = dt.datetime.fromtimestamp(NOW).strftime("%H:%M")
        self.assertEqual(call["fields"]["caption"], f"🟡 Suspicious · door · {clock}\nA person stands at the door.")
        self.assertEqual(call["fields"]["reply_markup"], feedback_keyboard("en", ai_label="suspicious"))
        self.assertEqual(call["fields"]["disable_notification"], "true")
        self.assertEqual(call["files"]["video"], ("door_1800000000_alert.mp4", b"video-bytes", "video/mp4"))
        self.assertEqual(self.index.lookup(CHAT, res["results"][0]["message_id"])["alert_id"], ALERT["alert_id"])

    def test_a_header_that_has_the_time_does_not_get_it_twice(self) -> None:
        tg = FakeTelegram()
        clock = dt.datetime.fromtimestamp(NOW).strftime("%H:%M")
        text = f"🟡 Suspicious · door · {clock}\nWhat's happening: A person stands at the door."
        send_alert(self.cfg, self.index, ALERT, text, image=b"jpg", post=tg.post,
                   post_multipart=tg.post_multipart, video=self.clip, silent=True)
        (call,) = tg.calls
        self.assertEqual(call["fields"]["caption"], text)

    def test_a_refused_video_falls_back_to_the_picture_with_the_same_buttons(self) -> None:
        tg = FakeTelegram(refuse_video=True)
        res = send_alert(self.cfg, self.index, ALERT, TEXT, image=b"jpg", post=tg.post,
                         post_multipart=tg.post_multipart, video=self.clip)
        self.assertTrue(res["sent"])
        self.assertEqual(tg.methods(), ["sendPhoto"])
        self.assertEqual(tg.calls[0]["fields"]["reply_markup"], feedback_keyboard("en", ai_label="suspicious"))

    def test_a_raised_alert_carries_the_rule_button(self) -> None:
        tg = FakeTelegram()
        alert = dict(ALERT, applied_fact_id="F3", softened=False)
        send_alert(self.cfg, self.index, alert, TEXT, post=tg.post, post_multipart=tg.post_multipart, video=self.clip)
        self.assertEqual(codes(tg.calls[0]["fields"]["reply_markup"])[-1], ["tag:rule_mismatch"])
        softened = dict(ALERT, applied_fact_id="F3", softened=True, label="normal")
        send_alert(self.cfg, self.index, softened, TEXT, post=tg.post, post_multipart=tg.post_multipart,
                   video=self.clip)
        self.assertNotIn(["tag:rule_mismatch"], codes(tg.calls[1]["fields"]["reply_markup"]))


class AssistantHoldsTheAlertTest(unittest.TestCase):
    """Inference sends the alert as soon as the AI has answered and the video once it is written: the assistant
    holds the alert until the video is there, so both go out as one message."""

    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = tmp.name
        self.index = AlertIndex(os.path.join(self.dir, "alert_index.json"))
        self.tg = FakeTelegram()
        self.clip = os.path.join(self.dir, "clip.mp4")
        with open(self.clip, "wb") as f:
            f.write(b"video-bytes")
        patcher = mock.patch.multiple(ta.telegram_notify, _http_post=self.tg.post,
                                      _http_post_multipart=self.tg.post_multipart)
        patcher.start()
        self.addCleanup(patcher.stop)
        timer = mock.patch.object(ta.threading, "Timer", _Timer)
        timer.start()
        self.addCleanup(timer.stop)
        _Timer.made = []
        self.assistant = OwnerAssistant(cfg=TelegramConfig(bot_token="T", chat_ids=[CHAT]), index=self.index,
                                        mute=None, inbox=None, feedback_dir=self.dir)

    def test_the_alert_waits_for_its_video_and_both_go_out_as_one(self) -> None:
        res = self.assistant.send_alert(ALERT, TEXT, b"jpg", silent=True, lang="en")
        self.assertTrue(inf.delivery(res)[0])
        self.assertEqual(self.tg.calls, [])
        out = self.assistant.send_clip(ALERT["alert_id"], self.clip, silent=True)
        self.assertTrue(out["sent"])
        self.assertEqual(self.tg.methods(), ["sendVideo"])
        self.assertEqual(self.tg.calls[0]["fields"]["disable_notification"], "true")
        _Timer.made[0].fire()                                  # the fallback finds nothing left to send
        self.assertEqual(self.tg.methods(), ["sendVideo"])

    def test_no_video_in_time_sends_the_picture(self) -> None:
        self.assistant.send_alert(ALERT, TEXT, b"jpg")
        (timer,) = _Timer.made
        self.assertEqual(timer.delay, ta.VIDEO_WAIT_SEC)
        timer.fire()
        self.assertEqual(self.tg.methods(), ["sendPhoto"])
        self.assistant.send_clip(ALERT["alert_id"], self.clip)       # a late video follows under the picture
        self.assertEqual(self.tg.methods(), ["sendPhoto", "sendVideo"])
        self.assertEqual(self.tg.calls[1]["fields"]["reply_to_message_id"], "901")

    def test_a_clip_that_could_not_be_written_sends_the_picture_at_once(self) -> None:
        self.assistant.send_alert(ALERT, TEXT, b"jpg")
        self.assistant.send_clip(ALERT["alert_id"], "")
        self.assertEqual(self.tg.methods(), ["sendPhoto"])

    def test_a_reminder_of_a_delivered_alert_goes_out_at_once(self) -> None:
        self.assistant.send_alert(ALERT, TEXT, b"jpg")
        self.assistant.send_clip(ALERT["alert_id"], self.clip)
        self.assistant.send_alert(ALERT, "reminder\n" + TEXT)
        self.assertEqual(self.tg.methods(), ["sendVideo", "sendMessage"])


class ClipWriterTest(unittest.TestCase):
    def test_a_clip_that_failed_to_write_still_releases_the_alert(self) -> None:
        sent = []

        class Assistant:
            def send_clip(self, alert_id, clip_path, silent=False):
                sent.append((alert_id, clip_path))
                return {"sent": True}

        delivered = {"channel": "telegram", "telegram": {"telegram": {"sent": True}}}
        job = inf.AlertJob(camera="door", stem="door_100_alert", ts=100.0, labels=["person"],
                           alert={"summary": "a person", "alert_command": "[send_message]", "labels": ["person"],
                                  "dispatch": delivered})
        job.ready.set()
        frames = [(100.0 + i * 0.2, encode_frame(np.zeros((48, 64, 3), dtype=np.uint8))) for i in range(5)]
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch("home_guard_project.box.alert_clips.write_alert_clip", return_value=None):
            inf._save_clip(job, frames, os.path.join(tmp, "p"), os.path.join(tmp, "t"), Assistant())
        self.assertEqual(sent, [("door_100_alert", "")])


if __name__ == "__main__":
    unittest.main()
