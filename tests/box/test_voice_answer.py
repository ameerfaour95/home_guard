"""The owner may answer "Other…" by voice: the recording is transcribed and saved like written words."""

from __future__ import annotations

import glob
import io
import json
import os
import tempfile
import unittest
from typing import Any, Dict, List
from unittest import mock

import numpy as np

from home_guard_project.box import voice
from home_guard_project.box.alert_clips import encode_frame, write_alert_clip
from home_guard_project.box.brain.i18n import t
from home_guard_project.box.feedback import AlertIndex, MuteState
from home_guard_project.box.telegram_agent import TelegramInbox
from home_guard_project.box.telegram_notify import TelegramConfig

NOW = 1_800_000_000.0
CHAT = "-1001"
ALERT_ID = "door_1800000000_alert"
ALERT = {"alert_id": ALERT_ID, "camera": "door", "summary": "a person", "label": "suspicious", "ts": NOW - 60}
DANA = {"id": 42, "first_name": "Dana"}
VOICE = {"file_id": "VOICE1", "duration": 3, "mime_type": "audio/ogg"}


class FakeTelegram:
    def __init__(self) -> None:
        self.calls: List[Dict[str, Any]] = []
        self._next = 900

    def post(self, token, method, fields, timeout=15.0):
        self._next += 1
        self.calls.append({"method": method, "fields": fields, "message_id": self._next})
        return {"ok": True, "result": {"message_id": self._next}}

    def post_multipart(self, token, method, fields, files, timeout=20.0):
        return self.post(token, method, fields, timeout)

    def texts(self) -> List[str]:
        return [c["fields"]["text"] for c in self.calls if c["method"] == "sendMessage"]


class FakeAgent:
    version = 1

    def __init__(self) -> None:
        self.seen: List[str] = []

    def handle(self, text, chat_id, who=None, alert=None):
        self.seen.append(text)
        from home_guard_project.box.agent import AgentReply
        return AgentReply(text="agent answer")


def tap(update_id: int, code: str, on_message: int) -> dict:
    return {"update_id": update_id, "callback_query": {
        "id": f"cb{update_id}", "data": code, "from": dict(DANA),
        "message": {"message_id": on_message, "chat": {"id": int(CHAT)}}}}


def spoken(update_id: int, reply_to: Any = None) -> dict:
    msg: Dict[str, Any] = {"message_id": 500 + update_id, "chat": {"id": int(CHAT)}, "voice": dict(VOICE),
                           "from": dict(DANA, is_bot=False)}
    if reply_to is not None:
        msg["reply_to_message"] = {"message_id": reply_to}
    return {"update_id": update_id, "message": msg}


class VoiceAnswerTest(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = tmp.name
        self.production = os.path.join(self.dir, "production_multi")
        self.training = os.path.join(self.dir, "dataset_multi")
        frames = [(NOW - 60 + i * 0.2, encode_frame(np.zeros((48, 64, 3), dtype=np.uint8))) for i in range(5)]
        with mock.patch("home_guard_project.box.alert_clips._to_h264", return_value=False):
            write_alert_clip(self.production, "door", ALERT_ID, frames,
                             {"summary": "a person", "labels": ["person"], "dispatch": {"sent": True}})
        self.index = AlertIndex(os.path.join(self.dir, "alert_index.json"))
        self.index.remember(CHAT, 77, ALERT)
        self.tg = FakeTelegram()
        self.fetched: List[str] = []
        self.heard: List[tuple] = []
        self.transcript = "it was the gardener"

    def _inbox(self, transcriber: Any = "default", agent: Any = None) -> TelegramInbox:
        def fetch(file_id: str):
            self.fetched.append(file_id)
            return b"ogg-bytes", "voice.oga"

        def transcribe(audio: bytes, name: str, language: str) -> str:
            self.heard.append((audio, name, language))
            return self.transcript

        return TelegramInbox(TelegramConfig(bot_token="T", chat_ids=[CHAT]), agent, self.index,
                             MuteState(os.path.join(self.dir, "mute.json")), self.production,
                             os.path.join(self.dir, "offset.json"), post=self.tg.post,
                             post_multipart=self.tg.post_multipart, now=lambda: NOW,
                             training_dir=self.training, archive_dir=os.path.join(self.dir, "archive"),
                             lang=lambda: "he", fetch_voice=fetch,
                             transcriber=transcribe if transcriber == "default" else transcriber)

    def _saved(self, root: str) -> List[dict]:
        found = []
        for path in sorted(glob.glob(os.path.join(root, "feedback", "*", "*", "*.feedback.json"))):
            with open(path, encoding="utf-8") as f:
                found.append(json.load(f))
        return found

    def test_a_voice_reply_to_the_question_is_transcribed_and_saved(self) -> None:
        inbox = self._inbox()
        inbox.handle_update(tap(1, "tag:other", 77))
        question = self.tg.calls[0]["message_id"]
        inbox.handle_update(spoken(2, reply_to=question))
        self.assertEqual(self.fetched, ["VOICE1"])
        self.assertEqual(self.heard, [(b"ogg-bytes", "voice.oga", "he")])
        (saved,) = self._saved(self.production)
        self.assertEqual((saved["owner_label"], saved["owner_text"], saved["source"], saved["transcript"]),
                         ("other", "it was the gardener", "voice", "it was the gardener"))
        (kept,) = self._saved(self.training)
        self.assertEqual(kept["training"]["answer"]["transcript"], "it was the gardener")
        self.assertTrue(self.tg.texts()[-1].startswith("✓"))

    def test_a_plain_voice_message_right_after_other_is_the_answer_too(self) -> None:
        inbox = self._inbox()
        inbox.handle_update(tap(1, "tag:other", 77))
        inbox.handle_update(spoken(2))
        self.assertEqual(self._saved(self.production)[0]["owner_text"], "it was the gardener")

    def test_a_voice_message_nobody_asked_for_is_left_alone(self) -> None:
        agent = FakeAgent()
        self._inbox(agent=agent).handle_update(spoken(1))
        self.assertEqual((self.fetched, self.heard, agent.seen, self.tg.calls), ([], [], [], []))

    def test_without_a_transcriber_the_owner_is_asked_to_write_and_the_wait_stays(self) -> None:
        inbox = self._inbox(transcriber=None)
        inbox.handle_update(tap(1, "tag:other", 77))
        inbox.handle_update(spoken(2))
        self.assertEqual(self.tg.texts()[-1], t("tag_voice_failed", "he"))
        self.assertEqual(self._saved(self.production), [])
        inbox.handle_update({"update_id": 3, "message": {"message_id": 600, "chat": {"id": int(CHAT)},
                                                         "text": "the gardener", "from": dict(DANA, is_bot=False)}})
        self.assertEqual(self._saved(self.production)[0]["owner_text"], "the gardener")

    def test_a_failed_transcription_or_an_empty_one_asks_to_write(self) -> None:
        def broken(audio, name, language):
            raise RuntimeError("service down")

        for transcriber in (broken, lambda audio, name, language: "  "):
            with self.subTest(transcriber=transcriber):
                self.tg.calls.clear()
                inbox = self._inbox(transcriber=transcriber)
                inbox.handle_update(tap(1, "tag:other", 77))
                inbox.handle_update(spoken(2))
                self.assertEqual(self.tg.texts()[-1], t("tag_voice_failed", "he"))
                self.assertEqual(self._saved(self.production), [])


class DownloadTest(unittest.TestCase):
    def test_getfile_then_the_file_url(self) -> None:
        calls = []

        def post(token, method, fields, timeout=15.0):
            calls.append((method, fields))
            return {"ok": True, "result": {"file_path": "voice/file_7.oga"}}

        opened = []

        def opener(url, timeout=30.0):
            opened.append(url)
            return io.BytesIO(b"ogg")

        self.assertEqual(voice.download("TOKEN", "VOICE1", post=post, opener=opener), (b"ogg", "file_7.oga"))
        self.assertEqual(calls, [("getFile", {"file_id": "VOICE1"})])
        self.assertEqual(opened, ["https://api.telegram.org/file/botTOKEN/voice/file_7.oga"])

    def test_no_file_raises(self) -> None:
        with self.assertRaises(OSError):
            voice.download("T", "X", post=lambda *a, **k: {"ok": False, "description": "file is too big"},
                           opener=lambda *a, **k: io.BytesIO(b""))

    def test_no_key_no_transcriber(self) -> None:
        self.assertIsNone(voice.make_transcriber({}))


if __name__ == "__main__":
    unittest.main()
