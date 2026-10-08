"""The Telegram conversation goes to S3 with the production clips (outbox.copy_chat_feed, run_upload)."""
from __future__ import annotations

import datetime as dt
import json
import os
import tempfile
import time
import unittest
from typing import Any, Dict, List
from unittest import mock

from home_guard_project.box.__main__ import run_upload
from home_guard_project.box.boxconfig import BoxConfig, production_prefix
from home_guard_project.box.chat_feed import FEED_NAME, ChatFeed
from home_guard_project.box.outbox import copy_chat_feed


def ts_at(day: int, hour: int) -> float:
    return dt.datetime(2026, 10, day, hour, 0, 0).timestamp()


def read_lines(path: str) -> List[Dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


class CopyChatFeedTest(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.logs = os.path.join(tmp.name, "logs")
        self.feed_path = os.path.join(self.logs, FEED_NAME)
        self.state = os.path.join(self.logs, "chat_upload_state.json")
        self.outbox = os.path.join(tmp.name, "archive", "house2")
        self.feed = ChatFeed(self.feed_path)

    def copy(self) -> int:
        return copy_chat_feed(self.feed_path, self.outbox, self.state)

    def day_file(self, day: str) -> str:
        return os.path.join(self.outbox, "chat", f"{day}.jsonl")

    def test_lines_go_to_their_local_day_with_hebrew_unescaped_and_pictures_copied(self) -> None:
        self.feed.add("box", "alert", "door: a person", camera="door", alert_id="door_1_alert",
                      image=b"\xff\xd8jpeg", now=ts_at(7, 23))
        self.feed.add("owner", "message", "מי זה?", name="Ameer", now=ts_at(8, 0))
        self.assertEqual(self.copy(), 2)
        first, second = self.day_file("2026-10-07"), self.day_file("2026-10-08")
        self.assertEqual([e["text"] for e in read_lines(first)], ["door: a person"])
        self.assertEqual([e["text"] for e in read_lines(second)], ["מי זה?"])
        with open(second, encoding="utf-8") as f:
            self.assertIn("מי זה?", f.read())                      # UTF-8, not מ escapes
        with open(os.path.join(self.outbox, "chat", "images", "door_1_alert.jpg"), "rb") as f:
            self.assertEqual(f.read(), b"\xff\xd8jpeg")

    def test_a_second_run_copies_only_the_new_lines_and_leaves_the_feed_alone(self) -> None:
        self.feed.add("owner", "message", "one", now=ts_at(8, 10))
        self.copy()
        self.feed.add("assistant", "answer", "two", now=ts_at(8, 11))
        with open(self.feed_path, encoding="utf-8") as f:
            before = f.read()
        self.assertEqual(self.copy(), 1)
        self.assertEqual(self.copy(), 0)
        self.assertEqual([e["text"] for e in read_lines(self.day_file("2026-10-08"))], ["one", "two"])
        with open(self.feed_path, encoding="utf-8") as f:
            self.assertEqual(f.read(), before)                      # never moved, trimmed or rewritten

    def test_a_trimmed_feed_neither_repeats_nor_loses_lines(self) -> None:
        feed = ChatFeed(self.feed_path, keep=3)
        for i in range(4):
            feed.add("owner", "message", f"m{i}", now=ts_at(8, 10) + i)
        self.copy()
        for i in range(4, 8):                                       # the feed trims itself back to 3 lines on the way
            feed.add("owner", "message", f"m{i}", now=ts_at(8, 10) + i)
        self.copy()
        self.assertEqual([e["text"] for e in read_lines(self.day_file("2026-10-08"))], [f"m{i}" for i in range(8)])

    def test_lines_with_the_same_time_are_each_copied_once(self) -> None:
        same = ts_at(8, 12)
        self.feed.add("owner", "button", "yes", now=same)
        self.copy()
        self.feed.add("assistant", "answer", "ok", now=same)
        self.assertEqual(self.copy(), 1)
        self.assertEqual([e["text"] for e in read_lines(self.day_file("2026-10-08"))], ["yes", "ok"])

    def test_a_lost_state_does_not_duplicate_what_the_day_file_has(self) -> None:
        self.feed.add("owner", "message", "one", now=ts_at(8, 10))
        self.copy()
        os.remove(self.state)
        self.assertEqual(self.copy(), 0)
        self.assertEqual(len(read_lines(self.day_file("2026-10-08"))), 1)

    def test_damaged_and_half_written_lines_are_skipped(self) -> None:
        os.makedirs(self.logs, exist_ok=True)
        good = json.dumps({"ts": ts_at(8, 9), "who": "owner", "text": "ok"})
        with open(self.feed_path, "w", encoding="utf-8") as f:
            f.write("{broken\n" + json.dumps({"ts": "x"}) + "\n" + good + "\n" + '{"ts": 1')
        self.assertEqual(self.copy(), 1)
        self.assertEqual(read_lines(self.day_file("2026-10-08")), [json.loads(good)])

    def test_no_feed_is_nothing_to_do(self) -> None:
        self.assertEqual(self.copy(), 0)
        self.assertFalse(os.path.exists(os.path.join(self.outbox, "chat")))


class FakeUploader:
    def __init__(self) -> None:
        self.calls: List[Dict[str, Any]] = []

    def __call__(self, **kwargs: Any) -> None:
        self.calls.append(kwargs)


class RunUploadChatTest(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = tmp.name
        self.live = os.path.join(tmp.name, "production_multi")
        self.archive = os.path.join(tmp.name, "production_archive")
        os.makedirs(self.live)
        self.feed_path = os.path.join(tmp.name, "logs", FEED_NAME)
        ChatFeed(self.feed_path).add("owner", "message", "שלום", now=ts_at(8, 10))
        self.cfg = BoxConfig(site="house2", min_age_minutes=10.0)

    def run_upload(self, uploader) -> None:
        run_upload(self.cfg, self.live, self.archive, "my-bucket", 2, uploader=uploader,
                   prefix_for=production_prefix, keep_local=True, chat_feed=self.feed_path)

    def test_the_chat_goes_into_the_site_archive_and_is_uploaded_with_it(self) -> None:
        uploader = FakeUploader()
        self.run_upload(uploader)
        (call,) = uploader.calls
        self.assertEqual(call["dataset_dir"], os.path.join(self.archive, "house2"))
        self.assertEqual(call["prefix"], "production_house2")
        self.assertFalse(call["delete_local"])
        self.assertTrue(os.path.isfile(os.path.join(self.archive, "house2", "chat", "2026-10-08.jsonl")))
        self.assertTrue(os.path.isfile(os.path.join(self.tmp, "logs", "chat_upload_state.json")))

    def test_without_a_feed_nothing_changes(self) -> None:
        uploader = FakeUploader()
        run_upload(self.cfg, self.live, self.archive, "my-bucket", 2, uploader=uploader,
                   prefix_for=production_prefix, keep_local=True)
        self.assertEqual(uploader.calls, [])

    def test_the_real_uploader_sends_chat_files_as_they_are(self) -> None:
        """s3_upload.run with a fake S3: chat files keep their path under the prefix, nothing re-encodes them."""
        from home_guard_project.s3_upload import s3_upload

        images = os.path.join(os.path.dirname(self.feed_path), "chat_images")
        ChatFeed(self.feed_path).add("box", "alert", "door", alert_id="door_1_alert", image=b"jpeg",
                                     now=ts_at(8, 11))
        self.assertTrue(os.path.isdir(images))
        uploads: List[tuple] = []

        class Paginator:
            def paginate(self, **_):
                return [{"Contents": []}]

        class S3:
            def get_paginator(self, _):
                return Paginator()

            def upload_file(self, path, bucket, key, ExtraArgs=None):
                with open(path, "rb") as f:
                    uploads.append((key, f.read(), ExtraArgs["ContentType"]))

        reencoded: List[list] = []
        with mock.patch.object(s3_upload.boto3, "client", return_value=S3()), \
                mock.patch.object(s3_upload, "reencode_videos",
                                  side_effect=lambda files, **_: reencoded.append(files) or
                                  {"encoded": 0, "skipped": 0, "failed": 0}):
            self.run_upload(s3_upload.run)
        keys = {key: (body, ctype) for key, body, ctype in uploads}
        day = keys["production_house2/chat/2026-10-08.jsonl"]
        with open(os.path.join(self.archive, "house2", "chat", "2026-10-08.jsonl"), "rb") as f:
            self.assertEqual(day[0], f.read())
        self.assertIn("שלום".encode("utf-8"), day[0])
        self.assertEqual(keys["production_house2/chat/images/door_1_alert.jpg"], (b"jpeg", "image/jpeg"))
        self.assertFalse(any("chat" in p for files in reencoded for p in files))
        self.assertTrue(os.path.isfile(os.path.join(self.archive, "house2", "chat", "2026-10-08.jsonl")))


if __name__ == "__main__":
    unittest.main()
