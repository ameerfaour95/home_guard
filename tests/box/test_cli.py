from __future__ import annotations

import os
import tempfile
import time
import unittest
from typing import Any, Dict

from helpers import make_clip

from home_guard_project.box.__main__ import run_upload
from home_guard_project.box.boxconfig import BoxConfig


class FakeUploader:
    def __init__(self) -> None:
        self.calls: list[Dict[str, Any]] = []

    def __call__(self, **kwargs: Any) -> None:
        self.calls.append(kwargs)


class RunUploadTest(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.live = os.path.join(tmp.name, "dataset_multi")
        self.outbox = os.path.join(tmp.name, "dataset_outbox")
        os.makedirs(self.live)
        self.cfg = BoxConfig(site="house2", min_age_minutes=10.0)
        self.uploader = FakeUploader()

    def _run(self) -> tuple[int, int]:
        return run_upload(
            self.cfg,
            live_dir=self.live,
            outbox_dir=self.outbox,
            bucket="my-bucket",
            workers=2,
            uploader=self.uploader,
        )

    def test_finished_clip_is_uploaded_from_outbox(self) -> None:
        make_clip(self.live, "front", "front_1_trigger", time.time() - 3600)
        self.assertEqual(self._run(), (1, 8))
        (call,) = self.uploader.calls
        self.assertEqual(call["dataset_dir"], self.outbox)
        self.assertEqual(call["bucket"], "my-bucket")
        self.assertEqual(call["prefix"], "dataset_house2")
        self.assertEqual(call["workers"], 2)
        self.assertTrue(call["no_cleanup"])
        self.assertTrue(call["delete_local"])
        self.assertEqual(call["allowed_labels"], frozenset())
        self.assertFalse(call["skip_reencode"])

    def test_nothing_to_upload_skips_uploader(self) -> None:
        self.assertEqual(self._run(), (0, 0))
        self.assertEqual(self.uploader.calls, [])
        self.assertTrue(os.path.isdir(self.outbox))

    def test_leftover_outbox_files_are_retried(self) -> None:
        make_clip(self.outbox, "front", "front_1_trigger", time.time() - 3600)
        self.assertEqual(self._run(), (0, 0))
        self.assertEqual(len(self.uploader.calls), 1)


if __name__ == "__main__":
    unittest.main()
