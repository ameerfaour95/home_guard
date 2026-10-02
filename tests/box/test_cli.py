from __future__ import annotations

import os
import tempfile
import time
import unittest
from typing import Any, Dict

from helpers import make_clip

from home_guard_project.box.__main__ import change_site, run_upload
from home_guard_project.box.boxconfig import BoxConfig, load_box_config

OLD = 3600  # seconds ago


class FakeUploader:
    def __init__(self) -> None:
        self.calls: list[Dict[str, Any]] = []

    def __call__(self, **kwargs: Any) -> None:
        self.calls.append(kwargs)


class BoxCliTest(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.live = os.path.join(tmp.name, "dataset_multi")
        self.outbox = os.path.join(tmp.name, "dataset_outbox")
        self.box_yaml = os.path.join(tmp.name, "box.yaml")
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

    def test_finished_clip_is_uploaded_from_its_site_folder(self) -> None:
        make_clip(self.live, "front", "front_1_trigger", time.time() - OLD)
        self.assertEqual(self._run(), (1, 8))
        (call,) = self.uploader.calls
        self.assertEqual(call["dataset_dir"], os.path.join(self.outbox, "house2"))
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
        make_clip(os.path.join(self.outbox, "house2"), "front", "front_1_trigger", time.time() - OLD)
        self.assertEqual(self._run(), (0, 0))
        self.assertEqual(len(self.uploader.calls), 1)

    def test_each_site_in_the_outbox_goes_to_its_own_folder(self) -> None:
        make_clip(os.path.join(self.outbox, "old_house"), "front", "front_1_trigger", time.time() - OLD)
        make_clip(self.live, "yard", "yard_2_trigger", time.time() - OLD)
        self._run()
        self.assertEqual(
            sorted((os.path.basename(c["dataset_dir"]), c["prefix"]) for c in self.uploader.calls),
            [("house2", "dataset_house2"), ("old_house", "dataset_old_house")],
        )

    def test_change_site_sets_aside_clips_saved_under_the_old_name(self) -> None:
        with open(self.box_yaml, "w", encoding="utf-8") as f:
            f.write('site: "old_house"\nmode: inference\n')
        files = make_clip(self.live, "front", "front_1_trigger", time.time() - 5)

        self.assertEqual(change_site("house2", self.live, self.outbox, self.box_yaml), (1, 8))

        cfg = load_box_config(self.box_yaml)
        self.assertEqual((cfg.site, cfg.mode), ("house2", "inference"))
        for rel in files:
            self.assertTrue(os.path.isfile(os.path.join(self.outbox, "old_house", rel)), rel)
            self.assertFalse(os.path.exists(os.path.join(self.live, rel)), rel)

    def test_change_site_to_the_same_name_moves_nothing(self) -> None:
        with open(self.box_yaml, "w", encoding="utf-8") as f:
            f.write('site: "house2"\n')
        make_clip(self.live, "front", "front_1_trigger", time.time() - 5)
        self.assertEqual(change_site("house2", self.live, self.outbox, self.box_yaml), (0, 0))

    def test_change_site_without_a_box_yaml_creates_it(self) -> None:
        self.assertEqual(change_site("house2", self.live, self.outbox, self.box_yaml), (0, 0))
        self.assertEqual(load_box_config(self.box_yaml).site, "house2")


if __name__ == "__main__":
    unittest.main()
