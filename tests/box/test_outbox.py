from __future__ import annotations

import os
import tempfile
import unittest

from helpers import make_clip

from home_guard_project.box.outbox import finished_clip_metas, move_finished_clips, move_orphans

NOW = 1_800_000_000.0
MIN_AGE = 600.0
ORPHAN_AGE = 3600.0
OLD = NOW - 3600
RECENT = NOW - 30


class OutboxTest(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.live = os.path.join(tmp.name, "dataset_multi")
        self.outbox = os.path.join(tmp.name, "dataset_outbox")
        os.makedirs(self.live)

    def _exists(self, root: str, rel: str) -> bool:
        return os.path.isfile(os.path.join(root, rel))

    def test_old_clip_moves_with_all_its_files(self) -> None:
        files = make_clip(self.live, "front", "front_100_trigger", OLD)
        result = move_finished_clips(self.live, self.outbox, MIN_AGE, now=NOW)
        self.assertEqual(result, (1, 8))
        for rel in files:
            self.assertTrue(self._exists(self.outbox, rel), rel)
            self.assertFalse(self._exists(self.live, rel), rel)

    def test_recent_clip_is_left_alone(self) -> None:
        files = make_clip(self.live, "front", "front_100_trigger", RECENT)
        self.assertEqual(move_finished_clips(self.live, self.outbox, MIN_AGE, now=NOW), (0, 0))
        for rel in files:
            self.assertTrue(self._exists(self.live, rel), rel)

    def test_similar_stem_of_recent_clip_is_not_moved(self) -> None:
        make_clip(self.live, "cam", "cam_100_trigger", OLD)
        recent = make_clip(self.live, "cam", "cam_1000_trigger", RECENT)
        self.assertEqual(move_finished_clips(self.live, self.outbox, MIN_AGE, now=NOW), (1, 8))
        for rel in recent:
            self.assertTrue(self._exists(self.live, rel), rel)
            self.assertFalse(self._exists(self.outbox, rel), rel)

    def test_clip_without_crop_still_moves(self) -> None:
        make_clip(self.live, "front", "front_100_random", OLD, with_crop=False)
        self.assertEqual(move_finished_clips(self.live, self.outbox, MIN_AGE, now=NOW), (1, 7))

    def test_missing_or_empty_live_dir(self) -> None:
        self.assertEqual(move_finished_clips(self.live, self.outbox, MIN_AGE, now=NOW), (0, 0))
        missing = os.path.join(self.live, "nope")
        self.assertEqual(move_finished_clips(missing, self.outbox, MIN_AGE, now=NOW), (0, 0))

    def _make_interrupted_clip(self, camera: str, stem: str, mtime: float) -> list[str]:
        """A clip the collector saved but never finished: every file except the meta."""
        files = make_clip(self.live, camera, stem, mtime)
        meta = files.pop()
        os.remove(os.path.join(self.live, meta))
        for rel in files:
            os.utime(os.path.join(self.live, rel), (mtime, mtime))
        return files

    def test_old_files_without_a_meta_are_moved_as_orphans(self) -> None:
        files = self._make_interrupted_clip("front", "front_100_trigger", OLD - ORPHAN_AGE)
        self.assertEqual(move_orphans(self.live, self.outbox, ORPHAN_AGE, now=NOW), len(files))
        for rel in files:
            self.assertTrue(self._exists(self.outbox, rel), rel)
            self.assertFalse(self._exists(self.live, rel), rel)

    def test_files_still_waiting_for_their_meta_are_not_orphans(self) -> None:
        files = self._make_interrupted_clip("front", "front_100_trigger", RECENT)
        self.assertEqual(move_orphans(self.live, self.outbox, ORPHAN_AGE, now=NOW), 0)
        for rel in files:
            self.assertTrue(self._exists(self.live, rel), rel)

    def test_files_of_a_clip_that_has_a_meta_are_not_orphans(self) -> None:
        files = make_clip(self.live, "front", "front_100_trigger", RECENT)
        for rel in files[:-1]:
            os.utime(os.path.join(self.live, rel), (OLD - ORPHAN_AGE, OLD - ORPHAN_AGE))
        self.assertEqual(move_orphans(self.live, self.outbox, ORPHAN_AGE, now=NOW), 0)
        for rel in files:
            self.assertTrue(self._exists(self.live, rel), rel)

    def test_orphans_with_missing_live_dir(self) -> None:
        self.assertEqual(move_orphans(os.path.join(self.live, "nope"), self.outbox, ORPHAN_AGE, now=NOW), 0)

    def test_finished_clip_metas_is_sorted_and_filtered(self) -> None:
        make_clip(self.live, "b", "b_2_trigger", OLD)
        make_clip(self.live, "a", "a_1_trigger", OLD)
        make_clip(self.live, "a", "a_3_trigger", RECENT)
        metas = finished_clip_metas(self.live, MIN_AGE, now=NOW)
        self.assertEqual(
            [os.path.basename(m) for m in metas],
            ["a_1_trigger.meta.json", "b_2_trigger.meta.json"],
        )


if __name__ == "__main__":
    unittest.main()
