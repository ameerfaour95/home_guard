from __future__ import annotations

import os
import tempfile
import unittest
from unittest import mock

from home_guard_project.box import live_view


class LookNowTest(unittest.TestCase):
    def setUp(self) -> None:
        self.out = tempfile.mkdtemp()
        self.env = {"OPENAI_API_KEY": "key"}

    def _look(self, camera="front_door", url="rtsp://u:p@10.0.0.1/s0", env=None, grab=None, describe=None):
        with mock.patch.object(live_view, "_camera_url", return_value=url):
            return live_view.look_now(
                camera, "cams.yaml", self.env if env is None else env, self.out,
                grab=grab if grab is not None else (lambda u, p: True),
                describe=describe if describe is not None else (lambda p, k: "A car is in the driveway."))

    def test_success_returns_description_and_image_path(self) -> None:
        r = self._look()
        self.assertEqual(r["camera"], "front_door")
        self.assertEqual(r["description"], "A car is in the driveway.")
        self.assertTrue(r["image"].endswith(".jpg"))

    def test_unknown_camera(self) -> None:
        r = self._look(url=None)
        self.assertIn("no camera named", r["error"])
        self.assertNotIn("image", r)

    def test_no_vision_key(self) -> None:
        r = self._look(env={})
        self.assertIn("not configured", r["error"])

    def test_grab_failure(self) -> None:
        r = self._look(grab=lambda u, p: False)
        self.assertIn("could not get a picture", r["error"])

    def test_describe_failure(self) -> None:
        r = self._look(describe=lambda p, k: None)
        self.assertIn("could not describe", r["error"])

    def test_make_look_now_none_without_key_or_dir(self) -> None:
        self.assertIsNone(live_view.make_look_now("cams.yaml", {}, self.out))
        self.assertIsNone(live_view.make_look_now("cams.yaml", {"OPENAI_API_KEY": "k"}, ""))

    def test_make_look_now_builds_a_callable(self) -> None:
        self.assertTrue(callable(live_view.make_look_now("cams.yaml", {"OPENAI_API_KEY": "k"}, self.out)))


class LookNowZoneMaskTest(unittest.TestCase):
    """The picture is masked to the camera's watch zone before the vision model or the owner sees it."""

    def setUp(self) -> None:
        self.out = tempfile.mkdtemp()
        self.zones = os.path.join(tempfile.mkdtemp(), "zones.yaml")
        self.env = {"OPENAI_API_KEY": "key"}
        self.seen = []
        self.describe_calls = 0

    @staticmethod
    def _grab(url, path) -> bool:
        import cv2
        import numpy as np

        return bool(cv2.imwrite(path, np.full((48, 64, 3), 255, dtype=np.uint8)))

    def _describe(self, path, key):
        import cv2

        self.describe_calls += 1
        self.seen.append(cv2.imread(path))
        return "A car is in the driveway."

    def _look(self, camera="front_door"):
        with mock.patch.object(live_view, "_camera_url", return_value="rtsp://x"):
            return live_view.look_now(camera, "cams.yaml", self.env, self.out, grab=self._grab,
                                      describe=self._describe, zones_path=self.zones)

    def test_picture_is_masked_for_the_model_and_for_the_owner(self) -> None:
        import cv2

        from home_guard_project.data_collection import zones

        zones.save_zone("front_door", [(0, 0), (0.5, 0), (0.5, 1), (0, 1)], self.zones)
        r = self._look()
        sent = cv2.imread(r["image"])
        for img in (self.seen[0], sent):
            self.assertLess(int(img[:, 34:].max()), 30)
            self.assertGreater(int(img[:, :30].min()), 220)

    def test_camera_without_a_zone_is_unchanged(self) -> None:
        r = self._look()
        self.assertNotIn("error", r)
        self.assertGreater(int(self.seen[0].min()), 220)

    def test_fails_closed_when_the_masked_picture_cannot_be_written(self) -> None:
        from home_guard_project.data_collection import zones

        zones.save_zone("front_door", [(0, 0), (0.5, 0), (0.5, 1), (0, 1)], self.zones)
        with mock.patch("cv2.imwrite", return_value=False):
            r = self._look()
        self.assertIn("error", r)
        self.assertEqual(self.describe_calls, 0)
        self.assertEqual(os.listdir(self.out), [])


if __name__ == "__main__":
    unittest.main()
