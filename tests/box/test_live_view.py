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


if __name__ == "__main__":
    unittest.main()
