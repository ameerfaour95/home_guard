from __future__ import annotations

import json
import unittest
from types import SimpleNamespace

import numpy as np

from home_guard_project.box.app.remote_cameras import RemoteCameras
from home_guard_project.box.find_cameras import looks_blank

# What the box really prints: decoder messages, then the result as indented JSON.
REAL_OUTPUT = (
    "[hevc @ 000001f0aa] Could not find ref with POC 12\n"
    "[rtsp @ 000001f0bb] method DESCRIBE failed: 500 ServerInternal\n"
    + json.dumps({"snapshots": [
        {"name": "house_ch2", "file": "C:\\Users\\ameer\\hg_snapshots\\house_ch2.jpg", "ok": True},
        {"name": "house_ch3", "file": "", "ok": False},
    ]}, indent=2)
    + "\n"
)


class RemoteCameraResultTest(unittest.TestCase):
    def setUp(self) -> None:
        self.cameras = RemoteCameras("ameer@100.64.0.9", runner=object())

    def _parse(self, stdout: str, key: str):
        return self.cameras.parse(SimpleNamespace(stdout=stdout, returncode=0), key)

    def test_the_boxs_indented_result_is_read(self) -> None:
        data = self._parse(REAL_OUTPUT, "snapshots")
        self.assertEqual([row["name"] for row in data["snapshots"]], ["house_ch2", "house_ch3"])

    def test_a_result_on_one_line_is_still_read(self) -> None:
        data = self._parse('note\n{"active": ["a"], "disabled": []}\n', "active")
        self.assertEqual(data["active"], ["a"])

    def test_an_error_result_and_no_result_both_fail(self) -> None:
        with self.assertRaises(RuntimeError):
            self._parse(json.dumps({"error": "duplicate name"}, indent=2), "active")
        with self.assertRaises(RuntimeError):
            self._parse("nothing useful here\n", "snapshots")


class BlankFrameTest(unittest.TestCase):
    def test_the_grey_first_frame_of_a_stream_is_blank(self) -> None:
        grey = np.full((540, 960, 3), 128, dtype=np.uint8)
        grey[::37, ::41] = 150                         # a few specks, as the decoder leaves
        self.assertTrue(looks_blank(grey))

    def test_real_pictures_are_not_blank(self) -> None:
        rng = np.random.default_rng(1)
        scene = rng.integers(0, 255, size=(540, 960, 3), dtype=np.uint8)
        night = np.full((540, 960, 3), 12, dtype=np.uint8)      # dark, but not mid-grey
        self.assertFalse(looks_blank(scene))
        self.assertFalse(looks_blank(night))


if __name__ == "__main__":
    unittest.main()
