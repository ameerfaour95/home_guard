from __future__ import annotations

import json
import os
import tempfile
import unittest

import cv2
import numpy as np

from home_guard_project.box.alert_clips import ClipRing, alert_stem, encode_frame, write_alert_clip
from home_guard_project.box.archive import load_records

NOW = 1_800_000_000.0


def frame(value: int) -> np.ndarray:
    return np.full((48, 64, 3), value, dtype=np.uint8)


class ClipRingTest(unittest.TestCase):
    def test_keeps_only_the_last_seconds_at_its_frame_rate(self) -> None:
        ring = ClipRing(seconds=2, fps=5)
        self.assertTrue(ring.wants(NOW))
        for i in range(30):                       # 6 seconds of frames
            ring.add(NOW + i * 0.2, b"x%d" % i)
        frames = ring.between(NOW, NOW + 100)
        self.assertEqual(len(frames), 10)
        self.assertEqual(frames[-1][1], b"x29")
        self.assertFalse(ring.wants(frames[-1][0] + 0.1))
        self.assertTrue(ring.wants(frames[-1][0] + 0.2))

    def test_between_selects_by_time_and_empty_frames_are_dropped(self) -> None:
        ring = ClipRing(seconds=10, fps=5)
        ring.add(NOW, b"")
        for i in range(10):
            ring.add(NOW + i, b"f%d" % i)
        self.assertEqual([d for _, d in ring.between(NOW + 2, NOW + 4)], [b"f2", b"f3", b"f4"])


class WriteAlertClipTest(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = os.path.join(tmp.name, "production_multi")
        self.frames = [(NOW + i * 0.2, encode_frame(frame(10 * i))) for i in range(20)]
        self.alert = {"summary": "a person at the door", "alert_command": "[send_message]", "alert_reason": ""}

    def test_stem(self) -> None:
        self.assertEqual(alert_stem("front_door", 1790951614.86), "front_door_1790951614_alert")

    def test_writes_a_playable_clip_and_then_a_meta_the_look_up_can_read(self) -> None:
        stem = alert_stem("front_door", NOW)
        meta_path = write_alert_clip(self.root, "front_door", stem, self.frames, self.alert, h264=False)

        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
        self.assertEqual((meta["camera_name"], meta["kind"], meta["frames_written"]), ("front_door", "alert", 20))
        self.assertEqual(meta["alert"], self.alert)
        self.assertEqual(meta["buffer"]["store_size"], [64, 48])
        self.assertAlmostEqual(meta["fps_estimated"], 5.0, places=1)
        self.assertEqual(meta["clip_end_ts"], self.frames[-1][0])

        clip = os.path.join(self.root, meta["clip_path"].replace("\\", os.sep))
        capture = cv2.VideoCapture(clip)
        self.assertEqual(int(capture.get(cv2.CAP_PROP_FRAME_COUNT)), 20)
        capture.release()
        self.assertFalse(os.path.exists(meta_path + ".tmp"))

        (record,) = load_records([self.root])
        self.assertEqual((record.alert_id, record.summary), (stem, "a person at the door"))
        self.assertEqual(record.clip_path, clip)

    def test_no_frames_writes_nothing(self) -> None:
        self.assertIsNone(write_alert_clip(self.root, "front_door", "x_1_alert", [], self.alert, h264=False))
        self.assertIsNone(write_alert_clip(self.root, "front_door", "x_1_alert", [(NOW, b"not a jpeg")], self.alert,
                                           h264=False))
        self.assertEqual(load_records([self.root]), [])


if __name__ == "__main__":
    unittest.main()
