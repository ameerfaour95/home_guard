# tests/box/test_brain_media.py
from __future__ import annotations

import os
import tempfile
import unittest
from unittest import mock

import numpy as np
import yaml

from home_guard_project.box.brain.media import bounds_text, clip_frames, clip_frames_at, cut_segment, record_live
from home_guard_project.box.brain.media import grab_photo
from home_guard_project.box import live_view


class FakeCapture:
    def __init__(self, frames: int) -> None:
        self.left = frames

    def read(self):
        if self.left <= 0:
            return False, None
        self.left -= 1
        return True, np.full((48, 64, 3), 120, dtype=np.uint8)

    def release(self) -> None:
        pass


class Clock:
    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        self.t += 0.1
        return self.t


class MediaTest(unittest.TestCase):
    def setUp(self) -> None:
        self.dir = tempfile.mkdtemp()
        self.cameras = os.path.join(self.dir, "cameras.yaml")
        with open(self.cameras, "w", encoding="utf-8") as f:
            yaml.safe_dump({"cameras": {"gate": "rtsp://x"}}, f)
        self.zones = os.path.join(self.dir, "zones.yaml")

    def test_record_live_writes_a_clip_of_the_asked_length(self) -> None:
        out = record_live("gate", 2, self.dir, self.cameras, zones_path=self.zones, now=lambda: 1000.0,
                          open_capture=lambda url: FakeCapture(100), clock=Clock(), h264=False)
        self.assertTrue(out["ok"], out)
        self.assertTrue(os.path.isfile(out["path"]))
        self.assertEqual((out["start"], out["end"]), (1000.0, 1002.0))
        self.assertGreater(len(clip_frames(out["path"], count=3)), 0)

    def test_record_live_errors(self) -> None:
        self.assertEqual(record_live("nope", 2, self.dir, self.cameras, zones_path=self.zones)["error"],
                         "camera_unknown")
        out = record_live("gate", 2, self.dir, self.cameras, zones_path=self.zones,
                          open_capture=lambda url: FakeCapture(0), clock=Clock(), h264=False)
        self.assertEqual(out["error"], "camera_offline")

    def test_record_live_refuses_an_unreadable_zone(self) -> None:
        with open(self.zones, "w", encoding="utf-8") as f:
            f.write("gate: [unclosed")
        out = record_live("gate", 2, self.dir, self.cameras, zones_path=self.zones,
                          open_capture=lambda url: FakeCapture(100), clock=Clock(), h264=False)
        self.assertEqual(out["error"], "error")

    def test_cut_segment_builds_the_ffmpeg_call(self) -> None:
        calls = []

        def run(cmd):
            calls.append(cmd)
            with open(cmd[-1], "wb") as f:
                f.write(b"mp4")
            return 0

        out = os.path.join(self.dir, "seg.mp4")
        span = cut_segment("clip.mp4", 100.0, 110.0, 102.5, 4, out, run=run, ffmpeg="ffmpeg")
        self.assertEqual(span, (102.5, 106.5))
        self.assertEqual(calls[0][:5], ["ffmpeg", "-y", "-ss", "2.50", "-i"])
        self.assertEqual(cut_segment("clip.mp4", 100.0, 110.0, 97.0, 5, out, run=run, ffmpeg="ffmpeg"),
                         (100.0, 102.0))                       # starts before the clip: only the part inside
        self.assertEqual(calls[1][calls[1].index("-t") + 1], "2.00")
        self.assertIsNone(cut_segment("clip.mp4", 100.0, 110.0, 90.0, 5, out, run=run, ffmpeg="ffmpeg"))
        self.assertIsNone(cut_segment("clip.mp4", 100.0, 110.0, 102.5, 4, out, run=lambda cmd: 1, ffmpeg="ffmpeg"))

    def test_cut_segment_removes_a_partial_file_on_failure(self) -> None:
        out = os.path.join(self.dir, "partial.mp4")

        def bad_exit(cmd):
            with open(cmd[-1], "wb") as f:
                f.write(b"half")
            return 1

        def boom(cmd):
            with open(cmd[-1], "wb") as f:
                f.write(b"half")
            raise RuntimeError("killed")

        for run in (bad_exit, boom):
            with self.subTest(run=run.__name__):
                self.assertIsNone(cut_segment("clip.mp4", 100.0, 110.0, 102.5, 4, out, run=run, ffmpeg="ffmpeg"))
                self.assertFalse(os.path.exists(out))

    def test_strict_zone_masks_a_numeric_camera_key(self) -> None:
        with open(self.zones, "w", encoding="utf-8") as f:
            f.write("zones:\n  101: [[0,0],[0.5,0],[0.5,1],[0,1]]\n")
        poly, readable = live_view.strict_zone("101", self.zones)
        self.assertTrue(readable)
        self.assertIsNotNone(poly)
        with open(self.zones, "w", encoding="utf-8") as f:
            f.write("zones:\n  101: [[0,0],[1,1]]\n")
        self.assertEqual(live_view.strict_zone("101", self.zones), (None, False))

    def test_bounds_text(self) -> None:
        self.assertRegex(bounds_text(1000.0, 1010.0), r"^\d\d:\d\d:\d\d–\d\d:\d\d:\d\d$")

    def test_strict_zone_rejects_wrong_types_and_invalid_utf8(self) -> None:
        for payload in (b"[]", b"false", b"zones: []", b"zones: {gate: nope}", b"\xff",
                        b"zones: {gate: [[0, 0], [.nan, 0], [1, 1]]}"):
            with self.subTest(payload=payload):
                with open(self.zones, "wb") as f:
                    f.write(payload)
                self.assertEqual(live_view.strict_zone("gate", self.zones), (None, False))
                grab = mock.Mock()
                self.assertIn("error", grab_photo("gate", self.dir, self.cameras, self.zones, grab=grab))
                grab.assert_not_called()

    def test_grab_photo_masks_and_uses_unique_names(self) -> None:
        import cv2
        with open(self.zones, "w", encoding="utf-8") as f:
            yaml.safe_dump({"zones": {"gate": [[0, 0], [.5, 0], [.5, 1], [0, 1]]}}, f)
        def grab(url, path):
            return cv2.imwrite(path, np.full((48, 64, 3), 255, dtype=np.uint8))
        paths = [grab_photo("gate", self.dir, self.cameras, self.zones,
                            now=lambda: 1000, grab=grab)["image"] for _ in range(2)]
        self.assertNotEqual(*paths)
        for path in paths:
            self.assertLess(int(cv2.imread(path)[:, 34:].max()), 30)

    def test_capture_callbacks_and_bad_config_never_escape(self) -> None:
        def broken_grab(url, path):
            with open(path, "wb") as f:
                f.write(b"unmasked")
            raise ValueError("broken grab")
        self.assertIn("error", grab_photo("gate", self.dir, self.cameras, self.zones, grab=broken_grab))
        self.assertFalse(any(p.endswith(".jpg") for p in os.listdir(self.dir)))
        self.assertIn("error", live_view.look_now("gate", self.cameras, [], self.dir, zones_path=self.zones))
        for payload in (b"[]", b"cameras: {gate: [bad]}", b"\xff"):
            with self.subTest(payload=payload):
                with open(self.cameras, "wb") as f:
                    f.write(payload)
                self.assertIn("error", grab_photo("gate", self.dir, self.cameras, self.zones,
                                                  grab=mock.Mock(side_effect=AssertionError("no capture"))))
                self.assertFalse(record_live("gate", 1, self.dir, self.cameras, zones_path=self.zones)["ok"])

    def test_bad_numbers_do_not_start_recording_or_cutting(self) -> None:
        for value in ("bad", [], float("nan"), float("inf"), -1, 0):
            with self.subTest(value=value):
                capture = mock.Mock(side_effect=AssertionError("no capture"))
                self.assertFalse(record_live("gate", value, self.dir, self.cameras, self.zones,
                                             open_capture=capture)["ok"])
                self.assertFalse(record_live("gate", 1, self.dir, self.cameras, self.zones,
                                             fps=value, open_capture=capture)["ok"])
                capture.assert_not_called()
                run = mock.Mock()
                self.assertIsNone(cut_segment("x", 100, 110, 100, value, "out", run=run, ffmpeg="ffmpeg"))
                run.assert_not_called()
        self.assertEqual(bounds_text(float("nan"), "bad"), "")

    def test_clip_frames_at_spreads_frames_over_a_time_range(self) -> None:
        class Capture(FakeCapture):
            def get(self, prop):
                return 10.0                                   # frames per second

        frames = clip_frames_at("x", count=4, open_video=lambda _: Capture(100))     # a 10-second clip
        self.assertEqual([sec for sec, _ in frames], [0.0, 3.3, 6.6, 9.9])
        self.assertTrue(all(isinstance(jpg, bytes) and jpg for _, jpg in frames))
        part = clip_frames_at("x", count=3, start_sec=2.0, end_sec=4.0, open_video=lambda _: Capture(100))
        self.assertEqual([sec for sec, _ in part], [2.0, 3.0, 4.0])
        self.assertEqual(clip_frames_at("x", count=3, start_sec=50, open_video=lambda _: Capture(100)), [])
        self.assertEqual(len(clip_frames_at("x", count=8, open_video=lambda _: FakeCapture(3))), 3)   # no fps
        self.assertEqual(clip_frames_at("x", count=0, open_video=lambda _: Capture(10)), [])

    def test_clip_frames_releases_on_bad_frame_and_bad_count(self) -> None:
        capture = mock.Mock()
        capture.read.side_effect = ValueError("bad frame")
        self.assertEqual(clip_frames("x", open_video=lambda _: capture), [])
        capture.release.assert_called_once()
        self.assertEqual(clip_frames("x", count="bad", open_video=lambda _: FakeCapture(1)), [])

    def test_recording_masks_and_releases_busy_after_failure(self) -> None:
        import cv2
        with open(self.zones, "w", encoding="utf-8") as f:
            yaml.safe_dump({"zones": {"gate": [[0, 0], [.5, 0], [.5, 1], [0, 1]]}}, f)
        cap = mock.Mock(wraps=FakeCapture(100))
        with mock.patch("home_guard_project.data_collection.zones.ZoneMask.apply", side_effect=ValueError("bad mask")):
            self.assertFalse(record_live("gate", 1, self.dir, self.cameras, self.zones,
                                         open_capture=lambda _: cap, clock=Clock(), h264=False)["ok"])
        cap.release.assert_called_once()
        out = record_live("gate", 1, self.dir, self.cameras, self.zones,
                          open_capture=lambda _: FakeCapture(100), clock=Clock(), h264=False)
        self.assertTrue(out["ok"], out)
        for jpeg in clip_frames(out["path"]):
            frame = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
            self.assertLess(int(frame[:, 36:].max()), 30)

    def test_recording_busy_and_failed_writer(self) -> None:
        from home_guard_project.box.brain import media
        with mock.patch.object(media, "_busy", {"gate"}):
            self.assertEqual(record_live("gate", 1, self.dir, self.cameras)["error"], "busy")
        writer = mock.Mock()
        writer.isOpened.return_value = False
        with mock.patch("cv2.VideoWriter", return_value=writer):
            out = record_live("gate", 1, self.dir, self.cameras, self.zones,
                              open_capture=lambda _: FakeCapture(100), clock=Clock(), h264=False)
        self.assertFalse(out["ok"])
        writer.release.assert_called_once()

    def test_recording_duration_starts_after_capture_opens(self) -> None:
        clock = Clock()
        def open_capture(url):
            clock.t += 10
            return FakeCapture(100)
        out = record_live("gate", 1, self.dir, self.cameras, self.zones,
                          now=lambda: 1000 + clock.t, open_capture=open_capture, clock=clock, h264=False)
        self.assertTrue(out["ok"], out)
        self.assertEqual((out["start"], out["end"]), (1010, 1011))


if __name__ == "__main__":
    unittest.main()
