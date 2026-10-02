from __future__ import annotations
import os
from pathlib import Path
import tempfile
import time
import unittest
from unittest import mock

import cv2
import numpy as np
from home_guard_project.box.preview import PreviewWriter, PreviewReader, camera_key
from home_guard_project.box.inference_preview import adapt_stream


class PreviewTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name)
        self.now = time.time() + 1
        self.writer = PreviewWriter(self.path, enabled=True, clock=lambda: self.now)
        self.reader = PreviewReader(self.path, clock=lambda: self.now)
        self.frame = np.full((80, 120, 3), 150, np.uint8)

    def test_viewer_gate_and_disabled_writer_do_not_encode(self):
        with mock.patch("cv2.imencode") as encode:
            self.assertFalse(self.writer.publish("front", self.frame))
            self.reader.touch()
            self.writer.enabled = False
            self.assertFalse(self.writer.publish("front", self.frame))
            encode.assert_not_called()
        self.writer.enabled = True
        self.now += 16
        self.assertFalse(self.writer.publish("front", self.frame))

    def test_throttle_is_per_camera(self):
        self.reader.touch()
        self.assertTrue(self.writer.publish("front", self.frame))
        self.assertFalse(self.writer.publish("front", self.frame))
        self.assertTrue(self.writer.publish("garden", self.frame))
        self.now += 0.49
        self.assertFalse(self.writer.publish("front", self.frame))
        self.now += 0.02
        self.assertTrue(self.writer.publish("front", self.frame))

    def test_atomic_replace_and_valid_jpeg(self):
        self.reader.touch()
        real_replace = os.replace

        def replace(temp, target):
            self.assertNotEqual(temp, target)
            decoded = cv2.imdecode(
                np.frombuffer(Path(temp).read_bytes(), np.uint8), cv2.IMREAD_COLOR
            )
            self.assertEqual(decoded.shape, self.frame.shape)
            real_replace(temp, target)

        with mock.patch(
            "home_guard_project.box.preview.os.replace", side_effect=replace
        ) as operation:
            self.assertTrue(self.writer.publish("front", self.frame))
            operation.assert_called_once()
        self.assertFalse(list(self.path.glob("*.tmp")))

    def test_windows_replace_failure_preserves_picture_and_retries(self):
        self.reader.touch()
        self.assertTrue(self.writer.publish("front", self.frame))
        before = self.reader.read("front")
        self.now += 1
        with mock.patch(
            "home_guard_project.box.preview.os.replace", side_effect=PermissionError
        ):
            self.assertFalse(self.writer.publish("front", np.zeros_like(self.frame)))
        self.assertEqual(before, self.reader.read("front"))
        self.assertFalse(list(self.path.glob("*.tmp")))
        self.now += 1
        self.assertTrue(self.writer.publish("front", np.zeros_like(self.frame)))
        self.assertNotEqual(before, self.reader.read("front"))

    def test_reader_missing_stale_and_future_pictures(self):
        self.assertIsNone(self.reader.read("front"))
        self.reader.touch()
        self.writer.publish("front", self.frame)
        self.assertIsNotNone(self.reader.read("front"))
        self.now += 13
        self.assertIsNone(self.reader.read("front"))
        self.now -= 30
        self.assertIsNone(self.reader.read("front"))

    def test_reader_closes_file_before_writer_replaces_it(self):
        self.reader.touch()
        self.writer.publish("front", self.frame)
        self.assertIsInstance(self.reader.read("front"), bytes)
        self.now += 1
        self.assertTrue(self.writer.publish("front", np.zeros_like(self.frame)))

    def test_frozen_capture_is_not_republished(self):
        self.reader.touch()
        self.assertTrue(self.writer.publish("front", self.frame, source=self.frame))
        self.now += 1
        self.assertFalse(self.writer.wanted("front", self.frame))
        self.assertFalse(self.writer.publish("front", self.frame, source=self.frame))
        fresh = self.frame.copy()
        self.assertTrue(self.writer.publish("front", fresh, source=fresh))

    def test_name_manifest_and_safe_paths(self):
        self.assertTrue(self.writer.set_cameras(["../front", "garden"]))
        self.assertEqual(self.reader.names(), ["../front", "garden"])
        self.assertNotIn("/", camera_key("../front"))
        (self.path / "cameras.json").write_text("broken")
        self.assertEqual(self.reader.names(), [])
        self.writer.enabled = False
        self.assertFalse(self.writer.set_cameras(["camera"]))

    def test_failed_encoder_does_not_publish(self):
        self.reader.touch()
        with mock.patch("cv2.imencode", return_value=(False, None)):
            self.assertFalse(self.writer.publish("front", self.frame))
        self.assertIsNone(self.reader.read("front"))

    def test_inference_adapter_reuses_existing_capture_and_preserves_read(self):
        frame = self.frame

        class Stream:
            def __init__(self, name, url):
                self.name = name
                self._frame = frame

            def read(self):
                return self._frame.copy()

        self.reader.touch()
        stream = adapt_stream(Stream, self.writer)("front", "synthetic")
        np.testing.assert_array_equal(stream.read(), frame)
        self.assertIsNotNone(self.reader.read("front"))
        self.assertEqual(self.reader.names(), ["front"])
        self.now += 1
        with mock.patch.object(self.writer, "publish") as publish:
            stream.read()
            publish.assert_not_called()

    def test_manifest_retries_after_a_sharing_failure(self):
        self.reader.touch()
        with mock.patch(
            "home_guard_project.box.preview.os.replace", side_effect=PermissionError
        ):
            self.assertFalse(self.writer.set_cameras(["front"]))
        self.assertTrue(self.writer.publish("front", self.frame))
        self.assertEqual(self.reader.names(), ["front"])

    def test_adapter_uses_actual_inference_read_without_opening_camera(self):
        import threading
        from home_guard_project.box import inference

        self.reader.touch()
        adapter = adapt_stream(inference._Stream, self.writer)
        stream = object.__new__(adapter)
        stream.name = "front"
        stream._frame = self.frame
        stream._lock = threading.Lock()
        with mock.patch(
            "cv2.VideoCapture", side_effect=AssertionError("camera access forbidden")
        ):
            np.testing.assert_array_equal(stream.read(), self.frame)
        self.assertIsNotNone(self.reader.read("front"))
