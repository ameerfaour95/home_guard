"""A frame the watch zone cannot mask is dropped, never kept unmasked, and the reader lives on."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest import mock

import cv2  # noqa: F401 - imported before sys.modules is patched, so the patch never unloads it
import numpy as np
import torch  # noqa: F401 - same reason

from home_guard_project.data_collection import config
from home_guard_project.data_collection.zones import ZoneMask

LEFT_HALF = [(0.0, 0.0), (0.5, 0.0), (0.5, 1.0), (0.0, 1.0)]


def _load_collector():
    name = "home_guard_project.data_collection._collector_mask_guard_test"
    source = Path(config.__file__).with_name("data_collection.py")
    spec = importlib.util.spec_from_file_location(name, source)
    module = importlib.util.module_from_spec(spec)
    fake_ultralytics = SimpleNamespace(YOLO=mock.Mock())
    with mock.patch.dict(sys.modules, {"config": config, "ultralytics": fake_ultralytics, name: module}):
        spec.loader.exec_module(module)
    return module


class MaskGuardTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.dc = _load_collector()

    def test_a_mask_that_raises_drops_the_frame_and_warns_once(self) -> None:
        mask = ZoneMask(LEFT_HALF)
        frame = np.full((4, 4, 3), 255, np.uint8)
        state = self.dc._MaskFailState("front sub-stream")
        with mock.patch.object(mask, "apply", side_effect=RuntimeError("bad shape")), \
                self.assertLogs(self.dc.log, level="WARNING") as logs:
            self.assertIsNone(self.dc._masked_or_none(mask, frame, state))
            self.assertIsNone(self.dc._masked_or_none(mask, frame, state))
        self.assertEqual(len(logs.records), 1)
        self.assertIn("watch zone could not be applied", logs.output[0])
        self.assertIn("bad shape", logs.output[0])

    def test_a_working_mask_returns_the_masked_frame(self) -> None:
        mask = ZoneMask(LEFT_HALF)
        frame = np.full((4, 4, 3), 255, np.uint8)
        out = self.dc._masked_or_none(mask, frame, self.dc._MaskFailState("front sub-stream"))
        self.assertIsNotNone(out)
        self.assertEqual(int(out[:, 3].max()), 0)      # right edge blacked out
        self.assertEqual(int(out[:, 0].min()), 255)    # left edge kept

    def test_both_readers_go_through_the_guard(self) -> None:
        import inspect
        for cls in (self.dc.SubStreamThread, self.dc.MainStreamThread):
            src = inspect.getsource(cls._reader)
            self.assertIn("_masked_or_none(", src, cls.__name__)
            self.assertNotIn("self.mask.apply(", src, cls.__name__)


    def test_each_reader_labels_its_warning_with_the_camera_name_not_the_url(self) -> None:
        cfg = config.Config()
        url = "rtsp://admin:s3cret@10.0.0.9/s0/live"
        self.assertEqual(self.dc.SubStreamThread.__init__.__defaults__[-1], "")   # name stays optional
        for cls, label in ((self.dc.SubStreamThread, "front sub-stream"), (self.dc.MainStreamThread, "front main-stream")):
            with (
                mock.patch.object(cls, "_open_capture", lambda self: None),   # no real RTSP connect
                mock.patch.object(self.dc.threading, "Thread", mock.Mock()),  # no reader thread
            ):
                reader = cls(cfg, url, mask=ZoneMask(None), name="front")
            self.assertEqual(reader._mask_state.label, label)
            self.assertNotIn("s3cret", reader._mask_state.label)


if __name__ == "__main__":
    unittest.main()
