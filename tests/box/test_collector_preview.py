"""Exercise the real collector main loop with synthetic capture/detection only.

No config credentials are read, no weights loaded, no camera/network calls made.
JPEG encoding, frame plotting/resizing, viewer gating and disk transport are real.
"""

from contextlib import ExitStack
import importlib.util
import logging
import math
from pathlib import Path
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np
from home_guard_project.data_collection.config import Config
from home_guard_project.data_collection import config
from home_guard_project.box import preview


class CollectorPreviewTest(unittest.TestCase):
    def test_real_main_loop_publishes_only_for_viewer_and_new_capture(self):
        for enabled, viewer, rounded_clock in (
            (True, True, False), (True, False, False), (False, True, False),
            (True, True, True),
        ):
            with (
                self.subTest(enabled=enabled, viewer=viewer, rounded_clock=rounded_clock),
                tempfile.TemporaryDirectory() as directory,
            ):
                root = Path(directory)
                frame = np.full((900, 1200, 3), 100, np.uint8)
                annotated = np.full_like(frame, 210)
                cfg = Config(
                    OUT_DIR=str(root / "clips"),
                    CAMERAS={"front": "synthetic"},
                    DEVICE="cpu",
                    SHOW_WINDOWS=False,
                    PREVIEW_ENABLED=enabled,
                    SHOW_PLOTTED_BOXES=True,
                    MAIN_STREAM_ENABLED=False,
                    RUN_VLM_ON_SAVED_CLIPS=False,
                    RANDOM_CLIP_ENABLED=False,
                    YOLO_EVERY_N_FRAMES_CPU=1,
                )
                stream = mock.Mock()
                # Initial readiness, one actual main-loop frame, then exit safely.
                stream.get_latest.side_effect = [
                    (True, frame),
                    (True, frame),
                    KeyboardInterrupt,
                ]
                stream.is_opened.return_value = True
                result = SimpleNamespace(
                    boxes=[], plot=mock.Mock(return_value=annotated)
                )
                detector = mock.Mock(return_value=[result])
                fake_ultralytics = SimpleNamespace(
                    YOLO=mock.Mock(return_value=detector)
                )
                name = "home_guard_project.data_collection._collector_preview_test"
                source = Path(config.__file__).with_name("data_collection.py")
                spec = importlib.util.spec_from_file_location(name, source)
                module = importlib.util.module_from_spec(spec)
                def one_ulp_behind_file():
                    # Deterministically reproduce Windows FILETIME conversion:
                    # first the viewer marker, then the just-published JPEG.
                    image = root / "preview" / (preview.camera_key("front") + ".jpg")
                    path = image if image.exists() else root / "preview" / "viewer.alive"
                    return math.nextafter(path.stat().st_mtime, -math.inf)

                clock = one_ulp_behind_file if rounded_clock else time.time
                writer = preview.PreviewWriter(root / "preview", enabled=enabled, clock=clock)
                reader = preview.PreviewReader(root / "preview", clock=clock)
                if viewer:
                    reader.touch()
                old_handlers = list(logging.getLogger().handlers)
                old_level = logging.getLogger().level
                try:
                    with ExitStack() as patches:
                        patches.enter_context(
                            mock.patch.dict(
                                sys.modules,
                                {
                                    "config": config,
                                    "ultralytics": fake_ultralytics,
                                    name: module,
                                },
                            )
                        )
                        spec.loader.exec_module(module)
                        patches.enter_context(
                            mock.patch.object(module, "load_config", return_value=cfg)
                        )
                        patches.enter_context(
                            mock.patch.object(
                                module, "SubStreamThread", return_value=stream
                            )
                        )
                        patches.enter_context(
                            mock.patch.object(
                                preview, "PreviewWriter", return_value=writer
                            )
                        )
                        with self.assertRaises(KeyboardInterrupt):
                            module.main()
                    if enabled and viewer:
                        # Distinguish failure to publish from rejection by the reader.
                        self.assertTrue((root / "preview" / (preview.camera_key("front") + ".jpg")).exists())
                    data = reader.read("front")
                    self.assertEqual(data is not None, enabled and viewer)
                    if data:
                        import cv2

                        decoded = cv2.imdecode(
                            np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR
                        )
                        self.assertEqual(max(decoded.shape[:2]), 1200)
                        self.assertGreater(
                            decoded.mean(), 200
                        )  # annotated image, not raw
                        result.plot.assert_called_once()
                    else:
                        result.plot.assert_not_called()
                    stream.release.assert_called_once()
                    fake_ultralytics.YOLO.assert_called_once()
                finally:
                    logging.getLogger().handlers = old_handlers
                    logging.getLogger().setLevel(old_level)
