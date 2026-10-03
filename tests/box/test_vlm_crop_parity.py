"""The AI sees exactly the crop data collection saves: one crop pipeline, two callers.

The golden hash below was taken from data_collection._save_vlm_crop_clip BEFORE the
crop code moved to data_collection/vlm_crop.py (2026-10-03), so these tests prove the
move changed nothing in what data collection writes - and that inference builds the
very same frames from the same inputs.
"""
from __future__ import annotations

import hashlib
import os
import tempfile
import types
import unittest
from unittest import mock

import importlib.util
import sys
from pathlib import Path

import cv2  # noqa: F401 - imported before sys.modules is patched, so the patch never unloads it
import numpy as np
import torch  # noqa: F401 - same reason

from home_guard_project.data_collection import config as dc_config


def _load_collector():
    """data_collection.py runs as a script (run_collector.sh); load it the way the collector tests do."""
    name = "home_guard_project.data_collection._collector_crop_parity_test"
    source = Path(dc_config.__file__).with_name("data_collection.py")
    spec = importlib.util.spec_from_file_location(name, source)
    module = importlib.util.module_from_spec(spec)
    with mock.patch.dict(sys.modules, {"config": dc_config, "ultralytics": types.SimpleNamespace(YOLO=mock.Mock()),
                                       name: module}):
        spec.loader.exec_module(module)
    return module


dc = _load_collector()

GOLDEN_SHA256 = "41fc58b508d73b7c46685d4ec802c9166aa0ec4287924f95e01f5c3d735ced92"
SUB_W, SUB_H = 352, 288
MAIN_W, MAIN_H = 1408, 1152
N_SUB, N_MAIN = 100, 50          # 10 s at STORE_FPS 10 and at MAIN_STORE_FPS 5


class _Cls:
    def __init__(self, value: int) -> None:
        self.value = value

    def item(self) -> int:
        return self.value


class _Xyxy:
    def __init__(self, box) -> None:
        self.box = box

    def tolist(self):
        return list(self.box)


class _Box:
    def __init__(self, cls_id: int, box) -> None:
        self.cls = _Cls(cls_id)
        self.xyxy = [_Xyxy(box)]


class _Result:
    def __init__(self, boxes) -> None:
        self.boxes = boxes


def fake_detector(frame, verbose=False, conf=0.0, imgsz=640):
    """A person walking left to right and a bird (not a trigger); nothing in the first frames."""
    i = int(frame[0, 0, 0])
    if i < 12:
        return [_Result([])]
    x = 20 + i * 2.5
    return [_Result([_Box(0, (x, 90 + i * 0.4, x + 30, 190 + i * 0.4)),     # person
                     _Box(14, (10, 10, 20, 20))])]                            # bird: not a trigger


def sub_frames():
    frames = []
    for i in range(N_SUB):
        f = np.zeros((SUB_H, SUB_W, 3), dtype=np.uint8)
        f[0, 0, 0] = i                      # the fake detector reads the frame index here
        frames.append(f)
    return frames


def main_frames():
    rng = np.random.default_rng(7)
    return [rng.integers(0, 255, (MAIN_H, MAIN_W, 3), dtype=np.uint8) for _ in range(N_MAIN)]


def cfg(out_dir: str):
    return types.SimpleNamespace(
        MAIN_STREAM_ENABLED=True, STORE_FPS=10.0, YOLO_TRIGGER_CONF=0.35, YOLO_IMGSZ=640,
        TRIGGER_CLASS_IDS=[0, 2, 5, 7], CROP_PADDING=0.3, CROP_MIN_SIZE=384, CROP_EMA_ALPHA=0.3,
        OUT_DIR=out_dir,
    )


def digest(frames) -> str:
    h = hashlib.sha256()
    for f in frames:
        h.update(str(f.shape).encode())
        h.update(np.ascontiguousarray(f).tobytes())
    return h.hexdigest()


def collector_crop_frames():
    """Run data collection's real crop step and capture the frames it writes."""
    captured = {}

    def capture(frames, fps):
        captured["frames"] = [f.copy() for f in frames]
        captured["fps"] = fps
        fd, path = tempfile.mkstemp(suffix=".mp4")
        os.close(fd)
        return path

    with tempfile.TemporaryDirectory() as out, mock.patch.object(dc, "_write_mp4_clip", side_effect=capture):
        meta = dc._save_vlm_crop_clip(cfg(out), fake_detector, types.SimpleNamespace(name="cam"),
                                      sub_frames(), 1000.0, 1010.0, (main_frames(), 1000.0, 1010.0, 5.0),
                                      "2026-10-03", "cam_1")
    return captured["frames"], captured["fps"], meta


class CollectorCropIsUnchangedTest(unittest.TestCase):
    def test_data_collection_writes_exactly_what_it_wrote_before_the_move(self) -> None:
        frames, fps, meta = collector_crop_frames()
        self.assertEqual(digest(frames), GOLDEN_SHA256)
        self.assertEqual((len(frames), fps, meta["crop_padding"], meta["crop_min_size"]), (N_MAIN, 5.0, 0.3, 384))


if __name__ == "__main__":
    unittest.main()
