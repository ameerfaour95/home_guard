"""A damaged main stream never reaches the Eye (box/frame_check.py; 2026-10-12: the 22:02 🔴 "covers the camera lens"
was a woman at the door, every main-stream frame of her crop a ghost or a grey field, the sub stream clean)."""
from __future__ import annotations

import unittest
from types import SimpleNamespace

import cv2
import numpy as np

from home_guard_project.box import frame_check as fc
from home_guard_project.box import inference

MAIN, SUB = (2592, 1520), (704, 576)
CROP = (900, 400, 1500, 1100)


def scene(t: float) -> np.ndarray:
    """A yard at sub size: tiles, a wall, and a person (a dark box) walking right."""
    rng = np.random.default_rng(7)
    img = (rng.integers(90, 160, (SUB[1], SUB[0], 3))).astype(np.uint8)
    img = cv2.GaussianBlur(img, (9, 9), 0)
    for x in range(0, SUB[0], 40):
        cv2.line(img, (x, 0), (x + 120, SUB[1]), (70, 70, 70), 2)
    cv2.rectangle(img, (0, 0), (SUB[0], 80), (40, 40, 40), -1)
    x = int(250 + 60 * t)
    cv2.rectangle(img, (x, 160), (x + 30, 280), (20, 20, 25), -1)
    cv2.circle(img, (x + 15, 145), 14, (30, 40, 60), -1)
    return img


def clip(n: int, seconds: float, size=SUB):
    return [cv2.resize(scene(i * seconds / (n - 1)), size) for i in range(n)]


class FrameCheckTests(unittest.TestCase):
    def check(self, main):
        sub = clip(30, 3.0)
        values = fc.scores(main, 0.0, 3.0, [CROP] * len(main), sub, 0.0, 3.0)
        return values, fc.damaged(values)

    def test_a_whole_main_stream_passes(self):
        values, bad = self.check(clip(15, 3.0, MAIN))
        self.assertFalse(bad, values)
        self.assertLess(max(values), fc.FRAME_ODD)

    def test_a_grey_field_is_damaged(self):
        main = clip(15, 3.0, MAIN)
        for i in range(10, 15):                      # the decoder lost the stream: a flat grey-purple picture
            main[i] = np.full_like(main[i], (150, 130, 140))
            main[i][700:716, 1100:1116] = 30
        self.assertTrue(self.check(main)[1])

    def test_a_ghost_is_damaged(self):
        main = clip(15, 3.0, MAIN)
        for i in range(4, 9):                        # the person smeared over a picture of a second earlier,
            ghost = cv2.addWeighted(main[i], 0.5, np.roll(main[i], 220, axis=1), 0.5, 0)   # blocks where data was lost
            ghost[800:1000:16, :] = ghost[780:980:16, :]
            ghost[600:1000, 1000:1400] = cv2.resize(cv2.resize(ghost[600:1000, 1000:1400], (25, 25)), (400, 400),
                                                    interpolation=cv2.INTER_NEAREST)
            main[i] = ghost
        values, bad = self.check(main)
        self.assertTrue(bad, values)

    def test_a_damaged_alert_is_described_from_the_sub_stream(self):
        sub = clip(30, 3.0)
        main = clip(15, 3.0, MAIN)
        main[7] = np.full_like(main[7], 140)
        job = SimpleNamespace(camera="cam", crop=SimpleNamespace(crops=[CROP] * 15), input_check=None)
        self.assertTrue(inference._main_damaged(job, (main, 0.0, 3.0, 5.0), sub, 0.0, 3.0))
        self.assertTrue(job.input_check["damaged"])
        job.crop.crops = [CROP] * 15
        self.assertFalse(inference._main_damaged(job, (clip(15, 3.0, MAIN), 0.0, 3.0, 5.0), sub, 0.0, 3.0))

    def test_a_failed_check_keeps_the_crop(self):
        job = SimpleNamespace(camera="cam", crop=None, input_check=None)
        self.assertFalse(inference._main_damaged(job, ([np.zeros((4, 4, 3), np.uint8)], 0, 1, 5), [], 0, 1))


if __name__ == "__main__":
    unittest.main()
