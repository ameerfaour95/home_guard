"""crop_policy clip_window: ONE still window per clip, so the person moves inside a still view.

The old per-frame crop panned and zoomed with the person (2026-10-09, seen while tagging); a video model reads
that as camera motion and loses the person's real movement. These tests pin the window's rules: every backed-up
box of the clip inside it, one-off boxes left out, padding of at least a quarter of a person, at most 2:1 grown
with more scene (never stretched), the whole frame when it covers over 80% of it, and the same box for every frame.
"""
from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np

from home_guard_project.data_collection import model_input, vlm_crop

MAIN_W, MAIN_H = 2592, 1520
SUB_W, SUB_H = 704, 576
SX, SY = MAIN_W / SUB_W, MAIN_H / SUB_H


def settings(**kw):
    base = dict(store_fps=10.0, trigger_class_ids=(0, 2, 5, 7), conf=0.35, imgsz=640, padding=0.3, min_size=384,
                ema_alpha=0.3)
    base.update(kw)
    return vlm_crop.CropSettings(**base)


class _Box:
    def __init__(self, cls_id, box):
        self.cls = cls_id
        self.xyxy = [np.array(box, dtype=float)]


def detector_from(boxes_at):
    """A fake YOLO: frame[0, 0, 0] (+ 256 * frame[0, 0, 1]) is the sub frame's index; boxes_at(i) its sub boxes."""
    calls = []

    def detect(frame, verbose=False, conf=0.0, imgsz=640):
        i = int(frame[0, 0, 0]) + 256 * int(frame[0, 0, 1])
        calls.append(i)
        return [SimpleNamespace(boxes=[_Box(c, b) for c, b in boxes_at(i)])]
    detect.calls = calls
    return detect


def subs(n=100):
    out = []
    for i in range(n):
        f = np.zeros((SUB_H, SUB_W, 3), np.uint8)
        f[0, 0, 0], f[0, 0, 1] = i % 256, i // 256
        out.append(f)
    return out


def mains(n=50):
    rng = np.random.default_rng(3)
    return [rng.integers(0, 255, (MAIN_H, MAIN_W, 3), dtype=np.uint8) for _ in range(n)]


def walk(i):
    """A person (sub pixels 40 wide, 100 tall) walking from x=200 to x=300 over the clip."""
    x = 200 + i
    return [(0, (x, 300, x + 40, 400))]


def main_box(b):
    return (b[0] * SX, b[1] * SY, b[2] * SX, b[3] * SY)


class ClipWindowTest(unittest.TestCase):
    def test_one_window_for_every_frame_and_the_frames_are_cut_not_resized(self):
        main = mains()
        r = vlm_crop.crop_clip(detector_from(walk), settings(), subs(), 1000., 1010., main, 1000.)
        self.assertEqual(r.policy, vlm_crop.POLICY_CLIP_WINDOW)
        self.assertEqual(len(set(r.crops)), 1)
        x1, y1, x2, y2 = r.first_crop
        self.assertEqual((r.width, r.height), (x2 - x1, y2 - y1))
        for frame, cut in zip(main, r.frames):
            np.testing.assert_array_equal(cut, frame[y1:y2, x1:x2])

    def test_the_window_holds_the_whole_path_not_just_its_ends(self):
        # The person dips down in the middle of the walk (around an obstacle): the window must hold the dip too.
        def path(i):
            dip = 120 if 40 <= i <= 60 else 0
            return [(0, (200 + i, 300 + dip, 240 + i, 400 + dip))]
        r = vlm_crop.crop_clip(detector_from(path), settings(), subs(), 1000., 1010., mains(), 1000.)
        x1, y1, x2, y2 = r.first_crop
        self.assertLessEqual(y1, 300 * SY)
        self.assertGreaterEqual(y2, 520 * SY)
        self.assertLessEqual(x1, 200 * SX)
        self.assertGreaterEqual(x2, 339 * SX)

    def test_the_last_frame_is_looked_at(self):
        det = detector_from(walk)
        vlm_crop.crop_clip(det, settings(), subs(97), 1000., 1010., mains(), 1000.)
        self.assertEqual(det.calls[-1], 96)
        self.assertEqual(det.calls[:-1], list(range(0, 97, 5)))

    def test_a_one_off_box_does_not_stretch_the_window(self):
        def with_shadow(i):
            boxes = walk(i)
            if i == 50:
                boxes.append((0, (620, 20, 660, 120)))      # a shadow in the far corner, seen once
            return boxes
        r = vlm_crop.crop_clip(detector_from(with_shadow), settings(), subs(), 1000., 1010., mains(), 1000.)
        clean = vlm_crop.crop_clip(detector_from(walk), settings(), subs(), 1000., 1010., mains(), 1000.)
        self.assertEqual(r.first_crop, clean.first_crop)
        self.assertLess(r.first_crop[2], 620 * SX)

    def test_a_person_seen_in_one_look_still_gets_a_window(self):
        looks = [[], [], [main_box((300, 300, 340, 400))], [], []]
        window, whole = vlm_crop.clip_window(looks, MAIN_H, MAIN_W, settings())
        self.assertIsNotNone(window)
        self.assertFalse(whole)

    def test_no_boxes_no_window(self):
        self.assertEqual(vlm_crop.clip_window([[], []], MAIN_H, MAIN_W, settings()), (None, False))
        r = vlm_crop.crop_clip(detector_from(lambda i: []), settings(), subs(), 1000., 1010., mains(), 1000.)
        self.assertIsNone(r)

    def test_a_person_standing_still_keeps_a_quarter_of_their_height_around_them(self):
        person = (1000., 600., 1150., 1000.)                 # 400 px tall
        window, _ = vlm_crop.clip_window([[person]] * 5, MAIN_H, MAIN_W, settings())
        x1, y1, x2, y2 = window
        self.assertLessEqual(x1, 1000 - 100)
        self.assertGreaterEqual(x2, 1150 + 100)
        self.assertLessEqual(y1, 600 - 100)
        self.assertGreaterEqual(y2, 1000 + 100 - 1)

    def test_a_long_walk_is_padded_by_its_own_size(self):
        looks = [[(400. + 60 * k, 700., 460. + 60 * k, 820.)] for k in range(20)]   # 1200 px wide, 120 tall
        window, whole = vlm_crop.clip_window(looks, MAIN_H, MAIN_W, settings())
        self.assertFalse(whole)
        self.assertLessEqual(window[0], 400 - 0.05 * 1200 + 1)
        self.assertGreaterEqual(window[2], 1600 + 0.05 * 1200 - 1)

    def test_shape_is_at_most_two_to_one_grown_with_more_scene(self):
        looks = [[(400. + 60 * k, 700., 460. + 60 * k, 820.)] for k in range(20)]
        window, _ = vlm_crop.clip_window(looks, MAIN_H, MAIN_W, settings())
        w, h = window[2] - window[0], window[3] - window[1]
        self.assertLessEqual(w, 2 * h + 2)
        self.assertGreater(h, 120 + 2 * 30)                  # taller than the padded walk itself: more scene

    def test_small_person_gets_at_least_min_size(self):
        window, _ = vlm_crop.clip_window([[(1000., 700., 1010., 730.)]] * 3, MAIN_H, MAIN_W, settings())
        self.assertGreaterEqual(window[2] - window[0], 384)
        self.assertGreaterEqual(window[3] - window[1], 384)

    def test_sides_are_even_and_the_window_stays_inside_the_frame(self):
        for box in [(0., 0., 37., 91.), (2550., 1400., 2591., 1519.), (1201., 333., 1263., 517.)]:
            window, _ = vlm_crop.clip_window([[box]] * 3, MAIN_H, MAIN_W, settings())
            x1, y1, x2, y2 = window
            self.assertEqual(((x2 - x1) % 2, (y2 - y1) % 2), (0, 0))
            self.assertTrue(0 <= x1 < x2 <= MAIN_W and 0 <= y1 < y2 <= MAIN_H, window)
            self.assertTrue(x1 <= box[0] and box[2] <= x2 + 1 and y1 <= box[1] and box[3] <= y2 + 1, window)

    def test_a_window_over_half_but_under_80_percent_stays_a_crop(self):
        # Sending the whole frame from half the frame on zoomed too little and lost alerts in the eval (2026-10-10).
        looks = [[(300. + 80 * k, 300., 420. + 80 * k, 1000.)] for k in range(20)]   # ~1900 x 700 + padding
        window, whole = vlm_crop.clip_window(looks, MAIN_H, MAIN_W, settings())
        area = (window[2] - window[0]) * (window[3] - window[1]) / float(MAIN_W * MAIN_H)
        self.assertFalse(whole)
        self.assertTrue(0.5 < area <= 0.8, area)

    def test_a_window_over_80_percent_of_the_frame_is_the_whole_frame(self):
        looks = [[(100. + 120 * k, 300., 300. + 120 * k, 1300.)] for k in range(20)]   # across the yard
        window, whole = vlm_crop.clip_window(looks, MAIN_H, MAIN_W, settings())
        self.assertEqual((window, whole), ((0, 0, MAIN_W, MAIN_H), True))

    def test_meta_says_clip_window(self):
        r = vlm_crop.crop_clip(detector_from(walk), settings(), subs(), 1000., 1010., mains(), 1000.)
        meta = vlm_crop.crop_meta(r, settings(), "rel.mp4", 5.0)
        self.assertEqual((meta["crop_policy"], meta["per_frame_tracking"], meta["whole_frame"]),
                         ("clip-window-1", False, False))
        self.assertEqual(meta["crop_region"], list(r.first_crop))

    def test_the_model_gets_the_same_view_in_every_frame(self):
        r = vlm_crop.crop_clip(detector_from(walk), settings(), subs(), 1000., 1010., mains(), 1000.)
        sent = model_input.render_model_input(
            mains(), {"vlm_input": model_input.VLM_INPUT_CROP, "fps": 5.0, "crops": r.crops,
                      "crop_size": (r.width, r.height)}, model_input.ModelInputConfig(1.0, max_side=1024))
        self.assertEqual(len(set(map(tuple, sent.crops))), 1)
        self.assertEqual(len({f.shape for f in sent.frames}), 1)
        self.assertEqual(tuple(sent.union_crop), r.first_crop)


class SettingsTest(unittest.TestCase):
    def cfg(self, **kw):
        base = dict(STORE_FPS=10.0, TRIGGER_CLASS_IDS=[0], YOLO_TRIGGER_CONF=0.35, YOLO_IMGSZ=640, CROP_PADDING=0.3,
                    CROP_MIN_SIZE=384, CROP_EMA_ALPHA=0.3)
        base.update(kw)
        return SimpleNamespace(**base)

    def test_clip_window_is_the_default(self):
        self.assertEqual(vlm_crop.settings_from_config(self.cfg()).policy, vlm_crop.POLICY_CLIP_WINDOW)

    def test_follow_can_be_chosen_and_an_unknown_policy_falls_back(self):
        self.assertEqual(vlm_crop.settings_from_config(self.cfg(CROP_POLICY="Follow")).policy,
                         vlm_crop.POLICY_FOLLOW)
        with self.assertLogs(vlm_crop.log, "WARNING"):
            s = vlm_crop.settings_from_config(self.cfg(CROP_POLICY="zoomy"))
        self.assertEqual(s.policy, vlm_crop.POLICY_CLIP_WINDOW)

    def test_the_collector_config_loads_the_window_settings(self):
        from home_guard_project.data_collection import config as dc_config
        cfg = dc_config.load_config()
        s = vlm_crop.settings_from_config(cfg)
        self.assertEqual((s.policy, s.window_padding, s.person_margin, s.max_aspect, s.whole_frame_above),
                         (vlm_crop.POLICY_CLIP_WINDOW, 0.05, 0.25, 2.0, 0.8))


if __name__ == "__main__":
    unittest.main()
