from __future__ import annotations

import unittest
from typing import List, Tuple

import numpy as np

from home_guard_project.box import plates as pl


class FakeBackend:
    """Plate boxes and OCR answers given per call, in pixels of the crop it is handed."""

    def __init__(self, boxes: List[List[Tuple[float, float, float, float, float]]], texts: List[Tuple[str, float]]):
        self.boxes, self.texts = list(boxes), list(texts)
        self.detect_calls = 0
        self.ocr_sizes: List[Tuple[int, int]] = []

    def detect(self, image: np.ndarray):
        self.detect_calls += 1
        return self.boxes.pop(0) if self.boxes else []

    def ocr(self, plate: np.ndarray):
        self.ocr_sizes.append(plate.shape[:2])
        return self.texts.pop(0) if self.texts else ("", 0.0)


def crop(w: int = 400, h: int = 300) -> np.ndarray:
    return np.zeros((h, w, 3), dtype=np.uint8)


class DigitsTest(unittest.TestCase):
    def test_digits_only_strips_separators_and_letters(self) -> None:
        self.assertEqual(pl.digits_only("12-345-67"), "1234567")
        self.assertEqual(pl.digits_only(" 106 52 002 "), "10652002")
        self.assertEqual(pl.digits_only(None), "")

    def test_israeli_plates_have_7_or_8_digits(self) -> None:
        self.assertTrue(pl.valid_plate("1234567"))
        self.assertTrue(pl.valid_plate("10652002"))
        self.assertFalse(pl.valid_plate("123456"))
        self.assertFalse(pl.valid_plate("123456789"))
        self.assertFalse(pl.valid_plate("12a4567"))

    def test_levenshtein(self) -> None:
        self.assertEqual(pl.levenshtein("5654230", "5654230"), 0)
        self.assertEqual(pl.levenshtein("5654230", "5654280"), 1)
        self.assertEqual(pl.levenshtein("5654230", "565423"), 1)
        self.assertEqual(pl.levenshtein("5533333", "5654230"), 5)

    def test_mask_keeps_the_last_three_digits(self) -> None:
        self.assertEqual(pl.mask("1234567"), "****567")
        self.assertEqual(pl.mask(""), "")


class VoteTest(unittest.TestCase):
    def read(self, digits: str, conf: float = 0.9) -> pl.PlateRead:
        return pl.PlateRead(digits, conf, 70.0)

    def test_two_agreeing_reads_win(self) -> None:
        self.assertEqual(pl.vote([self.read("5654230"), self.read("5654230")]), "5654230")

    def test_reads_one_digit_apart_agree_and_the_most_common_wins(self) -> None:
        reads = [self.read("5654230"), self.read("5654280"), self.read("5654230")]
        self.assertEqual(pl.vote(reads), "5654230")

    def test_one_read_is_not_enough(self) -> None:
        self.assertIsNone(pl.vote([self.read("5654230")]))
        self.assertIsNone(pl.vote([]))

    def test_disagreeing_reads_are_no_plate(self) -> None:
        self.assertIsNone(pl.vote([self.read("5654230"), self.read("1065200")]))

    def test_the_bigger_cluster_wins(self) -> None:
        reads = [self.read("1111111"), self.read("5654230"), self.read("5654230"), self.read("5654230"),
                 self.read("1111112")]
        self.assertEqual(pl.vote(reads), "5654230")


class MatchTest(unittest.TestCase):
    owners = [{"id": "v1", "name": "Toyota", "plate": "5654230"},
              {"id": "v2", "name": "Citroen", "plate": "10652002"},
              {"id": "v3", "name": "No plate", "plate": ""}]

    def test_exact_and_one_digit_off_match(self) -> None:
        self.assertEqual(pl.match("5654230", self.owners)["id"], "v1")
        self.assertEqual(pl.match("5654280", self.owners)["id"], "v1")
        self.assertEqual(pl.match("10652007", self.owners)["id"], "v2")

    def test_a_different_length_never_matches(self) -> None:
        self.assertIsNone(pl.match("565423", self.owners))
        self.assertIsNone(pl.match("56542300", self.owners))

    def test_two_digits_off_is_someone_else(self) -> None:
        self.assertIsNone(pl.match("5654999", self.owners))

    def test_exact_beats_near_and_two_near_owners_are_ambiguous(self) -> None:
        owners = [{"id": "a", "plate": "1234567"}, {"id": "b", "plate": "1234568"}]
        self.assertEqual(pl.match("1234567", owners)["id"], "a")
        self.assertIsNone(pl.match("1234569", owners))


class CropTest(unittest.TestCase):
    def test_crop_with_margin_and_inner_box(self) -> None:
        frame = np.zeros((1000, 2000, 3), dtype=np.uint8)
        crops = pl.vehicle_crops([frame], (0.25, 0.5, 0.75, 0.9))
        self.assertEqual(len(crops), 1)
        c = crops[0]
        # vehicle 1000 x 400 px, 20% margin each side (benchmark 2026-10-09: 15-25 %)
        self.assertEqual(c.image.shape[:2], (560, 1400))
        self.assertEqual(c.inner, (200, 80, 1200, 480))

    def test_at_most_six_frames_spread_over_the_list(self) -> None:
        frames = [np.full((100, 200, 3), i, dtype=np.uint8) for i in range(20)]
        crops = pl.vehicle_crops(frames, (0.0, 0.0, 1.0, 1.0), min_width=10)
        self.assertEqual(len(crops), 6)
        self.assertEqual(int(crops[0].image[0, 0, 0]), 0)
        self.assertEqual(int(crops[-1].image[0, 0, 0]), 19)

    def test_a_vehicle_too_small_for_a_readable_plate_gives_no_crop(self) -> None:
        frame = np.zeros((576, 704, 3), dtype=np.uint8)
        self.assertEqual(pl.vehicle_crops([frame], (0.1, 0.1, 0.15, 0.2)), [])

    def test_bad_boxes_and_frames_give_nothing(self) -> None:
        frame = np.zeros((100, 200, 3), dtype=np.uint8)
        self.assertEqual(pl.vehicle_crops([frame], (0.5, 0.5, 0.4, 0.6)), [])
        self.assertEqual(pl.vehicle_crops([None], (0.0, 0.0, 1.0, 1.0)), [])
        self.assertEqual(pl.vehicle_crops([], (0.0, 0.0, 1.0, 1.0)), [])

    def test_night_ir_frames_are_grey(self) -> None:
        grey = np.full((50, 50, 3), 120, dtype=np.uint8)
        colour = np.zeros((50, 50, 3), dtype=np.uint8)
        colour[..., 2] = 200
        self.assertTrue(pl.is_ir(grey))
        self.assertFalse(pl.is_ir(colour))


class BestFramesTest(unittest.TestCase):
    def test_the_biggest_sharpest_stillest_frames_win(self) -> None:
        boxes = [(100, 100, 200, 160), (100, 100, 400, 280), (110, 100, 410, 280), (600, 100, 900, 280)]
        sharp = [50.0, 900.0, 800.0, 10.0]
        self.assertEqual(pl.best_frames(boxes, sharp, k=2), [1, 2])

    def test_a_box_touching_the_frame_edge_comes_last(self) -> None:
        boxes = [(0, 100, 300, 280), (100, 100, 300, 260)]
        self.assertEqual(pl.best_frames(boxes, [900.0, 100.0], k=1, frame_size=(1000, 800)), [1])

    def test_fewer_frames_than_k_and_bad_input(self) -> None:
        self.assertEqual(pl.best_frames([(0, 0, 10, 10)], [1.0], k=5), [0])
        self.assertEqual(pl.best_frames([], [], k=5), [])
        self.assertEqual(pl.best_frames([(0, 0, 10, 10)], [], k=5), [])

    def test_sharpness_is_higher_for_edges_than_for_a_flat_picture(self) -> None:
        flat = np.full((40, 80, 3), 128, dtype=np.uint8)
        edges = flat.copy()
        edges[:, ::4] = 255
        self.assertGreater(pl.sharpness(edges), pl.sharpness(flat))
        self.assertEqual(pl.sharpness(None), 0.0)


class ReaderTest(unittest.TestCase):
    def test_without_the_packages_the_reader_reads_nothing(self) -> None:
        reader = pl.PlateReader(loader=lambda: None)
        self.assertFalse(reader.available())
        self.assertEqual(reader.read([crop()]), [])

    def test_a_missing_import_is_no_plate(self) -> None:
        def broken():
            raise ImportError("no module named open_image_models")
        reader = pl.PlateReader(loader=broken)
        self.assertFalse(reader.available())
        self.assertEqual(reader.read([crop()]), [])

    def test_reads_a_wide_plate_inside_the_vehicle(self) -> None:
        backend = FakeBackend([[(150, 200, 230, 225, 0.8)]], [("56-542-30", 0.95)])
        reader = pl.PlateReader(backend=backend)
        self.assertTrue(reader.available())
        reads = reader.read([pl.VehicleCrop(crop(), (0, 0, 400, 300))])
        self.assertEqual(reads, [pl.PlateRead("5654230", 0.95, 80.0)])

    def test_a_plate_under_50_px_is_dropped(self) -> None:
        backend = FakeBackend([[(150, 200, 195, 225, 0.8)]], [("5654230", 0.95)])
        self.assertEqual(pl.PlateReader(backend=backend).read([crop()]), [])
        self.assertEqual(backend.ocr_sizes, [])          # not even read

    def test_a_plate_outside_the_vehicle_box_is_dropped(self) -> None:
        # the plate sits in the crop's margin (the timestamp overlay, a board, the next car)
        backend = FakeBackend([[(0, 0, 80, 20, 0.9)]], [("5654230", 0.95)])
        self.assertEqual(pl.PlateReader(backend=backend).read([pl.VehicleCrop(crop(), (40, 30, 360, 270))]), [])

    def test_wrong_digit_counts_are_dropped(self) -> None:
        backend = FakeBackend([[(100, 100, 200, 130, 0.9)], [(100, 100, 200, 130, 0.9)]],
                              [("12345", 0.9), ("AB123456789", 0.9)])
        self.assertEqual(pl.PlateReader(backend=backend).read([crop(), crop()]), [])

    def test_a_backend_error_is_no_plate(self) -> None:
        class Broken(FakeBackend):
            def detect(self, image):
                raise RuntimeError("onnx failed")
        self.assertEqual(pl.PlateReader(backend=Broken([], [])).read([crop()]), [])

    def test_plain_arrays_count_as_the_whole_vehicle(self) -> None:
        backend = FakeBackend([[(10, 10, 90, 40, 0.9)]], [("10652002", 0.9)])
        self.assertEqual(pl.PlateReader(backend=backend).read([crop()])[0].digits, "10652002")


if __name__ == "__main__":
    unittest.main()
