from __future__ import annotations

import os
import tempfile
import unittest

import numpy as np

from home_guard_project.data_collection import zones as z

LEFT_HALF = [(0.0, 0.0), (0.5, 0.0), (0.5, 1.0), (0.0, 1.0)]


def white(h: int, w: int) -> np.ndarray:
    return np.full((h, w, 3), 255, dtype=np.uint8)


class ValidatePointsTest(unittest.TestCase):
    def test_three_to_thirty_two_points_in_range_are_accepted_and_rounded(self) -> None:
        self.assertEqual(z.validate_points([(0, 0), (1, 0), (0.123456, 1)]), [(0.0, 0.0), (1.0, 0.0), (0.1235, 1.0)])
        self.assertEqual(len(z.validate_points([(i / 40, 0.5) for i in range(32)])), 32)

    def test_too_few_or_too_many_points_are_refused(self) -> None:
        with self.assertRaises(ValueError):
            z.validate_points([(0, 0), (1, 1)])
        with self.assertRaises(ValueError):
            z.validate_points([(i / 40, 0.5) for i in range(33)])

    def test_points_outside_the_picture_or_not_numbers_are_refused(self) -> None:
        for bad in ([(0, 0), (1.2, 0), (0, 1)], [(0, 0), (1, -0.1), (0, 1)], [(0, 0), ("a", 0), (0, 1)], [(0, 0), (1,), (0, 1)]):
            with self.assertRaises(ValueError):
                z.validate_points(bad)


class ParsePointsTest(unittest.TestCase):
    def test_semicolon_pairs_are_the_command_line_form(self) -> None:
        self.assertEqual(z.parse_points("0.1,0.2;0.9,0.2;0.5,0.9"), [(0.1, 0.2), (0.9, 0.2), (0.5, 0.9)])

    def test_whitespace_between_pairs_is_accepted_too(self) -> None:
        self.assertEqual(z.parse_points("0.1,0.2 0.9,0.2\n0.5,0.9"), [(0.1, 0.2), (0.9, 0.2), (0.5, 0.9)])

    def test_garbage_is_refused_with_a_plain_message(self) -> None:
        with self.assertRaises(ValueError):
            z.parse_points("0.1,0.2;nope;0.5,0.9")
        with self.assertRaises(ValueError):
            z.parse_points("")


class ZoneMaskTest(unittest.TestCase):
    def test_no_polygon_returns_the_very_same_frame(self) -> None:
        frame = white(10, 20)
        self.assertIs(z.ZoneMask(None).apply(frame), frame)
        self.assertFalse(z.ZoneMask(None).active)

    def test_outside_the_polygon_is_black_and_the_original_is_untouched(self) -> None:
        frame = white(10, 20)
        out = z.ZoneMask(LEFT_HALF).apply(frame)
        self.assertEqual(int(out[:, :10].min()), 255)   # inside: kept
        self.assertEqual(int(out[:, 11:].max()), 0)     # outside: black
        self.assertEqual(int(frame.min()), 255)          # caller's frame not modified
        self.assertTrue(z.ZoneMask(LEFT_HALF).active)

    def test_one_normalised_polygon_fits_both_stream_resolutions(self) -> None:
        mask = z.ZoneMask(LEFT_HALF)
        small = mask.apply(white(10, 20))
        large = mask.apply(white(20, 40))               # same instance, bigger frame: mask rebuilt
        self.assertEqual(int(small[:, :10].min()), 255)
        self.assertEqual(int(small[:, 11:].max()), 0)
        self.assertEqual(int(large[:, :20].min()), 255)
        self.assertEqual(int(large[:, 22:].max()), 0)
        again = mask.apply(white(10, 20))               # and back again
        self.assertEqual(int(again[:, 11:].max()), 0)

    def test_mask_for_an_unknown_camera_watches_everything(self) -> None:
        zones = {"yard": LEFT_HALF}
        self.assertTrue(z.mask_for(zones, "yard").active)
        self.assertFalse(z.mask_for(zones, "street").active)


class ZoneFileTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tmp.name, "zones.yaml")

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_missing_file_means_no_zones(self) -> None:
        self.assertEqual(z.load_zones(self.path), {})

    def test_save_one_zone_then_load_it(self) -> None:
        saved = z.save_zone("yard", [(0.1, 0.2), (0.9, 0.2), (0.5, 0.9)], self.path)
        self.assertEqual(saved, [(0.1, 0.2), (0.9, 0.2), (0.5, 0.9)])
        self.assertEqual(z.load_zones(self.path), {"yard": [(0.1, 0.2), (0.9, 0.2), (0.5, 0.9)]})

    def test_saving_a_second_camera_keeps_the_first(self) -> None:
        z.save_zone("yard", LEFT_HALF, self.path)
        z.save_zone("gate", [(0, 0), (1, 0), (1, 1)], self.path)
        self.assertEqual(set(z.load_zones(self.path)), {"yard", "gate"})

    def test_clear_removes_only_that_camera(self) -> None:
        z.save_zone("yard", LEFT_HALF, self.path)
        z.save_zone("gate", [(0, 0), (1, 0), (1, 1)], self.path)
        self.assertTrue(z.clear_zone("yard", self.path))
        self.assertFalse(z.clear_zone("yard", self.path))     # already gone
        self.assertEqual(set(z.load_zones(self.path)), {"gate"})

    def test_rename_moves_the_zone_with_the_camera(self) -> None:
        z.save_zone("yard", LEFT_HALF, self.path)
        self.assertTrue(z.rename_zone("yard", "garden", self.path))
        self.assertFalse(z.rename_zone("yard", "garden", self.path))   # nothing left to move
        self.assertEqual(set(z.load_zones(self.path)), {"garden"})

    def test_a_damaged_file_loads_as_no_zones(self) -> None:
        with open(self.path, "w", encoding="utf-8") as f:
            f.write("zones: [1, 2\n")
        self.assertEqual(z.load_zones(self.path), {})

    def test_an_invalid_polygon_in_the_file_is_skipped(self) -> None:
        with open(self.path, "w", encoding="utf-8") as f:
            f.write("zones:\n  yard: [[0, 0], [1, 1]]\n  gate: [[0, 0], [1, 0], [1, 1]]\n  bad: [[0, 0], [2, 0], [1, 1]]\n")
        self.assertEqual(z.load_zones(self.path), {"gate": [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0)]})

    def test_save_zones_writes_the_whole_dict_for_the_laptop_editor(self) -> None:
        z.save_zones({"yard": [[0.1, 0.2], [0.9, 0.2], [0.5, 0.9]]}, self.path)
        self.assertEqual(z.load_zones(self.path), {"yard": [(0.1, 0.2), (0.9, 0.2), (0.5, 0.9)]})
