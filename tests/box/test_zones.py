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

    def test_booleans_are_not_numbers(self) -> None:
        with self.assertRaises(ValueError):
            z.validate_points([(True, 0), (1, 0), (1, 1)])
        with self.assertRaises(ValueError):
            z.validate_points([(0, False), (1, 0), (1, 1)])


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

    def test_a_corner_that_is_not_a_pair_does_not_crash_the_loader(self) -> None:
        with open(self.path, "w", encoding="utf-8") as f:
            f.write("zones:\n  yard: [{a: 1}, [0, 0], [1, 1]]\n  gate: [[0, 0], [1, 0], [1, 1]]\n")
        self.assertEqual(z.load_zones(self.path), {"gate": [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0)]})

    def test_a_number_too_large_for_a_float_does_not_crash_the_loader(self) -> None:
        with open(self.path, "w", encoding="utf-8") as f:
            f.write("zones:\n  yard: [[0, 0], [1, 0], [" + "9" * 400 + ", 1]]\n  gate: [[0, 0], [1, 0], [1, 1]]\n")
        self.assertEqual(z.load_zones(self.path), {"gate": [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0)]})

    def test_a_corner_must_be_exactly_two_numbers(self) -> None:
        with self.assertRaises(ValueError):
            z.validate_points(["01", "01", "01"])
        with self.assertRaises(ValueError):
            z.validate_points([(0, 0, 5), (1, 0), (1, 1)])

    def test_renaming_a_damaged_zone_reports_false_and_leaves_the_file_alone(self) -> None:
        with open(self.path, "w", encoding="utf-8") as f:
            f.write("zones:\n  bad: 5\n  gate: [[0, 0], [1, 0], [1, 1]]\n")
        before = open(self.path, encoding="utf-8").read()
        self.assertFalse(z.rename_zone("bad", "good", self.path))
        self.assertEqual(open(self.path, encoding="utf-8").read(), before)

    def test_a_file_that_is_not_utf8_loads_as_no_zones(self) -> None:
        with open(self.path, "wb") as f:
            f.write(b"zones:\n  yard: \xff\xfe\n")
        self.assertEqual(z.load_zones(self.path), {})

    def test_a_malformed_sibling_entry_does_not_stop_saving_clearing_or_renaming(self) -> None:
        with open(self.path, "w", encoding="utf-8") as f:
            f.write("zones:\n  bad: 5\n  other: null\n  gate: [[0, 0], [1, 0], [1, 1]]\n")
        z.save_zone("yard", LEFT_HALF, self.path)
        self.assertEqual(set(z.load_zones(self.path)), {"gate", "yard"})
        self.assertTrue(z.rename_zone("yard", "garden", self.path))
        self.assertTrue(z.clear_zone("gate", self.path))
        self.assertEqual(set(z.load_zones(self.path)), {"garden"})

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

    def test_a_deeply_nested_file_loads_as_no_zones(self) -> None:
        with open(self.path, "w", encoding="utf-8") as f:
            f.write("zones: " + "[" * 5000 + "]" * 5000 + "\n")
        self.assertEqual(z.load_zones(self.path), {})

    def test_a_camera_key_written_as_a_number_can_still_be_cleared(self) -> None:
        with open(self.path, "w", encoding="utf-8") as f:
            f.write("zones:\n  123: [[0, 0], [1, 0], [1, 1]]\n")
        self.assertEqual(set(z.load_zones(self.path)), {"123"})
        self.assertTrue(z.clear_zone("123", self.path))
        self.assertEqual(z.load_zones(self.path), {})


ZA = [(0.0, 0.0), (0.4, 0.0), (0.4, 1.0)]
ZB = [(0.6, 0.0), (1.0, 0.0), (1.0, 1.0)]
ZC = [(0.0, 0.5), (1.0, 0.5), (0.5, 1.0)]


class RemapZonesTest(unittest.TestCase):
    """Renames applied in one pass, so a swap or a chain never loses or misplaces a zone."""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tmp.name, "zones.yaml")

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _save(self, zones) -> None:
        z.save_zones(zones, self.path)

    def test_swapping_two_names_swaps_their_zones(self) -> None:
        self._save({"front": ZA, "back": ZB})
        z.remap_zones({"front": "back", "back": "front"}, self.path)
        self.assertEqual(z.load_zones(self.path), {"back": ZA, "front": ZB})

    def test_a_chain_moves_each_zone_one_step(self) -> None:
        self._save({"a": ZA, "b": ZB})
        z.remap_zones({"a": "b", "b": "c"}, self.path)
        self.assertEqual(z.load_zones(self.path), {"b": ZA, "c": ZB})

    def test_a_stale_zone_on_the_target_name_is_dropped_when_the_source_had_none(self) -> None:
        self._save({"b": ZB})
        z.remap_zones({"a": "b"}, self.path)
        self.assertEqual(z.load_zones(self.path), {})

    def test_a_rename_where_neither_camera_has_a_zone_changes_nothing(self) -> None:
        self._save({"yard": ZC})
        before = open(self.path, encoding="utf-8").read()
        z.remap_zones({"a": "b"}, self.path)
        self.assertEqual(open(self.path, encoding="utf-8").read(), before)

    def test_no_file_and_no_zones_stays_no_file(self) -> None:
        z.remap_zones({"a": "b"}, self.path)
        self.assertFalse(os.path.exists(self.path))

    def test_unrelated_cameras_are_untouched(self) -> None:
        self._save({"front": ZA, "back": ZB, "yard": ZC})
        z.remap_zones({"front": "back", "back": "front"}, self.path)
        self.assertEqual(z.load_zones(self.path)["yard"], ZC)

    def test_between_runs_while_both_old_and_new_names_are_covered(self) -> None:
        self._save({"a": ZA, "b": ZB, "s": ZC})
        seen = {}
        z.remap_zones({"a": "b", "b": "c", "x": "s"}, self.path,
                      between=lambda: seen.update(z.load_zones(self.path)))
        self.assertEqual(seen, {"a": ZA, "b": ZA, "c": ZB, "s": ZC})   # old a and stale s still masked mid-way
        self.assertEqual(z.load_zones(self.path), {"b": ZA, "c": ZB})

    def test_if_between_fails_the_file_still_covers_the_old_names(self) -> None:
        self._save({"a": ZA})

        def boom() -> None:
            raise OSError("disk full")

        with self.assertRaises(OSError):
            z.remap_zones({"a": "b"}, self.path, between=boom)
        self.assertEqual(z.load_zones(self.path), {"a": ZA, "b": ZA})


class RenameZoneRefusesOverwriteTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tmp.name, "zones.yaml")

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_rename_onto_a_camera_that_already_has_a_zone_is_refused(self) -> None:
        z.save_zones({"front": ZA, "back": ZB}, self.path)
        with self.assertLogs(z.log, level="WARNING"):
            self.assertFalse(z.rename_zone("front", "back", self.path))
        self.assertEqual(z.load_zones(self.path), {"front": ZA, "back": ZB})
