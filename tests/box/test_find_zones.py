from __future__ import annotations

import contextlib
import io
import json
import os
import tempfile
import unittest
from unittest import mock

from home_guard_project.box import find_cameras as fc
from home_guard_project.data_collection import zones as z

URL = "rtsp://admin:s3cret@192.168.1.50:554/unicast/c1/s0/live"


class ZoneCommandsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.cameras = os.path.join(self.tmp.name, "cameras.yaml")
        self.zones = os.path.join(self.tmp.name, "zones.yaml")
        fc._write_cameras({"yard": URL, "gate": URL}, {"old_cam": URL}, self.cameras)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_zone_rows_list_every_camera_with_an_empty_zone_by_default(self) -> None:
        rows = fc.zone_rows(self.cameras, self.zones)["cameras"]
        self.assertEqual(sorted(r["name"] for r in rows), ["gate", "old_cam", "yard"])   # YAML stores keys sorted
        self.assertEqual([r["points"] for r in rows], [[], [], []])

    def test_set_zone_stores_the_corners_and_reports_them(self) -> None:
        out = fc.set_zone("yard", "0.1,0.2;0.9,0.2;0.5,0.9", self.cameras, self.zones, restart=False)
        self.assertEqual(out, {"camera": "yard", "points": [[0.1, 0.2], [0.9, 0.2], [0.5, 0.9]]})
        self.assertEqual(z.load_zones(self.zones), {"yard": [(0.1, 0.2), (0.9, 0.2), (0.5, 0.9)]})
        rows = {r["name"]: r["points"] for r in fc.zone_rows(self.cameras, self.zones)["cameras"]}
        self.assertEqual(rows["yard"], [[0.1, 0.2], [0.9, 0.2], [0.5, 0.9]])

    def test_a_disabled_camera_can_have_a_zone_too(self) -> None:
        out = fc.set_zone("old_cam", "0,0;1,0;1,1", self.cameras, self.zones, restart=False)
        self.assertEqual(out["camera"], "old_cam")

    def test_an_unknown_camera_or_bad_corners_are_refused(self) -> None:
        with self.assertRaises(ValueError):
            fc.set_zone("street", "0,0;1,0;1,1", self.cameras, self.zones, restart=False)
        with self.assertRaises(ValueError):
            fc.set_zone("yard", "0,0;1,0", self.cameras, self.zones, restart=False)
        with self.assertRaises(ValueError):
            fc.set_zone("yard", "0,0;1,0;2,2", self.cameras, self.zones, restart=False)
        self.assertEqual(z.load_zones(self.zones), {})

    def test_clear_zone_means_the_whole_picture(self) -> None:
        fc.set_zone("yard", "0,0;1,0;1,1", self.cameras, self.zones, restart=False)
        out = fc.clear_zone_command("yard", self.cameras, self.zones, restart=False)
        self.assertEqual(out, {"camera": "yard", "points": []})
        self.assertEqual(z.load_zones(self.zones), {})
        with self.assertRaises(ValueError):
            fc.clear_zone_command("street", self.cameras, self.zones, restart=False)

    def test_renaming_a_camera_carries_its_zone(self) -> None:
        fc.set_zone("yard", "0,0;1,0;1,1", self.cameras, self.zones, restart=False)
        fc.apply_changes({"cameras": [{"name": "yard", "new_name": "garden", "enabled": True}]},
                         self.cameras, zones_path=self.zones, restart=False)
        self.assertEqual(set(z.load_zones(self.zones)), {"garden"})

    def test_disabling_a_camera_keeps_its_zone(self) -> None:
        fc.set_zone("yard", "0,0;1,0;1,1", self.cameras, self.zones, restart=False)
        fc.apply_changes({"cameras": [{"name": "yard", "enabled": False}]}, self.cameras, zones_path=self.zones,
                         restart=False)
        self.assertEqual(set(z.load_zones(self.zones)), {"yard"})



URL_A = "rtsp://admin:s3cret@192.168.1.51:554/unicast/c1/s0/live"
URL_B = "rtsp://admin:s3cret@192.168.1.52:554/unicast/c1/s0/live"
URL_C = "rtsp://admin:s3cret@192.168.1.53:554/unicast/c1/s0/live"
ZA = [(0.0, 0.0), (0.4, 0.0), (0.4, 1.0)]
ZB = [(0.6, 0.0), (1.0, 0.0), (1.0, 1.0)]


class RenameKeepsZonesOnTheirCameraTest(unittest.TestCase):
    """Each zone belongs to a physical camera (its URL); renames must not move it to another."""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.cameras = os.path.join(self.tmp.name, "cameras.yaml")
        self.zones = os.path.join(self.tmp.name, "zones.yaml")

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _zone_by_url(self) -> dict:
        cams = fc._read_cameras_raw(self.cameras)
        names = {**(cams.get("disabled") or {}), **(cams.get("cameras") or {})}
        zones = z.load_zones(self.zones)
        return {url: zones.get(name) for name, url in names.items()}

    def _rename(self, pairs) -> None:
        fc.apply_changes({"cameras": [{"name": o, "new_name": n, "enabled": True} for o, n in pairs]},
                         self.cameras, zones_path=self.zones, restart=False)

    def test_swapping_two_camera_names_keeps_each_zone_on_its_camera(self) -> None:
        fc._write_cameras({"front": URL_A, "back": URL_B}, {}, self.cameras)
        z.save_zones({"front": ZA, "back": ZB}, self.zones)
        self._rename([("front", "back"), ("back", "front")])
        self.assertEqual(self._zone_by_url(), {URL_A: ZA, URL_B: ZB})
        self.assertEqual(z.load_zones(self.zones), {"back": ZA, "front": ZB})

    def test_a_chain_of_renames_keeps_each_zone_on_its_camera(self) -> None:
        fc._write_cameras({"a": URL_A, "b": URL_B}, {}, self.cameras)
        z.save_zones({"a": ZA, "b": ZB}, self.zones)
        self._rename([("a", "b"), ("b", "c")])
        self.assertEqual(self._zone_by_url(), {URL_A: ZA, URL_B: ZB})

    def test_a_camera_without_a_zone_does_not_inherit_one_by_taking_a_name(self) -> None:
        fc._write_cameras({"a": URL_A, "b": URL_B, "c": URL_C}, {}, self.cameras)
        z.save_zones({"b": ZB}, self.zones)
        self._rename([("a", "b"), ("b", "c"), ("c", "a")])
        self.assertEqual(self._zone_by_url(), {URL_A: None, URL_B: ZB, URL_C: None})

    def test_a_failed_cameras_write_leaves_every_camera_masked(self) -> None:
        fc._write_cameras({"front": URL_A}, {}, self.cameras)
        z.save_zones({"front": ZA}, self.zones)
        with mock.patch.object(fc, "_write_cameras", side_effect=OSError("disk full")),                 self.assertRaises(OSError):
            self._rename([("front", "porch")])
        self.assertEqual(z.load_zones(self.zones), {"front": ZA, "porch": ZA})   # old and new names both covered


class ZoneCliFailureTest(unittest.TestCase):
    def test_an_unexpected_failure_still_prints_a_json_error(self) -> None:
        argv = ["find_cameras", "--json", "set-zone", "--camera", "yard", "--points", "0,0;1,0;1,1"]
        out = io.StringIO()
        with mock.patch("sys.argv", argv),                 mock.patch("home_guard_project.data_collection.zones.save_zone", side_effect=OSError("disk full")),                 mock.patch.object(fc, "_known_camera", side_effect=lambda name, *a, **k: name),                 mock.patch.object(fc, "_restart_running_mode", lambda *a, **k: None),                 contextlib.redirect_stdout(out), self.assertRaises(SystemExit) as cm:
            fc.main()
        self.assertEqual(cm.exception.code, 1)
        self.assertIn("disk full", json.loads(out.getvalue())["error"])
