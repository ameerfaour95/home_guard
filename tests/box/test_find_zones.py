from __future__ import annotations

import os
import tempfile
import unittest

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
