from __future__ import annotations

import contextlib
import io
import json
import os
import sys
import tempfile
import unittest
from unittest import mock

from home_guard_project.box import camera_alerts as ca
from home_guard_project.box import find_cameras as fc

URL = "rtsp://admin:s3cret@192.168.1.50:554/unicast/c1/s0/live"
NOW = 1_800_000_000.0


class ParseTypesTest(unittest.TestCase):
    def test_types_are_normalised_to_a_fixed_order(self) -> None:
        self.assertEqual(ca.parse_types("vehicle, Person"), ("person", "vehicle"))
        self.assertEqual(ca.parse_types(["animal", "person"]), ("person", "animal"))
        self.assertEqual(ca.parse_types("animal,vehicle,person"), ("person", "vehicle", "animal"))

    def test_empty_or_unknown_types_are_refused(self) -> None:
        for bad in ("", ",", "cats", "person,bird", None, 5, []):
            with self.assertRaises(ValueError, msg=repr(bad)):
                ca.parse_types(bad)


class StoreTest(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.path = os.path.join(tmp.name, "camera_alerts.yaml")

    def test_no_file_means_every_camera_uses_the_house_default(self) -> None:
        self.assertEqual(ca.load_camera_alerts(self.path), {})
        self.assertEqual(ca.effective({}, "yard", ("person",)), ("person",))

    def test_a_camera_keeps_its_own_choice_and_others_use_the_house(self) -> None:
        ca.set_camera_alerts("driveway", "person,vehicle", self.path)
        overrides = ca.load_camera_alerts(self.path)
        self.assertEqual(overrides, {"driveway": ("person", "vehicle")})
        self.assertEqual(ca.effective(overrides, "driveway", ("person",)), ("person", "vehicle"))
        self.assertEqual(ca.effective(overrides, "yard", ("person",)), ("person",))

    def test_back_to_the_house_default(self) -> None:
        ca.set_camera_alerts("driveway", "vehicle", self.path)
        self.assertTrue(ca.clear_camera_alerts("driveway", self.path))
        self.assertFalse(ca.clear_camera_alerts("driveway", self.path))
        self.assertEqual(ca.load_camera_alerts(self.path), {})

    def test_a_damaged_file_or_entry_falls_back_to_the_house_default(self) -> None:
        with open(self.path, "w", encoding="utf-8") as f:
            f.write("alerts: [oops\n")
        self.assertEqual(ca.load_camera_alerts(self.path), {})
        with open(self.path, "w", encoding="utf-8") as f:
            f.write("alerts:\n  yard: [person]\n  gate: [dragons]\n  porch: []\n")
        self.assertEqual(ca.load_camera_alerts(self.path), {"yard": ("person",)})

    def test_renames_carry_each_choice_swaps_included(self) -> None:
        ca.set_camera_alerts("a", "vehicle", self.path)
        ca.set_camera_alerts("b", "animal", self.path)
        ca.set_camera_alerts("c", "person,vehicle", self.path)
        ca.remap_camera_alerts({"a": "b", "b": "a", "c": "d"}, self.path)
        self.assertEqual(ca.load_camera_alerts(self.path),
                         {"a": ("animal",), "b": ("vehicle",), "d": ("person", "vehicle")})

    def test_a_name_taken_over_by_another_camera_loses_the_old_choice(self) -> None:
        ca.set_camera_alerts("gate", "vehicle", self.path)
        ca.remap_camera_alerts({"yard": "gate"}, self.path)          # yard had no choice of its own
        self.assertEqual(ca.load_camera_alerts(self.path), {})


class LiveCameraAlertsTest(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.path = os.path.join(tmp.name, "camera_alerts.yaml")

    def _write(self, text: str, mtime: float) -> None:
        with open(self.path, "w", encoding="utf-8") as f:
            f.write(text)
        os.utime(self.path, (mtime, mtime))

    def test_a_change_is_taken_over_while_running(self) -> None:
        self._write("alerts:\n  driveway: [person]\n", NOW - 100)
        live = ca.LiveCameraAlerts(self.path, poll_sec=2.0, now=NOW)
        self.assertEqual(live.overrides, {"driveway": ("person",)})
        self._write("alerts:\n  driveway: [person, vehicle]\n", NOW)
        self.assertFalse(live.check(NOW + 1.0))                   # too soon to look
        self.assertTrue(live.check(NOW + 2.5))
        self.assertEqual(live.for_camera("driveway", ("person",)), ("person", "vehicle"))
        self.assertFalse(live.check(NOW + 5.0))                   # untouched since

    def test_the_file_disappearing_means_the_house_default(self) -> None:
        self._write("alerts:\n  driveway: [vehicle]\n", NOW - 100)
        live = ca.LiveCameraAlerts(self.path, poll_sec=2.0, now=NOW)
        os.remove(self.path)
        self.assertTrue(live.check(NOW + 3))
        self.assertEqual(live.for_camera("driveway", ("person",)), ("person",))


class CommandsTest(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.cameras = os.path.join(tmp.name, "cameras.yaml")
        self.alerts = os.path.join(tmp.name, "camera_alerts.yaml")
        self.box = os.path.join(tmp.name, "box.yaml")
        with open(self.box, "w", encoding="utf-8") as f:
            f.write('site: "test"\nalert_on: "person,animal"\n')
        fc._write_cameras({"yard": URL, "driveway": URL}, {"old_cam": URL}, self.cameras)

    def test_rows_show_the_house_default_and_each_cameras_own_choice(self) -> None:
        fc.set_camera_alerts_command("driveway", "vehicle,person", self.cameras, self.alerts)
        out = fc.camera_alert_rows(self.cameras, self.alerts, self.box)
        self.assertEqual(out["house"], ["person", "animal"])
        rows = {r["name"]: r["alert_on"] for r in out["cameras"]}
        self.assertEqual(rows, {"driveway": ["person", "vehicle"], "yard": None, "old_cam": None})

    def test_unset_house_default_is_people(self) -> None:
        with open(self.box, "w", encoding="utf-8") as f:
            f.write('site: "test"\n')
        self.assertEqual(fc.camera_alert_rows(self.cameras, self.alerts, self.box)["house"], ["person"])

    def test_set_and_default_report_what_was_stored(self) -> None:
        self.assertEqual(fc.set_camera_alerts_command("yard", "animal", self.cameras, self.alerts),
                         {"camera": "yard", "alert_on": ["animal"]})
        self.assertEqual(fc.set_camera_alerts_command("yard", None, self.cameras, self.alerts),
                         {"camera": "yard", "alert_on": None})
        self.assertEqual(ca.load_camera_alerts(self.alerts), {})

    def test_unknown_camera_or_types_are_refused_and_nothing_is_written(self) -> None:
        with self.assertRaises(ValueError):
            fc.set_camera_alerts_command("street", "person", self.cameras, self.alerts)
        with self.assertRaises(ValueError):
            fc.set_camera_alerts_command("yard", "dragons", self.cameras, self.alerts)
        self.assertFalse(os.path.exists(self.alerts))

    def test_renaming_a_camera_carries_its_choice(self) -> None:
        fc.set_camera_alerts_command("driveway", "vehicle", self.cameras, self.alerts)
        with mock.patch.object(fc, "CAMERA_ALERTS_PATH", self.alerts):
            fc.apply_changes({"cameras": [{"name": "driveway", "new_name": "front_drive", "enabled": True}]},
                             self.cameras, zones_path=os.path.join(os.path.dirname(self.cameras), "zones.yaml"),
                             restart=False)
        self.assertEqual(ca.load_camera_alerts(self.alerts), {"front_drive": ("vehicle",)})

    def test_the_cli_prints_json_for_the_app(self) -> None:
        argv = ["find_cameras", "--json", "set-camera-alerts", "--camera", "yard", "--on", "person,vehicle"]
        out = io.StringIO()
        with mock.patch.object(sys, "argv", argv), mock.patch.object(fc, "CAMERAS_PATH", self.cameras), \
                mock.patch.object(fc, "CAMERA_ALERTS_PATH", self.alerts), contextlib.redirect_stdout(out):
            fc.main()
        self.assertEqual(json.loads(out.getvalue()), {"camera": "yard", "alert_on": ["person", "vehicle"]})


if __name__ == "__main__":
    unittest.main()
