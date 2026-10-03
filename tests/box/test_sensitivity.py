from __future__ import annotations

import os
import tempfile
import unittest

from home_guard_project.box import camera_alerts as ca
from home_guard_project.box import find_cameras as fc
from home_guard_project.box import inference as inf
from home_guard_project.box.boxconfig import BoxConfigError, LIVE_OPTIONS, set_option
from home_guard_project.box.inference import AlertSettings, apply_live_settings

URL = "rtsp://admin:s3cret@192.168.1.50:554/unicast/c1/s0/live"
NOW = 1_800_000_000.0
NAMES = {0: "person", 2: "car", 14: "bird", 15: "cat", 24: "backpack"}


class _Box:
    def __init__(self, cls_id: int, conf: float) -> None:
        self.cls = [cls_id]
        self.conf = [conf]
        self.xyxyn = [[0.1, 0.1, 0.2, 0.2]]


class _Result:
    def __init__(self, found) -> None:
        self.boxes = [_Box(c, p) for c, p in found]
        self.names = NAMES


class HouseSensitivityTest(unittest.TestCase):
    def test_each_type_falls_back_to_the_single_threshold(self) -> None:
        s = AlertSettings.from_box_settings({"inference_conf": 0.7})
        self.assertEqual(s.thresholds(), {"person": 0.7, "vehicle": 0.7, "animal": 0.7})
        s = AlertSettings.from_box_settings({"inference_conf": 0.7, "conf_person": 0.5, "conf_vehicle": 0.8})
        self.assertEqual(s.thresholds(), {"person": 0.5, "vehicle": 0.8, "animal": 0.7})

    def test_the_window_sees_each_type(self) -> None:
        s = AlertSettings(conf=0.6, conf_person=0.45)
        self.assertEqual(s.live_values()["sensitivity"], {"person": 0.45, "vehicle": 0.6, "animal": 0.6})

    def test_per_type_values_are_live_options_within_range(self) -> None:
        for key in ("conf_person", "conf_vehicle", "conf_animal"):
            self.assertIn(key, LIVE_OPTIONS)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "box.yaml")
            self.assertEqual(set_option("conf_person", "0.5", path), 0.5)
            with self.assertRaises(BoxConfigError):
                set_option("conf_vehicle", "0.99", path)

    def test_a_change_is_taken_over_while_running(self) -> None:
        s = AlertSettings(conf=0.7)
        self.assertEqual(apply_live_settings(s, {"inference_conf": 0.7, "conf_person": 0.5}), ["conf_person"])
        self.assertEqual(s.thresholds()["person"], 0.5)


class FilterTest(unittest.TestCase):
    TH = {"person": 0.5, "vehicle": 0.8, "animal": 0.6}

    def test_each_find_is_held_to_its_own_types_threshold(self) -> None:
        result = _Result([(0, 0.55), (2, 0.75), (2, 0.85), (15, 0.59), (14, 0.99)])
        kept = inf.filter_by_thresholds(result, self.TH, other=0.7)
        self.assertEqual([(int(b.cls[0]), b.conf[0]) for b in kept.boxes], [(0, 0.55), (2, 0.85), (14, 0.99)])
        self.assertIs(kept.names, NAMES)

    def test_a_person_the_detector_is_unsure_of_is_no_longer_hidden_behind_a_car(self) -> None:
        result = _Result([(0, 0.55), (2, 0.9)])
        person, vehicle, labels = inf.detect_trigger(inf.filter_by_thresholds(result, self.TH, other=0.7))
        self.assertTrue(person)

    def test_the_detector_is_asked_down_to_the_lowest_threshold(self) -> None:
        self.assertEqual(inf.detector_floor(self.TH, other=0.7), 0.5)


class CameraSensitivityTest(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.path = os.path.join(tmp.name, "camera_alerts.yaml")
        self.cameras = os.path.join(tmp.name, "cameras.yaml")
        self.box = os.path.join(tmp.name, "box.yaml")
        with open(self.box, "w", encoding="utf-8") as f:
            f.write('site: "test"\ninference_conf: 0.7\nconf_person: 0.5\n')
        fc._write_cameras({"yard": URL, "driveway": URL}, {}, self.cameras)

    def test_a_camera_overrides_some_types_and_inherits_the_rest(self) -> None:
        ca.set_camera_sensitivity("driveway", {"vehicle": 0.9}, self.path)
        own = ca.load_camera_sensitivity(self.path)
        self.assertEqual(own, {"driveway": {"vehicle": 0.9}})
        house = {"person": 0.5, "vehicle": 0.7, "animal": 0.7}
        self.assertEqual(ca.thresholds_for(own, "driveway", house), {"person": 0.5, "vehicle": 0.9, "animal": 0.7})
        self.assertEqual(ca.thresholds_for(own, "yard", house), house)

    def test_bad_values_are_refused_and_damaged_entries_ignored(self) -> None:
        for bad in ({}, {"person": 1.2}, {"dragon": 0.5}, {"person": "high"}):
            with self.assertRaises(ValueError, msg=repr(bad)):
                ca.set_camera_sensitivity("yard", bad, self.path)
        with open(self.path, "w", encoding="utf-8") as f:
            f.write("sensitivity:\n  yard: {person: 0.5}\n  gate: {person: 7}\n")
        self.assertEqual(ca.load_camera_sensitivity(self.path), {"yard": {"person": 0.5}})

    def test_alert_choices_and_sensitivity_live_side_by_side(self) -> None:
        ca.set_camera_alerts("yard", "person,animal", self.path)
        ca.set_camera_sensitivity("yard", {"animal": 0.4}, self.path)
        ca.clear_camera_alerts("yard", self.path)
        self.assertEqual(ca.load_camera_sensitivity(self.path), {"yard": {"animal": 0.4}})
        ca.clear_camera_sensitivity("yard", self.path)
        ca.set_camera_alerts("yard", "vehicle", self.path)
        self.assertEqual(ca.load_camera_alerts(self.path), {"yard": ("vehicle",)})
        self.assertEqual(ca.load_camera_sensitivity(self.path), {})

    def test_renames_carry_sensitivity_too(self) -> None:
        ca.set_camera_sensitivity("driveway", {"vehicle": 0.9}, self.path)
        ca.remap_camera_alerts({"driveway": "front_drive"}, self.path)
        self.assertEqual(ca.load_camera_sensitivity(self.path), {"front_drive": {"vehicle": 0.9}})

    def test_live_reload_sees_sensitivity(self) -> None:
        live = ca.LiveCameraAlerts(self.path, poll_sec=2.0, now=NOW)
        ca.set_camera_sensitivity("yard", {"person": 0.4}, self.path)
        os.utime(self.path, (NOW + 1, NOW + 1))
        self.assertTrue(live.check(NOW + 3))
        self.assertEqual(live.thresholds_for("yard", {"person": 0.5, "vehicle": 0.7, "animal": 0.7})["person"], 0.4)

    def test_commands_report_house_and_camera_sensitivity(self) -> None:
        out = fc.set_camera_sensitivity_command("driveway", "vehicle=0.9,person=0.45", self.cameras, self.path)
        self.assertEqual(out, {"camera": "driveway", "sensitivity": {"person": 0.45, "vehicle": 0.9}})
        rows = fc.camera_alert_rows(self.cameras, self.path, self.box)
        self.assertEqual(rows["house_sensitivity"], {"person": 0.5, "vehicle": 0.7, "animal": 0.7})
        by_name = {r["name"]: r["sensitivity"] for r in rows["cameras"]}
        self.assertEqual(by_name, {"driveway": {"person": 0.45, "vehicle": 0.9}, "yard": None})
        self.assertEqual(fc.set_camera_sensitivity_command("driveway", None, self.cameras, self.path),
                         {"camera": "driveway", "sensitivity": None})
        with self.assertRaises(ValueError):
            fc.set_camera_sensitivity_command("driveway", "person=2", self.cameras, self.path)


if __name__ == "__main__":
    unittest.main()
