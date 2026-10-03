from __future__ import annotations

import os
import tempfile
import unittest

from home_guard_project.box import alert_settings as al
from home_guard_project.box import camera_alerts as ca
from home_guard_project.box.boxconfig import get_option

CAMERAS = """cameras:
  main_entrance: rtsp://a
  back_door: rtsp://b
disabled:
  old_cam: rtsp://c
"""


class AlertSettingsTest(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.paths = dict(cameras_path=os.path.join(tmp.name, "cameras.yaml"),
                          alerts_path=os.path.join(tmp.name, "camera_alerts.yaml"),
                          box_path=os.path.join(tmp.name, "box.yaml"))
        with open(self.paths["cameras_path"], "w", encoding="utf-8") as f:
            f.write(CAMERAS)
        with open(self.paths["box_path"], "w", encoding="utf-8") as f:
            f.write('site: "test"\ninference_conf: 0.7\n')

    def test_camera_names_are_matched_the_way_the_owner_writes_them(self) -> None:
        self.assertEqual(al.resolve_camera("Main Entrance", self.paths["cameras_path"]), "main_entrance")
        self.assertEqual(al.resolve_camera("back-door", self.paths["cameras_path"]), "back_door")
        self.assertIsNone(al.resolve_camera("house", self.paths["cameras_path"]))
        self.assertIsNone(al.resolve_camera(None, self.paths["cameras_path"]))
        with self.assertRaises(ValueError) as err:
            al.resolve_camera("garage", self.paths["cameras_path"])
        self.assertIn("main_entrance", str(err.exception))

    def test_reading_shows_the_house_and_every_camera(self) -> None:
        out = al.get_alert_settings(**self.paths)
        self.assertEqual(out["house"], {"alert_on": ["person"],
                                        "sensitivity": {"person": 0.7, "vehicle": 0.7, "animal": 0.7}})
        self.assertEqual([r["camera"] for r in out["cameras"]], ["main_entrance", "back_door", "old_cam"])
        self.assertIsNone(out["cameras"][0]["own_alert_on"])

    def test_a_camera_gets_its_own_types_and_goes_back_to_the_default(self) -> None:
        state = al.set_alert_types("Main Entrance", ["vehicle", "person"], **self.paths)
        self.assertEqual(state["alert_on"], ["person", "vehicle"])
        self.assertEqual(state["own_alert_on"], ["person", "vehicle"])
        self.assertEqual(ca.load_camera_alerts(self.paths["alerts_path"]), {"main_entrance": ("person", "vehicle")})
        state = al.set_alert_types("main_entrance", ["default"], **self.paths)
        self.assertEqual(state["alert_on"], ["person"])
        self.assertIsNone(state["own_alert_on"])

    def test_changes_apply_to_the_current_list(self) -> None:
        state = al.set_alert_types("main_entrance", ["+vehicle"], **self.paths)    # people (house) + vehicles
        self.assertEqual(state["alert_on"], ["person", "vehicle"])
        state = al.set_alert_types("main_entrance", "-person", **self.paths)
        self.assertEqual(state["alert_on"], ["vehicle"])
        out = al.set_alert_types("house", ["+animal"], **self.paths)               # house: people -> people, animals
        self.assertEqual(out["house"]["alert_on"], ["person", "animal"])
        with self.assertRaises(ValueError):
            al.set_alert_types("main_entrance", ["-vehicle"], **self.paths)      # nothing would be left
        with self.assertRaises(ValueError):
            al.set_alert_types("main_entrance", ["+dragon"], **self.paths)

    def test_the_house_default_goes_to_box_yaml(self) -> None:
        out = al.set_alert_types(None, "animal,person", **self.paths)
        self.assertEqual(get_option("alert_on", self.paths["box_path"]), "person,animal")
        self.assertEqual(out["house"]["alert_on"], ["person", "animal"])
        with self.assertRaises(ValueError):
            al.set_alert_types("house", ["default"], **self.paths)

    def test_bad_types_are_refused_with_a_plain_message(self) -> None:
        with self.assertRaises(ValueError):
            al.set_alert_types("back_door", ["dragons"], **self.paths)
        with self.assertRaises(ValueError):
            al.set_alert_types("back_door", [], **self.paths)
        self.assertEqual(ca.load_camera_alerts(self.paths["alerts_path"]), {})

    def test_sensitivity_for_a_camera_merges_and_reads_percentages(self) -> None:
        al.set_sensitivity("back_door", {"vehicle": 80}, **self.paths)
        state = al.set_sensitivity("back_door", "person=50%", **self.paths)
        self.assertEqual(state["own_sensitivity"], {"person": 0.5, "vehicle": 0.8})
        self.assertEqual(state["sensitivity"], {"person": 0.5, "vehicle": 0.8, "animal": 0.7})
        state = al.set_sensitivity("back_door", "default", **self.paths)
        self.assertIsNone(state["own_sensitivity"])

    def test_house_sensitivity_goes_to_box_yaml(self) -> None:
        out = al.set_sensitivity("house", {"person": 0.5}, **self.paths)
        self.assertEqual(get_option("conf_person", self.paths["box_path"]), 0.5)
        self.assertEqual(out["house"]["sensitivity"]["person"], 0.5)
        with self.assertRaises(ValueError):
            al.set_sensitivity(None, {"person": 2.5}, **self.paths)     # 250%: out of range

    def test_describe_is_one_plain_line(self) -> None:
        state = al.set_alert_types("main_entrance", ["person", "vehicle"], **self.paths)
        self.assertEqual(al.describe(state), "main_entrance alerts on people and vehicles (its own choice); "
                                             "the detector must be sure: people 70%, vehicles 70%, animals 70%.")


if __name__ == "__main__":
    unittest.main()
