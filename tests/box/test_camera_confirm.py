from __future__ import annotations

import os
import tempfile
import unittest
from unittest import mock

import yaml

from home_guard_project.box import find_cameras as fc


def _write(path, data):
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(data, f)


class ApplyChangesTest(unittest.TestCase):
    def setUp(self) -> None:
        self.dir = tempfile.mkdtemp()
        self.path = os.path.join(self.dir, "cameras.yaml")
        _write(self.path, {"cameras": {
            "house_ch1": "rtsp://u:p@10.0.0.1/s0/1",
            "house_ch2": "rtsp://u:p@10.0.0.1/s0/2",
            "house_ch3": "rtsp://u:p@10.0.0.1/s0/3",
        }})

    def _read(self):
        with open(self.path, encoding="utf-8") as f:
            return yaml.safe_load(f)

    def test_rename_and_disable(self) -> None:
        changes = {"cameras": [
            {"name": "house_ch1", "new_name": "front_door", "enabled": True},
            {"name": "house_ch2", "new_name": "house_ch2", "enabled": False},
        ]}
        with mock.patch("home_guard_project.box.control.request_restart"):
            res = fc.apply_changes(changes, self.path)
        data = self._read()
        self.assertIn("front_door", data["cameras"])
        self.assertNotIn("house_ch1", data["cameras"])
        self.assertNotIn("house_ch2", data["cameras"])
        self.assertIn("house_ch2", data.get("disabled", {}))  # recoverable, not thrown away
        self.assertEqual(data["disabled"]["house_ch2"], "rtsp://u:p@10.0.0.1/s0/2")
        self.assertIn("house_ch3", data["cameras"])            # unmentioned kept active
        self.assertEqual(res["active"], ["front_door", "house_ch3"])

    def test_reenable_from_disabled(self) -> None:
        _write(self.path, {"cameras": {"a": "urlA"}, "disabled": {"b": "urlB"}})
        with mock.patch("home_guard_project.box.control.request_restart"):
            fc.apply_changes({"cameras": [{"name": "b", "new_name": "b", "enabled": True}]}, self.path)
        data = self._read()
        self.assertIn("b", data["cameras"])
        self.assertNotIn("b", data.get("disabled", {}) or {})

    def test_duplicate_name_refused(self) -> None:
        changes = {"cameras": [
            {"name": "house_ch1", "new_name": "dup", "enabled": True},
            {"name": "house_ch2", "new_name": "dup", "enabled": True},
        ]}
        with self.assertRaises(ValueError):
            fc.apply_changes(changes, self.path)
        # nothing was written on error
        self.assertIn("house_ch1", self._read()["cameras"])

    def test_invalid_name_refused(self) -> None:
        with self.assertRaises(ValueError):
            fc.apply_changes({"cameras": [{"name": "house_ch1", "new_name": "Front Door"}]}, self.path)

    def test_unknown_camera_refused(self) -> None:
        with self.assertRaises(ValueError):
            fc.apply_changes({"cameras": [{"name": "nope", "new_name": "x"}]}, self.path)

    def test_restart_requested_on_success(self) -> None:
        with mock.patch("home_guard_project.box.control.request_restart") as rr:
            fc.apply_changes({"cameras": [{"name": "house_ch1", "new_name": "x", "enabled": True}]}, self.path)
        rr.assert_called_once()

    def test_atomic_write_uses_temp(self) -> None:
        # after a successful apply there is no leftover .tmp file
        with mock.patch("home_guard_project.box.control.request_restart"):
            fc.apply_changes({"cameras": [{"name": "house_ch1", "new_name": "x", "enabled": True}]}, self.path)
        self.assertFalse(os.path.exists(self.path + ".tmp"))


if __name__ == "__main__":
    unittest.main()
