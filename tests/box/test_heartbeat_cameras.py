"""The heartbeat's camera_list: every configured camera with the name the owner reads, never its stream URL."""
from __future__ import annotations

import json
import os
import tempfile
import unittest

from home_guard_project.box.heartbeat import build_heartbeat, camera_list

NOW = 1_800_000_000.0
CAMERAS = """\
cameras:
  ameer_week_0_1_ch1: rtsp://admin:secret@192.168.1.10:554/ch1
  ameer_week_0_1_ch6: rtsp://admin:secret@192.168.1.10:554/ch6
  front_door: rtsp://admin:secret@192.168.1.11:554/main
disabled:
  ameer_week_0_1_ch3: rtsp://admin:secret@192.168.1.10:554/ch3
"""
ALIASES = """\
ameer_week_0_1_ch1:
- old name
- פרגולה
"""


class CameraListTest(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = tmp.name
        self.cameras = os.path.join(tmp.name, "cameras.yaml")
        self.aliases = os.path.join(tmp.name, "camera_aliases.yaml")
        with open(self.cameras, "w", encoding="utf-8") as f:
            f.write(CAMERAS)
        with open(self.aliases, "w", encoding="utf-8") as f:
            f.write(ALIASES)

    def test_names_channels_and_disabled_cameras(self) -> None:
        self.assertEqual(camera_list(self.cameras, "he", self.aliases), [
            {"id": "ameer_week_0_1_ch1", "name": "פרגולה", "channel": "1", "enabled": True},
            {"id": "ameer_week_0_1_ch6", "name": "מצלמה 6", "channel": "6", "enabled": True},
            {"id": "front_door", "name": "front door", "channel": None, "enabled": True},
            {"id": "ameer_week_0_1_ch3", "name": "מצלמה 3", "channel": "3", "enabled": False},
        ])
        self.assertEqual(camera_list(self.cameras, "en", self.aliases)[1]["name"], "Camera 6")

    def test_the_heartbeat_carries_it_beside_the_clip_counts_and_no_login(self) -> None:
        live = os.path.join(self.tmp, "live")
        os.makedirs(live)
        hb = build_heartbeat("house2", live, os.path.join(self.tmp, "outbox"), os.path.join(self.tmp, "alive"),
                             now=NOW, cameras_path=self.cameras, lang="he", aliases_path=self.aliases)
        self.assertEqual(hb["cameras"], {})                          # the existing field is unchanged
        self.assertEqual([c["id"] for c in hb["camera_list"]],
                         ["ameer_week_0_1_ch1", "ameer_week_0_1_ch6", "front_door", "ameer_week_0_1_ch3"])
        text = json.dumps(hb, ensure_ascii=False)
        self.assertNotIn("secret", text)
        self.assertNotIn("rtsp", text)

    def test_without_a_cameras_file_the_list_is_empty_and_without_a_path_absent(self) -> None:
        self.assertEqual(camera_list(os.path.join(self.tmp, "missing.yaml")), [])
        live = os.path.join(self.tmp, "live")
        os.makedirs(live)
        hb = build_heartbeat("house2", live, os.path.join(self.tmp, "outbox"), os.path.join(self.tmp, "alive"), now=NOW)
        self.assertNotIn("camera_list", hb)

    def test_a_damaged_cameras_file_is_an_empty_list(self) -> None:
        with open(self.cameras, "w", encoding="utf-8") as f:
            f.write("cameras: [unclosed\n")
        self.assertEqual(camera_list(self.cameras, "he", self.aliases), [])


if __name__ == "__main__":
    unittest.main()
