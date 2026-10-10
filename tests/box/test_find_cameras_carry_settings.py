"""2026-10-10: a camera search named every camera afresh under the new site (ameer_week_0_1_* -> ameer_v2_*).

The family's names ("כניסה ראשית", "פרגולה") stayed on the old ids, so the assistant no longer knew the main
entrance, and the zones, scene maps and alert choices of the old ids were left behind too. When a search gives a
stream a fresh name and a setting is still kept under an id that is no longer a camera, for the same channel, the
setting moves to the new name - the same carry-over a rename in the app does.
"""

from __future__ import annotations

import os
import tempfile
import unittest

import yaml

from home_guard_project.box.find_cameras import orphan_renames, write_found

HOST = "192.168.68.109"
SQUARE = [[0.1, 0.1], [0.9, 0.1], [0.9, 0.9], [0.1, 0.9]]


def url(channel: int, host: str = HOST) -> str:
    return f"rtsp://admin:pw@{host}:554/unicast/c{channel}/s0/live"


def stream(channel: int, host: str = HOST) -> dict:
    return {"channel": channel, "url": url(channel, host), "pattern": "x", "w": 1920, "h": 1080}


class CarrySettingsTest(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        d = tmp.name
        self.cameras = os.path.join(d, "cameras.yaml")
        self.aliases = os.path.join(d, "camera_aliases.yaml")
        self.zones = os.path.join(d, "zones.yaml")
        self.scenes = os.path.join(d, "scene_maps.yaml")
        self.alerts = os.path.join(d, "camera_alerts.yaml")
        self._dump(self.aliases, {"ameer_week_0_1_ch3": ["פרגולה"], "ameer_week_0_1_ch6": ["כניסה ראשית"]})
        self._dump(self.zones, {"zones": {"ameer_week_0_1_ch6": SQUARE}})
        self._dump(self.scenes, {"scene_maps": {"ameer_week_0_1_ch3": {"areas": []}}})
        self._dump(self.alerts, {"alerts": {"ameer_week_0_1_ch6": ["person"]}})

    @staticmethod
    def _dump(path: str, data: dict) -> None:
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(data, f, allow_unicode=True)

    @staticmethod
    def _load(path: str) -> dict:
        with open(path, encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    def test_a_search_that_names_cameras_afresh_carries_every_setting_to_the_new_names(self) -> None:
        # No cameras.yaml (lost): every stream gets a fresh name under the new site.
        write_found({HOST: [stream(3), stream(6)]}, "ameer_v2", self.cameras)
        self.assertEqual(self._load(self.aliases), {"ameer_v2_ch3": ["פרגולה"], "ameer_v2_ch6": ["כניסה ראשית"]})
        self.assertEqual(list(self._load(self.zones)["zones"]), ["ameer_v2_ch6"])
        self.assertEqual(list(self._load(self.scenes)["scene_maps"]), ["ameer_v2_ch3"])
        self.assertEqual(self._load(self.alerts)["alerts"], {"ameer_v2_ch6": ["person"]})

    def test_a_known_camera_keeps_its_name_and_its_settings_stay_put(self) -> None:
        self._dump(self.cameras, {"cameras": {"ameer_week_0_1_ch6": url(6)}})
        write_found({HOST: [stream(3), stream(6)]}, "ameer_v2", self.cameras)
        aliases = self._load(self.aliases)
        self.assertEqual(aliases["ameer_week_0_1_ch6"], ["כניסה ראשית"])      # still a camera: untouched
        self.assertEqual(aliases["ameer_v2_ch3"], ["פרגולה"])                  # the fresh one got its old names

    def test_an_old_setting_for_a_channel_not_found_stays_where_it_is(self) -> None:
        write_found({HOST: [stream(3)]}, "ameer_v2", self.cameras)
        self.assertIn("ameer_week_0_1_ch6", self._load(self.aliases))
        self.assertIn("ameer_week_0_1_ch6", self._load(self.zones)["zones"])


class OrphanRenamesTest(unittest.TestCase):
    def test_the_same_channel_ending_matches_one_to_one(self) -> None:
        self.assertEqual(orphan_renames(["ameer_v2_ch3", "ameer_v2_ch6"], ["ameer_v2_ch3", "ameer_v2_ch6"],
                                        {"ameer_week_0_1_ch3", "ameer_week_0_1_ch6"}),
                         {"ameer_week_0_1_ch3": "ameer_v2_ch3", "ameer_week_0_1_ch6": "ameer_v2_ch6"})

    def test_two_old_ids_for_one_channel_are_ambiguous_and_nothing_moves(self) -> None:
        self.assertEqual(orphan_renames(["new_ch3"], ["new_ch3"], {"old_ch3", "older_ch3"}), {})

    def test_a_device_tag_must_match_too(self) -> None:
        # Two recorders: names carry the last number of the address.
        self.assertEqual(orphan_renames(["new_109_ch3", "new_110_ch3"], ["new_109_ch3", "new_110_ch3"],
                                        {"old_109_ch3"}), {"old_109_ch3": "new_109_ch3"})

    def test_owner_names_and_current_cameras_never_move(self) -> None:
        self.assertEqual(orphan_renames(["new_ch3"], ["new_ch3", "main_door"], {"main_door", "front"}), {})


if __name__ == "__main__":
    unittest.main()
