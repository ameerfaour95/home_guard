# tests/box/test_brain_registry.py
from __future__ import annotations

import datetime as dt
import json
import os
import tempfile
import unittest

import yaml

from home_guard_project.box.brain.aliases import add_alias
from home_guard_project.box.brain.registry import (
    CameraState,
    HouseRegistry,
    HouseSnapshot,
    render_block,
    resolve_camera,
)
from home_guard_project.data_collection.zones import save_zones
from home_guard_project.box.feedback import Feedback, MuteState

NOW = dt.datetime(2026, 10, 3, 23, 0).timestamp()


def snap(*cams: CameraState) -> HouseSnapshot:
    return HouseSnapshot(now=NOW, mode="guard", mode_ends=NOW + 7 * 3600, mode_started=NOW - 3600,
                         start_hour=22, end_hour=6, cameras=tuple(cams))


CAMS = snap(
    CameraState("main_entrance", True, ("entrance", "front door", "כניסה"), live=True),
    CameraState("back_door", False, ("back", "אחורית"), live=False),
    CameraState("front_side", True, ("front", "street", "קדמית"), live=True),
    CameraState("test_ch8", True, (), live=True),
)


class ResolveTest(unittest.TestCase):
    def test_exact_name_and_alias_in_any_language(self) -> None:
        self.assertEqual(resolve_camera(CAMS, "main_entrance").camera, "main_entrance")
        self.assertEqual(resolve_camera(CAMS, "Main Entrance").camera, "main_entrance")
        self.assertEqual(resolve_camera(CAMS, "כניסה").camera, "main_entrance")
        self.assertEqual(resolve_camera(CAMS, "back").camera, "back_door")

    def test_alias_inside_a_phrase(self) -> None:
        self.assertEqual(resolve_camera(CAMS, "the back camera").camera, "back_door")
        self.assertEqual(resolve_camera(CAMS, "המצלמה אחורית").camera, "back_door")

    def test_hebrew_prefixes(self) -> None:
        self.assertEqual(resolve_camera(CAMS, "המצלמה הקדמית").camera, "front_side")
        self.assertEqual(resolve_camera(CAMS, "בכניסה").camera, "main_entrance")

    def test_a_bare_number_matches_the_channel(self) -> None:
        self.assertEqual(resolve_camera(CAMS, "8").camera, "test_ch8")

    def test_ambiguity_returns_the_candidates(self) -> None:
        res = resolve_camera(CAMS, "front door and street")
        self.assertIsNone(res.camera)
        self.assertEqual(set(res.candidates), {"main_entrance", "front_side"})

    def test_no_match(self) -> None:
        self.assertEqual(resolve_camera(CAMS, "garage"), resolve_camera(CAMS, "garage"))
        self.assertIsNone(resolve_camera(CAMS, "garage").camera)
        self.assertEqual(resolve_camera(CAMS, "garage").candidates, ())
        self.assertIsNone(resolve_camera(CAMS, "").camera)


class RenderTest(unittest.TestCase):
    def test_block_lists_state_aliases_and_mode(self) -> None:
        block = render_block(snap(
            CameraState("main_entrance", True, ("entrance",), live=True, sees="driveway and the front gate",
                        zone=True),
            CameraState("back_door", False, ("back",), live=False),
            CameraState("front_side", True, (), live=False, last_seen=NOW - 600, muted_until=NOW + 9 * 3600),
        ))
        self.assertIn("CAMERAS (3 · 1 live)", block)
        self.assertIn("main_entrance  aka: entrance  live  alerts on", block)
        self.assertIn("  sees: driveway and the front gate", block)
        self.assertIn("  watches only the drawn area", block)
        self.assertIn("back_door  aka: back  OFF", block)
        self.assertIn("front_side  offline (last seen 22:50)  alerts paused until 08:00", block)
        self.assertIn("MODE: Guard until 06:00", block)


class RegistryTest(unittest.TestCase):
    def setUp(self) -> None:
        d = tempfile.mkdtemp()
        self.cameras = os.path.join(d, "cameras.yaml")
        with open(self.cameras, "w", encoding="utf-8") as f:
            yaml.safe_dump({"cameras": {"main_entrance": "rtsp://a", "front_side": "rtsp://b"},
                            "disabled": {"back_door": "rtsp://c"}}, f, sort_keys=False)
        self.aliases = os.path.join(d, "aliases.yaml")
        add_alias("main_entrance", "entrance", ["main_entrance", "front_side", "back_door"], self.aliases)
        self.status = os.path.join(d, "ai_status.json")
        with open(self.status, "w", encoding="utf-8") as f:
            json.dump({"cameras": {"main_entrance": {"checked_ts": NOW - 5},
                                   "front_side": {"checked_ts": NOW - 300}}}, f)
        self.sees = os.path.join(d, "sees.json")
        with open(self.sees, "w", encoding="utf-8") as f:
            json.dump({"cameras": {"main_entrance": {"text": "the front gate", "ts": NOW}}}, f)
        self.mute = MuteState(os.path.join(d, "mute.json"))
        self.mute.apply(Feedback(action="mute", mute_until=NOW + 3600, camera="front_side"), NOW)
        self.zones = os.path.join(d, "zones.yaml")
        save_zones({"main_entrance": [[0.1, 0.1], [0.9, 0.1], [0.5, 0.9]]}, self.zones)

    def _registry(self, **kw) -> HouseRegistry:
        return HouseRegistry(self.mute, hours=lambda: (22, 6), cameras_path=self.cameras,
                             aliases_path=self.aliases, status_path=self.status, sees_path=self.sees,
                             now=lambda: NOW, zones_path=self.zones, **kw)

    def test_snapshot_reads_every_source(self) -> None:
        s = self._registry().snapshot()
        self.assertEqual(s.mode, "guard")
        self.assertEqual(s.names, ["main_entrance", "front_side", "back_door"])
        main, front, back = (s.camera(n) for n in s.names)
        self.assertEqual((main.live, main.aliases, main.sees, main.zone), (True, ("entrance",), "the front gate", True))
        self.assertFalse(front.zone)
        self.assertEqual((front.live, front.muted_until), (False, NOW + 3600))
        self.assertEqual((back.enabled, back.live), (False, False))
        self.assertEqual(s.offline_names, ["front_side"])
        self.assertEqual(s.paused, [("front_side", NOW + 3600)])

    def test_missing_status_means_live_is_unknown(self) -> None:
        os.remove(self.status)
        s = self._registry().snapshot()
        self.assertIsNone(s.camera("main_entrance").live)
        self.assertEqual(s.offline_names, [])

    def test_unreadable_cameras_file_is_reported(self) -> None:
        with open(self.cameras, "w", encoding="utf-8") as f:
            f.write("cameras: [unclosed")
        s = self._registry().snapshot()
        self.assertFalse(s.state_known)
        self.assertIn("camera state unknown", render_block(s))

    def _write(self, path: str, text: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)

    def test_cameras_section_as_a_list_does_not_raise(self) -> None:
        self._write(self.cameras, "cameras: [a, b]\ndisabled: {back_door: rtsp://c}\n")
        s = self._registry().snapshot()
        self.assertTrue(set(s.names) <= {"back_door"})

    def test_disabled_scalar_does_not_raise(self) -> None:
        self._write(self.cameras, "cameras: {main_entrance: rtsp://a}\ndisabled: 5\n")
        self.assertEqual(self._registry().snapshot().names, ["main_entrance"])

    def test_bad_box_hours_guard_all_day(self) -> None:
        from home_guard_project.box.brain.registry import hours_from_box_yaml

        d = tempfile.mkdtemp()
        for text in ('alert_start_hour: "x"\nalert_end_hour: 6\n', "alert_start_hour: 25\nalert_end_hour: 6\n"):
            path = os.path.join(d, "box.yaml")
            self._write(path, text)
            self.assertEqual(hours_from_box_yaml(path)(), (0, 0))

    def test_bad_hours_callable_does_not_raise(self) -> None:
        reg = HouseRegistry(self.mute, hours=lambda: (_ for _ in ()).throw(ValueError("x")),
                            cameras_path=self.cameras, aliases_path=self.aliases, status_path=self.status,
                            now=lambda: NOW, zones_path=self.zones)
        s = reg.snapshot()
        self.assertEqual((s.start_hour, s.end_hour, s.mode), (0, 0, "guard"))
        self.assertFalse(s.state_known)
        self.assertEqual(s.cameras, ())

    def test_a_new_picture_decides_live_not_the_detectors_last_look(self) -> None:
        # The detector looked a moment ago, but at a frozen picture: the camera sent nothing new for minutes.
        self._write(self.status, json.dumps({"cameras": {
            "main_entrance": {"checked_ts": NOW - 1, "frame_ts": NOW - 300},
            "front_side": {"checked_ts": NOW - 300, "frame_ts": NOW - 2}}}))
        s = self._registry().snapshot()
        self.assertEqual((s.camera("main_entrance").live, s.camera("front_side").live), (False, True))
        self.assertEqual(s.camera("front_side").last_seen, NOW - 2)

    def test_non_numeric_checked_ts_means_no_timestamp(self) -> None:
        self._write(self.status, json.dumps({"cameras": {"main_entrance": {"checked_ts": "soon"},
                                                         "front_side": "junk"}}))
        s = self._registry().snapshot()
        main = s.camera("main_entrance")
        self.assertIsNone(main.last_seen)
        self.assertFalse(main.live)

    def test_sees_file_holding_a_list_is_ignored(self) -> None:
        self._write(self.sees, '["x"]')
        s = self._registry().snapshot()
        self.assertEqual([c.sees for c in s.cameras], ["", "", ""])
        self._write(self.sees, json.dumps({"cameras": {"main_entrance": "text"}}))
        self.assertEqual(self._registry().snapshot().camera("main_entrance").sees, "")

    def test_bad_quiet_since_is_none(self) -> None:
        path = os.path.join(os.path.dirname(self.cameras), "quiet.json")
        self._write(path, '{"since": "later"}')
        s = self._registry(quiet_log=lambda: True, quiet_since_path=path).snapshot()
        self.assertTrue(s.quiet_log)
        self.assertIsNone(s.quiet_since)


if __name__ == "__main__":
    unittest.main()
