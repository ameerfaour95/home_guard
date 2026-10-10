"""The scene interview's command line for the app's map editor: --embed, --text-b64 and confirm --map-b64."""
from __future__ import annotations

import base64
import contextlib
import io
import json
import os
import tempfile
import unittest
import zlib
from unittest import mock

import numpy as np

from home_guard_project.box import find_cameras
from home_guard_project.box import scene_interview as si
from home_guard_project.box import scene_map as sm
from home_guard_project.data_collection import zones as z

LAWN = [[0.0, 0.5], [0.5, 0.5], [0.5, 1.0], [0.0, 1.0]]
GATE = [[0.6, 0.2], [0.9, 0.2], [0.9, 0.6], [0.6, 0.6]]


def picture(h: int = 120, w: int = 160) -> np.ndarray:
    img = np.full((h, w, 3), 128, np.uint8)
    img[60:, :80] = (40, 160, 40)
    return img


def b64(data) -> str:
    return base64.b64encode(json.dumps(data, ensure_ascii=False).encode("utf-8")).decode("ascii")


def b64z(data) -> str:
    return base64.b64encode(zlib.compress(json.dumps(data, ensure_ascii=False).encode("utf-8"))).decode("ascii")


class AppCommandLineTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.cameras = os.path.join(self.tmp.name, "cameras.yaml")
        with open(self.cameras, "w", encoding="utf-8") as f:
            f.write("cameras:\n  front: rtsp://x\n  back: rtsp://y\ndisabled:\n  ameer_week_0_1_ch3: rtsp://z\n")
        self.zones = os.path.join(self.tmp.name, "zones.yaml")
        self.out = os.path.join(self.tmp.name, "interview")
        self.picture_path = os.path.join(self.tmp.name, "front_clean.jpg")
        import cv2

        cv2.imwrite(self.picture_path, picture())
        self.restart = mock.Mock()

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def run_main(self, *argv):
        out = io.StringIO()
        with mock.patch.object(find_cameras, "CAMERAS_PATH", self.cameras), \
                mock.patch.object(find_cameras, "_restart_running_mode", self.restart), \
                mock.patch.object(z, "ZONES_PATH", self.zones), \
                mock.patch.object(si, "interview_picture", return_value=(picture(), self.picture_path)), \
                mock.patch("home_guard_project.box.camera_names._load", return_value={"front": ["הכניסה"]}), \
                contextlib.redirect_stdout(out):
            code = si.main(list(argv))
        text = out.getvalue()
        return code, text, json.loads(text)

    def test_propose_embed_carries_both_pictures_points_and_the_current_map(self) -> None:
        z.save_zone("front", [(0, 0), (0.5, 0), (0.5, 1), (0, 1)], self.zones)
        code, text, data = self.run_main("propose", "--camera", "front", "--json", "--embed", "--grid", "--out", self.out)
        self.assertEqual(code, 0)
        self.assertEqual(len(text.strip().splitlines()), 1)                  # one line
        text.encode("ascii")                                                 # ASCII, whatever the code page
        self.assertEqual(data["method"], "grid")
        self.assertEqual(base64.b64decode(data["picture_b64"]), open(self.picture_path, "rb").read())
        self.assertEqual(base64.b64decode(data["image_b64"]), open(data["image"], "rb").read())
        self.assertEqual(len(data["regions"]), si.GRID_ROWS * si.GRID_COLS)
        first = data["regions"][0]
        self.assertEqual(set(first), {"number", "area", "points"})
        self.assertGreaterEqual(len(first["points"]), 3)
        self.assertEqual(data["current_map"]["watched"], [[0.0, 0.0], [0.5, 0.0], [0.5, 1.0], [0.0, 1.0]])

    def test_propose_without_embed_is_unchanged(self) -> None:
        code, _text, data = self.run_main("propose", "--camera", "front", "--json", "--grid", "--out", self.out)
        self.assertEqual(code, 0)
        self.assertNotIn("picture_b64", data)
        self.assertEqual(set(data["regions"][0]), {"number", "area"})

    def test_answer_reads_base64_hebrew_words_and_embeds_the_ownership_picture(self) -> None:
        self.run_main("propose", "--camera", "front", "--json", "--embed", "--grid", "--out", self.out)
        words = base64.b64encode("1 שלנו 'הדשא', 2 של השכן".encode("utf-8")).decode("ascii")
        code, text, data = self.run_main("answer", "--camera", "front", "--text-b64", words, "--json", "--embed",
                                         "--out", self.out)
        self.assertEqual(code, 0)
        text.encode("ascii")
        self.assertEqual([a["name"] for a in data["map"]["areas"]][:1], ["הדשא"])
        self.assertEqual(data["map"]["areas"][1]["owner"], "neighbour")
        self.assertTrue(base64.b64decode(data["image_b64"]).startswith(b"\xff\xd8"))   # a JPEG

    def test_answer_with_unreadable_base64_is_a_plain_error(self) -> None:
        self.run_main("propose", "--camera", "front", "--json", "--grid", "--out", self.out)
        code, _text, data = self.run_main("answer", "--camera", "front", "--text-b64", "not base64!", "--json",
                                          "--out", self.out)
        self.assertEqual(code, 1)
        self.assertEqual(data["error"], "the answers could not be read")

    def app_map(self, **extra):
        data = {"camera": "front",
                "areas": [{"name": "הדשא", "kind": "mine", "zone": "yard", "points": LAWN},
                          {"name": "gate", "kind": "watch_no_alert", "owner": "neighbour", "zone": "gate",
                           "points": GATE},
                          {"name": "window", "kind": "black", "zone": "window",
                           "points": [[0.1, 0.1], [0.2, 0.1], [0.2, 0.2]]}],
                "lines": [{"name": "המעקה", "a": [0.55, 0.0], "b": [0.55, 1.0], "inward": "left"}]}
        data.update(extra)
        return data

    def test_confirm_map_b64_saves_the_apps_map_and_restarts_once_when_the_mask_changed(self) -> None:
        code, text, data = self.run_main("confirm", "--camera", "front", "--map-b64", b64z(self.app_map()), "--json",
                                         "--out", self.out)
        self.assertEqual(code, 0, data)
        text.encode("ascii")
        self.assertTrue(data["restart_needed"])                  # a new black area
        self.restart.assert_called_once()
        saved = sm.load_scene_map("front", self.zones)
        self.assertEqual([(a.name, a.kind, a.ground) for a in saved.areas],
                         [("הדשא", "mine", "mine"), ("gate", "watch_no_alert", "neighbour"), ("window", "black", "")])
        self.assertEqual([(ln.name, ln.inward) for ln in saved.lines], [("המעקה", "left")])
        self.assertTrue(saved.confirmed)
        self.assertFalse(os.path.exists(si.draft_path("front", self.out)))

    def test_confirm_map_b64_plain_json_and_no_restart(self) -> None:
        code, _text, data = self.run_main("confirm", "--camera", "front", "--map-b64", b64(self.app_map()), "--json",
                                          "--out", self.out,
                                          "--no-restart")
        self.assertEqual(code, 0, data)
        self.assertTrue(data["restart_needed"])
        self.restart.assert_not_called()

    def test_confirm_map_b64_is_the_whole_truth(self) -> None:
        # Today's zone becomes an explicit mine area, its zones.yaml entry goes, the rest becomes the neighbour's.
        z.save_zone("front", [(0, 0), (0.5, 0), (0.5, 1), (0, 1)], self.zones)
        z.save_zone("back", [(0, 0), (1, 0), (1, 1)], self.zones)
        mine_only = {"camera": "front", "areas": [{"name": "lawn", "kind": "mine", "zone": "yard", "points": LAWN}]}
        code, _text, data = self.run_main("confirm", "--camera", "front", "--map-b64", b64z(mine_only), "--json",
                                          "--out", self.out)
        self.assertEqual(code, 0, data)
        self.assertTrue(data["restart_needed"])                  # the zone's black outside is gone
        self.assertEqual(set(z.load_zones(self.zones)), {"back"})
        saved = sm.load_scene_map("front", self.zones)
        self.assertEqual([a.name for a in saved.areas], ["lawn", sm.WATCHED_NAME])
        self.assertEqual((saved.rest, saved.rest_owner), (sm.WATCH, sm.NEIGHBOUR))

    def test_confirm_map_b64_rejects_an_invalid_map_and_saves_nothing(self) -> None:
        for bad in (self.app_map(areas=[{"name": "x", "kind": "mine", "zone": "moon", "points": LAWN}]),
                    self.app_map(areas=[{"name": "x", "kind": "maybe", "zone": "yard", "points": LAWN}]),
                    self.app_map(areas=[{"name": "x", "kind": "mine", "zone": "yard", "points": [[0, 0], [1, 1]]}]),
                    self.app_map(camera="back")):
            code, _text, data = self.run_main("confirm", "--camera", "front", "--map-b64", b64z(bad), "--json",
                                              "--out", self.out)
            self.assertEqual(code, 1)
            self.assertIn("error", data)
        code, _text, data = self.run_main("confirm", "--camera", "front", "--map-b64", "%%%", "--json", "--out", self.out)
        self.assertEqual((code, data["error"]), (1, "the map could not be read"))
        self.assertEqual(sm.load_scene_map("front", self.zones).areas, ())
        self.restart.assert_not_called()

    def test_answer_asks_for_a_walls_side_then_takes_it_as_sides_b64(self) -> None:
        self.run_main("propose", "--camera", "front", "--json", "--grid", "--out", self.out)
        words = base64.b64encode("5 המעקה ביני לבין השכן".encode("utf-8")).decode("ascii")
        code, _text, data = self.run_main("answer", "--camera", "front", "--text-b64", words, "--json",
                                          "--out", self.out)
        self.assertEqual(code, 0, data)
        self.assertEqual([s["number"] for s in data["sides_needed"]], [5])
        self.assertEqual(data["map"]["lines"], [])                       # no line until the owner says the side
        sides = base64.b64encode(json.dumps({"5": "left"}).encode("utf-8")).decode("ascii")
        code, text, data = self.run_main("answer", "--camera", "front", "--text-b64", words, "--sides-b64", sides,
                                         "--json", "--out", self.out)
        self.assertEqual(code, 0, data)
        text.encode("ascii")
        self.assertEqual(data["sides_needed"], [])
        self.assertEqual([ln["inward"] for ln in data["map"]["lines"]], ["left"])
        bad = base64.b64encode(json.dumps({"5": "up"}).encode("utf-8")).decode("ascii")
        code, _text, data = self.run_main("answer", "--camera", "front", "--text-b64", words, "--sides-b64", bad,
                                          "--json", "--out", self.out)
        self.assertEqual((code, data["error"]), (1, "a side is left or right"))

    def test_restore_check_then_restore_the_previous_map(self) -> None:
        code, _text, data = self.run_main("restore", "--camera", "front", "--check", "--json")
        self.assertEqual((code, data["has_previous"], data["saved_at"]), (0, False, None))
        code, _text, data = self.run_main("restore", "--camera", "front", "--json")
        self.assertEqual((code, data["error"]), (1, "there is no previous map of this camera"))
        z.save_zone("front", [(0, 0), (0.5, 0), (0.5, 1), (0, 1)], self.zones)
        self.run_main("confirm", "--camera", "front", "--map-b64", b64z(self.app_map()), "--json", "--out", self.out)
        self.restart.reset_mock()
        code, _text, data = self.run_main("restore", "--camera", "front", "--check", "--json")
        self.assertTrue(data["has_previous"])
        self.assertIsInstance(data["saved_at"], float)
        code, text, data = self.run_main("restore", "--camera", "front", "--json")
        self.assertEqual(code, 0, data)
        text.encode("ascii")
        self.assertTrue(data["restart_needed"])                        # the zone and its black outside are back
        self.restart.assert_called_once()
        self.assertEqual(data["map"]["areas"], [])
        self.assertEqual(data["map"]["watched"], [[0.0, 0.0], [0.5, 0.0], [0.5, 1.0], [0.0, 1.0]])
        self.restart.reset_mock()
        code, _text, data = self.run_main("restore", "--camera", "front", "--json", "--no-restart")
        self.assertEqual(code, 0, data)                                 # a restore is undone the same way
        self.assertEqual(len(data["map"]["areas"]), 4)                 # its three, and the zone as ours
        self.restart.assert_not_called()

    def test_the_box_names_the_cameras_the_app_never_shows_an_id(self) -> None:
        code, text, data = self.run_main("names", "--json")
        self.assertEqual(code, 0)
        text.encode("ascii")
        self.assertEqual(data["names"], {"front": "הכניסה", "back": "back", "ameer_week_0_1_ch3": "מצלמה 3"})
        self.assertEqual(self.run_main("names", "--json", "--lang", "en")[2]["names"]["ameer_week_0_1_ch3"], "Camera 3")
        _code, _text, data = self.run_main("propose", "--camera", "front", "--json", "--embed", "--grid", "--out",
                                           self.out)
        self.assertEqual((data["display_name"], data["display_name_en"]), ("הכניסה", "הכניסה"))
        _code, _text, data = self.run_main("propose", "--camera", "ameer_week_0_1_ch3", "--json", "--embed", "--grid",
                                           "--out", self.out)
        self.assertEqual((data["display_name"], data["display_name_en"]), ("מצלמה 3", "Camera 3"))
        _code, _text, data = self.run_main("confirm", "--camera", "front", "--map-b64", b64z(self.app_map()), "--json",
                                           "--out", self.out)
        self.assertEqual(data["display_name"], "הכניסה")
        for argv in (("restore", "--camera", "front", "--check", "--json"), ("restore", "--camera", "front", "--json")):
            self.assertEqual(self.run_main(*argv)[2]["display_name"], "הכניסה")

    def test_after_a_site_rename_propose_brings_the_map_saved_under_the_old_id(self) -> None:
        # ch3 was mapped as ameer_test_ch3; the site was renamed and the camera is ameer_week_0_1_ch3 now.
        old = sm.SceneMap.from_dict("ameer_test_ch3", self.app_map(camera="ameer_test_ch3"))
        sm.confirm_scene_map(old, self.zones, now=1791640000.0)
        code, _text, data = self.run_main("propose", "--camera", "ameer_week_0_1_ch3", "--json", "--embed", "--grid",
                                          "--out", self.out)
        self.assertEqual(code, 0, data)
        self.assertIs(data["from_old_name"], True)
        current = data["current_map"]
        self.assertEqual(current["camera"], "ameer_week_0_1_ch3")
        self.assertEqual([a["name"] for a in current["areas"]], ["הדשא", "gate", "window"])
        self.assertEqual([ln["name"] for ln in current["lines"]], ["המעקה"])
        self.assertEqual(current["confirmed"], 1791640000.0)
        # Saving it keeps it under the new id and drops the old key.
        code, _text, data = self.run_main("confirm", "--camera", "ameer_week_0_1_ch3", "--map-b64",
                                          b64z(dict(current, camera="ameer_week_0_1_ch3")), "--json", "--out", self.out)
        self.assertEqual(code, 0, data)
        self.assertEqual(data["stale_removed"], ["ameer_test_ch3"])
        self.assertEqual(sorted(z.read_scene_maps(z.scene_maps_path_for(self.zones))), ["ameer_week_0_1_ch3"])
        _code, _text, data = self.run_main("propose", "--camera", "ameer_week_0_1_ch3", "--json", "--embed", "--grid",
                                           "--out", self.out)
        self.assertNotIn("from_old_name", data)                          # its own map now
        self.assertEqual(len(data["current_map"]["areas"]), 3)

    def test_a_camera_with_its_own_map_or_no_old_one_gets_no_old_map(self) -> None:
        code, _text, data = self.run_main("propose", "--camera", "front", "--json", "--embed", "--grid", "--out",
                                          self.out)
        self.assertEqual(code, 0, data)
        self.assertNotIn("from_old_name", data)                          # no _chN: no channel to look under
        self.assertEqual(data["current_map"]["areas"], [])
        sm.confirm_scene_map(sm.SceneMap.from_dict("ameer_test_ch3", self.app_map(camera="ameer_test_ch3")),
                             self.zones)
        sm.confirm_scene_map(sm.SceneMap.from_dict("ameer_week_0_1_ch3", {"areas": [
            {"name": "x", "kind": "mine", "zone": "yard", "points": LAWN}]}), self.zones)
        _code, _text, data = self.run_main("propose", "--camera", "ameer_week_0_1_ch3", "--json", "--embed", "--grid",
                                           "--out", self.out)
        self.assertNotIn("from_old_name", data)
        self.assertEqual([a["name"] for a in data["current_map"]["areas"]], ["x"])

    def test_the_apps_save_keeps_the_map_it_replaces_and_the_rest_beyond_the_boundary(self) -> None:
        # The editor saves only through confirm --map-b64: confirm keeps the map it replaces (scene_maps_backup.json)
        # before writing, so a cleaned-up map (duplicates merged) can always come back with "restore".
        piled = self.app_map()
        piled["areas"] = piled["areas"] + piled["areas"][:2]           # as the reopen bug left it
        self.run_main("confirm", "--camera", "front", "--map-b64", b64z(piled), "--json", "--out", self.out)
        self.assertEqual(len(sm.load_scene_map("front", self.zones).areas), 5)
        cleaned = self.app_map(rest="watch_no_alert", rest_owner="neighbour")
        code, _text, data = self.run_main("confirm", "--camera", "front", "--map-b64", b64z(cleaned), "--json",
                                          "--out", self.out)
        self.assertEqual(code, 0, data)
        backup = si.previous_map("front", self.zones)
        self.assertEqual(len(backup["scene"]["areas"]), 5)              # the piled-up map, kept
        saved = sm.load_scene_map("front", self.zones)
        self.assertEqual(len(saved.areas), 3)
        self.assertEqual((saved.rest, saved.rest_owner), ("watch_no_alert", "neighbour"))
        self.assertEqual((data["map"]["rest"], data["map"]["rest_owner"]), ("watch_no_alert", "neighbour"))
        code, _text, data = self.run_main("restore", "--camera", "front", "--json", "--no-restart")
        self.assertEqual((code, len(data["map"]["areas"])), (0, 5))

    def test_the_app_map_round_trips_through_from_dict_and_to_dict(self) -> None:
        data = self.app_map(rest="watch_no_alert", rest_owner="public")
        scene = sm.SceneMap.from_dict("front", data)
        again = sm.SceneMap.from_dict("front", json.loads(json.dumps(scene.to_dict())))
        self.assertEqual(again, scene)
        self.assertEqual([a["points"] for a in scene.to_dict()["areas"]][:2], [LAWN, GATE])
        self.assertEqual(si.decode_map_b64(b64z(scene.to_dict())), si.decode_map_b64(b64(scene.to_dict())))
        draft = si.confirm_app_map("front", data, self.out)
        self.assertEqual(draft, scene)
        with open(si.draft_path("front", self.out), encoding="utf-8") as f:
            self.assertEqual(sm.SceneMap.from_dict("front", json.load(f)), scene)


if __name__ == "__main__":
    unittest.main()
