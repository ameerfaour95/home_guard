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
            f.write("cameras:\n  front: rtsp://x\n  back: rtsp://y\n")
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
