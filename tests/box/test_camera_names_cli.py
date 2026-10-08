"""camera_names set: the app gives a camera its name (the newest alias, brain/aliases)."""
from __future__ import annotations

import base64
import contextlib
import io
import json
import os
import tempfile
import unittest
from unittest import mock

from home_guard_project.box import camera_names, find_cameras
from home_guard_project.box.brain import aliases


def b64(text: str) -> str:
    return base64.b64encode(text.encode("utf-8")).decode("ascii")


class SetNameTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.cameras = os.path.join(self.tmp.name, "cameras.yaml")
        with open(self.cameras, "w", encoding="utf-8") as f:
            f.write("cameras:\n  ameer_week_0_1_ch3: rtsp://x\n  front_door: rtsp://y\ndisabled:\n  back_ch5: rtsp://z\n")
        self.aliases = os.path.join(self.tmp.name, "camera_aliases.yaml")
        for patcher in (mock.patch.object(find_cameras, "CAMERAS_PATH", self.cameras),
                        mock.patch.object(aliases, "ALIASES_PATH", self.aliases)):
            patcher.start(); self.addCleanup(patcher.stop)

    def run_main(self, *argv):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = camera_names.main(list(argv))
        text = out.getvalue()
        return code, text, json.loads(text)

    def test_set_makes_the_name_the_cameras_name(self) -> None:
        code, text, data = self.run_main("set", "--camera", "ameer_week_0_1_ch3", "--name-b64", b64("הגינה"), "--json")
        self.assertEqual(code, 0, data)
        self.assertEqual(len(text.strip().splitlines()), 1)
        text.encode("ascii")                                    # one line of ASCII JSON, any console code page
        self.assertEqual(data, {"camera": "ameer_week_0_1_ch3", "display_name": "הגינה", "display_name_en": "הגינה"})
        self.assertEqual(aliases.load_aliases(self.aliases)["ameer_week_0_1_ch3"], ["הגינה"])
        self.run_main("set", "--camera", "ameer_week_0_1_ch3", "--name-b64", b64("the garden"), "--json")
        code, _text, data = self.run_main("set", "--camera", "ameer_week_0_1_ch3", "--name-b64", b64("הגינה"), "--json")
        self.assertEqual(data["display_name"], "הגינה")         # chosen again: the newest once more
        self.assertEqual(aliases.load_aliases(self.aliases)["ameer_week_0_1_ch3"], ["the garden", "הגינה"])

    def test_a_switched_off_camera_can_be_named_too(self) -> None:
        code, _text, data = self.run_main("set", "--camera", "back_ch5", "--name-b64", b64("Back yard"), "--json")
        self.assertEqual((code, data["display_name_en"]), (0, "Back yard"))

    def test_plain_errors(self) -> None:
        self.run_main("set", "--camera", "front_door", "--name-b64", b64("הכניסה"), "--json")
        cases = [(("--camera", "ameer_week_0_1_ch3", "--name-b64", b64("הכניסה")), "already names front_door"),
                 (("--camera", "nope", "--name-b64", b64("x")), "unknown camera"),
                 (("--camera", "front_door", "--name-b64", b64("")), "1 to 40 characters"),
                 (("--camera", "front_door", "--name-b64", "%%%"), "the name could not be read")]
        for args, words in cases:
            code, text, data = self.run_main("set", *args, "--json")
            self.assertEqual(code, 1)
            self.assertIn(words, data["error"])
            text.encode("ascii")
        self.assertEqual(camera_names.display_name("front_door", "he", aliases.load_aliases(self.aliases)), "הכניסה")


if __name__ == "__main__":
    unittest.main()
