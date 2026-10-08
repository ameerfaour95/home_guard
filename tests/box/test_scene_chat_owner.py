"""The three live bugs of the owner's camera-1 answer (2026-10-08), through the Telegram map change."""
from __future__ import annotations

import json
import os
import shutil
import tempfile
import unittest
from unittest import mock

import cv2
import numpy as np

from home_guard_project.box import camera_names
from home_guard_project.box import scene_chat as sc
from home_guard_project.box import scene_interview as si
from home_guard_project.box import scene_map as sm
from test_scene_owner_answer import ANSWER, CAM, REGIONS

CHAT = "-5"
NOW = 1_800_000_000.0


class OwnerAnswerChatTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.zones = os.path.join(self.tmp, "zones.yaml")
        self.photos, self.texts = [], []
        names = mock.patch.object(camera_names, "_load", side_effect=lambda a: {CAM: ["פרגולה"]} if a is None else a)
        names.start()
        self.addCleanup(names.stop)
        self.chat = sc.SceneChat(lambda c, path, caption, rows: self.photos.append({"caption": caption, "rows": rows}),
                                 lambda c, text, rows: self.texts.append({"text": text, "rows": rows}),
                                 lambda: [CAM], lambda: "he", state_path=os.path.join(self.tmp, "chat.json"),
                                 out_dir=os.path.join(self.tmp, "interview"), zones_path=self.zones,
                                 picture=self.grab, segmenter=si.grid_segmenter, background=False, now=lambda: NOW)
        self.chat.on_text(CHAT, "מפה")
        # The numbers the owner saw on the box: the real regions of camera 1.
        s = self.chat.session(CHAT)
        shutil.copy(REGIONS, s["regions_path"])
        with open(REGIONS, encoding="utf-8") as f:
            s["numbers"] = [r["number"] for r in json.load(f)["regions"]]
        self.chat._put(CHAT, s)

    def grab(self, camera):
        path = os.path.join(self.tmp, "pic.jpg")
        img = np.full((562, 960, 3), 90, np.uint8)
        cv2.imwrite(path, img)
        return img, path

    def token(self):
        return self.chat.session(CHAT)["token"]

    def test_the_railing_is_a_boundary_of_ours_and_7_is_understood(self):
        self.assertTrue(self.chat.on_text(CHAT, ANSWER))
        stage = self.chat.session(CHAT)["stage"]
        if stage == "side":                              # 7's side was unclear: asked, never guessed
            rows = self.photos[-1]["rows"]
            self.assertEqual(len(rows[0]), 2)
            self.chat.on_button(CHAT, rows[0][0][1])
        caption = self.photos[-1]["caption"]
        self.assertIn("ככה?", caption)
        self.assertIn("קווים: המעקה 4", caption)
        self.assertNotIn("לא הבנתי", caption)
        self.chat.on_button(CHAT, f"sm:s:{self.token()}")
        saved = sm.load_scene_map(CAM, self.zones)
        self.assertFalse(any(a.kind == sm.WATCH and a.zone == "fence" for a in saved.areas))
        self.assertEqual(sum(a.zone == "fence" and a.kind == sm.MINE for a in saved.areas), 2)
        self.assertIn("המעקה 4", [ln.name for ln in saved.lines])

    def test_a_correction_while_the_coloured_picture_waits(self):
        self.chat.on_text(CHAT, "2 שלי 1 שלי 3 שלי 6 שלי 9 של השכן")
        self.assertEqual(self.chat.session(CHAT)["stage"], "confirm")
        self.assertFalse(self.chat.on_text(CHAT, "יש 2 אנשים בחוץ?"))     # not a correction: the assistant's
        self.assertTrue(self.chat.on_text(CHAT, "שכחת את 7"))               # never the assistant's (it sent a clip)
        self.assertIn("מה 7?", self.texts[-1]["text"])
        self.assertTrue(self.chat.on_text(CHAT, "7 שלי"))
        self.assertEqual(self.chat.session(CHAT)["stage"], "confirm")
        self.chat.on_button(CHAT, f"sm:s:{self.token()}")
        kinds = {a.name: a.kind for a in sm.load_scene_map(CAM, self.zones).areas}
        self.assertEqual(kinds.get("area 7"), sm.MINE)                     # the correction was added ...
        self.assertEqual(kinds.get("area 2"), sm.MINE)                     # ... to what was already said
        self.assertEqual(kinds.get("של השכן"), sm.WATCH)

    def test_a_correction_that_already_says_what(self):
        self.chat.on_text(CHAT, "2 שלי 1 שלי")
        self.assertTrue(self.chat.on_text(CHAT, "שכחת: 7 של השכן"))
        self.assertEqual(self.chat.session(CHAT)["stage"], "confirm")
        self.assertIn("של השכן", self.photos[-1]["caption"])

    def test_a_number_that_made_nothing_is_said_back(self):
        self.chat.on_text(CHAT, "2 שלי 1 שלי 7 בבקשה")
        self.assertIn("לא הבנתי את 7", self.photos[-1]["caption"])

    def test_an_unclear_side_is_asked_with_two_buttons(self):
        self.chat.on_text(CHAT, "4 המעקה ביני לבין השכן")
        self.assertEqual(self.chat.session(CHAT)["stage"], "side")
        question = self.photos[-1]
        self.assertIn("באיזה צד של המעקה (4) השטח שלכם?", question["caption"])
        labels = [label for label, _ in question["rows"][0]]
        self.assertEqual(sorted(labels), sorted(si.side_words((0.3151, 0.7302), (0.6824, 0.012)).values()))
        self.chat.on_button(CHAT, question["rows"][0][1][1])
        self.assertEqual(self.chat.session(CHAT)["stage"], "confirm")
        self.chat.on_button(CHAT, f"sm:s:{self.token()}")
        self.assertEqual([ln.inward for ln in sm.load_scene_map(CAM, self.zones).lines], ["right"])


if __name__ == "__main__":
    unittest.main()
