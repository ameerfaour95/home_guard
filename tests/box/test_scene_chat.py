"""Changing one camera's map over Telegram, on demand (owner's decision 2026-10-08 22:55)."""
from __future__ import annotations

import json
import os
import tempfile
import unittest
from unittest import mock

import cv2
import numpy as np

from home_guard_project.box import camera_names
from home_guard_project.box import scene_chat as sc
from home_guard_project.box import scene_map as sm
from home_guard_project.data_collection import zones as z

CH1, CH3 = "ameer_week_0_1_ch1", "ameer_week_0_1_ch3"
ALIASES = {CH1: ["פרגולה"], CH3: ["חניה"]}
CHAT = "-5"
NOW = 1_800_000_000.0
LEFT = ((0.0, 0.0), (0.5, 0.0), (0.5, 1.0), (0.0, 1.0))
RIGHT = ((0.5, 0.0), (1.0, 0.0), (1.0, 1.0), (0.5, 1.0))


def halves(image):
    h, w = image.shape[:2]
    left = np.zeros((h, w), bool)
    left[:, : w // 2] = True
    return [left, ~left]


class Fake:
    def __init__(self):
        self.photos, self.texts = [], []

    def photo(self, chat_id, path, caption, rows):
        assert os.path.isfile(path)
        self.photos.append({"chat": chat_id, "caption": caption, "rows": rows})

    def text(self, chat_id, text, rows):
        self.texts.append({"chat": chat_id, "text": text, "rows": rows})


class SceneChatTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.zones = os.path.join(self.tmp.name, "zones.yaml")
        self.state = os.path.join(self.tmp.name, "state", "scene_chat.json")
        self.clock = NOW
        self.restarts = 0
        self.fake = Fake()
        names = mock.patch.object(camera_names, "_load", side_effect=lambda a: ALIASES if a is None else a)
        names.start()
        self.addCleanup(names.stop)
        self.chat = sc.SceneChat(self.fake.photo, self.fake.text, lambda: [CH1, CH3], lambda: "he",
                                 state_path=self.state, out_dir=os.path.join(self.tmp.name, "interview"),
                                 zones_path=self.zones, picture=self.grab, segmenter=halves, restart=self.restart,
                                 background=False, now=lambda: self.clock)

    def tearDown(self):
        self.tmp.cleanup()

    def grab(self, camera):
        path = os.path.join(self.tmp.name, f"{camera}.jpg")
        img = np.full((90, 120, 3), 120, np.uint8)
        cv2.imwrite(path, img)
        return img, path

    def restart(self):
        self.restarts += 1

    def token(self):
        return self.chat.session(CHAT)["token"]

    # ---------- starting ----------
    def test_only_an_explicit_request_starts_it(self):
        for text in ("מה המצב?", "יצאנו", "תן לי תמונה", "יש 2 אנשים בחוץ?", "1 שלי"):
            self.assertFalse(self.chat.on_text(CHAT, text), text)
        self.assertEqual(self.fake.photos + self.fake.texts, [])
        for text in ("מפה", "/map", "תגדיר אזור חדש", "אני רוצה לשנות את האזור של השכן במצלמה 3"):
            self.assertTrue(sc.is_trigger(text), text)

    def test_a_named_camera_starts_at_once(self):
        self.assertTrue(self.chat.on_text(CHAT, "אני רוצה לשנות את האזור של השכן במצלמה 3"))
        self.assertEqual(len(self.fake.photos), 1)
        self.assertTrue(self.fake.photos[0]["caption"].startswith("חניה: מה משתנה?"))
        self.assertNotIn(CH3, self.fake.photos[0]["caption"])          # the family's name, never the id
        self.assertTrue(self.chat.on_text(CHAT, "מפה פרגולה"))
        self.assertEqual(self.chat.session(CHAT)["camera"], CH1)

    def test_without_a_camera_one_button_per_camera(self):
        self.chat.on_text(CHAT, "תגדיר אזור חדש")
        ask = self.fake.texts[-1]
        self.assertEqual(ask["text"], "איזו מצלמה?")
        self.assertEqual([row[0][0] for row in ask["rows"]], ["פרגולה", "חניה"])
        self.assertEqual(self.fake.photos, [])                          # never walks the cameras
        self.chat.on_button(CHAT, ask["rows"][1][0][1])
        self.assertTrue(self.fake.photos[0]["caption"].startswith("חניה"))
        self.assertEqual(len(self.fake.photos), 1)

    # ---------- the change ----------
    def test_the_change_merges_and_only_save_saves(self):
        sm.save_scene_map(sm.SceneMap(CH1, areas=(sm.Area("yard", sm.MINE, "yard", LEFT),
                                                  sm.Area("trees", sm.BLACK, "other", ((0.9, 0.9), (1, 0.9), (1, 1))))),
                          self.zones)
        self.chat.on_text(CHAT, "מפה פרגולה")
        self.assertTrue(self.chat.on_text(CHAT, "2 של השכן"))
        confirm = self.fake.photos[-1]
        self.assertIn("ככה?", confirm["caption"])
        self.assertEqual([label for label, _ in confirm["rows"][0]], ["שמור", "תקן"])
        self.assertEqual(len(sm.load_scene_map(CH1, self.zones).areas), 2)     # nothing saved before [שמור]
        self.chat.on_button(CHAT, f"sm:s:{self.token()}")
        saved = sm.load_scene_map(CH1, self.zones)
        kinds = {a.name: a.kind for a in saved.areas}
        self.assertEqual(kinds["yard"], "mine")                 # not mentioned: kept as it was
        self.assertEqual(kinds["trees"], "black")
        self.assertIn("watch_no_alert", kinds.values())
        self.assertTrue(saved.confirmed)
        self.assertIsNone(self.chat.session(CHAT))
        self.assertIn("נשמר: פרגולה", self.fake.texts[-1]["text"])
        self.assertEqual(self.restarts, 0)                      # the black areas did not change

    def test_the_newest_word_replaces_what_lay_inside_its_region(self):
        sm.save_scene_map(sm.SceneMap(CH1, areas=(sm.Area("old", sm.MINE, "yard", ((0.6, 0.1), (0.9, 0.1), (0.9, 0.9))),)),
                          self.zones)
        self.chat.on_text(CHAT, "מפה פרגולה")
        self.chat.on_text(CHAT, "2 של השכן")
        self.chat.on_button(CHAT, f"sm:s:{self.token()}")
        self.assertEqual([a.ground for a in sm.load_scene_map(CH1, self.zones).areas], ["neighbour"])

    def test_hiding_restarts_once_after_the_session_is_closed(self):
        self.chat.on_text(CHAT, "מפה פרגולה")
        self.chat.on_text(CHAT, "להסתיר את 1")
        self.chat.on_button(CHAT, f"sm:s:{self.token()}")
        self.assertEqual(self.restarts, 1)
        self.assertIsNone(self.chat.session(CHAT))
        self.assertIn("מתחילה מחדש", self.fake.texts[-1]["text"])
        self.assertEqual(len(z.load_black(z.scene_maps_path_for(self.zones))[CH1]), 1)

    def test_fix_and_lines(self):
        self.chat.on_text(CHAT, "מפה פרגולה")
        self.chat.on_text(CHAT, "1 שלי")
        self.chat.on_button(CHAT, f"sm:f:{self.token()}")
        self.assertIn("כתבו שוב", self.fake.texts[-1]["text"])
        self.assertTrue(self.chat.on_text(CHAT, "1 שלי, 2 של השכן, המעקה בין 2 ל-1"))
        self.chat.on_button(CHAT, f"sm:s:{self.token()}")
        self.assertEqual([ln.name for ln in sm.load_scene_map(CH1, self.zones).lines], ["המעקה"])

    # ---------- never swallow the owner's other messages ----------
    def test_ordinary_messages_reach_the_assistant_while_a_change_is_open(self):
        self.chat.on_text(CHAT, "מפה פרגולה")
        for text in ("מה המצב?", "יצאנו", "תן לי תמונה", "יש 2 אנשים בחוץ?", "9 של השכן"):
            self.assertFalse(self.chat.on_text(CHAT, text), text)     # 9: no such number in this picture
        reminders = [t for t in self.fake.texts if "עדיין פתוח" in t["text"]]
        self.assertEqual(len(reminders), 1)                           # one short reminder at most
        self.assertTrue(self.chat.on_text(CHAT, "2 של השכן"))         # a real answer still counts

    def test_an_answer_counts_only_within_the_window(self):
        self.chat.on_text(CHAT, "מפה פרגולה")
        self.clock += sc.ANSWER_WINDOW + 1
        self.assertFalse(self.chat.on_text(CHAT, "2 של השכן"))
        self.clock += sc.SESSION_TTL
        self.assertIsNone(self.chat.session(CHAT))

    def test_stop_ends_any_session_and_is_the_assistants_without_one(self):
        self.assertFalse(self.chat.on_text(CHAT, "עצור"))
        self.chat.on_text(CHAT, "מפה פרגולה")
        self.assertTrue(self.chat.on_text(CHAT, "עצור"))
        self.assertIsNone(self.chat.session(CHAT))
        self.chat.on_text(CHAT, "תגדיר אזור חדש")                    # the which-camera question
        self.assertTrue(self.chat.on_text(CHAT, "עצור"))
        self.assertIsNone(self.chat.session(CHAT))

    def test_an_old_button_or_an_expired_one(self):
        self.chat.on_text(CHAT, "מפה פרגולה")
        self.chat.on_text(CHAT, "2 של השכן")
        token = self.token()
        self.assertTrue(self.chat.on_button(CHAT, "sm:s:ffffffff"))
        self.assertEqual(sm.load_scene_map(CH1, self.zones).areas, ())
        self.clock += sc.SESSION_TTL + 1
        self.chat.on_button(CHAT, f"sm:s:{token}")
        self.assertIn("פג הזמן", self.fake.texts[-1]["text"])
        self.assertEqual(sm.load_scene_map(CH1, self.zones).areas, ())

    # ---------- the stage-2c state left on the box ----------
    def test_a_stale_multi_camera_session_is_dropped_harmlessly(self):
        os.makedirs(os.path.dirname(self.state), exist_ok=True)
        with open(self.state, "w", encoding="utf-8") as f:
            json.dump({CHAT: {"cameras": [CH1, CH3], "index": 0, "camera": CH1, "stage": "answer", "token": "abc",
                              "restart": False, "started": NOW - 60, "regions_path": "x.json", "picture": "x.jpg",
                              "touched": NOW - 60}}, f)
        self.assertIsNone(self.chat.session(CHAT))
        for text in ("1 שלי", "מה המצב?", "יש 2 אנשים בחוץ?"):
            self.assertFalse(self.chat.on_text(CHAT, text), text)
        self.assertTrue(self.chat.on_button(CHAT, "sm:s:abc"))       # its old [שמור]: nothing happens
        self.assertEqual(self.fake.photos + self.fake.texts, [])
        self.assertEqual(sm.load_scene_map(CH1, self.zones).areas, ())

    def test_a_damaged_state_file_is_harmless(self):
        os.makedirs(os.path.dirname(self.state), exist_ok=True)
        for payload in ("{not json", "[]", '{"-5": "x"}', '{"-5": {"v": 2}}'):
            with open(self.state, "w", encoding="utf-8") as f:
                f.write(payload)
            self.assertIsNone(self.chat.session(CHAT))
            self.assertFalse(self.chat.on_text(CHAT, "מה המצב?"))
        self.assertTrue(self.chat.on_text(CHAT, "מפה פרגולה"))       # and a new change still starts

    def test_the_cli_needs_a_camera(self):
        self.assertIn("error", sc.start_from_cli(""))


if __name__ == "__main__":
    unittest.main()
