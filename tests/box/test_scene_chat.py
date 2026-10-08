"""The install interview over Telegram (scene map stage 2c), with fake senders and a fake camera."""
from __future__ import annotations

import os
import tempfile
import unittest
from unittest import mock

import numpy as np

from home_guard_project.box import camera_names
from home_guard_project.box import scene_chat as sc
from home_guard_project.box import scene_map as sm
from home_guard_project.data_collection import zones as z

CH1, CH2 = "ameer_week_0_1_ch1", "ameer_week_0_1_ch2"
ALIASES = {CH1: ["פרגולה"], CH2: ["מעבר צד"]}
CHAT = "-5"


def picture(h: int = 120, w: int = 160) -> np.ndarray:
    img = np.full((h, w, 3), 128, np.uint8)
    img[:, w // 2:] = (40, 160, 40)
    return img


def halves(image):
    h, w = image.shape[:2]
    left = np.zeros((h, w), bool)
    left[:, : w // 2] = True
    right = np.zeros((h, w), bool)
    right[:, w // 2:] = True
    return [left, right]


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
        self.restarts = 0
        self.fake = Fake()
        self.names = mock.patch.object(camera_names, "_load", side_effect=lambda a: ALIASES if a is None else a)
        self.names.start()
        self.chat = sc.SceneChat(self.fake.photo, self.fake.text, lambda: [CH1, CH2], lambda: "he",
                                 state_path=os.path.join(self.tmp.name, "state", "scene_chat.json"),
                                 out_dir=os.path.join(self.tmp.name, "interview"), zones_path=self.zones,
                                 picture=self.grab, segmenter=halves, restart=self.restart, background=False)

    def tearDown(self):
        self.names.stop()
        self.tmp.cleanup()

    def grab(self, camera):
        path = os.path.join(self.tmp.name, f"{camera}.jpg")
        import cv2

        cv2.imwrite(path, picture())
        return picture(), path

    def restart(self):
        self.restarts += 1

    def token(self):
        return self.chat.session(CHAT)["token"]

    def test_a_trigger_starts_and_other_words_go_to_the_assistant(self):
        self.assertFalse(self.chat.on_text(CHAT, "מה קורה בחצר?"))
        self.assertTrue(sc.is_trigger("מפה"))
        self.assertTrue(sc.is_trigger("בוא נגדיר את המצלמות"))
        self.assertTrue(sc.is_trigger("/map"))
        self.assertFalse(sc.is_trigger("מפהק"))
        self.assertTrue(self.chat.on_text(CHAT, "מפה"))
        self.assertIn("2 מצלמות", self.fake.texts[0]["text"])
        first = self.fake.photos[0]
        self.assertTrue(first["caption"].startswith("פרגולה (1/2)"))
        self.assertNotIn(CH1, first["caption"])                      # never the camera's id
        self.assertEqual(self.chat.session(CHAT)["stage"], "answer")
        self.assertFalse(self.chat.on_text(CHAT, "מי זה היה?"))      # no number: the assistant answers it

    def test_answer_confirm_save_and_next_camera(self):
        self.chat.on_text(CHAT, "מפה")
        self.assertTrue(self.chat.on_text(CHAT, "1 שלי, 2 של השכן, המעקה בין 2 ל-1"))
        confirm = self.fake.photos[-1]
        self.assertIn("ככה?", confirm["caption"])
        self.assertIn("של השכן", confirm["caption"])
        self.assertEqual([label for label, _ in confirm["rows"][0]], ["שמור", "תקן"])
        self.assertEqual(sm.load_scene_map(CH1, self.zones).areas, ())     # nothing saved before [שמור]
        self.assertTrue(self.chat.on_button(CHAT, f"sm:s:{self.token()}"))
        saved = sm.load_scene_map(CH1, self.zones)
        self.assertTrue(saved.confirmed)
        self.assertEqual([ln.name for ln in saved.lines], ["המעקה"])
        self.assertTrue(self.fake.photos[-1]["caption"].startswith("מעבר צד (2/2)"))
        self.assertTrue(self.chat.on_button(CHAT, f"sm:k:{self.token()}"))  # skip the second camera
        self.assertIsNone(self.chat.session(CHAT))
        self.assertIn("סיימנו", self.fake.texts[-1]["text"])
        self.assertEqual(self.restarts, 0)                                  # nothing the mask holds changed

    def test_fix_asks_again_and_an_old_button_does_nothing(self):
        self.chat.start(CHAT, [CH1])
        self.chat.on_text(CHAT, "1 שלי")
        old = self.token()
        self.chat.on_button(CHAT, f"sm:f:{old}")
        self.assertIn("כתבו שוב", self.fake.texts[-1]["text"])
        self.assertEqual(self.chat.session(CHAT)["stage"], "answer")
        self.chat.on_text(CHAT, "1 שלי, 2 רחוב")
        self.assertTrue(self.chat.on_button(CHAT, "sm:s:ffffffff"))         # someone else's button
        self.assertEqual(sm.load_scene_map(CH1, self.zones).areas, ())
        self.chat.on_button(CHAT, f"sm:s:{old}")
        self.assertEqual(len(sm.load_scene_map(CH1, self.zones).areas), 2)

    def test_opening_todays_black_outside_restarts_once_at_the_end(self):
        z.save_zone(CH1, [(0, 0), (0.5, 0), (0.5, 1), (0, 1)], self.zones)
        z.save_zone("ameer_test_ch1", [(0, 0), (1, 0), (1, 1)], self.zones)      # stale, same channel
        self.chat.start(CHAT, [CH1])
        self.assertIn("הקו הלבן", self.fake.photos[0]["caption"])
        self.chat.on_text(CHAT, "1 שלי")
        self.assertIn("כל השאר: של השכן", self.fake.photos[-1]["caption"])
        self.chat.on_button(CHAT, f"sm:s:{self.token()}")
        self.assertEqual(self.restarts, 1)
        self.assertIn("מתחילה מחדש", self.fake.texts[-1]["text"])
        self.assertEqual(z.load_zones(self.zones), {})

    def test_stop_and_skip_words(self):
        self.chat.on_text(CHAT, "מפה")
        self.assertTrue(self.chat.on_text(CHAT, "דלג"))
        self.assertTrue(self.fake.photos[-1]["caption"].startswith("מעבר צד"))
        self.assertTrue(self.chat.on_text(CHAT, "עצור"))
        self.assertIsNone(self.chat.session(CHAT))
        self.assertIn("עצרתי", self.fake.texts[-1]["text"])

    def test_a_camera_named_in_the_request_is_the_only_one(self):
        self.chat.on_text(CHAT, "בוא נגדיר את המצלמות: מעבר צד")
        self.assertEqual(self.chat.session(CHAT)["cameras"], [CH2])

    def test_a_camera_without_a_picture_is_passed(self):
        def broken(camera):
            if camera == CH1:
                raise RuntimeError("offline")
            return self.grab(camera)

        self.chat._picture = broken
        with self.assertLogs("box.scene_chat", level="WARNING"):
            self.chat.on_text(CHAT, "מפה")
        self.assertIn("לא הצלחתי לקבל תמונה מפרגולה", self.fake.texts[1]["text"])
        self.assertTrue(self.fake.photos[0]["caption"].startswith("מעבר צד (2/2)"))

    def test_notes_reach_the_owner_in_hebrew(self):
        self.chat.start(CHAT, [CH1])
        self.chat.on_text(CHAT, "1 שלי, 9 של השכן")
        self.assertIn("אין 9 בתמונה", self.fake.photos[-1]["caption"])


if __name__ == "__main__":
    unittest.main()
