"""Re-defining one camera's map over Telegram, on demand (owner's decisions 2026-10-08 22:55 and 23:10)."""
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
from home_guard_project.box import scene_interview as si
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


def buttons(message):
    """{label: callback} of a sent message."""
    return {label: code for row in message["rows"] for label, code in row}


class SceneChatCase(unittest.TestCase):
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

    def start(self, text="מפה פרגולה"):
        """Ask, then [כן, להגדיר מחדש]: the numbered picture."""
        self.assertTrue(self.chat.on_text(CHAT, text))
        self.chat.on_button(CHAT, buttons(self.fake.texts[-1])["כן, להגדיר מחדש"])
        self.assertEqual(self.chat.session(CHAT)["stage"], "answer")


class StartTest(SceneChatCase):
    def test_rule_1_only_a_request_to_define_starts_it(self):
        for text in ("מה המצב?", "יצאנו", "תן לי תמונה", "יש 2 אנשים בחוץ?", "1 שלי", "מה קורה במצלמה 3?",
                     "תראה לי את הפרגולה", "המפה של הבית יפה"):
            self.assertFalse(self.chat.on_text(CHAT, text), text)
        self.assertEqual(self.fake.photos + self.fake.texts, [])
        for text in ("מפה", "/map", "מפה פרגולה", "תגדיר אזור חדש", "אני רוצה לשנות את האזור של השכן במצלמה 3",
                     "תגדיר מחדש את מצלמה 3", "redefine the area of camera 3"):
            self.assertTrue(sc.is_trigger(text), text)

    def test_rule_2_nothing_starts_without_yes(self):
        self.assertTrue(self.chat.on_text(CHAT, "אני רוצה לשנות את האזור של השכן במצלמה 3"))
        sure = self.fake.texts[-1]
        self.assertEqual(sure["text"], "אתה בטוח שאתה רוצה להגדיר מחדש את חניה? המפה הנוכחית ממשיכה לעבוד עד "
                                       "שתאשר את החדשה.")
        self.assertEqual(list(buttons(sure)), ["כן, להגדיר מחדש", "לא"])
        self.assertEqual(self.fake.photos, [])                         # no picture before [כן]
        self.assertFalse(self.chat.on_text(CHAT, "2 של השכן"))         # no answers before [כן] either
        self.chat.on_button(CHAT, buttons(sure)["לא"])
        self.assertIsNone(self.chat.session(CHAT))
        self.assertIn("לא שיניתי כלום: המפה של חניה נשארת כמו שהיא", self.fake.texts[-1]["text"])
        self.assertEqual(self.fake.photos, [])

    def test_yes_shows_the_numbered_picture_with_cancel(self):
        self.start()
        picture = self.fake.photos[-1]
        self.assertTrue(picture["caption"].startswith("פרגולה: מה משתנה?"))
        self.assertNotIn(CH1, picture["caption"])
        self.assertIn("בטל", buttons(picture))

    def test_without_a_camera_one_button_per_camera_and_cancel(self):
        self.chat.on_text(CHAT, "תגדיר אזור חדש")
        ask = self.fake.texts[-1]
        self.assertEqual(ask["text"], "איזו מצלמה?")
        self.assertEqual(list(buttons(ask)), ["פרגולה", "חניה", "בטל"])
        self.chat.on_button(CHAT, buttons(ask)["חניה"])
        self.assertIn("להגדיר מחדש את חניה", self.fake.texts[-1]["text"])   # then the "are you sure"
        self.assertEqual(self.fake.photos, [])


class ExitTest(SceneChatCase):
    def test_rule_3_every_step_has_cancel_and_cancel_changes_nothing(self):
        sm.save_scene_map(sm.SceneMap(CH1, areas=(sm.Area("yard", sm.MINE, "yard", LEFT),)), self.zones)
        before = sm.load_scene_map(CH1, self.zones)
        self.start()
        self.assertIn("בטל", buttons(self.fake.photos[-1]))                   # the numbered picture
        self.chat.on_text(CHAT, "2 של השכן")
        confirm = self.fake.photos[-1]
        self.assertEqual(list(buttons(confirm)), ["שמור", "תקן", "בטל"])     # the coloured picture
        self.chat.on_button(CHAT, buttons(confirm)["בטל"])
        self.assertIsNone(self.chat.session(CHAT))
        self.assertIn("לא שיניתי כלום: המפה של פרגולה נשארת כמו שהיא", self.fake.texts[-1]["text"])
        self.assertEqual(sm.load_scene_map(CH1, self.zones), before)
        self.assertFalse(os.path.exists(si.draft_path(CH1, self.chat.out_dir)))

    def test_rule_3_exit_words(self):
        for word in ("עצור", "בטל", "לא משנה"):
            self.start()
            self.assertTrue(self.chat.on_text(CHAT, word), word)
            self.assertIsNone(self.chat.session(CHAT))
            self.assertIn("לא שיניתי כלום", self.fake.texts[-1]["text"])
        self.assertFalse(self.chat.on_text(CHAT, "בטל"))                     # no session: the assistant's
        self.chat.on_text(CHAT, "תגדיר אזור חדש")                            # the which-camera question
        self.assertTrue(self.chat.on_text(CHAT, "לא משנה"))
        self.assertIsNone(self.chat.session(CHAT))


class DraftTest(SceneChatCase):
    def test_rule_4_the_live_map_and_mask_are_untouched_until_save(self):
        z.save_zone(CH1, [(0, 0), (0.6, 0), (0.6, 1), (0, 1)], self.zones)
        sm.save_scene_map(sm.SceneMap(CH1, areas=(sm.Area("yard", sm.MINE, "yard", LEFT),)), self.zones)
        live = (sm.load_scene_map(CH1, self.zones), z.load_zones(self.zones), z.load_black(z.scene_maps_path_for(self.zones)))
        self.start()
        self.chat.on_text(CHAT, "להסתיר את 2")
        self.assertEqual(self.chat.session(CHAT)["stage"], "confirm")
        self.assertEqual((sm.load_scene_map(CH1, self.zones), z.load_zones(self.zones),
                          z.load_black(z.scene_maps_path_for(self.zones))), live)
        self.assertEqual(self.restarts, 0)

    def test_the_change_merges_and_only_save_saves(self):
        sm.save_scene_map(sm.SceneMap(CH1, areas=(sm.Area("yard", sm.MINE, "yard", LEFT),
                                                  sm.Area("trees", sm.BLACK, "other", ((0.9, 0.9), (1, 0.9), (1, 1))))),
                          self.zones)
        self.start()
        self.assertTrue(self.chat.on_text(CHAT, "2 של השכן"))
        self.assertEqual(len(sm.load_scene_map(CH1, self.zones).areas), 2)
        self.chat.on_button(CHAT, buttons(self.fake.photos[-1])["שמור"])
        saved = sm.load_scene_map(CH1, self.zones)
        kinds = {a.name: a.kind for a in saved.areas}
        self.assertEqual(kinds["yard"], "mine")
        self.assertEqual(kinds["trees"], "black")
        self.assertIn("watch_no_alert", kinds.values())
        self.assertTrue(saved.confirmed)
        self.assertIsNone(self.chat.session(CHAT))
        self.assertIn("נשמר: המפה החדשה של פרגולה פועלת", self.fake.texts[-1]["text"])

    def test_the_newest_word_replaces_what_lay_inside_its_region(self):
        sm.save_scene_map(sm.SceneMap(CH1, areas=(sm.Area("old", sm.MINE, "yard", ((0.6, 0.1), (0.9, 0.1), (0.9, 0.9))),)),
                          self.zones)
        self.start()
        self.chat.on_text(CHAT, "2 של השכן")
        self.chat.on_button(CHAT, f"sm:s:{self.token()}")
        self.assertEqual([a.ground for a in sm.load_scene_map(CH1, self.zones).areas], ["neighbour"])

    def test_hiding_restarts_once_after_the_session_is_closed(self):
        self.start()
        self.chat.on_text(CHAT, "להסתיר את 1")
        self.chat.on_button(CHAT, f"sm:s:{self.token()}")
        self.assertEqual(self.restarts, 1)
        self.assertIsNone(self.chat.session(CHAT))
        self.assertIn("מתחילה מחדש", self.fake.texts[-1]["text"])

    def test_fix_and_lines(self):
        self.start()
        self.chat.on_text(CHAT, "1 שלי")
        self.chat.on_button(CHAT, f"sm:f:{self.token()}")
        self.assertIn("כתבו שוב", self.fake.texts[-1]["text"])
        self.assertIn("בטל", buttons(self.fake.texts[-1]))
        self.assertTrue(self.chat.on_text(CHAT, "1 שלי, 2 של השכן, המעקה בין 2 ל-1"))
        self.chat.on_button(CHAT, f"sm:s:{self.token()}")
        self.assertEqual([ln.name for ln in sm.load_scene_map(CH1, self.zones).lines], ["המעקה"])


class RestoreTest(SceneChatCase):
    def test_rule_5_the_previous_map_comes_back_with_its_own_confirm(self):
        z.save_zone(CH1, [(0, 0), (0.6, 0), (0.6, 1), (0, 1)], self.zones)
        sm.save_scene_map(sm.SceneMap(CH1, areas=(sm.Area("yard", sm.MINE, "yard", LEFT),)), self.zones)
        before = (sm.load_scene_map(CH1, self.zones), z.load_zones(self.zones))
        self.start()
        self.chat.on_text(CHAT, "2 של השכן")
        self.chat.on_button(CHAT, f"sm:s:{self.token()}")
        self.assertNotEqual(sm.load_scene_map(CH1, self.zones), before[0])
        self.assertEqual(z.load_zones(self.zones), {})                       # the confirmed map opened the zone
        self.assertTrue(self.chat.on_text(CHAT, "תחזיר את המפה הקודמת של פרגולה"))
        question = self.fake.texts[-1]
        self.assertIn("להחזיר את המפה הקודמת של פרגולה", question["text"])
        self.assertEqual(list(buttons(question)), ["כן, להחזיר", "לא"])
        self.assertNotEqual(sm.load_scene_map(CH1, self.zones), before[0])  # nothing before [כן]
        self.chat.on_button(CHAT, buttons(question)["כן, להחזיר"])
        self.assertEqual((sm.load_scene_map(CH1, self.zones), z.load_zones(self.zones)), before)
        self.assertIn("החזרתי את המפה הקודמת של פרגולה", self.fake.texts[-1]["text"])
        self.assertIsNone(self.chat.session(CHAT))

    def test_no_previous_map_says_so_and_no_restores_nothing(self):
        self.chat.on_text(CHAT, "תחזיר את המפה הקודמת של פרגולה")
        self.assertIn("אין מפה קודמת של פרגולה", self.fake.texts[-1]["text"])
        sm.save_scene_map(sm.SceneMap(CH1, areas=(sm.Area("yard", sm.MINE, "yard", LEFT),)), self.zones)
        self.start()
        self.chat.on_text(CHAT, "2 של השכן")
        self.chat.on_button(CHAT, f"sm:s:{self.token()}")
        after = sm.load_scene_map(CH1, self.zones)
        self.chat.on_text(CHAT, "תחזיר את המפה הקודמת של פרגולה")
        self.chat.on_button(CHAT, buttons(self.fake.texts[-1])["לא"])
        self.assertEqual(sm.load_scene_map(CH1, self.zones), after)


class NeverSwallowTest(SceneChatCase):
    def test_ordinary_messages_reach_the_assistant_while_a_change_is_open(self):
        self.start()
        for text in ("מה המצב?", "יצאנו", "תן לי תמונה", "יש 2 אנשים בחוץ?", "9 של השכן"):
            self.assertFalse(self.chat.on_text(CHAT, text), text)
        reminders = [t for t in self.fake.texts if "עדיין פתוחה" in t["text"]]
        self.assertEqual(len(reminders), 1)
        self.assertTrue(self.chat.on_text(CHAT, "2 של השכן"))

    def test_rule_6_the_flow_expires_after_30_minutes_without_a_change(self):
        self.start()
        self.clock += 25 * 60
        self.assertTrue(self.chat.on_text(CHAT, "2 של השכן"))       # still open: answers count up to 30 min
        self.clock += sc.SESSION_TTL + 1
        self.assertIsNone(self.chat.session(CHAT))
        self.assertFalse(self.chat.on_text(CHAT, "2 של השכן"))      # after it, nothing is swallowed
        self.assertFalse(self.chat.on_text(CHAT, "עצור"))

    def test_an_old_button_or_an_expired_one(self):
        self.start()
        self.chat.on_text(CHAT, "2 של השכן")
        token = self.token()
        self.assertTrue(self.chat.on_button(CHAT, "sm:s:ffffffff"))
        self.assertEqual(sm.load_scene_map(CH1, self.zones).areas, ())
        self.clock += sc.SESSION_TTL + 1
        self.chat.on_button(CHAT, f"sm:s:{token}")
        self.assertIn("פג הזמן", self.fake.texts[-1]["text"])
        self.assertIn("לא שיניתי כלום", self.fake.texts[-1]["text"])
        self.assertEqual(sm.load_scene_map(CH1, self.zones).areas, ())

    def test_a_stale_multi_camera_session_is_dropped_harmlessly(self):
        os.makedirs(os.path.dirname(self.state), exist_ok=True)
        with open(self.state, "w", encoding="utf-8") as f:
            json.dump({CHAT: {"cameras": [CH1, CH3], "index": 0, "camera": CH1, "stage": "answer", "token": "abc",
                              "restart": False, "started": NOW - 60, "regions_path": "x.json", "picture": "x.jpg",
                              "touched": NOW - 60}}, f)
        self.assertIsNone(self.chat.session(CHAT))
        for text in ("1 שלי", "מה המצב?", "יש 2 אנשים בחוץ?"):
            self.assertFalse(self.chat.on_text(CHAT, text), text)
        self.assertTrue(self.chat.on_button(CHAT, "sm:s:abc"))
        self.assertEqual(self.fake.photos + self.fake.texts, [])
        self.assertEqual(sm.load_scene_map(CH1, self.zones).areas, ())

    def test_a_damaged_state_file_is_harmless(self):
        os.makedirs(os.path.dirname(self.state), exist_ok=True)
        for payload in ("{not json", "[]", '{"-5": "x"}', '{"-5": {"v": 2}}'):
            with open(self.state, "w", encoding="utf-8") as f:
                f.write(payload)
            self.assertIsNone(self.chat.session(CHAT))
            self.assertFalse(self.chat.on_text(CHAT, "מה המצב?"))
        self.start()

    def test_the_cli_needs_a_camera(self):
        self.assertIn("error", sc.start_from_cli(""))


if __name__ == "__main__":
    unittest.main()
