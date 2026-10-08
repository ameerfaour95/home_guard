"""The on-demand map change inside the Telegram inbox: the request, its answer and its buttons never reach the
agent; the owner's other messages always do."""
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
from home_guard_project.box.agent import AgentReply
from home_guard_project.box.feedback import AlertIndex, MuteState
from home_guard_project.box.telegram_agent import TelegramInbox
from home_guard_project.box.telegram_notify import TelegramConfig
from test_telegram_agent import CHAT, NOW, FakeAgent, FakeTelegram

CAM = "ameer_week_0_1_ch1"


def halves(image):
    h, w = image.shape[:2]
    left = np.zeros((h, w), bool)
    left[:, : w // 2] = True
    return [left, ~left]


class InboxInterviewTest(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = tmp.name
        names = mock.patch.object(camera_names, "_load", side_effect=lambda a: {CAM: ["פרגולה"]} if a is None else a)
        names.start()
        self.addCleanup(names.stop)
        self.tg = FakeTelegram()
        self.agent = FakeAgent(AgentReply(text="the agent answered"))
        self.inbox = TelegramInbox(TelegramConfig(bot_token="T", chat_ids=[CHAT]), self.agent,
                                   AlertIndex(os.path.join(self.dir, "i.json")), MuteState(os.path.join(self.dir, "m.json")),
                                   os.path.join(self.dir, "fb"), os.path.join(self.dir, "offset.json"),
                                   post=self.tg.post, post_multipart=self.tg.post_multipart, now=lambda: NOW)
        send_photo, _ = sc.telegram_senders("T", self.tg.post, self.tg.post_multipart)
        self.zones = os.path.join(self.dir, "zones.yaml")
        self.inbox.scene = sc.SceneChat(send_photo, lambda c, t, rows: self.inbox._say(c, t, rows=rows),
                                        lambda: [CAM], lambda: "he",
                                        state_path=os.path.join(self.dir, "scene_chat.json"),
                                        out_dir=os.path.join(self.dir, "interview"), zones_path=self.zones,
                                        picture=self.grab, segmenter=halves, background=False, now=lambda: NOW)

    def grab(self, camera):
        path = os.path.join(self.dir, "pic.jpg")
        img = np.full((90, 120, 3), 120, np.uint8)
        cv2.imwrite(path, img)
        return img, path

    def message(self, text, uid):
        return {"update_id": uid, "message": {"message_id": uid, "chat": {"id": int(CHAT)}, "from": {"id": 7, "first_name": "Ameer"},
                                              "text": text}}

    def tap(self, data, uid):
        return {"update_id": uid, "callback_query": {"id": f"q{uid}", "data": data, "from": {"id": 7},
                                                     "message": {"message_id": 1, "chat": {"id": int(CHAT)}}}}

    def start(self, uid: int = 1) -> None:
        self.inbox.handle_update(self.message("מפה", uid))
        sure = self.tg.sent("sendMessage")[-1]
        yes = json.loads(sure["fields"]["reply_markup"])["inline_keyboard"][0][0]
        self.assertEqual(yes["text"], "כן, להגדיר מחדש")
        self.inbox.handle_update(self.tap(yes["callback_data"], uid + 100))

    def test_the_whole_interview_runs_without_the_agent(self) -> None:
        self.start()
        photos = self.tg.sent("sendPhoto")
        self.assertEqual(len(photos), 1)
        self.assertTrue(photos[0]["fields"]["caption"].startswith("פרגולה: מה משתנה?"))   # the box's one camera
        self.inbox.handle_update(self.message("1 שלי, 2 של השכן", 2))
        confirm = self.tg.sent("sendPhoto")[-1]
        buttons = json.loads(confirm["fields"]["reply_markup"])["inline_keyboard"][0]
        self.assertEqual([b["text"] for b in buttons], ["שמור", "תקן"])
        self.inbox.handle_update(self.tap(buttons[0]["callback_data"], 3))
        self.assertTrue(sm.load_scene_map(CAM, self.zones).confirmed)
        self.assertEqual(self.agent.seen, [])                       # nothing of it reached the assistant
        self.assertTrue(any("נשמר: המפה החדשה של פרגולה" in c["fields"].get("text", "") for c in self.tg.sent("sendMessage")))

    def test_forgot_7_while_the_coloured_picture_waits_never_reaches_the_agent(self) -> None:
        self.start()
        self.inbox.handle_update(self.message("1 שלי", 2))
        self.inbox.handle_update(self.message("שכחת את 2", 3))
        self.assertEqual(self.agent.seen, [])
        texts = [c["fields"].get("text", "") for c in self.tg.sent("sendMessage")]
        self.assertTrue(any("מה 2?" in t for t in texts), texts)

    def test_other_messages_still_reach_the_agent(self) -> None:
        self.inbox.handle_update(self.message("מה קורה בחצר?", 1))
        self.assertEqual(len(self.agent.seen), 1)

    def test_ordinary_messages_reach_the_agent_while_a_change_is_open(self) -> None:
        self.start()
        for uid, text in enumerate(("מה המצב?", "יצאנו", "תן לי תמונה", "יש 2 אנשים בחוץ?"), 2):
            self.inbox.handle_update(self.message(text, uid))
        self.assertEqual([s["text"] for s in self.agent.seen], ["מה המצב?", "יצאנו", "תן לי תמונה", "יש 2 אנשים בחוץ?"])
        self.inbox.handle_update(self.message("עצור", 9))
        self.assertIsNone(self.inbox.scene.session(CHAT))
        self.assertEqual(len(self.agent.seen), 4)

    def test_a_stale_stage_2c_state_file_is_harmless(self) -> None:
        with open(os.path.join(self.dir, "scene_chat.json"), "w", encoding="utf-8") as f:
            json.dump({CHAT: {"cameras": [CAM], "index": 0, "camera": CAM, "stage": "answer", "token": "t",
                              "touched": NOW}}, f)
        self.inbox.handle_update(self.message("1 שלי", 1))
        self.inbox.handle_update(self.tap("sm:s:t", 2))
        self.assertEqual([s["text"] for s in self.agent.seen], ["1 שלי"])
        self.assertEqual(sm.load_scene_map(CAM, self.zones).areas, ())


if __name__ == "__main__":
    unittest.main()
