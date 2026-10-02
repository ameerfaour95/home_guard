from __future__ import annotations

import json
import os
import tempfile
import unittest
from typing import Any, Dict, List, Optional

from home_guard_project.box.agent import UNAVAILABLE_REPLY, AgentReply
from home_guard_project.box.feedback import FEEDBACK_QUESTION, AlertIndex, MuteState
from home_guard_project.box.telegram_agent import TelegramInbox, alert_roots, feedback_keyboard, send_alert, start
from home_guard_project.box.telegram_notify import TelegramConfig

NOW = 1_800_000_000.0
CHAT = "-1001"
ALERT = {"alert_id": "front_door_3_alert", "camera": "front_door", "summary": "a person at the door", "ts": NOW - 60}


class FakeTelegram:
    """Records every Bot API call and answers like Telegram would."""

    def __init__(self, updates: Optional[List[Dict[str, Any]]] = None) -> None:
        self.calls: List[Dict[str, Any]] = []
        self.updates = updates or []
        self._next_message_id = 100

    def post(self, token: str, method: str, fields: Dict[str, str], timeout: float = 15.0) -> Dict[str, Any]:
        self.calls.append({"method": method, "fields": fields})
        if method == "getUpdates":
            offset = int(fields["offset"])
            return {"ok": True, "result": [u for u in self.updates if u["update_id"] >= offset]}
        self._next_message_id += 1
        return {"ok": True, "result": {"message_id": self._next_message_id}}

    def post_multipart(self, token: str, method: str, fields: Dict[str, str], files: Dict[str, Any],
                       timeout: float = 20.0) -> Dict[str, Any]:
        self.calls.append({"method": method, "fields": fields, "files": {k: v[0] for k, v in files.items()}})
        self._next_message_id += 1
        return {"ok": True, "result": {"message_id": self._next_message_id}}

    def sent(self, method: str) -> List[Dict[str, Any]]:
        return [c for c in self.calls if c["method"] == method]


class FakeAgent:
    def __init__(self, reply: AgentReply) -> None:
        self.reply = reply
        self.seen: List[Dict[str, Any]] = []

    def handle(self, text: str, chat_id: Any, who: Any = None, alert: Any = None) -> AgentReply:
        self.seen.append({"text": text, "chat_id": chat_id, "who": who, "alert": alert})
        return self.reply


def message(update_id: int, text: str, chat: str = CHAT, reply_to: Optional[int] = None, bot: bool = False) -> dict:
    msg: Dict[str, Any] = {"message_id": 500 + update_id, "chat": {"id": int(chat)}, "text": text,
                           "from": {"id": 42, "first_name": "Dana", "is_bot": bot}}
    if reply_to is not None:
        msg["reply_to_message"] = {"message_id": reply_to}
    return {"update_id": update_id, "message": msg}


def button(update_id: int, code: str, on_message: int, chat: str = CHAT) -> dict:
    return {"update_id": update_id, "callback_query": {
        "id": f"cb{update_id}", "data": code, "from": {"id": 42, "first_name": "Dana"},
        "message": {"message_id": on_message, "chat": {"id": int(chat)}},
    }}


class TelegramAgentTest(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = tmp.name
        self.cfg = TelegramConfig(bot_token="T", chat_ids=[CHAT])
        self.index = AlertIndex(os.path.join(self.dir, "alert_index.json"))
        self.mute = MuteState(os.path.join(self.dir, "alert_mute.json"))
        self.feedback_dir = os.path.join(self.dir, "production_multi")

    def _inbox(self, tg: FakeTelegram, agent: Any) -> TelegramInbox:
        return TelegramInbox(self.cfg, agent, self.index, self.mute, self.feedback_dir,
                             os.path.join(self.dir, "telegram_offset.json"),
                             post=tg.post, post_multipart=tg.post_multipart, now=lambda: NOW)

    def _saved(self) -> List[dict]:
        found = []
        for dirpath, _, names in os.walk(os.path.join(self.feedback_dir, "feedback")):
            for name in sorted(names):
                with open(os.path.join(dirpath, name), encoding="utf-8") as f:
                    found.append(json.load(f))
        return found

    def test_an_alert_goes_out_with_the_question_and_buttons_and_is_remembered(self) -> None:
        tg = FakeTelegram()
        result = send_alert(self.cfg, self.index, ALERT, "front_door: a person at the door", image=b"jpg",
                            post=tg.post, post_multipart=tg.post_multipart)
        (call,) = tg.sent("sendPhoto")
        self.assertTrue(result["sent"])
        self.assertIn("a person at the door", call["fields"]["caption"])
        self.assertIn(FEEDBACK_QUESTION, call["fields"]["caption"])
        self.assertEqual(call["fields"]["reply_markup"], feedback_keyboard())
        self.assertEqual(self.index.lookup(CHAT, result["results"][0]["message_id"])["alert_id"], "front_door_3_alert")

    def test_dry_run_sends_nothing(self) -> None:
        tg = FakeTelegram()
        cfg = TelegramConfig(bot_token="T", chat_ids=[CHAT], dry_run=True)
        self.assertFalse(send_alert(cfg, self.index, ALERT, "x", post=tg.post, post_multipart=tg.post_multipart)["sent"])
        self.assertEqual(tg.calls, [])

    def test_a_tapped_button_is_saved_with_its_alert_and_confirmed(self) -> None:
        self.index.remember(CHAT, 77, ALERT)
        tg = FakeTelegram([button(1, "fb:false", on_message=77)])
        self.assertEqual(self._inbox(tg, agent=None).poll_once(), 1)

        (saved,) = self._saved()
        self.assertEqual((saved["verdict"], saved["source"]), ("false_alarm", "button"))
        self.assertEqual(saved["alert"]["alert_id"], "front_door_3_alert")
        self.assertEqual(len(tg.sent("answerCallbackQuery")), 1)
        (said,) = tg.sent("sendMessage")
        self.assertIn("false alarm", said["fields"]["text"])

    def test_the_pause_button_pauses(self) -> None:
        tg = FakeTelegram([button(1, "fb:mute60", on_message=77)])
        self._inbox(tg, agent=None).poll_once()
        self.assertTrue(self.mute.is_muted(NOW + 60, "front_door"))
        self.assertFalse(self.mute.is_muted(NOW + 3700, "front_door"))

    def test_a_reply_goes_to_the_agent_with_the_alert_it_answers_and_clips_are_sent(self) -> None:
        clip = os.path.join(self.dir, "clip.mp4")
        with open(clip, "wb") as f:
            f.write(b"mp4")
        self.index.remember(CHAT, 77, ALERT)
        agent = FakeAgent(AgentReply(text="Here it is.", clips=(clip,)))
        tg = FakeTelegram([message(1, "send me that video", reply_to=77)])
        self._inbox(tg, agent).poll_once()

        (seen,) = agent.seen
        self.assertEqual((seen["text"], seen["alert"]["alert_id"]), ("send me that video", "front_door_3_alert"))
        self.assertEqual(seen["who"], {"user_id": 42, "name": "Dana"})
        (said,) = tg.sent("sendMessage")
        self.assertEqual((said["fields"]["text"], said["fields"]["reply_to_message_id"]), ("Here it is.", "501"))
        (video,) = tg.sent("sendVideo")
        self.assertEqual(video["files"], {"video": "clip.mp4"})

    def test_a_message_that_replies_to_nothing_is_about_the_latest_alert(self) -> None:
        self.index.remember(CHAT, 77, ALERT)
        agent = FakeAgent(AgentReply(text="ok"))
        self._inbox(FakeTelegram([message(1, "false alarm")]), agent).poll_once()
        self.assertEqual(agent.seen[0]["alert"]["alert_id"], "front_door_3_alert")

    def test_strangers_and_bots_are_ignored(self) -> None:
        agent = FakeAgent(AgentReply(text="ok"))
        tg = FakeTelegram([message(1, "stop all alerts", chat="999"), button(2, "fb:mute60", 77, chat="999"),
                           message(3, "hello", bot=True)])
        self.assertEqual(self._inbox(tg, agent).poll_once(), 3)
        self.assertEqual(agent.seen, [])
        self.assertFalse(self.mute.is_muted(NOW + 60, "front_door"))
        self.assertEqual(tg.sent("sendMessage"), [])
        self.assertEqual(self._saved(), [])

    def test_without_an_agent_the_message_is_still_saved(self) -> None:
        tg = FakeTelegram([message(1, "no there was nothing")])
        self._inbox(tg, agent=None).poll_once()
        (saved,) = self._saved()
        self.assertEqual(saved["raw_text"], "no there was nothing")
        self.assertEqual(tg.sent("sendMessage")[0]["fields"]["text"], UNAVAILABLE_REPLY)

    def test_handled_updates_are_not_fetched_again_after_a_restart(self) -> None:
        agent = FakeAgent(AgentReply(text="ok"))
        tg = FakeTelegram([message(1, "one"), message(2, "two")])
        self._inbox(tg, agent).poll_once()
        self.assertEqual(self._inbox(tg, agent).poll_once(), 0)   # a new inbox reads the saved offset
        self.assertEqual([s["text"] for s in agent.seen], ["one", "two"])

    def test_start_without_a_token_listens_to_nothing_but_still_answers_is_muted(self) -> None:
        assistant = start({"telegram_chat_ids": CHAT}, {}, ["front_door"], log_dir=self.dir,
                          live_dir=self.feedback_dir, archive_dir=os.path.join(self.dir, "production_archive"))
        self.assertIsNone(assistant.thread)
        self.assertFalse(assistant.is_muted("front_door"))
        self.assertFalse(assistant.send_alert(ALERT, "x")["sent"])

    def test_alert_roots_are_the_live_folder_and_each_archived_site(self) -> None:
        archive = os.path.join(self.dir, "production_archive")
        os.makedirs(os.path.join(archive, "house2"))
        os.makedirs(os.path.join(archive, "old_house"))
        self.assertEqual(alert_roots("live", archive),
                         ["live", os.path.join(archive, "house2"), os.path.join(archive, "old_house")])
        self.assertEqual(alert_roots("live", os.path.join(self.dir, "nope")), ["live"])

    def test_one_bad_update_does_not_stop_the_next(self) -> None:
        class Exploding(FakeAgent):
            def handle(self, text: str, *args: Any, **kwargs: Any) -> AgentReply:
                if text == "boom":
                    raise RuntimeError("bug")
                return super().handle(text, *args, **kwargs)

        agent = Exploding(AgentReply(text="ok"))
        self._inbox(FakeTelegram([message(1, "boom"), message(2, "fine")]), agent).poll_once()
        self.assertEqual([s["text"] for s in agent.seen], ["fine"])


if __name__ == "__main__":
    unittest.main()
