from __future__ import annotations

import io
import json
import os
import tempfile
import unittest
import urllib.error

from home_guard_project.box.agent import AgentReply
from home_guard_project.box.chat_feed import ChatFeed, read_feed
from home_guard_project.box.feedback import AlertIndex, MuteState
from home_guard_project.box.telegram_agent import TelegramInbox, send_alert, send_clip
from home_guard_project.box.telegram_notify import TelegramConfig

CHAT = "-100200"
NOW = 1_800_000_000.0
ALERT = {"alert_id": "door_100_alert", "camera": "door", "summary": "a person at the door", "ts": NOW}


class ChatFeedTest(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.path = os.path.join(tmp.name, "logs", "telegram_chat.jsonl")

    def test_messages_come_back_in_order_with_their_picture(self) -> None:
        feed = ChatFeed(self.path)
        feed.add("box", "alert", "door: a person", camera="door", alert_id="door_100_alert", image=b"jpg", now=NOW)
        feed.add("owner", "message", "מי זה?", name="Ameer", alert_id="door_100_alert", now=NOW + 5)
        first, second = read_feed(self.path)
        self.assertEqual((first["who"], first["kind"], first["camera"], first["delivered"]), ("box", "alert", "door", True))
        with open(os.path.join(os.path.dirname(self.path), "chat_images", first["image"]), "rb") as f:
            self.assertEqual(f.read(), b"jpg")
        self.assertEqual((second["name"], second["text"], second["image"]), ("Ameer", "מי זה?", ""))

    def test_the_file_is_trimmed_and_unused_pictures_go_with_it(self) -> None:
        feed = ChatFeed(self.path, keep=5)
        for i in range(10):
            feed.add("box", "alert", f"alert {i}", alert_id=f"door_{i}_alert", image=b"jpg", now=NOW + i)
        entries = read_feed(self.path)
        self.assertEqual([e["text"] for e in entries], [f"alert {i}" for i in range(5, 10)])
        images = sorted(os.listdir(os.path.join(os.path.dirname(self.path), "chat_images")))
        self.assertEqual(images, sorted(e["image"] for e in entries))

    def test_a_damaged_line_and_a_missing_file_are_harmless(self) -> None:
        self.assertEqual(read_feed(self.path), [])
        feed = ChatFeed(self.path)
        feed.add("box", "alert", "one", now=NOW)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write('{"ts": 1, "who": "bo')
        self.assertEqual([e["text"] for e in read_feed(self.path)], ["one"])

    def test_the_newest_are_returned_up_to_the_limit(self) -> None:
        feed = ChatFeed(self.path)
        for i in range(8):
            feed.add("owner", "message", str(i), now=NOW + i)
        self.assertEqual([e["text"] for e in read_feed(self.path, limit=3)], ["5", "6", "7"])


class _Agent:
    def handle(self, text, chat_id, who, alert):
        return AgentReply(text="That was the gardener, noted.", clips=[])


class ConversationInTheWindowTest(unittest.TestCase):
    """Everything said in the Telegram group about the alerts ends up in the window's feed."""

    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = tmp.name
        self.path = os.path.join(self.tmp, "telegram_chat.jsonl")
        self.feed = ChatFeed(self.path)
        self.cfg = TelegramConfig(bot_token="T", chat_ids=[CHAT])
        self.index = AlertIndex(os.path.join(self.tmp, "alerts.json"))

    def _post(self, token, method, fields, files=None, timeout=20.0):
        return {"ok": True, "result": {"message_id": 41}}

    def _inbox(self, agent=None) -> TelegramInbox:
        return TelegramInbox(self.cfg, agent, self.index, MuteState(os.path.join(self.tmp, "mute.json")),
                             os.path.join(self.tmp, "production"), os.path.join(self.tmp, "offset.json"),
                             post=self._post, post_multipart=self._post, now=lambda: NOW + 60, feed=self.feed)

    def test_an_alert_its_video_a_tap_and_a_written_answer(self) -> None:
        send_alert(self.cfg, self.index, ALERT, "door: a person at the door", image=b"jpg",
                   post=self._post, post_multipart=self._post, feed=self.feed)
        clip = os.path.join(self.tmp, "door_100_alert.mp4")
        with open(clip, "wb") as f:
            f.write(b"video")
        send_clip(self.cfg, self.index, "door_100_alert", clip, post_multipart=self._post, feed=self.feed)
        inbox = self._inbox(_Agent())
        inbox.handle_update({"update_id": 1, "callback_query": {
            "id": "q", "data": "fb:false", "from": {"id": 5, "first_name": "Ameer"},
            "message": {"message_id": 41, "chat": {"id": int(CHAT)}}}})
        inbox.handle_update({"update_id": 2, "message": {
            "message_id": 50, "chat": {"id": int(CHAT)}, "from": {"id": 5, "first_name": "Ameer"},
            "text": "it was the gardener", "reply_to_message": {"message_id": 41}}})

        feed = read_feed(self.path)
        self.assertEqual([(e["who"], e["kind"]) for e in feed], [
            ("box", "alert"), ("box", "video"),
            ("owner", "button"), ("assistant", "answer"),
            ("owner", "message"), ("assistant", "answer"),
        ])
        alert, video, tap, _, written, answer = feed
        self.assertEqual((alert["text"], alert["camera"], alert["delivered"]), ("door: a person at the door", "door", True))
        self.assertTrue(alert["image"])
        self.assertEqual((video["alert_id"], video["delivered"]), ("door_100_alert", True))
        self.assertEqual((tap["name"], tap["alert_id"]), ("Ameer", "door_100_alert"))
        self.assertTrue(tap["text"] and not tap["text"].startswith("fb:"))        # the button's words, not its code
        self.assertEqual((written["text"], written["name"], written["camera"]), ("it was the gardener", "Ameer", "door"))
        self.assertEqual(answer["text"], "That was the gardener, noted.")

    def test_a_refused_alert_is_in_the_feed_as_not_delivered(self) -> None:
        def refuse(*args, **kwargs):
            body = json.dumps({"ok": False, "description": "Forbidden: bot is not a member of the group chat"}).encode()
            raise urllib.error.HTTPError("https://api.telegram.org/x", 403, "Forbidden", {}, io.BytesIO(body))

        send_alert(self.cfg, self.index, ALERT, "door: a person at the door", image=b"jpg",
                   post=refuse, post_multipart=refuse, feed=self.feed)
        (entry,) = read_feed(self.path)
        self.assertFalse(entry["delivered"])
        self.assertIn("not a member", entry["error"])

    def test_a_live_picture_the_assistant_took_is_sent_and_recorded(self) -> None:
        sent = []

        def post(token, method, fields, files=None, timeout=20.0):
            sent.append((method, files))
            return {"ok": True, "result": {"message_id": 70}}

        picture = os.path.join(self.tmp, "front_door_now.jpg")
        with open(picture, "wb") as f:
            f.write(b"jpeg-bytes")

        class LookingAgent:
            def handle(self, text, chat_id, who, alert):
                return AgentReply(text="Nobody is at the front door right now.", clips=(), photos=(picture,))

        inbox = TelegramInbox(self.cfg, LookingAgent(), self.index, MuteState(os.path.join(self.tmp, "m.json")),
                              os.path.join(self.tmp, "production"), os.path.join(self.tmp, "offset.json"),
                              post=post, post_multipart=post, now=lambda: NOW, feed=self.feed)
        inbox.handle_update({"update_id": 9, "message": {
            "message_id": 80, "chat": {"id": int(CHAT)}, "from": {"id": 5, "first_name": "Ameer"},
            "text": "what is happening at the front door now?"}})

        self.assertEqual([m for m, _ in sent], ["sendMessage", "sendPhoto"])
        self.assertEqual(sent[1][1]["photo"], ("front_door_now.jpg", b"jpeg-bytes", "image/jpeg"))
        self.assertEqual([(e["who"], e["kind"]) for e in read_feed(self.path)],
                         [("owner", "message"), ("assistant", "answer"), ("assistant", "photo")])

    def test_the_answer_is_retried_and_a_restart_waits_for_it(self) -> None:
        from unittest import mock

        order = []
        failures = [OSError(10054, "connection reset")]

        def post(token, method, fields, files=None, timeout=20.0):
            if method == "sendMessage" and failures:
                raise failures.pop()
            order.append(method)
            return {"ok": True, "result": {"message_id": 71}}

        class TurningOff:
            def handle(self, text, chat_id, who, alert):
                return AgentReply(text="back_door is off.", clips=(), restart=True)

        inbox = TelegramInbox(self.cfg, TurningOff(), self.index, MuteState(os.path.join(self.tmp, "m2.json")),
                              os.path.join(self.tmp, "production"), os.path.join(self.tmp, "offset2.json"),
                              post=post, post_multipart=post, now=lambda: NOW, feed=self.feed)
        with mock.patch("home_guard_project.box.telegram_agent.time.sleep"),                 mock.patch("home_guard_project.box.control.request_restart",
                           side_effect=lambda *a, **k: order.append("restart")):
            inbox.handle_update({"update_id": 11, "message": {
                "message_id": 90, "chat": {"id": int(CHAT)}, "from": {"id": 5, "first_name": "Ameer"},
                "text": "turn off the back door camera"}})
        self.assertEqual(order, ["sendMessage", "restart"])      # retried once, then the restart

    def test_a_pause_for_all_cameras_says_so(self) -> None:
        import time as _time

        from home_guard_project.box.feedback import Feedback, confirmation_text

        text = confirmation_text(Feedback(action="mute", mute_until=_time.time() + 600))
        self.assertIn("ALL cameras", text)
        text = confirmation_text(Feedback(action="mute", mute_until=_time.time() + 600, camera="back_door"))
        self.assertIn("for back_door", text)

    def test_strangers_are_not_recorded(self) -> None:
        self._inbox(_Agent()).handle_update({"update_id": 3, "message": {
            "message_id": 60, "chat": {"id": 999}, "from": {"id": 9, "first_name": "X"}, "text": "hello"}})
        self.assertEqual(read_feed(self.path), [])


if __name__ == "__main__":
    unittest.main()
