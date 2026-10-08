"""An event's updates reply in its thread: telegram_notify and dispatch_alert carry an optional reply_to."""
import unittest
from unittest import mock

from home_guard_project.box import inference as inf
from home_guard_project.box import telegram_notify as tg

CFG = tg.TelegramConfig(bot_token="T", chat_ids=["-5", "-6"])
FIRST = {"alert_id": "cam_1_alert", "chat_id": "-5", "message_id": 41, "ts": 1.0}


class ReplyFieldsTest(unittest.TestCase):
    def test_only_the_chat_of_the_first_message_replies(self):
        self.assertEqual(tg.reply_fields("-5", FIRST),
                         {"reply_to_message_id": "41", "allow_sending_without_reply": "true"})
        self.assertEqual(tg.reply_fields("-6", FIRST), {})
        self.assertEqual(tg.reply_fields("-6", {"message_id": 9}), {"reply_to_message_id": "9",
                                                                    "allow_sending_without_reply": "true"})

    def test_nothing_to_reply_to(self):
        for reply_to in (None, {}, {"chat_id": "-5"}, {"chat_id": "-5", "message_id": "x"}, "41"):
            self.assertEqual(tg.reply_fields("-5", reply_to), {}, reply_to)


class SendTest(unittest.TestCase):
    def test_send_message_replies_and_returns_message_ids(self):
        posts = []

        def post(token, method, fields, timeout=15.0):
            posts.append(fields)
            return {"ok": True, "result": {"message_id": 100 + len(posts)}}

        with mock.patch.object(tg, "_http_post", side_effect=post):
            res = tg.send_message(CFG, "hello", reply_to=FIRST)
        self.assertEqual(posts[0]["reply_to_message_id"], "41")
        self.assertNotIn("reply_to_message_id", posts[1])
        self.assertEqual([r["message_id"] for r in res["results"]], [101, 102])

    def test_send_photo_replies(self):
        with mock.patch.object(tg, "_http_post_multipart",
                               return_value={"ok": True, "result": {"message_id": 7}}) as post:
            res = tg.send_photo(CFG, b"jpg", "cap", reply_to=FIRST)
        self.assertEqual(post.call_args_list[0].args[2]["reply_to_message_id"], "41")
        self.assertEqual(res["results"][0]["message_id"], 7)

    def test_without_reply_to_nothing_changes(self):
        with mock.patch.object(tg, "_http_post", return_value={"ok": True}) as post:
            tg.send_message(CFG, "hello")
        self.assertNotIn("reply_to_message_id", post.call_args.args[2])

    def test_notify_passes_the_thread_on(self):
        with mock.patch.object(tg, "send_message", return_value={"sent": True}) as send:
            tg.notify(CFG, "[send_message]", "s", reply_to=FIRST)
        self.assertEqual(send.call_args.kwargs["reply_to"], FIRST)
        with mock.patch.object(tg, "send_message", return_value={"sent": True}) as send:
            tg.notify(CFG, "[send_message]", "s")
        self.assertNotIn("reply_to", send.call_args.kwargs)


class DispatchTest(unittest.TestCase):
    def test_an_assistant_that_takes_reply_to_gets_it(self):
        class Assistant:
            def send_alert(self, alert, text, image=None, silent=False, lang="en", reply_to=None):
                self.reply_to = reply_to
                return {"sent": True}

        assistant = Assistant()
        inf.dispatch_alert({"alert_channel": "telegram"}, {}, "[send_message]", "s", "", assistant=assistant,
                           alert={"alert_id": "a"}, graded="text", reply_to=FIRST)
        self.assertEqual(assistant.reply_to, FIRST)

    def test_an_assistant_without_it_sends_as_before(self):
        class Assistant:
            def send_alert(self, alert, text, image=None, silent=False, lang="en"):
                self.text = text
                return {"sent": True}

        assistant = Assistant()
        res = inf.dispatch_alert({"alert_channel": "telegram"}, {}, "[send_message]", "s", "", assistant=assistant,
                                 alert={"alert_id": "a"}, graded="text", reply_to=FIRST)
        self.assertEqual(assistant.text, "text")
        self.assertNotIn("error", res)

    def test_without_an_assistant_the_thread_reaches_telegram_notify(self):
        with mock.patch.object(tg, "notify", return_value={"sent": True}) as notify:
            inf.dispatch_alert({"alert_channel": "telegram"}, {}, "[send_message]", "s", "", reply_to=FIRST)
        self.assertEqual(notify.call_args.kwargs["reply_to"], FIRST)
        with mock.patch.object(tg, "notify", return_value={"sent": True}) as notify:
            inf.dispatch_alert({"alert_channel": "telegram"}, {}, "[send_message]", "s", "")
        self.assertNotIn("reply_to", notify.call_args.kwargs)

    def test_first_message_of_a_dispatch(self):
        res = {"channel": "telegram", "telegram": {"telegram": {"sent": True, "results": [
            {"chat_id": "-5", "ok": False}, {"chat_id": "-6", "ok": True, "message_id": 8}]}}}
        self.assertEqual(inf._first_message(res), ("-6", 8))
        self.assertIsNone(inf._first_message({"telegram": {"sent": True, "held": "waiting for the video"}}))


if __name__ == "__main__":
    unittest.main()
