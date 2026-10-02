from __future__ import annotations

import unittest
from unittest import mock

from home_guard_project.box import telegram_notify as tg
from home_guard_project.box.telegram_notify import TelegramConfig, load_telegram_config, _as_chat_ids


class ChatIdParsingTest(unittest.TestCase):
    def test_list(self) -> None:
        self.assertEqual(_as_chat_ids([123, "456", " 789 "]), ["123", "456", "789"])

    def test_comma_string(self) -> None:
        self.assertEqual(_as_chat_ids("111, 222 ,333"), ["111", "222", "333"])

    def test_space_string(self) -> None:
        self.assertEqual(_as_chat_ids("111 222"), ["111", "222"])

    def test_empty(self) -> None:
        self.assertEqual(_as_chat_ids(""), [])
        self.assertEqual(_as_chat_ids(None), [])


class LoadConfigTest(unittest.TestCase):
    def test_token_from_env_chats_from_settings(self) -> None:
        cfg = load_telegram_config({"telegram_chat_ids": "100,200"}, {"TELEGRAM_BOT_TOKEN": "T"})
        self.assertEqual(cfg.bot_token, "T")
        self.assertEqual(cfg.chat_ids, ["100", "200"])
        self.assertTrue(cfg.enabled)

    def test_not_enabled_without_token_or_chat(self) -> None:
        self.assertFalse(load_telegram_config({"telegram_chat_ids": "100"}, {}).enabled)
        self.assertFalse(load_telegram_config({}, {"TELEGRAM_BOT_TOKEN": "T"}).enabled)

    def test_dry_run_flag(self) -> None:
        cfg = load_telegram_config({"telegram_chat_ids": "1", "notify_dry_run": True}, {"TELEGRAM_BOT_TOKEN": "T"})
        self.assertTrue(cfg.dry_run)


class SendMessageTest(unittest.TestCase):
    def test_dry_run_sends_nothing(self) -> None:
        cfg = TelegramConfig(bot_token="T", chat_ids=["1"], dry_run=True)
        with mock.patch.object(tg, "_http_post") as post:
            res = tg.send_message(cfg, "hi")
        post.assert_not_called()
        self.assertEqual(res["reason"], "dry_run")

    def test_not_configured_sends_nothing(self) -> None:
        with mock.patch.object(tg, "_http_post") as post:
            res = tg.send_message(TelegramConfig(), "hi")
        post.assert_not_called()
        self.assertEqual(res["reason"], "not_configured")

    def test_sends_to_every_chat(self) -> None:
        cfg = TelegramConfig(bot_token="T", chat_ids=["100", "200", "300"])
        with mock.patch.object(tg, "_http_post", return_value={"ok": True}) as post:
            res = tg.send_message(cfg, "hello")
        self.assertTrue(res["sent"])
        self.assertEqual(post.call_count, 3)
        # token + method + fields carry the chat id and text
        _token, method, fields = post.call_args.args
        self.assertEqual(method, "sendMessage")
        self.assertEqual(fields["text"], "hello")

    def test_partial_failure_still_counts_as_sent(self) -> None:
        cfg = TelegramConfig(bot_token="T", chat_ids=["1", "2"])
        with mock.patch.object(tg, "_http_post", side_effect=[{"ok": False, "description": "blocked"}, {"ok": True}]):
            res = tg.send_message(cfg, "hi")
        self.assertTrue(res["sent"])

    def test_network_error_swallowed(self) -> None:
        cfg = TelegramConfig(bot_token="T", chat_ids=["1"])
        with mock.patch.object(tg, "_http_post", side_effect=OSError("boom")):
            res = tg.send_message(cfg, "hi")
        self.assertFalse(res["sent"])


class DispatchTest(unittest.TestCase):
    def test_send_message_command(self) -> None:
        cfg = TelegramConfig(bot_token="T", chat_ids=["1"])
        with mock.patch.object(tg, "_http_post", return_value={"ok": True}) as post:
            res = tg.notify(cfg, "[send_message]", "person at gate")
        self.assertTrue(res["telegram"]["sent"])
        self.assertIn("person at gate", post.call_args.args[2]["text"])

    def test_call_owner_sends_urgent_message(self) -> None:
        cfg = TelegramConfig(bot_token="T", chat_ids=["1"])
        with mock.patch.object(tg, "_http_post", return_value={"ok": True}) as post:
            res = tg.notify(cfg, "[call_owner]", "forced entry", "window broken")
        self.assertTrue(res["telegram"]["sent"])
        self.assertIn("ALERT", post.call_args.args[2]["text"])

    def test_none_sends_nothing(self) -> None:
        cfg = TelegramConfig(bot_token="T", chat_ids=["1"])
        with mock.patch.object(tg, "_http_post") as post:
            res = tg.notify(cfg, "[none]")
        post.assert_not_called()
        self.assertFalse(res["sent"])


class DiscoverChatsTest(unittest.TestCase):
    def test_distinct_chats_from_updates(self) -> None:
        updates = {"ok": True, "result": [
            {"message": {"chat": {"id": -100, "type": "group", "title": "Family"}}},
            {"message": {"chat": {"id": -100, "type": "group", "title": "Family"}}},
            {"message": {"chat": {"id": 55, "type": "private", "first_name": "Ameer"}}},
        ]}
        with mock.patch.object(tg, "_http_get", return_value=updates):
            chats = tg.discover_chats("T")
        ids = {c["id"] for c in chats}
        self.assertEqual(ids, {"-100", "55"})


if __name__ == "__main__":
    unittest.main()
