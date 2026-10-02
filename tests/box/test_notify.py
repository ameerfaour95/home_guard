from __future__ import annotations

import unittest
from unittest import mock

from home_guard_project.box import notify
from home_guard_project.box.notify import NotifyConfig, load_notify_config


def _full() -> NotifyConfig:
    return NotifyConfig(
        account_sid="AC123",
        auth_token="tok",
        whatsapp_from="whatsapp:+1000",
        owner_whatsapp="whatsapp:+2000",
        voice_from="+1000",
        owner_phone="+2000",
    )


class LoadConfigTest(unittest.TestCase):
    def test_secrets_from_env_numbers_from_settings(self) -> None:
        env = {"TWILIO_ACCOUNT_SID": "ACx", "TWILIO_AUTH_TOKEN": "tokx"}
        settings = {
            "twilio_whatsapp_from": "whatsapp:+1",
            "owner_whatsapp": "whatsapp:+2",
            "twilio_voice_from": "+1",
            "owner_phone": "+2",
        }
        cfg = load_notify_config(settings, env)
        self.assertEqual(cfg.account_sid, "ACx")
        self.assertEqual(cfg.auth_token, "tokx")
        self.assertTrue(cfg.can_message)
        self.assertTrue(cfg.can_call)
        self.assertFalse(cfg.dry_run)

    def test_notify_dry_run_flag_forces_dry_run(self) -> None:
        cfg = load_notify_config({"notify_dry_run": True}, {})
        self.assertTrue(cfg.dry_run)

    def test_missing_creds_means_cannot_send(self) -> None:
        cfg = load_notify_config({}, {})
        self.assertFalse(cfg.can_message)
        self.assertFalse(cfg.can_call)

    def test_api_key_from_env(self) -> None:
        env = {
            "TWILIO_ACCOUNT_SID": "AC9",
            "TWILIO_API_KEY_SID": "SKabc",
            "TWILIO_API_KEY_SECRET": "sek",
        }
        settings = {"twilio_whatsapp_from": "whatsapp:+1", "owner_whatsapp": "whatsapp:+2"}
        cfg = load_notify_config(settings, env)
        self.assertEqual(cfg._auth, ("SKabc", "sek"))
        self.assertTrue(cfg.can_message)

    def test_api_key_without_account_sid_cannot_send(self) -> None:
        # The Account SID (AC...) is required for the URL even with an API key.
        cfg = NotifyConfig(api_key_sid="SK", api_key_secret="s",
                           whatsapp_from="whatsapp:+1", owner_whatsapp="whatsapp:+2")
        self.assertFalse(cfg.can_message)


class ApiKeyAuthTest(unittest.TestCase):
    def test_api_key_is_used_for_auth_account_sid_for_url(self) -> None:
        cfg = NotifyConfig(account_sid="AC9", api_key_sid="SKabc", api_key_secret="sek",
                           whatsapp_from="whatsapp:+1", owner_whatsapp="whatsapp:+2")
        with mock.patch.object(notify, "_http_post", return_value={"sid": "SM1"}) as post:
            notify.send_whatsapp(cfg, "hi")
        url, _fields, auth_user, auth_pass = post.call_args.args
        self.assertEqual(url, "https://api.twilio.com/2010-04-01/Accounts/AC9/Messages.json")
        self.assertEqual((auth_user, auth_pass), ("SKabc", "sek"))


class SendWhatsappTest(unittest.TestCase):
    def test_dry_run_sends_nothing(self) -> None:
        cfg = NotifyConfig(account_sid="AC", auth_token="t", whatsapp_from="whatsapp:+1",
                           owner_whatsapp="whatsapp:+2", dry_run=True)
        with mock.patch.object(notify, "_http_post") as post:
            res = notify.send_whatsapp(cfg, "hi")
        post.assert_not_called()
        self.assertFalse(res["sent"])
        self.assertEqual(res["reason"], "dry_run")

    def test_not_configured_sends_nothing(self) -> None:
        with mock.patch.object(notify, "_http_post") as post:
            res = notify.send_whatsapp(NotifyConfig(), "hi")
        post.assert_not_called()
        self.assertEqual(res["reason"], "not_configured")

    def test_configured_posts_to_messages_endpoint(self) -> None:
        cfg = _full()
        with mock.patch.object(notify, "_http_post", return_value={"sid": "SM1"}) as post:
            res = notify.send_whatsapp(cfg, "hello")
        self.assertTrue(res["sent"])
        self.assertEqual(res["sid"], "SM1")
        url, fields, sid, token = post.call_args.args
        self.assertEqual(url, "https://api.twilio.com/2010-04-01/Accounts/AC123/Messages.json")
        self.assertEqual(fields["From"], "whatsapp:+1000")
        self.assertEqual(fields["To"], "whatsapp:+2000")
        self.assertEqual(fields["Body"], "hello")
        self.assertEqual((sid, token), ("AC123", "tok"))

    def test_network_error_is_swallowed(self) -> None:
        cfg = _full()
        with mock.patch.object(notify, "_http_post", side_effect=OSError("boom")):
            res = notify.send_whatsapp(cfg, "hello")
        self.assertFalse(res["sent"])
        self.assertEqual(res["reason"], "error")


class CallOwnerTest(unittest.TestCase):
    def test_configured_posts_twiml_to_calls_endpoint(self) -> None:
        cfg = _full()
        with mock.patch.object(notify, "_http_post", return_value={"sid": "CA1"}) as post:
            res = notify.call_owner(cfg, "intruder at the door")
        self.assertTrue(res["called"])
        url, fields, _sid, _token = post.call_args.args
        self.assertEqual(url, "https://api.twilio.com/2010-04-01/Accounts/AC123/Calls.json")
        self.assertEqual(fields["From"], "+1000")
        self.assertEqual(fields["To"], "+2000")
        self.assertIn("<Say>intruder at the door</Say>", fields["Twiml"])


class DispatchTest(unittest.TestCase):
    def test_send_message_sends_whatsapp_only(self) -> None:
        cfg = _full()
        with mock.patch.object(notify, "_http_post", return_value={"sid": "SM1"}) as post:
            res = notify.notify(cfg, "[send_message]", "person at gate", "")
        self.assertTrue(res["whatsapp"]["sent"])
        self.assertNotIn("call", res)
        self.assertEqual(post.call_count, 1)

    def test_call_owner_sends_whatsapp_and_calls(self) -> None:
        cfg = _full()
        with mock.patch.object(notify, "_http_post", return_value={"sid": "X"}) as post:
            res = notify.notify(cfg, "[call_owner]", "forced entry", "window broken")
        self.assertTrue(res["whatsapp"]["sent"])
        self.assertTrue(res["call"]["called"])
        self.assertEqual(post.call_count, 2)

    def test_call_owner_without_voice_falls_back_to_whatsapp(self) -> None:
        cfg = NotifyConfig(account_sid="AC", auth_token="t",
                           whatsapp_from="whatsapp:+1", owner_whatsapp="whatsapp:+2")  # no voice
        with mock.patch.object(notify, "_http_post", return_value={"sid": "SM1"}) as post:
            res = notify.notify(cfg, "[call_owner]", "loitering", "")
        self.assertTrue(res["whatsapp"]["sent"])
        self.assertFalse(res["call"]["called"])
        self.assertEqual(res["call"]["reason"], "fallback_to_whatsapp")
        self.assertEqual(post.call_count, 1)

    def test_none_sends_nothing(self) -> None:
        cfg = _full()
        with mock.patch.object(notify, "_http_post") as post:
            res = notify.notify(cfg, "[none]")
        post.assert_not_called()
        self.assertFalse(res["sent"])


if __name__ == "__main__":
    unittest.main()
