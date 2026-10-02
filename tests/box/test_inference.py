from __future__ import annotations

import unittest
from unittest import mock

from home_guard_project.box import inference as inf
from home_guard_project.box.inference import AlertSettings, NullBackend


class WindowTest(unittest.TestCase):
    def test_normal_window(self) -> None:
        self.assertTrue(inf.in_alert_window(10, 9, 17))
        self.assertFalse(inf.in_alert_window(8, 9, 17))
        self.assertFalse(inf.in_alert_window(17, 9, 17))  # end exclusive

    def test_wraps_midnight(self) -> None:
        self.assertTrue(inf.in_alert_window(23, 22, 6))
        self.assertTrue(inf.in_alert_window(3, 22, 6))
        self.assertFalse(inf.in_alert_window(12, 22, 6))

    def test_equal_means_always(self) -> None:
        self.assertTrue(inf.in_alert_window(0, 0, 0))
        self.assertTrue(inf.in_alert_window(13, 5, 5))


class ParseTest(unittest.TestCase):
    def test_plain_json(self) -> None:
        self.assertEqual(inf.parse_vlm_json('{"a": 1}'), {"a": 1})

    def test_json_with_surrounding_text(self) -> None:
        self.assertEqual(inf.parse_vlm_json('sure: {"a": 1} done'), {"a": 1})

    def test_garbage(self) -> None:
        self.assertIsNone(inf.parse_vlm_json("not json"))
        self.assertIsNone(inf.parse_vlm_json(""))


class PolicyOverrideTest(unittest.TestCase):
    def test_outside_window_forces_none(self) -> None:
        out = inf.apply_policy_override({"alert_command": "[call_owner]"}, in_window=False, person=True, vehicle=False)
        self.assertEqual(out["alert_command"], "[none]")

    def test_inside_person_none_upgrades(self) -> None:
        out = inf.apply_policy_override({"alert_command": "[none]"}, in_window=True, person=True, vehicle=False)
        self.assertEqual(out["alert_command"], "[send_message]")
        self.assertTrue(out["alert_reason"])

    def test_inside_call_owner_unchanged(self) -> None:
        out = inf.apply_policy_override({"alert_command": "[call_owner]"}, in_window=True, person=True, vehicle=False)
        self.assertEqual(out["alert_command"], "[call_owner]")

    def test_inside_no_trigger_stays_none(self) -> None:
        out = inf.apply_policy_override({"alert_command": "[none]"}, in_window=True, person=False, vehicle=False)
        self.assertEqual(out["alert_command"], "[none]")


class SettingsTest(unittest.TestCase):
    def test_defaults(self) -> None:
        s = AlertSettings.from_box_settings({})
        self.assertEqual(s.alert_channel, "telegram")
        self.assertEqual(s.vlm_backend, "gpt")
        self.assertEqual(s.model, "yolo11s.pt")

    def test_custom(self) -> None:
        s = AlertSettings.from_box_settings({
            "alert_start_hour": 22, "alert_end_hour": 6, "alert_cooldown_sec": 300,
            "alert_channel": "both", "notify_dry_run": True,
        })
        self.assertEqual((s.alert_start_hour, s.alert_end_hour), (22, 6))
        self.assertEqual(s.cooldown_sec, 300.0)
        self.assertEqual(s.alert_channel, "both")
        self.assertTrue(s.dry_run)


class BackendSelectionTest(unittest.TestCase):
    def test_dry_run_is_null(self) -> None:
        s = AlertSettings(dry_run=True)
        self.assertIsInstance(inf.make_backend(s, {"OPENAI_API_KEY": "k"}), NullBackend)

    def test_gpt_without_key_is_null(self) -> None:
        s = AlertSettings(vlm_backend="gpt", dry_run=False)
        self.assertIsInstance(inf.make_backend(s, {}), NullBackend)

    def test_unknown_backend_is_null(self) -> None:
        s = AlertSettings(vlm_backend="llava", dry_run=False)
        self.assertIsInstance(inf.make_backend(s, {}), NullBackend)

    def test_null_backend_returns_none_verdict(self) -> None:
        raw, parsed = NullBackend().analyze([], "cam", 0, 0, 0)
        self.assertEqual(parsed["alert_command"], "[none]")


class _FakeBox:
    def __init__(self, cls_id: int) -> None:
        self.cls = [cls_id]


class _FakeResult:
    def __init__(self, cls_ids, names) -> None:
        self.boxes = [_FakeBox(c) for c in cls_ids]
        self.names = names


class DetectTriggerTest(unittest.TestCase):
    NAMES = {0: "person", 2: "car", 15: "cat"}

    def test_person_and_car(self) -> None:
        person, vehicle, labels = inf.detect_trigger(_FakeResult([0, 2], self.NAMES))
        self.assertTrue(person)
        self.assertTrue(vehicle)
        self.assertEqual(labels, ["car", "person"])

    def test_only_cat_no_trigger(self) -> None:
        person, vehicle, labels = inf.detect_trigger(_FakeResult([15], self.NAMES))
        self.assertFalse(person)
        self.assertFalse(vehicle)
        self.assertEqual(labels, [])

    def test_empty(self) -> None:
        self.assertEqual(inf.detect_trigger(_FakeResult([], self.NAMES)), (False, False, []))


class DispatchTest(unittest.TestCase):
    def test_telegram_channel_routes_to_telegram(self) -> None:
        from home_guard_project.box import telegram_notify
        box_settings = {"alert_channel": "telegram", "telegram_chat_ids": "1"}
        with mock.patch.object(telegram_notify, "notify", return_value={"telegram": "ok"}) as n:
            res = inf.dispatch_alert(box_settings, {"TELEGRAM_BOT_TOKEN": "T"}, "[send_message]", "sum", "why")
        n.assert_called_once()
        self.assertEqual(res["channel"], "telegram")
        self.assertEqual(res["telegram"], {"telegram": "ok"})

    def test_dispatch_never_raises(self) -> None:
        from home_guard_project.box import telegram_notify
        with mock.patch.object(telegram_notify, "notify", side_effect=RuntimeError("boom")):
            res = inf.dispatch_alert({"alert_channel": "telegram"}, {}, "[send_message]", "s", "r")
        self.assertIn("error", res)


if __name__ == "__main__":
    unittest.main()
