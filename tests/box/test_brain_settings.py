# tests/box/test_brain_settings.py
from __future__ import annotations

import os
import tempfile
import unittest
from dataclasses import replace

from test_brain_tools_read import snapshot

from home_guard_project.box.brain.claims import unbacked_claims
from home_guard_project.box.brain.memory import ChatState
from home_guard_project.box.brain.receipts import DONE, FAILED, ReceiptBook
from home_guard_project.box.brain.render import receipt_line
from home_guard_project.box.brain.tools import _hours, TOOLS, Services, ToolContext, change_setting, settings_line, settings_view

NOW = 1_790_000_000.0


class SettingsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.dir = tempfile.mkdtemp()
        self.store = {"alert_start_hour": 22, "alert_end_hour": 6, "alert_cooldown_sec": 120.0,
                      "inference_conf": 0.4, "owner_language": "en"}

    def set_option(self, key, value):
        if key == "quiet_log":
            self.store[key] = str(value).lower() in ("true", "yes", "on", "1")
            return
        if key == "inference_conf" and not 0.05 <= float(value) <= 0.95:
            raise ValueError("inference_conf must be a number from 0.05 to 0.95")
        text = str(value)
        self.store[key] = text if key == "owner_language" else (float(text) if "." in text else int(text))

    def ctx(self, text) -> ToolContext:
        services = Services(roots=lambda: [], desc_dir=self.dir, feedback_dir=self.dir, work_dir=self.dir,
                            mute=None, deliver=None, set_option=self.set_option,
                            read_settings=lambda: dict(self.store), now=lambda: NOW)
        return ToolContext(turn_id="t1", chat_id="-5", speaker={}, text=text, lang="en", mode="assistant",
                           snapshot=snapshot("assistant"), state=ChatState(), services=services,
                           book=ReceiptBook(os.path.join(self.dir, "r"), now=lambda: NOW))

    def test_registered(self) -> None:
        self.assertIn("change_setting", TOOLS)
        self.assertEqual(len(TOOLS), 18)   # + ask_vision (§10)

    def test_settings_line(self) -> None:
        self.assertEqual(settings_line(self.store),
                         "SETTINGS: alert hours 22:00–06:00 · time between alerts per camera 2 min · "
                         "detector sensitivity medium (0.40) · box language English (alerts and announcements)"
                         " · quiet log outside the hours off")

    def test_quiet_log_receipt(self) -> None:
        ctx = self.ctx("turn on quiet logging")
        out = change_setting(ctx, {"setting": "quiet_log", "value": "on", "owner_words": "turn on quiet logging"})
        self.assertEqual(out["status"], DONE)
        self.assertIs(self.store["quiet_log"], True)
        self.assertEqual(receipt_line(ctx.receipts[0], "en"), "✓ Quiet log outside the alert hours: off → on")

    def test_quiet_log_owner_view_and_receipt_in_english_and_hebrew(self) -> None:
        from home_guard_project.box.brain.i18n import t
        ctx = self.ctx("turn on quiet logging")
        change_setting(ctx, {"setting": "quiet_log", "value": "on", "owner_words": "turn on quiet logging"})
        for lang, on, off in (("en", "on", "off"), ("he", "פעיל", "כבוי")):
            with self.subTest(lang=lang):
                self.assertEqual(settings_view(self.store, lang=lang)["quiet_log"], on)
                self.assertIn(on, settings_line(self.store, lang=lang))
                self.assertIn(t("setting_quiet_log", lang), receipt_line(ctx.receipts[0], lang))
                self.assertIn(on, receipt_line(ctx.receipts[0], lang))
                self.assertIn(off, receipt_line(ctx.receipts[0], lang))
                self.assertIn(off, settings_line({"quiet_log": False}, lang=lang))

    def test_alert_hours_need_the_owners_words(self) -> None:
        ctx = self.ctx("from now on watch from 23 to 7")
        self.assertFalse(change_setting(ctx, {"setting": "alert_hours", "value": "23-07",
                                              "owner_words": "change it"})["ok"])
        out = change_setting(ctx, {"setting": "alert_hours", "value": "23-07", "owner_words": "watch from 23 to 7"})
        self.assertEqual(out["status"], DONE)
        self.assertEqual((self.store["alert_start_hour"], self.store["alert_end_hour"]), (23, 7))
        self.assertEqual(ctx.receipts[0].detail["setting"], "alert_hours")
        self.assertEqual(ctx.receipts[0].detail["old"], "22:00–06:00")
        self.assertEqual(ctx.receipts[0].detail["new"], "23:00–07:00")
        self.assertEqual(ctx.receipts[0].detail["restore"], {"alert_start_hour": 22, "alert_end_hour": 6})
        self.assertEqual(receipt_line(ctx.receipts[0], "en"), "✓ Alert hours: 22:00–06:00 → 23:00–07:00")

    def test_cooldown_and_sensitivity(self) -> None:
        ctx = self.ctx("make it more sensitive and alert every 5 minutes")
        change_setting(ctx, {"setting": "sensitivity", "value": "high", "owner_words": "more sensitive"})
        change_setting(ctx, {"setting": "cooldown_minutes", "value": 5, "owner_words": "every 5 minutes"})
        self.assertEqual((self.store["inference_conf"], self.store["alert_cooldown_sec"]), (0.25, 300))

    def test_box_language(self) -> None:
        ctx = self.ctx("from now on send the alerts in Hebrew")
        out = change_setting(ctx, {"setting": "language", "value": "Hebrew", "owner_words": "alerts in Hebrew"})
        self.assertEqual((out["status"], self.store["owner_language"]), (DONE, "he"))
        self.assertEqual(receipt_line(ctx.receipts[0], "en"), "✓ Box language: English → Hebrew")
        self.assertFalse(change_setting(ctx, {"setting": "language", "value": "French",
                                              "owner_words": "alerts in Hebrew"})["ok"])

    def test_bad_values(self) -> None:
        ctx = self.ctx("set the hours to whenever please")
        self.assertFalse(change_setting(ctx, {"setting": "alert_hours", "value": "whenever",
                                              "owner_words": "set the hours"})["ok"])
        self.assertFalse(change_setting(ctx, {"setting": "volume", "value": 3, "owner_words": "set the hours"})["ok"])
        self.assertFalse(change_setting(ctx, {"setting": "alert_hours", "value": "99-88",
                                              "owner_words": "set the hours"})["ok"])
        out = change_setting(ctx, {"setting": "sensitivity", "value": "0.99", "owner_words": "set the hours"})
        self.assertEqual(out["status"], FAILED)

    def test_claim_without_receipt(self) -> None:
        self.assertEqual(unbacked_claims("I changed the alert hours to 23-07.", []), ["setting"])

    def test_malformed_settings_view_and_line(self) -> None:
        for value in (None, [], "bad", {"alert_start_hour": []}, {"alert_end_hour": "nan"},
                      {"inference_conf": float("inf")}, {"alert_cooldown_sec": "bad"},
                      {"owner_language": {}}, {"owner_language": "unknown"}):
            for entrypoint in (settings_view, settings_line):
                with self.subTest(value=value, entrypoint=entrypoint.__name__):
                    with self.assertLogs("box.brain.tools", level="WARNING") as logs:
                        result = entrypoint(value)
                    self.assertEqual(len(logs.output), 1)
                    self.assertTrue(result)

    def test_malformed_tool_arguments_never_write(self) -> None:
        for value in (None, [], "bad", {"setting": []},
                      {"setting": "sensitivity", "value": object(), "owner_words": "set sensitivity"},
                      {"setting": "sensitivity", "value": float("nan"), "owner_words": "set sensitivity"},
                      {"setting": "sensitivity", "value": float("inf"), "owner_words": "set sensitivity"}):
            ctx = self.ctx("set sensitivity high")
            before = dict(self.store)
            result = change_setting(ctx, value)
            self.assertFalse(result["ok"])
            self.assertEqual(self.store, before)

    def test_failed_service_reads_and_writes(self) -> None:
        args = {"setting": "language", "value": "he", "owner_words": "in Hebrew"}
        for value in ([], {"owner_language": []}, {"inference_conf": "nan"}):
            ctx = self.ctx("send alerts in Hebrew")
            ctx.services.read_settings = lambda: value
            with self.assertLogs("box.brain.tools", level="WARNING") as logs:
                self.assertFalse(change_setting(ctx, args)["ok"])
            self.assertEqual(len(logs.output), 1)
            self.assertEqual(self.store["owner_language"], "en")
        def unreadable():
            raise UnicodeDecodeError("utf-8", b"\xff", 0, 1, "invalid byte")
        ctx.services.read_settings = unreadable
        with self.assertLogs("box.brain.tools", level="WARNING") as logs:
            self.assertFalse(change_setting(ctx, args)["ok"])
        self.assertEqual(len(logs.output), 1)
        ctx = self.ctx("send alerts in Hebrew")
        ctx.services.set_option = lambda key, value: unreadable()
        with self.assertLogs("box.brain.tools", level="WARNING") as logs:
            result = change_setting(ctx, args)
        self.assertEqual(result["status"], FAILED)
        self.assertEqual(len(logs.output), 1)

    def test_malformed_post_write_read_cannot_confirm(self) -> None:
        ctx = self.ctx("send alerts in Hebrew")
        reads = iter((dict(self.store), {"owner_language": []}))
        ctx.services.read_settings = lambda: next(reads)
        with self.assertLogs("box.brain.tools", level="WARNING"):
            result = change_setting(ctx, {"setting": "language", "value": "he", "owner_words": "in Hebrew"})
        self.assertFalse(result["ok"])
        self.assertFalse(any(r.status == DONE for r in ctx.receipts))

    def test_real_boxconfig_language_roundtrip(self) -> None:
        from home_guard_project.box import boxconfig
        path = os.path.join(self.dir, "box.yaml")
        with open(path, "w", encoding="utf-8") as stream:
            stream.write("site: house2\nalert_start_hour: 22\nalert_end_hour: 6\n")
        ctx = self.ctx("send alerts in Hebrew")
        ctx.services.set_option = lambda key, value: boxconfig.set_option(key, value, path)
        ctx.services.read_settings = lambda: boxconfig.load_box_settings(path)
        result = change_setting(ctx, {"setting": "language", "value": "Hebrew", "owner_words": "in Hebrew"})
        self.assertEqual(result["status"], DONE)
        self.assertEqual(boxconfig.get_option("owner_language", path), "he")
        self.assertIn("owner_language", boxconfig.LIVE_OPTIONS)
        self.assertNotIn("owner_language", boxconfig.RESTART_OPTIONS)
        self.assertIn("alert_on", boxconfig.LIVE_OPTIONS)
        with self.assertRaises(boxconfig.BoxConfigError):
            boxconfig.set_option("owner_language", "ar", path)

    def test_zero_to_24_is_all_day_but_equal_hours_are_refused(self) -> None:
        for value in ("0-24", "00-24", "0:00-24:00"):
            self.assertEqual(_hours(value), (0, 0), value)
        self.assertIsNone(_hours("6-6"))

    def test_boundary_values_and_pointer_rejection(self) -> None:
        ctx = self.ctx("set alert hours all day and cooldown please")
        self.assertTrue(change_setting(ctx, {"setting": "alert_hours", "value": "24-06",
                                             "owner_words": "set alert hours"})["ok"])
        self.assertEqual((self.store["alert_start_hour"], self.store["alert_end_hour"]), (0, 6))
        self.assertTrue(change_setting(ctx, {"setting": "alert_hours", "value": "all day",
                                             "owner_words": "set alert hours"})["ok"])
        self.assertEqual(self.store["alert_end_hour"], 0)
        for value in (0, -1, 1441, "nan", "inf", [], True):
            self.assertFalse(change_setting(ctx, {"setting": "cooldown_minutes", "value": value,
                                                  "owner_words": "cooldown please"})["ok"])
        ctx = self.ctx("that one please")
        self.assertFalse(change_setting(ctx, {"setting": "language", "value": "he",
                                              "owner_words": "that one"})["ok"])

    def test_setting_claims_preserve_save_and_negation(self) -> None:
        cases = {
            "I changed the alias to driveway.": ["save"],
            "I updated the verdict.": ["save"],
            "I changed the alert hours and I saved the alias.": ["save", "setting"],
            "I changed the alias and I changed the alert hours.": ["save", "setting"],
            "The alert hours changed yesterday.": [],
            "The settings were updated yesterday.": [],
            "It has been updated now to the new settings.": ["save"],
            "I changed the alias, the hours are now set to 23-07.": ["save", "setting"],
            "The hours are now set to 23-07.": ["setting"],
            "I have just set the sensitivity.": ["setting"],
            "I set nothing aside.": [],
            "I have not changed the hours.": [],
            "\u05dc\u05d0 \u05e9\u05d9\u05e0\u05d9\u05ea\u05d9 \u05e9\u05e2\u05d5\u05ea": [],
            "\u05e9\u05d9\u05e0\u05d9\u05ea\u05d9 \u05e9\u05e2\u05d5\u05ea": ["setting"],
            "\u063a\u064a\u0631\u062a \u0633\u0627\u0639\u0627\u062a": ["setting"],
        }
        for answer, expected in cases.items():
            with self.subTest(answer=answer):
                self.assertEqual(unbacked_claims(answer, []), expected)
        ctx = self.ctx("set alert hours please")
        change_setting(ctx, {"setting": "alert_hours", "value": "23-07", "owner_words": "set alert hours"})
        self.assertEqual(unbacked_claims("I changed the alert hours.", ctx.receipts), [])
        malformed = replace(ctx.receipts[0], detail={"setting": [], "old": {}, "new": float("nan")})
        with self.assertLogs("box.brain.render", level="WARNING") as logs:
            self.assertEqual(receipt_line(malformed, "en"), "")
        self.assertEqual(len(logs.output), 1)

    def test_setting_claim_needs_a_settings_subject(self) -> None:
        from home_guard_project.box.brain.receipts import Receipt
        def receipt(tool):
            return Receipt(id="r1", turn="t1", tool=tool, status=DONE, target="x", detail={}, reason="", ts=NOW)
        backed = {"I changed the alias to driveway.": "set_alias", "I updated the verdict.": "record_verdict",
                  "I switched the entrance camera off.": "set_camera_active"}
        for answer, tool in backed.items():
            with self.subTest(answer=answer):
                self.assertEqual(unbacked_claims(answer, [receipt(tool)]), [])
                self.assertNotIn("setting", unbacked_claims(answer, []))
        for answer in ("I changed the alert hours to 23-07.", "The hours are now set to 23-07.",
                       "I switched the language to Hebrew.", "I set the quiet log to off.",
                       "שיניתי את שעות ההתראה"):
            with self.subTest(answer=answer):
                self.assertIn("setting", unbacked_claims(answer, []))
        for answer in ("The entrance camera is now set to record.", "The alias is now set to driveway."):
            with self.subTest(answer=answer):
                self.assertNotIn("setting", unbacked_claims(answer, []))

    def hours_args(self, value="23-07"):
        return {"setting": "alert_hours", "value": value, "owner_words": "set alert hours"}

    def test_end_hour_write_fails_start_hour_is_put_back(self) -> None:
        ctx = self.ctx("set alert hours please")
        real = self.set_option
        def flaky(key, value):
            if key == "alert_end_hour":
                raise ValueError("config write failed")
            real(key, value)
        ctx.services.set_option = flaky
        result = change_setting(ctx, self.hours_args())
        self.assertEqual(result["status"], FAILED)
        self.assertEqual((self.store["alert_start_hour"], self.store["alert_end_hour"]), (22, 6))

    def test_end_hour_and_rollback_fail_reason_says_start_changed(self) -> None:
        ctx = self.ctx("set alert hours please")
        real = self.set_option
        calls = []
        def flaky(key, value):
            calls.append(key)
            if key == "alert_end_hour" or calls.count("alert_start_hour") > 1:
                raise ValueError("config write failed")
            real(key, value)
        ctx.services.set_option = flaky
        result = change_setting(ctx, self.hours_args())
        self.assertEqual(result["status"], FAILED)
        self.assertIn("start hour changed to 23, end hour unchanged", ctx.receipts[0].reason)
        self.assertEqual(self.store["alert_start_hour"], 23)

    def test_hours_readback_failure_after_both_writes_is_done(self) -> None:
        ctx = self.ctx("set alert hours please")
        reads = [dict(self.store)]
        def read():
            if reads:
                return reads.pop()
            raise OSError("unreadable")
        ctx.services.read_settings = read
        with self.assertLogs("box.brain.tools", level="WARNING"):
            result = change_setting(ctx, self.hours_args())
        self.assertEqual(result["status"], DONE)
        self.assertEqual(ctx.receipts[0].detail["new"], "23:00–07:00")
        self.assertEqual((self.store["alert_start_hour"], self.store["alert_end_hour"]), (23, 7))

    def test_same_start_and_end_needs_an_all_day_word(self) -> None:
        for value in ("6-6", "06-06"):
            with self.subTest(value=value):
                ctx = self.ctx("set alert hours please")
                self.assertFalse(change_setting(ctx, self.hours_args(value))["ok"])
                self.assertEqual((self.store["alert_start_hour"], self.store["alert_end_hour"]), (22, 6))
        for value in ("all day", "24h", "always", "כל היום", "طوال اليوم"):
            with self.subTest(value=value):
                ctx = self.ctx("set alert hours please")
                self.assertEqual(change_setting(ctx, self.hours_args(value))["status"], DONE)
                self.assertEqual((self.store["alert_start_hour"], self.store["alert_end_hour"]), (0, 0))

    def test_other_receipts_ignore_setting_detail_fields(self) -> None:
        ctx = self.ctx("set alert hours please")
        change_setting(ctx, {"setting": "alert_hours", "value": "23-07", "owner_words": "set alert hours"})
        receipt = replace(ctx.receipts[0], tool="set_alias", target="driveway",
                          detail={"alias": "entry", "camera": "driveway", "old": [], "new": {}, "setting": []})
        self.assertTrue(receipt_line(receipt, "en"))

    def test_alias_named_after_setting_still_requires_save_receipt(self) -> None:
        ctx = self.ctx("set alert hours please")
        change_setting(ctx, {"setting": "alert_hours", "value": "23-07", "owner_words": "set alert hours"})
        for answer in ("I changed the alias to alert hours.",
                       "I changed the alias but the alert hours are now set to 23-07.",
                       "I changed the alias: the hours are now set to 23-07."):
            with self.subTest(answer=answer):
                self.assertEqual(unbacked_claims(answer, ctx.receipts), ["save"])


if __name__ == "__main__":
    unittest.main()
