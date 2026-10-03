# tests/box/test_brain_render.py
from __future__ import annotations

import unittest

from home_guard_project.box.brain.claims import unbacked_claims
from home_guard_project.box.brain.i18n import t
from home_guard_project.box.brain.receipts import DONE, FAILED, REQUESTED, Receipt
from home_guard_project.box.brain.render import receipt_line, render_reply


def r(tool, status, target="", reason="", **detail):
    return Receipt(id="R1", turn="t", tool=tool, status=status, target=target, detail=detail, reason=reason)


class ClaimsTest(unittest.TestCase):
    def test_claims_without_receipts_are_caught_in_three_languages(self) -> None:
        self.assertEqual(unbacked_claims("Here are the two videos", []), ["send"])
        self.assertEqual(unbacked_claims("המצלמה הקדמית כובתה עד 01:35", []), ["off"])
        self.assertEqual(unbacked_claims("סימנתי את ההתרעה הזו כהתרעה שגויה", []), ["save"])
        self.assertEqual(unbacked_claims("تم إيقاف التنبيهات حتى السادسة", []), ["pause"])

    def test_receipts_back_the_claim(self) -> None:
        self.assertEqual(unbacked_claims("Here is the video.", [r("send_media", DONE)]), [])
        self.assertEqual(unbacked_claims("turned off", [r("set_camera_active", REQUESTED)]), ["off"])
        self.assertEqual(unbacked_claims("turned off", [r("set_camera_active", DONE)]), [])
        self.assertEqual(unbacked_claims("I sent it", [r("send_media", FAILED)]), ["send"])

    def test_plain_facts_pass(self) -> None:
        self.assertEqual(unbacked_claims("Two events tonight, both at the entrance.", []), [])
        self.assertEqual(unbacked_claims("היו שני אירועים היום במצלמה test_ch6.", []), [])

    def test_malformed_claim_inputs_are_skipped_and_logged_once(self) -> None:
        with self.assertLogs("box.brain.claims", level="WARNING") as logs:
            self.assertEqual(unbacked_claims("I sent it", [None, {}, r([], DONE)]), ["send"])
        self.assertEqual(len(logs.output), 1)
        with self.assertLogs("box.brain.claims", level="WARNING"):
            self.assertEqual(unbacked_claims("I sent it", None), ["send"])
        with self.assertLogs("box.brain.claims", level="WARNING"):
            self.assertEqual(unbacked_claims(["I sent it"], []), [])
        with self.assertLogs("box.brain.claims", level="WARNING"):
            self.assertEqual(unbacked_claims("I sent it", [None, r("send_media", DONE)]), [])


class RenderTest(unittest.TestCase):
    def test_lines(self) -> None:
        self.assertEqual(receipt_line(r("send_media", DONE, kind="video", bounds="01:24:03–01:24:13"), "en"),
                         "✓ Video sent (01:24:03–01:24:13)")
        self.assertEqual(receipt_line(r("send_media", FAILED, reason="not_on_box"), "en"),
                         "✗ Sending the video could not be done: the video is no longer on the box (older than 14 days)")
        self.assertEqual(receipt_line(r("pause_alerts", DONE, camera="", until="06:00"), "en"),
                         "✓ Alerts paused until 06:00. The cameras keep watching.")
        self.assertEqual(receipt_line(r("set_camera_active", REQUESTED, camera="back_door", active=False), "en"),
                         "⏳ Turning back_door off - the box restarts for a moment.")
        self.assertEqual(receipt_line(r("set_camera_active", DONE, camera="back_door", active=False), "en"),
                         "✓ back_door is off.")
        self.assertEqual(receipt_line(r("record_verdict", DONE, verdict="expected"), "en"),
                         "✓ Noted: expected activity")
        self.assertEqual(receipt_line(r("set_alias", FAILED, reason='"x" already names y'), "en"),
                         '✗ Saving the camera name could not be done: "x" already names y')
        self.assertIn("נשלחה", receipt_line(r("check_camera", DONE, camera="gate"), "he"))

    def test_reply_is_answer_then_receipt_lines(self) -> None:
        text = render_reply("Two events tonight.", [r("send_media", DONE, kind="video", bounds="b")], "en")
        self.assertEqual(text, "Two events tonight.\n✓ Video sent (b)")
        self.assertEqual(render_reply("", [], "en"), "")

    def test_malformed_receipt_inputs_are_skipped_and_logged_once(self) -> None:
        bad_detail = r("send_media", DONE)
        bad_detail.detail = []
        cases = [
            (None, "en", 14),
            (bad_detail, "en", 14),
            (r("send_media", DONE, bounds=object()), "en", 14),
            (r("send_media", DONE), [], 14),
            (r("set_camera_active", DONE, active="false"), "en", 14),
            (r("record_verdict", DONE, verdict="unknown"), "en", 14),
            (r("send_media", "invalid"), "en", 14),
            (r("send_media", REQUESTED), "en", 14),
        ]
        for value in ("invalid", float("nan"), float("inf"), -1, None, []):
            cases.append((r("send_media", FAILED, reason="not_on_box"), "en", value))
            cases.append((r("record_clip", DONE, seconds=value), "en", 14))
        for receipt, lang, days in cases:
            with self.subTest(receipt=receipt, lang=lang, days=days):
                with self.assertLogs("box.brain.render", level="WARNING") as logs:
                    self.assertEqual(receipt_line(receipt, lang, days), "")
                self.assertEqual(len(logs.output), 1)

    def test_malformed_reply_inputs_preserve_valid_receipts_and_log_once(self) -> None:
        good = r("send_media", DONE, kind="video", bounds="b")
        bad = r("check_camera", DONE)
        bad.detail = []
        with self.assertLogs("box.brain.render", level="WARNING") as logs:
            self.assertEqual(render_reply({"text": "bad"}, [None, bad, good], "en"), "✓ Video sent (b)")
        self.assertEqual(len(logs.output), 1)
        for receipts in (None, {}, "bad", 42):
            with self.subTest(receipts=receipts):
                with self.assertLogs("box.brain.render", level="WARNING"):
                    self.assertEqual(render_reply(" Facts. ", receipts, "en"), "Facts.")

    def test_unknown_detail_keys_are_ignored(self) -> None:
        receipt = r("send_media", DONE, kind="video", bounds="b", by="owner", extra=object())
        self.assertEqual(receipt_line(receipt, "en"), "✓ Video sent (b)")

    def test_nothing_done_in_three_languages(self) -> None:
        self.assertEqual(t("nothing_done", "en"), "I did not do anything yet - please tell me again what you need.")
        self.assertEqual(t("nothing_done", "he"), "עדיין לא עשיתי כלום - תכתוב לי שוב מה צריך.")
        self.assertEqual(t("nothing_done", "ar"), "لم أقم بأي إجراء بعد - أخبرني مرة أخرى بما تحتاجه.")


if __name__ == "__main__":
    unittest.main()
