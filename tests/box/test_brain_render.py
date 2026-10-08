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
        # Claims per action (2026-10-08): a verdict wording is its own kind as well as a save.
        self.assertEqual(unbacked_claims("סימנתי את ההתרעה הזו כהתרעה שגויה", []), ["verdict", "save"])
        self.assertEqual(unbacked_claims("أوقفت التنبيهات حتى السادسة", []), ["pause"])

    def test_plain_facts_and_negations_are_not_claims(self) -> None:
        for text in ["An alert was sent at 02:10", "Nothing was sent", "No events were noted",
                     "The alert was marked as expected earlier", "The camera is enabled in settings",
                     "היו שני אירועים, נשלח אליך אתמול", "لم يتم إرسال شيء", "I did not send anything",
                     "לא שלחתי כלום"]:
            self.assertEqual(unbacked_claims(text, []), [], text)

    def test_first_person_and_now_claims_are_caught(self) -> None:
        self.assertEqual(unbacked_claims("I sent you the video.", []), ["send"])
        self.assertEqual(unbacked_claims("I've paused the alerts.", []), ["pause"])
        self.assertEqual(unbacked_claims("Alerts are now paused until 06:00.", []), ["pause"])
        self.assertEqual(unbacked_claims("Here are the two latest videos", []), ["send"])

    def test_fix2_english_claims_after_an_action_are_caught(self) -> None:
        cases = {
            "Alerts are paused until 06:00.": ["pause"],
            "Done, alerts paused until 6.": ["pause"],
            "Muted for an hour.": ["pause"],
            "Done - paused until six.": ["pause"],
            "The back camera is turned off now.": ["off"],
            "Done - back camera turned off.": ["off"],
            "Turned off the back camera.": ["off"],
            "The camera is off until 6.": ["off"],
            "Sent.": ["send"],
            "Marked as a false alarm.": ["verdict", "save"],
            "Two events. Done - sent.": ["send"],
        }
        for text, kinds in cases.items():
            self.assertEqual(unbacked_claims(text, []), kinds, text)

    def test_fix2_negation_stops_at_clause_boundaries(self) -> None:
        self.assertEqual(unbacked_claims("No problem, I sent it", []), ["send"])
        self.assertEqual(unbacked_claims("I'm not sure why, but I sent it", []), ["send"])
        self.assertEqual(unbacked_claims("No problem I sent it", []), ["send"])
        self.assertEqual(unbacked_claims("Not sure, I paused the alerts", []), ["pause"])

    def test_fix2_plain_facts_still_pass(self) -> None:
        for text in ["An alert was sent at 02:10", "The camera was turned off at 3pm", "The back camera is off.",
                     "No video was saved for that time.", "I noted two events at the entrance",
                     "I recorded no events", "I recorded nothing at the gate", "I sent nothing",
                     "I set nothing aside"]:
            self.assertEqual(unbacked_claims(text, []), [], text)

    def test_fix2_probe_is_fast(self) -> None:
        import time
        start = time.perf_counter()
        unbacked_claims("done - turned " * 5000, [])
        unbacked_claims("done - " * 5000 + "x", [])
        self.assertLess(time.perf_counter() - start, 1.0)

    def test_receipts_back_the_claim(self) -> None:
        self.assertEqual(unbacked_claims("Here is the video.", [r("send_media", DONE)]), [])
        self.assertEqual(unbacked_claims("I turned it off", [r("set_camera_active", REQUESTED)]), ["off"])
        self.assertEqual(unbacked_claims("I turned it off", [r("set_camera_active", DONE)]), [])
        self.assertEqual(unbacked_claims("I sent it", [r("send_media", FAILED)]), ["send"])

    def test_plain_facts_pass(self) -> None:
        self.assertEqual(unbacked_claims("Two events tonight, both at the entrance.", []), [])
        self.assertEqual(unbacked_claims("היו שני אירועים היום במצלמה test_ch6.", []), [])

    def test_house_state_claims_need_a_house_receipt(self) -> None:
        for text in ("I've set the house to vacation mode.", "I set you to asleep.", "You're now marked as away.",
                     "The house is now in night mode.", "סימנתי שאתם ישנים.", "העברתי את הבית למצב חופשה",
                     "הבית עכשיו במצב לילה", "I've noted that you're expecting a package."):
            with self.subTest(text=text):
                self.assertIn("house", unbacked_claims(text, []))
                self.assertEqual(unbacked_claims(text, [r("house_state", DONE)]), [])
        for text in ("The house is asleep until 06:00 by the night schedule.", "You are away since 08:30.",
                     "I did not set the house to away.", "הבית במצב רגיל"):
            with self.subTest(text=text):
                self.assertEqual(unbacked_claims(text, []), [])

    def test_remember_promises_need_a_save_receipt(self) -> None:
        # The pergola bug (2026-10-05): "I'll remember" passed although set_alias never ran.
        for text in ("אזכור שהפרגולה זה מצלמה 3.", "בסדר, אני זוכר: פרגולה = מצלמה 3", "אני אזכור את זה",
                     "סבבה, זוכרת!", "I'll remember that the pergola is camera 3.", "I will remember it.",
                     "Got it - I'll keep that in mind.", "Noted: pergola means camera 3.", "I've noted that."):
            with self.subTest(text=text):
                self.assertIn(unbacked_claims(text, []), (["save"], ["alias", "save"]))   # a name: alias too
                self.assertEqual(unbacked_claims(text, [r("set_alias", DONE)]), [])
        for text in ("אני לא זוכר שהיה שם מישהו", "I don't remember any event there", "I noted two events at 22:00",
                     "Remember to lock the gate", "עדיין לא שמרתי את זה"):
            with self.subTest(text=text):
                self.assertEqual(unbacked_claims(text, []), [])

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
                         "⏳ Turning back door off - the box restarts for a moment.")      # a name, never the id
        self.assertEqual(receipt_line(r("set_camera_active", DONE, camera="back_door", active=False), "en"),
                         "✓ back door is off.")
        self.assertEqual(receipt_line(r("record_verdict", DONE, verdict="expected"), "en"),
                         "✓ Noted: expected activity")
        self.assertEqual(receipt_line(r("set_alias", FAILED, reason='"x" already names y'), "en"),
                         '✗ Saving the camera name could not be done: "x" already names y')
        self.assertIn("נשלחה", receipt_line(r("check_camera", DONE, camera="gate"), "he"))

    def test_lines_name_the_camera_the_way_the_owner_does(self) -> None:
        self.assertEqual(receipt_line(r("check_camera", DONE, camera="camera_3", aka="פרגולה"), "he"),
                         "✓ התמונה נשלחה (camera 3 (פרגולה))")
        self.assertEqual(receipt_line(r("record_clip", DONE, camera="camera_3", aka="pergola", seconds=10), "en"),
                         "✓ New 10-second video from camera 3 (pergola) sent")
        self.assertEqual(receipt_line(r("set_alias", DONE, camera="camera_3", alias="פרגולה", photo=True), "he"),
                         '✓ "פרגולה" מעכשיו זה camera 3')
        self.assertEqual(receipt_line(r("set_alias", DONE, camera="camera_3", alias="pergola", undo_of="set_alias"),
                                      "en"), '✓ "pergola" no longer means camera 3')

    def test_lines_never_show_a_camera_id(self) -> None:
        # Owner rule (2026-10-06, again 2026-10-08): the family's name, else "מצלמה 6", never ameer_week_0_1_ch6.
        from home_guard_project.box.brain.registry import CameraState, HouseSnapshot
        snap = HouseSnapshot(now=0.0, mode="guard", mode_ends=None, mode_started=None, start_hour=0, end_hour=0,
                             cameras=(CameraState("ameer_week_0_1_ch6", True, aliases=("כניסה ראשית",)),
                                      CameraState("ameer_week_0_1_ch2", True)))
        self.assertEqual(receipt_line(r("pause_alerts", DONE, camera="ameer_week_0_1_ch6", until="06:00"), "he", 14,
                                      snap), receipt_line(r("pause_alerts", DONE, camera="X", until="06:00"), "he")
                         .replace("X", "כניסה ראשית"))
        line = receipt_line(r("check_camera", DONE, camera="ameer_week_0_1_ch2"), "he", 14, snap)
        self.assertIn("מצלמה 2", line)
        self.assertNotIn("ameer", line)
        # A new name's receipt says which camera got it by its other name, not by the new one twice.
        self.assertEqual(receipt_line(r("set_alias", DONE, camera="ameer_week_0_1_ch6", alias="כניסה ראשית"), "he",
                                      14, snap), '✓ "כניסה ראשית" מעכשיו זה מצלמה 6')
        self.assertNotIn("ameer", render_reply("", [r("set_camera_active", DONE, camera="ameer_week_0_1_ch2",
                                                      active=True)], "en", 14, snap))

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
