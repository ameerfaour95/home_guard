# tests/box/test_bot_human_1010.py
"""The owner's chat of 2026-10-09 / 2026-10-10: the bot leaked its bookkeeping, asked "until when?" about the
neighbour's house, dumped its memory, talked about the workers out of nowhere, asked generic questions and answered
a 402 in English. Models are scripted; nothing reaches a network."""

from __future__ import annotations

import unittest

from home_guard_project.box.agent import AgentContext, OwnerAgent, UNAVAILABLE_REPLY, unavailable_reply
from home_guard_project.box.brain.models import no_ai_access
from home_guard_project.box.brain.style import clean_outgoing, strip_internals

LEAK_0909 = ("במצלמת הפרגולה יש אדם אחד שנראה עובד או זז ליד הטריילר וחומרי הבניין. התמונה ברורה.\n"
             "✓ התמונה נשלחה (פרגולה)\n"
             "[handles: E30=photo פרגולה Fri 09 Oct 18:41 | receipts: R1 check_camera פרגולה done]")


class InternalsGuardTests(unittest.TestCase):
    def test_the_0909_leak_is_stripped(self) -> None:
        out = clean_outgoing(LEAK_0909, [], "he")
        self.assertEqual(out, "במצלמת הפרגולה יש אדם אחד שנראה עובד או זז ליד הטריילר וחומרי הבניין.")
        for word in ("handles", "receipts", "R1", "check_camera", "✓", "התמונה ברורה", "E30"):
            self.assertNotIn(word, out)

    def test_bookkeeping_lines_and_inline_brackets(self) -> None:
        self.assertEqual(strip_internals("כן, יש שני אנשים. [receipts: R1 check_camera x done]"), "כן, יש שני אנשים.")
        self.assertEqual(strip_internals("שלום\nreceipts: R2 send_media done\nR3 look_around הכל done"), "שלום")
        self.assertEqual(strip_internals("The picture is clear. Two men by the gate."), "Two men by the gate.")

    def test_a_photo_receipt_alone_stays(self) -> None:
        self.assertEqual(strip_internals("✓ התמונה נשלחה (פרגולה)"), "✓ התמונה נשלחה (פרגולה)")

    def test_plain_text_is_untouched(self) -> None:
        text = "הבנתי, זה הבית של השכן.\nמה שקורה אצלו לא יגיע אליך."
        self.assertEqual(strip_internals(text), text)



class Refused402(Exception):
    pass


ERR_402 = Refused402("Error code: 402 - {'error': {'message': 'This request requires more credits, or fewer "
                     "max_tokens.'}}")
ERR_429 = Refused402("Error code: 429 - {'error': {'message': 'You have no credits remaining.', 'type': "
                     "'insufficient_quota'}}")


class FailureTextTests(unittest.TestCase):
    def test_402_and_quota_are_no_access(self) -> None:
        self.assertTrue(no_ai_access(ERR_402))
        self.assertTrue(no_ai_access(ERR_429))
        self.assertFalse(no_ai_access(TimeoutError("read timed out")))

    def test_hebrew_and_honest(self) -> None:
        self.assertEqual(unavailable_reply("he", ERR_402), "אין לי כרגע גישה ל-AI, אני בודק ומעדכן.")
        self.assertEqual(unavailable_reply("he"), "לא הצלחתי לטפל בזה כרגע, אבל ההודעה שלך נשמרה.")
        self.assertNotEqual(unavailable_reply("he", ERR_402), UNAVAILABLE_REPLY)

    def test_v1_agent_answers_a_402_in_hebrew(self) -> None:
        import tempfile  # noqa: PLC0415

        from home_guard_project.box.feedback import MuteState  # noqa: PLC0415

        class Broke:
            def chat(self, *a, **k):
                raise ERR_429

        root = tempfile.mkdtemp()
        ctx = AgentContext(camera_names=["ameer_v2_ch6"], mute_state=MuteState(root + "/mute.json"),
                           feedback_dir=root, roots=lambda: [root], embedder=False, look_now=lambda c: {},
                           set_camera=lambda c, a: {}, conversations_dir=root + "/conv", language=lambda: "he")
        reply = OwnerAgent(Broke(), ctx).handle("מה קורה הם יש משהו מחוץ לכניסה הראשית", "-1", {}, None)
        self.assertEqual(reply.text, "אין לי כרגע גישה ל-AI, אני בודק ומעדכן.")

    def test_detector_only_summary_is_in_the_box_language(self) -> None:
        from home_guard_project.box.inference import DETECTED_ONLY, owner_summary  # noqa: PLC0415

        self.assertEqual(owner_summary(DETECTED_ONLY, "", "he"), "זוהה אדם או רכב")
        self.assertEqual(owner_summary(DETECTED_ONLY, "", "en"), DETECTED_ONLY)


if __name__ == "__main__":
    unittest.main()
