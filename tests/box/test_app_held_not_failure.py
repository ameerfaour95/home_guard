"""2026-10-10: the box app showed "ההתראות לא מגיעות לטלגרם" for an alert the event book kept on purpose
("normal: kept in the event, not sent"). A held decision is neutral and never raises the delivery banner."""
import time
import unittest

from home_guard_project.box.app.ai_view import decisions, undelivered_alert


def status(*entries):
    return {"decisions": [dict(ts=time.time() - i, camera="cam", labels=["person"], summary="s", command="[send_message]",
                               **e) for i, e in enumerate(entries)]}


class HeldIsNotFailureTest(unittest.TestCase):
    def test_a_held_decision_raises_no_banner(self):
        data = status(dict(sent=False, held="normal: kept in the event, not sent"))
        self.assertIsNone(undelivered_alert(data))
        self.assertEqual(decisions(data)[0].outcome()[0], "muted")

    def test_an_old_record_with_the_reason_in_error_is_held_too(self):
        data = status(dict(sent=False, error="normal: kept in the event, not sent"))
        self.assertIsNone(undelivered_alert(data))

    def test_a_real_delivery_failure_still_raises_it(self):
        data = status(dict(sent=False, error="Telegram: 401 Unauthorized"))
        self.assertIsNotNone(undelivered_alert(data))
        self.assertEqual(decisions(data)[0].outcome()[0], "error")


if __name__ == "__main__":
    unittest.main()
