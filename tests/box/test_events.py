import json
import os
import tempfile
import unittest

from home_guard_project.box.events import ESCALATION_REPEAT_SEC, EventBook

CAM = "ameer_week_0_1_ch3"
T0 = 1_791_355_000.0            # 2026-10-07 ~09:56


class EventBookTest(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.book = EventBook(self.dir)

    def sent(self, d, label, people, ts, mid=1):
        self.book.record_sent(d.session_id, label, people, ts, alert_id=f"a{ts}", chat_id=-1, message_id=mid)

    def test_normal_is_never_sent(self):
        for i in range(10):
            d = self.book.decide(CAM, T0 + i * 120, "normal", 2, "two men carry a board")
            self.assertFalse(d.notify, d.reason)

    def test_normal_can_be_turned_on_for_data_collection(self):
        book = EventBook(self.dir, notify_normal=True)
        self.assertTrue(book.decide(CAM, T0, "normal", 1).notify)

    def test_suspicious_once_per_event_then_only_when_more_people(self):
        d = self.book.decide(CAM, T0, "suspicious", 2)
        self.assertTrue(d.notify)
        self.sent(d, "suspicious", 2, T0)
        self.assertFalse(self.book.decide(CAM, T0 + 40, "suspicious", 2).notify)
        d3 = self.book.decide(CAM, T0 + 80, "suspicious", 3)
        self.assertTrue(d3.notify)
        self.assertEqual(d3.new_people, 1)
        self.assertEqual(d3.reply_to["message_id"], 1)       # the update goes in the event's thread

    def test_session_ends_after_idle_and_activity_keeps_it(self):
        d = self.book.decide(CAM, T0, "suspicious", 1)
        self.sent(d, "suspicious", 1, T0)
        for i in range(1, 30):                                # a person in view the whole time
            self.book.activity(CAM, T0 + i * 10, people=1)
        self.assertFalse(self.book.decide(CAM, T0 + 300, "suspicious", 1).notify)
        self.book.activity(CAM, T0 + 500, people=0)           # gone for more than a minute
        d2 = self.book.decide(CAM, T0 + 520, "suspicious", 1)
        self.assertTrue(d2.notify)
        self.assertNotEqual(d2.session_id, d.session_id)

    def test_owner_known_silences_suspicious_but_not_escalation(self):
        d = self.book.decide(CAM, T0, "suspicious", 3)
        self.sent(d, "suspicious", 3, T0)
        receipt = self.book.mark_known(CAM, "עובדים בפרגולה", "owner", until=T0 + 8 * 3600, now=T0 + 60)
        self.assertEqual(receipt["people"], 3)
        quiet = self.book.decide(CAM, T0 + 600, "suspicious", 2, "man with a mask loads the van")
        self.assertFalse(quiet.notify)
        self.assertEqual(quiet.known_text, "עובדים בפרגולה")
        self.assertTrue(self.book.decide(CAM, T0 + 700, "escalation", 2).notify)

    def test_known_survives_a_new_session_the_same_day(self):
        self.book.decide(CAM, T0, "normal", 3)
        self.book.mark_known(CAM, "workers", "owner", until=T0 + 8 * 3600, now=T0 + 10)
        later = self.book.decide(CAM, T0 + 3 * 3600, "suspicious", 2)      # a new session after lunch
        self.assertFalse(later.notify)
        tomorrow = self.book.decide(CAM, T0 + 24 * 3600, "suspicious", 2)
        self.assertTrue(tomorrow.notify)

    def test_known_does_not_cover_more_people(self):
        self.book.decide(CAM, T0, "normal", 2)
        self.book.mark_known(CAM, "workers", "owner", until=T0 + 3600, now=T0 + 5)
        self.assertFalse(self.book.decide(CAM, T0 + 50, "suspicious", 4).notify)
        self.assertTrue(self.book.decide(CAM, T0 + 60, "suspicious", 5).notify)

    def test_known_is_per_camera(self):
        self.book.mark_known(CAM, "workers", "owner", until=T0 + 3600, now=T0, people=3)
        self.assertTrue(self.book.decide("ameer_week_0_1_ch6", T0 + 10, "suspicious", 1).notify)

    def test_escalation_repeat_needs_new_people_or_time(self):
        d = self.book.decide(CAM, T0, "escalation", 1)
        self.sent(d, "escalation", 1, T0)
        self.assertFalse(self.book.decide(CAM, T0 + 30, "escalation", 1).notify)
        self.assertTrue(self.book.decide(CAM, T0 + 40, "escalation", 2).notify)
        self.book.activity(CAM, T0 + 50, people=1)
        for i in range(1, 70):
            self.book.activity(CAM, T0 + 50 + i * 10, people=1)
        self.assertTrue(self.book.decide(CAM, T0 + ESCALATION_REPEAT_SEC + 60, "escalation", 1).notify)

    def test_roll_over_keeps_the_story(self):
        d = self.book.decide(CAM, T0, "suspicious", 2)
        self.sent(d, "suspicious", 2, T0)
        for i in range(1, 200):
            self.book.activity(CAM, T0 + i * 10, people=2)
        d2 = self.book.decide(CAM, T0 + 2000, "suspicious", 2)
        self.assertFalse(d2.notify)                           # linked session, nothing new
        self.assertNotEqual(d2.session_id, d.session_id)
        self.assertEqual(self.book.recent(T0, CAM)[-1]["parent"], d.session_id)

    def test_closed_sessions_are_archived_and_found_by_alert(self):
        self.book.decide(CAM, T0, "normal", 1, "a man walks", alert_id="ameer_week_0_1_ch3_1_alert")
        self.book.tick(T0 + 120)
        with open(os.path.join(self.dir, "events.jsonl"), encoding="utf-8") as f:
            rows = [json.loads(line) for line in f]
        self.assertEqual(len(rows), 1)
        self.assertEqual(self.book.session_of_alert("ameer_week_0_1_ch3_1_alert")["id"], rows[0]["id"])
        self.assertIn("a man walks", self.book.recent(T0 - 1)[0]["summary_line"])

    def test_known_is_persisted_and_can_be_undone(self):
        r = self.book.mark_known(CAM, "workers", "owner", until=T0 + 3600, now=T0)
        again = EventBook(self.dir)
        self.assertEqual(len(again.list_known(T0 + 1)), 1)
        self.assertTrue(again.cancel_known(r["id"]))
        self.assertEqual(again.list_known(T0 + 1), [])

    def test_bad_known_requests(self):
        with self.assertRaises(ValueError):
            self.book.mark_known(CAM, "", "owner", until=T0 + 10, now=T0)
        with self.assertRaises(ValueError):
            self.book.mark_known(CAM, "x", "owner", until=T0 - 10, now=T0)

    def test_workers_day_oct_7(self):
        """The real pergola day: an alert job every ~3 minutes from 09:42 to 17:03, all normal except a few
        suspicious (masks, hoodies) after the owner said at 09:52 these are his workers. Nothing is sent."""
        ts = T0 - 840
        sent = 0
        labels = ["normal"] * 9 + ["suspicious"]
        i = 0
        while ts < T0 + 7 * 3600:
            if abs(ts - (T0 - 240)) < 90:
                self.book.mark_known(CAM, "עובדים בפרגולה", "owner", until=T0 + 8 * 3600, now=ts)
            d = self.book.decide(CAM, ts, labels[i % 10], 3)
            if d.notify:
                sent += 1
                self.sent(d, labels[i % 10], 3, ts)
            for k in range(1, 18):
                self.book.activity(CAM, ts + k * 10, people=3)
            ts += 180
            i += 1
        self.assertEqual(sent, 0)


if __name__ == "__main__":
    unittest.main()
