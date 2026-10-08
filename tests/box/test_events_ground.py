"""events.decide with the scene map's ground (stage 2c): additive, the decision stays in code."""
import tempfile
import unittest

from home_guard_project.box.events import EventBook

CAM = "ameer_week_0_1_ch1"
T0 = 1_791_355_000.0
OFF = {"on": "neighbour", "entered": False, "off_our_ground": True}
ENTERED = {"on": "mine", "entered": True, "off_our_ground": False, "crossed_inward": True, "line": "railing"}
OURS = {"on": "mine", "entered": False, "off_our_ground": False}


class GroundDecisionTest(unittest.TestCase):
    def setUp(self):
        self.book = EventBook(tempfile.mkdtemp())

    def test_without_a_ground_nothing_changes(self):
        self.assertTrue(self.book.decide(CAM, T0, "suspicious", 1).notify)
        self.assertTrue(EventBook(tempfile.mkdtemp()).decide(CAM, T0, "suspicious", 1, ground=None).notify)
        self.assertTrue(EventBook(tempfile.mkdtemp()).decide(CAM, T0, "suspicious", 1, ground=OURS).notify)

    def test_a_suspicious_on_the_neighbours_ground_for_no_action_is_not_sent(self):
        d = self.book.decide(CAM, T0, "suspicious", 1, "a man walks at night", ground=OFF)
        self.assertFalse(d.notify)
        self.assertIn("not ours", d.reason)

    def test_an_action_on_the_neighbours_ground_is_sent(self):
        d = self.book.decide(CAM, T0, "suspicious", 1, "tries the car door", ground=dict(OFF, action=True))
        self.assertTrue(d.notify)

    def test_an_escalation_is_sent_wherever_it_is(self):
        self.assertTrue(self.book.decide(CAM, T0, "escalation", 1, ground=OFF).notify)

    def test_normal_off_our_ground_stays_quiet_even_with_notify_normal(self):
        book = EventBook(tempfile.mkdtemp(), notify_normal=True)
        self.assertFalse(book.decide(CAM, T0, "normal", 1, ground=OFF).notify)

    def test_coming_onto_our_ground_is_sent_even_when_the_eye_said_normal(self):
        d = self.book.decide(CAM, T0, "normal", 1, "a man walks", ground=ENTERED)
        self.assertTrue(d.notify)
        self.assertIn("came onto our ground", d.reason)
        self.book.record_sent(d.session_id, "suspicious", 1, T0, alert_id="a1", chat_id=-1, message_id=5)
        again = self.book.decide(CAM, T0 + 20, "normal", 1, ground=ENTERED)
        self.assertFalse(again.notify)                       # once per event, like any suspicious

    def test_the_owners_known_people_cover_a_crossing_too(self):
        self.book.mark_known(CAM, "the workers", "owner", until=T0 + 3600, now=T0 - 10, people=3)
        self.assertFalse(self.book.decide(CAM, T0, "normal", 2, ground=ENTERED).notify)


if __name__ == "__main__":
    unittest.main()
