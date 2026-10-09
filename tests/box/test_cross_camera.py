"""Stage 3.3: one event across cameras (box/events.py ``cross_camera``), and the owner's line for it."""
import shutil
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

from home_guard_project.box import events as ev
from home_guard_project.box import inference as inf
from home_guard_project.box import story
from home_guard_project.box.events import EventBook

GATE, ENTRANCE, YARD = "ameer_week_0_1_ch1", "ameer_week_0_1_ch2", "ameer_week_0_1_ch5"
T0 = 1_791_355_000.0
FIRST = {"chat_id": "-5", "message_id": 41}


def trk(track_id, first, last, start=(0.5, 0.6), end=None, active=True, entry="", exit_=""):
    end = start if end is None else end
    return {"id": track_id, "kind": "person", "first_seen": float(first), "last_seen": float(last), "prev_id": None,
            "first_foot": tuple(start), "last_foot": tuple(end), "moved": 0.3, "active": active, "path": [],
            "entry_edge": entry, "exit_edge": exit_}


class FakeReid:
    """What the clothes say across cameras: a fixed cosine (None: no embedding yet)."""

    def __init__(self, score, mode="on"):
        self.mode = mode
        self.score = score
        self.settings = SimpleNamespace(link=0.70, margin=0.08, veto=0.35)

    def appearance(self, camera):
        return None

    def cross_score(self, cam_a, keys_a, cam_b, keys_b):
        return self.score


class CrossCameraTest(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.dir, ignore_errors=True)
        # The person at the gate: in view T0..T0+20, then out by the right edge (the tracker's latest snapshot).
        self.gate_now = [trk(1, T0, T0 + 20, end=(0.97, 0.6), active=False, exit_="right")]

    def book(self, mode="on", neighbours=None, reid=None):
        book = EventBook(self.dir, cross_camera=mode,
                         neighbours={GATE: [ENTRANCE]} if neighbours is None else neighbours)
        book.configure(tracks_source=lambda cam, t0, t1: self.gate_now if cam == GATE else [], reid=reid)
        return book

    def gate_reported(self, book):
        """The gate's alert went out at T0+10 with P1 in view (message 41)."""
        d = book.decide(GATE, T0 + 10, "suspicious", 1, "a man at the gate", "gate_1",
                        tracks=[trk(1, T0, T0 + 10)], since=T0)
        self.assertTrue(d.notify)
        book.record_sent(d.session_id, "suspicious", 1, T0 + 10, "gate_1", chat_id="-5", message_id=41,
                         entities=d.entities)
        return d

    def entrance(self, book, label="suspicious", start=T0 + 30, entry="left", alert="ent_1"):
        return book.decide(ENTRANCE, start + 10, label, 1, "a man at the entrance", alert,
                           tracks=[trk(7, start, start + 10, start=(0.03, 0.7), entry=entry)], since=start)

    def test_gate_to_entrance_walk_is_one_incident_in_the_gate_thread(self):
        book = self.book()
        gate = self.gate_reported(book)
        d = self.entrance(book)
        self.assertTrue(d.notify)
        self.assertEqual(d.reply_to, FIRST | {"alert_id": "gate_1", "ts": T0 + 10})
        self.assertEqual((d.incident["mode"], d.incident["camera"], d.incident["entity"], d.incident["to_entity"]),
                         ("on", GATE, "P1", "P1"))
        self.assertTrue(d.incident["thread"] and d.incident["announce"])
        self.assertEqual(d.incident["gap_s"], 10.0)
        self.assertIn("continues the incident", d.reason)
        session = book.session_of_alert("ent_1")
        self.assertEqual(session["incident_from"]["session"], gate.session_id)
        self.assertEqual(book.session_of_alert("gate_1")["incident_id"], session["incident_id"])
        self.assertEqual(session["entities"][0]["from_entity"], "P1")

    def test_later_updates_stay_in_the_thread_without_repeating_the_line(self):
        book = self.book()
        self.gate_reported(book)
        d = self.entrance(book)
        book.record_sent(d.session_id, "suspicious", 1, T0 + 40, "ent_1", chat_id="-5", message_id=55,
                         entities=d.entities)
        d2 = book.decide(ENTRANCE, T0 + 50, "suspicious", 2, "two men", "ent_2",
                         tracks=[trk(7, T0 + 30, T0 + 50, start=(0.03, 0.7), entry="left"),
                                 trk(8, T0 + 45, T0 + 50, start=(0.5, 0.2))], since=T0 + 40)
        self.assertTrue(d2.notify)                             # a new person at the entrance: an update
        self.assertEqual(d2.reply_to["message_id"], 55)
        self.assertFalse(d2.incident["announce"])
        self.assertNotIn("thread", d2.incident)

    def test_too_late_at_the_second_camera_is_a_separate_event(self):
        book = self.book()
        self.gate_reported(book)
        d = self.entrance(book, start=T0 + 20 + ev.CROSS_SEC + 15)
        self.assertTrue(d.notify)
        self.assertIsNone(d.reply_to)
        self.assertEqual(d.incident, {})
        self.assertEqual(book.session_of_alert("ent_1")["incident_id"], "")

    def test_someone_still_at_the_gate_or_gone_mid_picture_is_not_the_one(self):
        book = self.book()
        self.gate_reported(book)
        self.gate_now = [trk(1, T0, T0 + 40, active=True)]                       # still at the gate
        self.assertEqual(self.entrance(book).incident, {})
        book2 = EventBook(tempfile.mkdtemp(), cross_camera="on", neighbours={GATE: [ENTRANCE]})
        self.addCleanup(shutil.rmtree, book2.directory, ignore_errors=True)
        self.gate_now = [trk(1, T0, T0 + 20, end=(0.5, 0.5), active=False, exit_="")]   # vanished mid-picture
        book2.configure(tracks_source=lambda cam, t0, t1: self.gate_now if cam == GATE else [])
        self.gate_reported(book2)
        self.assertEqual(self.entrance(book2).incident, {})

    def test_different_clothes_are_a_different_person_with_reid_on(self):
        book = self.book(reid=FakeReid(0.1))
        self.gate_reported(book)
        d = self.entrance(book)
        self.assertEqual(d.incident, {})
        self.assertIsNone(d.reply_to)

    def test_matching_clothes_confirm_and_are_recorded(self):
        book = self.book(reid=FakeReid(0.82))
        self.gate_reported(book)
        self.assertEqual(self.entrance(book).incident["score"], 0.82)

    def test_two_people_left_the_gate_need_the_clothes_to_choose(self):
        self.gate_now = [trk(1, T0, T0 + 20, end=(0.97, 0.6), active=False, exit_="right"),
                         trk(2, T0, T0 + 22, start=(0.3, 0.6), end=(0.97, 0.5), active=False, exit_="right")]
        book = self.book()
        d = book.decide(GATE, T0 + 10, "suspicious", 2, "two men", "gate_1",
                        tracks=[trk(1, T0, T0 + 10), trk(2, T0, T0 + 10, start=(0.3, 0.6))], since=T0)
        book.record_sent(d.session_id, "suspicious", 2, T0 + 10, "gate_1", chat_id="-5", message_id=41,
                         entities=d.entities)
        self.assertEqual(self.entrance(book).incident, {})        # ambiguous: never claimed

    def test_escalation_always_goes_out_as_its_own_message(self):
        book = self.book()
        self.gate_reported(book)
        d = self.entrance(book, label="escalation")
        self.assertTrue(d.notify)
        self.assertEqual(d.reason, "escalation")
        self.assertIsNone(d.reply_to)
        self.assertEqual(d.incident["mode"], "on")
        self.assertNotIn("thread", d.incident)

    def test_a_normal_at_the_second_camera_stays_quiet(self):
        book = self.book()
        self.gate_reported(book)
        self.assertFalse(self.entrance(book, label="normal").notify)

    def test_shadow_only_logs(self):
        book = self.book(mode="shadow")
        self.gate_reported(book)
        with self.assertLogs("box.events", "INFO") as logs:
            d = self.entrance(book)
        self.assertTrue(d.notify)
        self.assertIsNone(d.reply_to)
        self.assertEqual(d.incident["mode"], "shadow")
        self.assertEqual(book.session_of_alert("ent_1")["incident_id"], "")
        self.assertTrue(any("cross-camera: would link P1 at" in line for line in logs.output))

    def test_off_does_nothing(self):
        book = self.book(mode="off")
        self.gate_reported(book)
        d = self.entrance(book)
        self.assertEqual((d.incident, d.reply_to), ({}, None))

    def test_cameras_not_named_neighbours_are_only_a_suggestion(self):
        book = self.book(neighbours={GATE: [YARD]})
        self.gate_reported(book)
        with self.assertLogs("box.events", "INFO") as logs:
            d = self.entrance(book)
        self.assertEqual((d.incident, d.reply_to), ({}, None))
        self.assertEqual(book.cross_suggested, {(GATE, ENTRANCE): 1})
        self.assertTrue(any("cross-camera suggestion" in line for line in logs.output))

    def test_the_first_camera_not_reported_means_a_message_of_its_own(self):
        book = self.book()
        book.decide(GATE, T0 + 10, "normal", 1, "a man at the gate", "gate_1", tracks=[trk(1, T0, T0 + 10)],
                    since=T0)
        d = self.entrance(book)
        self.assertTrue(d.notify)
        self.assertIsNone(d.reply_to)
        self.assertEqual(d.incident["mode"], "on")
        self.assertTrue(d.incident["announce"])

    def test_neighbours_are_both_ways_and_box_yaml_parsing(self):
        self.assertEqual(ev.neighbours_of({GATE: [ENTRANCE], ENTRANCE: "door", "x": 5}),
                         {GATE: {ENTRANCE}, ENTRANCE: {GATE, "door"}, "door": {ENTRANCE}})
        self.assertEqual(ev.cross_mode_of({}), "shadow")
        self.assertEqual(ev.cross_mode_of({"cross_camera": "on"}), "on")
        self.assertEqual(ev.cross_mode_of({"cross_camera": False}), "off")
        self.assertEqual(ev.cross_sec_of({"cross_sec": 30}), 30.0)
        self.assertEqual(ev.cross_sec_of({"cross_sec": "x"}), ev.CROSS_SEC)

    def test_open_track_keys_are_what_reid_may_keep(self):
        book = self.book()
        self.gate_reported(book)
        self.assertEqual(book.open_track_keys(), {GATE: {f"1@{T0:.3f}"}})


class IncidentLineTest(unittest.TestCase):
    def test_hebrew_and_english(self):
        self.assertEqual(story.incident_line("P1", "השער", "הכניסה", "he"), "אותו אדם (P1) עבר מהשער לכניסה")
        self.assertEqual(story.incident_line("P1", "Gate", "Entrance", "en"), "The same person (P1) went from Gate to Entrance")

    def test_inference_uses_display_names_and_only_when_on_and_first(self):
        event = SimpleNamespace(incident={"mode": "on", "announce": True, "camera": GATE, "entity": "P1"})
        names = {GATE: "השער", ENTRANCE: "הכניסה"}
        with mock.patch.object(inf, "camera_display", side_effect=lambda cam, lang: names[cam]):
            self.assertEqual(inf._incident_text(event, ENTRANCE, "he"), "אותו אדם (P1) עבר מהשער לכניסה")
            event.incident["announce"] = False
            self.assertEqual(inf._incident_text(event, ENTRANCE, "he"), "")
            event.incident.update(announce=True, mode="shadow")
            self.assertEqual(inf._incident_text(event, ENTRANCE, "he"), "")
            self.assertEqual(inf._incident_text(SimpleNamespace(), ENTRANCE, "he"), "")


if __name__ == "__main__":
    unittest.main()
