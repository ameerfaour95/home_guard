"""Stage 2a: the event's story for the owner's updates (box/story.py), from the event book's entities."""
import shutil
import tempfile
import unittest

from home_guard_project.box import story
from home_guard_project.box.events import EventBook

CAM = "ameer_week_0_1_ch3"
T0 = 1_791_355_000.0


def trk(track_id, first, last, start=(0.5, 0.8), active=True, path=(), prev=None):
    return {"id": track_id, "kind": "person", "first_seen": float(first), "last_seen": float(last), "prev_id": prev,
            "first_foot": tuple(start), "last_foot": tuple(start), "moved": 0.2, "active": active, "path": list(path),
            "entry_edge": "", "exit_edge": ""}


class StoryCase(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.book = EventBook(self.dir)

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def step(self, ts, tracks, per_entity=None, note="", label="suspicious", send=True):
        d = self.book.decide(CAM, ts, label, len(tracks), "summary", f"a{int(ts)}", tracks=tracks, since=ts - 10,
                             note=note, per_entity=per_entity)
        session = self.book.session_of_alert(f"a{int(ts)}")
        lines = {lang: story.story_line(session, lang, now=ts, in_view=d.entities, announced=d.fresh)
                 for lang in ("he", "en")}
        if send:
            self.book.record_sent(d.session_id, label, len(tracks), ts, alert_id=f"a{int(ts)}", chat_id=-5,
                                  message_id=int(ts) % 1000, entities=d.entities)
        return d, lines


class StoryTest(StoryCase):
    def test_no_entities_no_story(self):
        self.assertEqual(story.story_line({"entities": [], "observations": []}, "he"), "")
        d = self.book.decide(CAM, T0, "suspicious", 2, alert_id="a1")       # no tracker data: no entities
        self.assertEqual(d.counted_by, "head-count")
        self.assertEqual(story.story_line(self.book.session_of_alert("a1"), "en"), "")

    def test_both_moved_toward_the_same_new_area(self):
        self.step(T0, [trk(1, T0 - 2, T0, path=("כניסה",)), trk(2, T0 - 2, T0, start=(0.2, 0.8), path=("כניסה",))])
        _, lines = self.step(T0 + 60, [trk(1, T0 - 2, T0 + 60, path=("כניסה", "פרגולה")),
                                       trk(2, T0 - 2, T0 + 60, start=(0.2, 0.8), path=("כניסה", "פרגולה"))])
        self.assertEqual(lines["he"], "שניהם (P1, P2) עברו לכיוון פרגולה.")
        self.assertEqual(lines["en"], "Both (P1, P2) moved toward פרגולה.")

    def test_one_moved_one_stayed_is_not_both(self):
        self.step(T0, [trk(1, T0 - 2, T0, path=("gate",)), trk(2, T0 - 2, T0, start=(0.2, 0.8), path=("gate",))])
        _, lines = self.step(T0 + 60, [trk(1, T0 - 2, T0 + 60, path=("gate", "pergola")),
                                       trk(2, T0 - 2, T0 + 60, start=(0.2, 0.8), path=("gate",))])
        self.assertNotIn("Both", lines["en"])
        self.assertTrue(lines["en"].startswith("P1 moved toward pergola."), lines["en"])
        self.assertIn("P2", lines["en"])

    def test_without_a_map_no_movement_is_claimed(self):
        self.step(T0, [trk(1, T0 - 2, T0), trk(2, T0 - 2, T0, start=(0.2, 0.8))])
        _, lines = self.step(T0 + 60, [trk(1, T0 - 2, T0 + 60), trk(2, T0 - 2, T0 + 60, start=(0.2, 0.8))])
        self.assertEqual(lines["en"], "In view now: P1 and P2.")
        self.assertEqual(lines["he"], "בתמונה עכשיו: P1 ו-P2.")

    def test_the_one_who_was_cleaning(self):
        two = [trk(1, T0 - 2, T0), trk(2, T0 - 2, T0, start=(0.2, 0.8))]
        self.step(T0, two, per_entity=[{"id": "P1", "action": "הולך ליד הכניסה"}, {"id": "P2", "action": "מנקה את הרצפה"}])
        _, lines = self.step(T0 + 60, [trk(1, T0 - 2, T0 + 60), trk(2, T0 - 2, T0 + 60, start=(0.2, 0.8))],
                             per_entity=[{"id": "P2", "action": "שם את המטאטא בטנדר"}])
        self.assertIn("P2 (קודם: מנקה את הרצפה): שם את המטאטא בטנדר.", lines["he"])
        self.assertIn("P1 (earlier: הולך ליד הכניסה) is still in view.", lines["en"])

    def test_an_ambiguous_return_is_never_the_one_who(self):
        two = [trk(1, T0 - 2, T0, start=(0.5, 0.8)), trk(2, T0 - 2, T0, start=(0.52, 0.8))]
        self.step(T0, two, per_entity=[{"id": "P1", "action": "cleans the floor"}, {"id": "P2", "action": "sweeps"}])
        gone = [dict(t, last_seen=T0 + 5, active=False) for t in two]
        d, lines = self.step(T0 + 40, gone + [trk(3, T0 + 30, T0 + 40, start=(0.51, 0.8))],
                             per_entity=[{"id": "P3", "action": "carries a broom"}])
        self.assertEqual(d.entities, ["P3"])
        self.assertIn("P3 (maybe P1 or P2 back, not sure): carries a broom.", lines["en"])
        self.assertIn("אולי P1 או P2 שחזר, לא בטוח", lines["he"])
        self.assertNotIn("earlier", lines["en"])
        self.assertNotIn("קודם", lines["he"])

    def test_who_left_since_the_last_message(self):
        self.step(T0, [trk(1, T0 - 2, T0), trk(2, T0 - 2, T0, start=(0.2, 0.8))])
        _, lines = self.step(T0 + 60, [trk(1, T0 - 2, T0 + 30, active=False), trk(2, T0 - 2, T0 + 60, start=(0.2, 0.8))])
        self.assertEqual(lines["en"], "P2 is still in view. P1 left the view.")
        self.assertEqual(lines["he"], "P2 עדיין בתמונה. P1 יצא מהתמונה.")
        _, again = self.step(T0 + 120, [trk(1, T0 - 2, T0 + 30, active=False), trk(2, T0 - 2, T0 + 120, start=(0.2, 0.8))])
        self.assertNotIn("left", again["en"])                         # said once

    def test_new_people_line(self):
        self.assertEqual(story.new_people_line(["P4"], True, "עובדים בפרגולה", "he"),
                         "אדם חדש הגיע (P4), לא מאלה שסימנת (עובדים בפרגולה)")
        self.assertEqual(story.new_people_line(["P4"], True, "workers", "en"),
                         "A new person arrived (P4), not one of those you marked (workers)")
        self.assertEqual(story.new_people_line(["P3", "P4"], False, "", "he"), "עוד 2 אנשים הגיעו (P3 ו-P4)")
        self.assertEqual(story.new_people_line(["P3"], False, "", "en"), "1 more person arrived (P3)")
        self.assertEqual(story.new_people_line([], False, "", "en"), "")

    def test_a_new_one_already_announced_is_not_repeated(self):
        self.step(T0, [trk(1, T0 - 2, T0)])
        d, lines = self.step(T0 + 60, [trk(1, T0 - 2, T0 + 60), trk(2, T0 + 50, T0 + 60, start=(0.1, 0.2))])
        self.assertEqual(d.fresh, ["P2"])
        self.assertNotIn("P2 is new", lines["en"])
        self.assertEqual(lines["en"], "In view now: P1 and P2.")


if __name__ == "__main__":
    unittest.main()
