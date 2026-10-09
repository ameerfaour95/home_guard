"""2026-10-09 (the owner's morning chat): a corrected mark replaces the old one, and the workers marked at one
camera cover the camera they walk to only through a cross-camera link of that session (on, or seen in shadow)."""
import shutil
import tempfile
import unittest

from home_guard_project.box.events import EventBook

from test_cross_camera import T0, trk

PERGOLA, ENTRANCE, YARD = "ameer_week_0_1_ch3", "ameer_week_0_1_ch6", "ameer_week_0_1_ch5"


class ReplaceKnownTest(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.dir, ignore_errors=True)

    def test_a_correction_replaces_the_old_mark(self):
        book = EventBook(self.dir)
        old = book.mark_known(PERGOLA, "העובדים", "Ameer", until=T0 + 14 * 3600, now=T0)
        new = book.replace_known(old["id"], PERGOLA, "העובדים", "Ameer", until=T0 + 8 * 3600, now=T0 + 60)
        live = book.list_known(T0 + 120)
        self.assertEqual([k["id"] for k in live], [new["id"]])
        self.assertEqual(live[0]["until"], T0 + 8 * 3600)
        self.assertEqual([r["id"] for r in new["replaced"]], [old["id"]])
        self.assertEqual(new["replaced"][0]["until"], T0 + 14 * 3600)
        # Saved to disk: a restart sees one mark.
        self.assertEqual([k["id"] for k in EventBook(self.dir).list_known(T0 + 120)], [new["id"]])

    def test_a_house_wide_correction_replaces_several_camera_marks(self):
        book = EventBook(self.dir)
        a = book.mark_known(PERGOLA, "העובדים", "Ameer", until=T0 + 3600, now=T0)
        b = book.mark_known(ENTRANCE, "העובדים", "Ameer", until=T0 + 3600, now=T0)
        new = book.replace_known([a["id"], b["id"]], "", "העובדים", "Ameer", until=T0 + 7200, now=T0 + 60)
        self.assertEqual([(k["id"], k["camera"]) for k in book.list_known(T0 + 120)], [(new["id"], "")])
        self.assertFalse(book.decide(YARD, T0 + 200, "suspicious", 2, "workers").notify)

    def test_a_bad_request_leaves_the_old_mark(self):
        book = EventBook(self.dir)
        old = book.mark_known(PERGOLA, "העובדים", "Ameer", until=T0 + 3600, now=T0)
        with self.assertRaises(ValueError):
            book.replace_known(old["id"], PERGOLA, "העובדים", "Ameer", until=T0 - 1, now=T0 + 60)
        self.assertEqual([k["id"] for k in book.list_known(T0 + 120)], [old["id"]])


class KnownThroughLinkTest(unittest.TestCase):
    """The workers at the pergola (marked) walk to the main entrance: P1 leaves the pergola's picture by the right
    edge at T0+20 and P7 starts at the entrance's left edge at T0+30."""

    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.dir, ignore_errors=True)
        self.pergola_now = [trk(1, T0, T0 + 20, end=(0.97, 0.6), active=False, exit_="right")]

    def book(self, mode, neighbours=None):
        book = EventBook(self.dir, cross_camera=mode,
                         neighbours={PERGOLA: [ENTRANCE]} if neighbours is None else neighbours)
        book.configure(tracks_source=lambda cam, t0, t1: self.pergola_now if cam == PERGOLA else [])
        book.decide(PERGOLA, T0 + 10, "suspicious", 1, "a worker", "perg_1", tracks=[trk(1, T0, T0 + 10)], since=T0)
        book.mark_known(PERGOLA, "העובדים", "Ameer", until=T0 + 8 * 3600, now=T0 + 12)
        return book

    def entrance(self, book, label="suspicious", start=T0 + 30):
        return book.decide(ENTRANCE, start + 10, label, 1, "a man at the entrance", "ent_1",
                           tracks=[trk(7, start, start + 10, start=(0.03, 0.7), entry="left")], since=start)

    def test_shadow_link_carries_the_pergola_mark_to_the_entrance(self):
        book = self.book("shadow")
        d = self.entrance(book)
        self.assertFalse(d.notify)
        self.assertEqual(d.known_text, "העובדים")
        seen = book.session_of_alert("ent_1")["cross_seen"]
        self.assertEqual((seen["camera"], seen["mode"]), (PERGOLA, "shadow"))
        self.assertEqual(book.session_of_alert("ent_1")["incident_id"], "")     # shadow still links nothing

    def test_on_link_carries_it_too(self):
        d = self.entrance(self.book("on"))
        self.assertFalse(d.notify)
        self.assertEqual(d.known_text, "העובדים")

    def test_never_an_escalation(self):
        self.assertTrue(self.entrance(self.book("shadow"), label="escalation").notify)

    def test_no_link_no_cover(self):
        # Not neighbours (only a suggestion), cross-camera off, or too late to be the same person: it alerts.
        self.assertTrue(self.entrance(self.book("shadow", neighbours={PERGOLA: [YARD]})).notify)

    def test_off_no_cover(self):
        self.assertTrue(self.entrance(self.book("off")).notify)

    def test_too_late_no_cover(self):
        self.assertTrue(self.entrance(self.book("shadow"), start=T0 + 200).notify)

    def test_only_within_the_marks_time(self):
        book = self.book("shadow")
        (mark,) = book.list_known(T0 + 20)
        book.cancel_known(mark["id"])
        book.mark_known(PERGOLA, "העובדים", "Ameer", until=T0 + 35, now=T0 + 15)    # over before the alert
        self.assertTrue(self.entrance(book).notify)


if __name__ == "__main__":
    unittest.main()
