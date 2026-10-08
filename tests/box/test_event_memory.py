"""Stage 2b (2026-10-08): the event memory - the archive of closed events (caption, change from the previous event,
keyframe, 30 days) and its search (keywords with Hebrew words, time words, camera names; embeddings with a fake
embedder). Everything is local: a temp folder, no network."""

import datetime as dt
import json
import os
import shutil
import tempfile
import unittest

from home_guard_project.box import event_memory as em
from home_guard_project.box.events import EventBook

PERGOLA, GATE = "ameer_week_0_1_ch6", "ameer_week_0_1_ch2"
PLACES = {PERGOLA: "פרגולה", GATE: "Camera 2"}
NOON = dt.datetime(2026, 10, 7, 12, 10).timestamp()
MORNING = dt.datetime(2026, 10, 7, 8, 0).timestamp()
NIGHT = dt.datetime(2026, 10, 7, 23, 30).timestamp()
TODAY_LATE = dt.datetime(2026, 10, 8, 18, 0).timestamp()


class MemoryCase(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.dir, ignore_errors=True)
        self.book = EventBook(self.dir)
        self.memory = em.EventMemory(self.dir, place=lambda cam: PLACES.get(cam, ""), clock=lambda: TODAY_LATE)
        em.attach(self.book, self.memory)

    def event(self, camera, t0, looks, known=None, sent=False):
        """Looks: [(seconds after t0, label, people, summary)]. Closes the session after the idle time."""
        first = None
        for i, (dt_s, label, people, summary) in enumerate(looks):
            d = self.book.decide(camera, t0 + dt_s, label, people, summary, alert_id=f"{camera}_{int(t0 + dt_s)}_alert")
            first = first or d
            if sent and i == 0:
                self.book.record_sent(d.session_id, label, people, t0, alert_id="x", chat_id=-5, message_id=7)
        if known:
            self.book.mark_known(camera, known, "Ameer", until=t0 + 8 * 3600, now=t0 + looks[-1][0])
        self.book.tick(t0 + looks[-1][0] + 120)
        return first.session_id


class ArchiveTest(MemoryCase):
    def test_a_closed_session_becomes_a_record_with_a_caption(self):
        sid = self.event(PERGOLA, NOON, [(0, "suspicious", 2, "Two men carry a ladder onto the pergola."),
                                          (60, "normal", 2, "Two men carry a ladder onto the pergola."),
                                          (90, "normal", 3, "Three men fix the pergola roof.")],
                         known="העובדים", sent=True)
        (rec,) = self.memory.records()
        self.assertEqual(rec["event_id"], sid)
        self.assertEqual((rec["camera"], rec["camera_name"], rec["people_max"]), (PERGOLA, "פרגולה", 3))
        self.assertEqual(rec["labels"], ["suspicious", "normal"])
        self.assertTrue(rec["reported"])
        self.assertEqual(rec["owner_known"], ["העובדים"])
        self.assertEqual(len(rec["alert_ids"]), 3)
        self.assertEqual(rec["outcome"], "left")
        self.assertEqual(rec["caption"],
                         "3 people at פרגולה. Seen: Two men carry a ladder onto the pergola; Three men fix the pergola "
                         "roof. Judged suspicious. The owner was told. The owner said: העובדים. They left.")
        self.assertNotRegex(rec["caption"], r"\d{1,2}:\d{2}")             # no timestamps in a caption
        self.assertEqual(rec["change_from_previous"], "The first event at this camera in the memory.")
        self.assertEqual(rec["keyframe"], "")
        with open(os.path.join(self.dir, "events.jsonl"), encoding="utf-8") as f:   # the old file is as it was
            self.assertEqual(json.loads(f.readline())["id"], sid)

    def test_at_most_three_things_they_did_the_last_one_kept(self):
        looks = [(i * 20, "normal", 1, text) for i, text in enumerate(
            ["A man opens the gate.", "A man opens the gate.", "A man waters the plants.", "A man sweeps the path.",
             "A man reads the electricity meter.", "A man walks out to the street."])]
        self.event(GATE, NOON, looks)
        caption = self.memory.records()[0]["caption"]
        self.assertIn("A man opens the gate; A man waters the plants; A man walks out to the street.", caption)
        self.assertEqual(caption.count("opens the gate"), 1)

    def test_change_from_previous_and_to_next(self):
        self.event(PERGOLA, MORNING, [(0, "normal", 2, "Two workers measure the pergola beams.")], known="העובדים")
        self.event(PERGOLA, NOON, [(0, "normal", 2, "Two workers measure the pergola beams again.")],
                   known="העובדים")
        self.event(PERGOLA, NIGHT, [(0, "suspicious", 1, "A man looks into a parked car.")])
        first, second, third = self.memory.records()
        self.assertIn("4.2 h after the previous one", second["change_from_previous"])
        self.assertIn("the same number of people (2)", second["change_from_previous"])
        self.assertIn("the same group the owner named (העובדים)", second["change_from_previous"])
        self.assertIn("similar activity", second["change_from_previous"])
        self.assertIn("fewer people (1, before 2)", third["change_from_previous"])
        self.assertIn("different activity", third["change_from_previous"])
        self.assertIn("now judged suspicious", third["change_from_previous"])
        self.assertEqual(first["change_to_next"], second["change_from_previous"])
        self.assertEqual(third["change_to_next"], "")

    def test_same_count_and_activity_without_the_owners_words_is_likely_the_same_group(self):
        self.event(GATE, MORNING, [(0, "normal", 2, "Two gardeners trim the hedge.")])
        self.event(GATE, NOON, [(0, "normal", 2, "Two gardeners trim the hedge by the gate.")])
        self.event(GATE, NIGHT, [(0, "normal", 4, "Four people talk by the gate.")])
        _, again, more = self.memory.records()
        self.assertIn("likely the same group back", again["change_from_previous"])
        self.assertIn("more people (4, before 2)", more["change_from_previous"])
        self.assertIn("new people", more["change_from_previous"])

    def test_a_rolled_session_continues_and_links_its_parent(self):
        book = EventBook(self.dir, roll_sec=300)
        em.attach(book, self.memory)
        for i in range(12):
            book.decide(PERGOLA, NOON + i * 50, "normal", 2, "Two men work on the pergola.")
        book.tick(NOON + 1000)
        first, second = self.memory.records()
        self.assertEqual(first["outcome"], "continued")
        self.assertIn("goes on in the next event", first["caption"])
        self.assertEqual(second["parent"], first["event_id"])
        self.assertTrue(second["change_from_previous"].startswith("Continues the previous event"))

    def test_the_keyframe_is_the_first_jobs_snapshot(self):
        d = self.book.decide(GATE, NOON, "suspicious", 1, "A man at the gate.", alert_id="a1")
        self.assertTrue(em.save_keyframe(self.dir, d.session_id, b"first"))
        self.assertEqual(em.save_keyframe(self.dir, d.session_id, b"second"), "")      # the first one wins
        self.book.tick(NOON + 300)
        rec = self.memory.get(d.session_id)
        with open(rec["keyframe"], "rb") as f:
            self.assertEqual(f.read(), b"first")

    def test_records_older_than_thirty_days_are_dropped_with_their_keyframes(self):
        old = NOON - 40 * 86400
        d = self.book.decide(GATE, old, "normal", 1, "A man at the gate.")
        em.save_keyframe(self.dir, d.session_id, b"old")
        self.book.tick(old + 300)
        os.utime(em.keyframe_path(self.dir, d.session_id), (old, old))
        self.event(GATE, NOON, [(0, "normal", 1, "A courier leaves a package.")])
        self.memory.prune(TODAY_LATE)          # (the archive itself already pruned the old record)
        self.assertEqual([r["observations"][0]["summary"] for r in self.memory.records()], ["A courier leaves a package."])
        self.assertFalse(os.path.exists(em.keyframe_path(self.dir, d.session_id)))

    def test_a_failing_memory_never_stops_the_book(self):
        self.memory.place = lambda cam: 1 / 0          # the name lookup breaks: the record is still written
        self.event(GATE, NOON, [(0, "normal", 1, "A man at the gate.")])
        self.assertEqual(self.memory.records()[0]["camera_name"], "")
        self.book.archive_hooks.append(lambda s: 1 / 0)
        self.event(GATE, NIGHT, [(0, "normal", 1, "A man at the gate.")])
        self.assertEqual(len(self.memory.records()), 2)

    def test_attach_is_idempotent(self):
        em.attach(self.book, self.memory)
        self.assertEqual(len(self.book.archive_hooks), 1)


class FakeEmbedder:
    """Vectors from a tiny vocabulary: texts sharing these words point the same way."""

    VOCAB = ("car", "white", "dog", "package", "ladder", "worker")

    def __init__(self, broken=False):
        self.calls, self.broken = 0, broken

    def embed(self, texts):
        self.calls += 1
        if self.broken:
            return None
        out = []
        for text in texts:
            low = text.lower()
            out.append([1.0 if w in low else 0.0 for w in self.VOCAB] + [0.05])
        return out


class SearchTest(MemoryCase):
    def setUp(self):
        super().setUp()
        self.event(PERGOLA, MORNING, [(0, "normal", 3, "Three workers carry a ladder onto the pergola.")],
                   known="העובדים")
        self.event(GATE, NOON, [(0, "normal", 1, "A courier leaves a package at the gate.")])
        self.event(GATE, NOON + 1800, [(0, "suspicious", 1, "A white pickup truck parks by the gate.")])
        self.event(PERGOLA, NIGHT, [(0, "normal", 1, "A dog walks across the yard.")])
        self.event(PERGOLA, TODAY_LATE - 3600, [(0, "normal", 2, "Two workers fix the pergola roof.")])

    def summaries(self, hits):
        return [h["observations"][0]["summary"] for h in hits]

    def test_hebrew_words_find_english_captions(self):
        self.assertEqual(self.summaries(self.memory.search("היה פה טנדר לבן?"))[0],
                         "A white pickup truck parks by the gate.")
        self.assertEqual(self.summaries(self.memory.search("הגיע שליח?"))[0], "A courier leaves a package at the gate.")
        self.assertEqual(self.summaries(self.memory.search("ראית כלב?"))[0], "A dog walks across the yard.")

    def test_day_and_time_words_filter(self):
        yesterday = self.summaries(self.memory.search("were the workers here yesterday?"))
        self.assertEqual(yesterday, ["Three workers carry a ladder onto the pergola."])
        today = self.summaries(self.memory.search("העובדים היו היום?"))
        self.assertEqual(today, ["Two workers fix the pergola roof."])
        at_noon = self.summaries(self.memory.search("מה קרה בצהריים?"))
        self.assertEqual(at_noon, ["A white pickup truck parks by the gate.", "A courier leaves a package at the gate."])
        at_night = self.summaries(self.memory.search("what happened at night"))
        self.assertEqual(at_night, ["A dog walks across the yard."])

    def test_the_cameras_name_picks_its_events(self):
        hits = self.memory.search("מה קרה בפרגולה בבוקר?")
        self.assertEqual(self.summaries(hits), ["Three workers carry a ladder onto the pergola."])
        self.assertEqual(hits[0]["match"], "time and place")
        names = {GATE: ["שער", "Camera 2"]}
        gate = self.memory.search("מה היה בשער?", names=names)
        self.assertEqual({h["camera"] for h in gate}, {GATE})

    def test_since_until_camera_and_k(self):
        hits = self.memory.search("workers", camera=PERGOLA, since=TODAY_LATE - 7200, until=TODAY_LATE)
        self.assertEqual(self.summaries(hits), ["Two workers fix the pergola roof."])
        self.assertEqual(len(self.memory.search("people", k=2)), 2)          # every caption counts people
        self.assertEqual(self.memory.search("bicycle"), [])
        self.assertLessEqual(len(self.memory.search("what happened", k=2)), 2)
        self.assertEqual(self.memory.search("a unicorn"), [])

    def test_embeddings_when_they_work(self):
        embedder = FakeEmbedder()
        hits = self.memory.search("did the white car leave?", embedder=embedder)
        self.assertEqual(hits[0]["match"], "embedding")
        self.assertEqual(self.summaries(hits)[0], "A white pickup truck parks by the gate.")
        self.assertTrue(all(h["score"] >= 0.3 for h in hits))

    def test_no_embeddings_falls_back_to_keywords_and_stops_asking(self):
        embedder = FakeEmbedder(broken=True)
        hits = self.memory.search("package", embedder=embedder)
        self.assertEqual(hits[0]["match"], "keywords")
        self.memory.search("ladder", embedder=embedder)
        self.assertEqual(embedder.calls, 1)                                   # off for an hour after a failure

    def test_live_records_are_searched_too(self):
        d = self.book.decide(GATE, TODAY_LATE - 60, "normal", 1, "A neighbour waves at the gate.")
        live = [self.memory.live_record(s) for s in self.book.recent(TODAY_LATE - 600) if not s.get("closed")]
        hits = self.memory.search("neighbour", extra=live)
        self.assertEqual(hits[0]["event_id"], d.session_id)
        self.assertEqual(hits[0]["outcome"], "open")


class WordsTest(unittest.TestCase):
    def test_hebrew_prefixes_and_map(self):
        (group,) = em.query_groups("והרכב")
        self.assertIn("car", group)
        self.assertEqual(em.query_groups("מה קרה היום בצהריים?"), [])
        self.assertEqual(em.time_filter("מה קרה אתמול בערב?"), ((17, 22), 1))
        self.assertEqual(em.time_filter("אחה\"צ"), ((13, 18), None))
        self.assertIn("pickup", em.english_of("טנדר לבן"))


if __name__ == "__main__":
    unittest.main()
