"""Stage 2a of the alert fix: the event's entities (P1, P2, CAR1) from the tracker's tracks (box/entities.py) and the
event book's decisions with them (box/events.py)."""
import json
import shutil
import tempfile
import unittest

from home_guard_project.box import entities as ent
from home_guard_project.box.events import EventBook

CAM = "ameer_week_0_1_ch3"
T0 = 1_791_355_000.0


def trk(track_id, first, last, start=(0.5, 0.8), end=None, kind="person", active=True, prev=None, path=(),
        moved=0.2):
    end = start if end is None else end
    return {"id": track_id, "kind": kind, "first_seen": float(first), "last_seen": float(last), "prev_id": prev,
            "first_foot": tuple(start), "last_foot": tuple(end), "moved": moved, "active": active, "path": list(path),
            "entry_edge": "", "exit_edge": ""}


class IngestTest(unittest.TestCase):
    def test_two_people_get_p1_and_p2(self):
        rows = []
        seen = ent.ingest(rows, [trk(1, T0, T0 + 5), trk(2, T0 + 1, T0 + 5, start=(0.1, 0.9))], T0 + 5)
        self.assertEqual(seen, ["P1", "P2"])
        self.assertEqual([e["state"] for e in rows], ["active", "active"])

    def test_the_same_track_again_is_the_same_entity(self):
        rows = []
        ent.ingest(rows, [trk(1, T0, T0 + 5)], T0 + 5)
        ent.ingest(rows, [trk(1, T0, T0 + 40, end=(0.6, 0.7))], T0 + 40)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["last_seen"], T0 + 40)
        self.assertEqual(rows[0]["last_foot"], [0.6, 0.7])

    def test_came_back_keeps_the_entity(self):
        rows = []
        ent.ingest(rows, [trk(1, T0, T0 + 5, start=(0.5, 0.8))], T0 + 5)
        ent.ingest(rows, [trk(1, T0, T0 + 5, active=False), trk(2, T0 + 300, T0 + 310, start=(0.9, 0.9), prev=1)],
                   T0 + 310)
        self.assertEqual([e["id"] for e in rows], ["P1"])          # the tracker linked it: far and late is fine
        self.assertEqual(rows[0]["track_ids"], [1, 2])

    def test_a_lone_candidate_nearby_reattaches(self):
        rows = []
        tracks = [trk(1, T0, T0 + 5, start=(0.5, 0.8), active=False), trk(2, T0 + 60, T0 + 70, start=(0.55, 0.82))]
        self.assertEqual(ent.ingest(rows, tracks, T0 + 70), ["P1"])
        self.assertEqual(rows[0]["state"], "active")

    def test_too_far_or_too_late_is_a_new_person(self):
        rows = []
        tracks = [trk(1, T0, T0 + 5, active=False), trk(2, T0 + 30, T0 + 40, start=(0.1, 0.2), active=False),
                  trk(3, T0 + 5 + ent.REATTACH_SEC + 60, T0 + 5 + ent.REATTACH_SEC + 70, start=(0.5, 0.8))]
        ent.ingest(rows, tracks, T0 + 300)
        self.assertEqual([e["id"] for e in rows], ["P1", "P2", "P3"])
        self.assertFalse(any(e["maybe_of"] for e in rows))

    def test_two_candidates_make_a_new_entity_that_may_be_either(self):
        rows = []
        tracks = [trk(1, T0, T0 + 5, start=(0.5, 0.8), active=False),
                  trk(2, T0, T0 + 6, start=(0.52, 0.8), active=False),
                  trk(3, T0 + 20, T0 + 30, start=(0.51, 0.81))]
        self.assertEqual(ent.ingest(rows, tracks, T0 + 30, since=T0 + 15), ["P3"])
        self.assertEqual(rows[2]["maybe_of"], ["P1", "P2"])
        self.assertEqual([e["track_ids"] for e in rows], [[1], [2], [3]])     # never merged

    def test_someone_still_in_view_is_never_a_candidate(self):
        rows = []
        tracks = [trk(1, T0, T0 + 30, start=(0.5, 0.8)), trk(2, T0 + 10, T0 + 30, start=(0.52, 0.8))]
        ent.ingest(rows, tracks, T0 + 30)
        self.assertEqual([e["id"] for e in rows], ["P1", "P2"])

    def test_parked_vehicles_are_never_entities(self):
        rows = []
        seen = ent.ingest(rows, [trk(5, T0, T0 + 60, kind="vehicle", moved=0.01),
                                 trk(6, T0, T0 + 20, kind="vehicle", moved=0.3), trk(7, T0, T0 + 20)], T0 + 20)
        self.assertEqual(seen, ["P1", "CAR1"])
        self.assertEqual([e["kind"] for e in rows], ["vehicle", "person"])

    def test_states_age_from_active_to_lost_to_gone(self):
        rows = []
        ent.ingest(rows, [trk(1, T0, T0 + 5, active=False)], T0 + 10)
        self.assertEqual(rows[0]["state"], "lost")
        ent.ingest(rows, [], T0 + 5 + ent.REATTACH_SEC + 1)
        self.assertEqual(rows[0]["state"], "gone")

    def test_path_is_areas_with_a_map_else_edges(self):
        rows = []
        mapped = trk(1, T0, T0 + 5, path=("gate", "patio"))
        edges = dict(trk(2, T0, T0 + 5, start=(0.1, 0.1)), entry_edge="left", exit_edge="")
        ent.ingest(rows, [mapped, edges], T0 + 5)
        self.assertEqual((rows[0]["path"], rows[0]["mapped"]), (["gate", "patio"], True))
        self.assertEqual((rows[1]["path"], rows[1]["mapped"]), (["left"], False))

    def test_entities_are_json(self):
        rows = []
        ent.ingest(rows, [trk(1, T0, T0 + 5, path=("gate",))], T0 + 5)
        self.assertEqual(json.loads(json.dumps(rows)), json.loads(json.dumps(rows, ensure_ascii=False)))

    def test_per_entity_actions_and_a_lone_person_note(self):
        rows = []
        view = ent.ingest(rows, [trk(1, T0, T0 + 5), trk(2, T0, T0 + 5, start=(0.1, 0.9))], T0 + 5)
        got = ent.attribute(rows, view, T0 + 5, "two men", "normal",
                            [{"id": "p2", "action": "cleans the floor"}, {"id": "P9", "action": "x"}])
        self.assertEqual(got, ["P2"])
        self.assertEqual(rows[1]["notes"][-1]["text"], "cleans the floor")
        self.assertEqual(ent.attribute(rows, view, T0 + 6, "two men", "normal"), [])     # two in view: no guess
        self.assertEqual(ent.attribute(rows, ["P1"], T0 + 7, "a man walks", "normal"), ["P1"])

    def test_roster_line(self):
        rows = []
        ent.ingest(rows, [trk(1, T0, T0 + 200, path=("gate", "patio"))], T0 + 200)
        view = ent.ingest(rows, [trk(1, T0, T0 + 200, path=("gate", "patio")), trk(2, T0 + 195, T0 + 200,
                                                                                    start=(0.1, 0.9))],
                          T0 + 200, since=T0 + 190)
        self.assertEqual(ent.roster_line(rows, view, T0 + 200, since=T0 + 190),
                         "PEOPLE/VEHICLES IN VIEW (from the tracker): P1 in view 3 min, gate>patio; P2 new")
        self.assertEqual(ent.roster_line(rows, [], T0), "")


class BookCase(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.book = EventBook(self.dir)

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def decide(self, ts, label, tracks, people=None, since=None, **kw):
        return self.book.decide(CAM, ts, label, len(tracks) if people is None else people, "summary",
                                f"a{int(ts)}", tracks=tracks, since=ts - 10 if since is None else since, **kw)

    def sent(self, d, label, ts, mid=1):
        self.book.record_sent(d.session_id, label, len(d.entities), ts, alert_id=f"a{int(ts)}", chat_id=-5,
                              message_id=mid, entities=d.entities)

    def keep_alive(self, start, end):
        for t in range(int(start), int(end), 20):
            self.book.activity(CAM, t, people=1)


class SessionEntitiesTest(BookCase):
    def test_ids_restart_in_a_new_session(self):
        d = self.decide(T0, "normal", [trk(1, T0 - 2, T0)])
        self.assertEqual(d.entities, ["P1"])
        d2 = self.decide(T0 + 600, "normal", [trk(9, T0 + 598, T0 + 600, start=(0.1, 0.1))])
        self.assertNotEqual(d2.session_id, d.session_id)
        self.assertEqual(d2.entities, ["P1"])
        self.assertEqual([e["track_ids"] for e in self.book.session_of_alert(f"a{int(T0 + 600)}")["entities"]], [[9]])

    def test_a_rolled_over_session_keeps_its_entities_and_ids(self):
        book = self.book = EventBook(self.dir, roll_sec=100)
        a = trk(1, T0 - 2, T0)
        d = self.decide(T0, "suspicious", [a])
        self.sent(d, "suspicious", T0)
        self.keep_alive(T0, T0 + 150)
        later = [trk(1, T0 - 2, T0 + 150), trk(2, T0 + 148, T0 + 150, start=(0.1, 0.9))]
        d2 = self.decide(T0 + 150, "suspicious", later)
        session = book.session_of_alert(f"a{int(T0 + 150)}")
        self.assertEqual(session["parent"], d.session_id)
        self.assertEqual(d2.entities, ["P1", "P2"])
        self.assertEqual(d2.fresh, ["P2"])
        self.assertEqual(session["reported_entities"], ["P1"])

    def test_closed_sessions_archive_their_entities(self):
        self.decide(T0, "normal", [trk(1, T0 - 2, T0, path=("gate",))])
        self.book.close_all(T0 + 500)
        with open(self.book.events_path, encoding="utf-8") as f:
            row = json.loads(f.readline())
        self.assertEqual(row["entities"][0]["id"], "P1")
        self.assertEqual(row["observations"][0]["entities"], ["P1"])

    def test_roster_does_not_change_the_book(self):
        r = self.book.roster(CAM, T0, [trk(1, T0 - 2, T0)], since=T0 - 10)
        self.assertEqual(r["in_view"], ["P1"])
        self.assertIn("P1 new", r["line"])
        self.assertEqual(self.book.recent(0), [])


class DecideWithEntitiesTest(BookCase):
    def workers(self, ts, extra=()):
        tracks = [trk(1, T0 - 2, ts, start=(0.2, 0.8)), trk(2, T0 - 2, ts, start=(0.5, 0.8)),
                  trk(3, T0 - 1, ts, start=(0.8, 0.8))]
        return tracks + list(extra)

    def test_without_tracker_data_the_head_count_decides_as_before(self):
        d = self.decide(T0, "suspicious", [], people=2)
        self.assertEqual(d.counted_by, "head-count")
        self.sent(d, "suspicious", T0)
        self.assertTrue(self.decide(T0 + 40, "suspicious", [], people=3).notify)

    def test_the_eyes_noisy_head_count_does_not_make_new_people(self):
        d = self.decide(T0, "suspicious", self.workers(T0))
        self.assertTrue(d.notify)
        self.sent(d, "suspicious", T0)
        again = self.decide(T0 + 40, "suspicious", self.workers(T0 + 40), people=6)
        self.assertFalse(again.notify, again.reason)
        self.assertEqual(again.counted_by, "entities")

    def test_someone_else_arriving_after_the_first_left_is_new(self):
        d = self.decide(T0, "suspicious", [trk(1, T0 - 2, T0, start=(0.2, 0.8))])
        self.sent(d, "suspicious", T0)
        tracks = [trk(1, T0 - 2, T0 + 5, start=(0.2, 0.8), active=False), trk(2, T0 + 30, T0 + 40, start=(0.9, 0.3))]
        d2 = self.decide(T0 + 40, "suspicious", tracks, people=1)
        self.assertTrue(d2.notify, d2.reason)
        self.assertEqual((d2.fresh, d2.new_people), (["P2"], 1))
        self.assertEqual(d2.reply_to["message_id"], 1)

    def test_workers_marked_then_a_fourth_person_arrives(self):
        d = self.decide(T0, "suspicious", self.workers(T0))
        self.sent(d, "suspicious", T0)
        receipt = self.book.mark_known(CAM, "עובדים בפרגולה", "owner", until=T0 + 8 * 3600, now=T0 + 20)
        session = self.book.session_of_alert(f"a{int(T0)}")
        self.assertEqual([e["owner_label"] for e in session["entities"]], ["עובדים בפרגולה"] * 3)
        quiet = self.decide(T0 + 60, "suspicious", self.workers(T0 + 60))
        self.assertFalse(quiet.notify)
        self.assertEqual(quiet.known_text, "עובדים בפרגולה")
        stranger = trk(4, T0 + 100, T0 + 120, start=(0.05, 0.3))
        normal = self.decide(T0 + 120, "normal", self.workers(T0 + 120, [stranger]))
        self.assertFalse(normal.notify)                                   # normal never sends
        sus = self.decide(T0 + 130, "suspicious", self.workers(T0 + 130, [dict(stranger, last_seen=T0 + 130)]),
                          people=4)
        self.assertTrue(sus.notify, sus.reason)
        self.assertTrue(sus.unmarked)
        self.assertEqual(sus.fresh, ["P4"])
        self.assertEqual(sus.known_text, receipt["text"])
        self.sent(sus, "suspicious", T0 + 130, mid=2)
        after = self.decide(T0 + 170, "suspicious", self.workers(T0 + 170, [dict(stranger, last_seen=T0 + 170)]))
        self.assertFalse(after.notify, after.reason)                      # told once, not again
        self.assertTrue(self.decide(T0 + 180, "escalation", self.workers(T0 + 180)).notify)

    def test_a_worker_back_from_behind_the_pergola_is_still_covered(self):
        d = self.decide(T0, "suspicious", [trk(1, T0 - 2, T0, start=(0.5, 0.8))])
        self.sent(d, "suspicious", T0)
        self.book.mark_known(CAM, "workers", "owner", until=T0 + 3600, now=T0 + 5)
        tracks = [trk(1, T0 - 2, T0 + 10, start=(0.5, 0.8), active=False), trk(2, T0 + 50, T0 + 60, start=(0.52, 0.8))]
        back = self.decide(T0 + 60, "suspicious", tracks)
        self.assertEqual(back.entities, ["P1"])
        self.assertFalse(back.notify, back.reason)

    def test_an_ambiguous_return_among_told_people_is_not_new(self):
        d = self.decide(T0, "suspicious", [trk(1, T0 - 2, T0, start=(0.5, 0.8)), trk(2, T0 - 2, T0, start=(0.52, 0.8))])
        self.sent(d, "suspicious", T0)
        tracks = [trk(1, T0 - 2, T0 + 5, start=(0.5, 0.8), active=False),
                  trk(2, T0 - 2, T0 + 5, start=(0.52, 0.8), active=False), trk(3, T0 + 30, T0 + 40, start=(0.51, 0.8))]
        d2 = self.decide(T0 + 40, "suspicious", tracks)
        self.assertEqual(d2.entities, ["P3"])
        self.assertEqual(d2.fresh, [])
        self.assertFalse(d2.notify, d2.reason)

    def test_known_said_in_another_session_counts_the_trackers_people(self):
        self.book.mark_known(CAM, "workers", "owner", until=T0 + 8 * 3600, now=T0 - 3600, people=2)
        self.assertFalse(self.decide(T0, "suspicious", self.workers(T0), people=9).notify)   # 3 <= 2 + 2
        five = self.workers(T0 + 30, [trk(4, T0 + 20, T0 + 30, start=(0.1, 0.1)), trk(5, T0 + 20, T0 + 30,
                                                                                       start=(0.9, 0.1))])
        self.assertTrue(self.decide(T0 + 30, "suspicious", five, people=1).notify)


if __name__ == "__main__":
    unittest.main()
