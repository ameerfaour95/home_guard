"""Stage 3.2: the same-day appearance memory (box/reid.py) and what the clothes do to the entities (box/entities.py)."""
import hashlib
import io
import os
import shutil
import tempfile
import unittest
from unittest import mock

import numpy as np

from home_guard_project.box import entities as ent
from home_guard_project.box import reid
from home_guard_project.box.tracker import CameraTracker

CAM = "ameer_week_0_1_ch3"
T0 = 1_791_355_000.0
RED, BLUE = [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]


def trk(track_id, first, last, start=(0.5, 0.8), end=None, active=True, prev=None, entry="", exit_=""):
    end = start if end is None else end
    return {"id": track_id, "kind": "person", "first_seen": float(first), "last_seen": float(last), "prev_id": prev,
            "first_foot": tuple(start), "last_foot": tuple(end), "moved": 0.2, "active": active, "path": [],
            "entry_edge": entry, "exit_edge": exit_}


def looks(vectors):
    """An Appearance whose score is the cosine of fixed vectors per track id (entities: the mean of their tracks)."""
    def score(track, entity):
        mine = vectors.get(int(track["id"]))
        theirs = reid.ema(vectors[i] for i in entity.get("track_ids", ()) if i in vectors and i != int(track["id"]))
        return reid.cosine(mine, theirs)
    return score


class FakeEmbedder:
    def __init__(self):
        self.calls = 0

    def embed(self, crop):
        self.calls += 1
        return [float(crop.mean()), 1.0, 0.0]


class VectorsTest(unittest.TestCase):
    def test_cosine_and_missing(self):
        self.assertAlmostEqual(reid.cosine(RED, RED), 1.0)
        self.assertAlmostEqual(reid.cosine(RED, BLUE), 0.0)
        self.assertIsNone(reid.cosine(None, RED))
        self.assertIsNone(reid.cosine([], []))

    def test_ema_weights_the_new_look_by_rho(self):
        mean = reid.ema([RED, BLUE])
        self.assertAlmostEqual(mean[1] / mean[0], 0.3 / 0.7, places=6)
        self.assertAlmostEqual(sum(x * x for x in mean), 1.0, places=6)


class SettingsTest(unittest.TestCase):
    def test_defaults_are_shadow_and_the_agreed_thresholds(self):
        s = reid.ReidSettings.from_box_settings({})
        self.assertEqual((s.mode, s.device, s.link, s.margin, s.veto), ("shadow", "AUTO", 0.70, 0.08, 0.35))

    def test_box_yaml_values(self):
        s = reid.ReidSettings.from_box_settings({"reid": "on", "reid_device": "cpu", "reid_link": 0.8,
                                                 "reid_margin": "0.1", "reid_veto": 0.3})
        self.assertEqual((s.mode, s.device, s.link, s.margin, s.veto), ("on", "CPU", 0.8, 0.1, 0.3))
        self.assertEqual(reid.ReidSettings.from_box_settings({"reid": False}).mode, "off")
        self.assertEqual(reid.ReidSettings.from_box_settings({"reid": "maybe", "reid_link": 7}).link, 0.70)
        self.assertEqual(reid.ReidSettings.from_box_settings({"reid": "maybe"}).mode, "shadow")

    def test_off_starts_nothing(self):
        self.assertIsNone(reid.start({"reid": "off"}))

    def test_missing_model_files_mean_reid_off(self):
        empty = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, empty, ignore_errors=True)
        with mock.patch.object(reid, "model_path", return_value=os.path.join(empty, reid.MODEL_FILE)):
            self.assertIsNone(reid.start({"reid": "on"}))


class CropTest(unittest.TestCase):
    def test_whole_box_resized_to_the_model_input(self):
        frame = np.zeros((360, 640, 3), np.uint8)
        crop = reid.crop_person(frame, (0.4, 0.2, 0.5, 0.9))
        self.assertEqual(crop.shape, (256, 128, 3))

    def test_a_tiny_person_says_nothing_about_clothes(self):
        frame = np.zeros((360, 640, 3), np.uint8)
        self.assertIsNone(reid.crop_person(frame, (0.4, 0.4, 0.42, 0.45)))


class StoreTest(unittest.TestCase):
    def setUp(self):
        self.frame = np.full((360, 640, 3), 100, np.uint8)
        self.embedder = FakeEmbedder()
        self.store = reid.Reid(reid.ReidSettings(mode="on"), self.embedder, clock=lambda: T0)

    def look(self, track_id=1, conf=0.9, confirmed=True, first=T0):
        return {"id": track_id, "first_seen": first, "box": (0.4, 0.2, 0.5, 0.9), "conf": conf, "confirmed": confirmed}

    def test_only_confirmed_tracks_and_at_most_per_track_looks(self):
        self.assertEqual(self.store.offer(CAM, T0, self.frame, [self.look(confirmed=False)]), 0)
        for i in range(10):
            self.store.offer(CAM, T0 + i, self.frame, [self.look(conf=0.6)])
            self.store.step(T0 + 10 * i)
        key = reid.track_key(1, T0)
        self.assertEqual(len(self.store._looks[(CAM, key)].embeds), reid.PER_TRACK)
        self.assertEqual(self.embedder.calls, reid.PER_TRACK)          # same score: never re-embedded

    def test_a_clearly_better_look_replaces_the_worst_but_attempts_are_capped(self):
        for i, conf in enumerate([0.5, 0.55, 0.6, 0.9, 0.95, 0.99, 0.999, 1.0]):
            self.store.offer(CAM, T0 + i, self.frame, [self.look(conf=conf)])
            self.store.step(T0 + 10 * i)
        rec = self.store._looks[(CAM, reid.track_key(1, T0))]
        self.assertLessEqual(rec.attempts, reid.MAX_ATTEMPTS)
        self.assertGreaterEqual(min(e[1] for e in rec.embeds), 0.6)

    def test_rate_limited_across_cameras(self):
        store = reid.Reid(reid.ReidSettings(mode="on"), self.embedder, rate_per_sec=2.0)
        for i in range(6):
            store.offer(f"cam{i}", T0, self.frame, [self.look(track_id=i)])
        done = sum(store.step(T0) for _ in range(6))
        self.assertEqual(done, 2)
        self.assertTrue(store.step(T0 + 0.5))
        self.assertFalse(store.step(T0 + 0.5))

    def test_nothing_on_disk_and_forgotten_when_the_event_closes(self):
        self.store.offer(CAM, T0, self.frame, [self.look()])
        self.store.step(T0)
        key = reid.track_key(1, T0)
        self.store.prune(T0 + 5, {CAM: {key}})                     # held by an open event
        self.assertIsNotNone(self.store.track_vector(CAM, key))
        self.store.prune(T0 + 6, {})                               # the event closed
        self.assertIsNone(self.store.track_vector(CAM, key))

    def test_unheld_tracks_expire_with_the_tracker_history_and_everything_after_a_day(self):
        self.store.offer(CAM, T0, self.frame, [self.look(1)])
        self.store.offer(CAM, T0, self.frame, [self.look(2)])
        self.store.step(T0)
        self.store.step(T0 + 1)
        k1, k2 = reid.track_key(1, T0), reid.track_key(2, T0)
        self.store.prune(T0 + reid.HISTORY_KEEP_SEC + 1, {CAM: {k2}})
        self.assertIsNone(self.store.track_vector(CAM, k1))
        self.assertIsNotNone(self.store.track_vector(CAM, k2))
        self.store.prune(T0 + reid.MAX_AGE_SEC + 1, {CAM: {k2}})
        self.assertIsNone(self.store.track_vector(CAM, k2))

    def test_appearance_scores_a_track_against_an_entity(self):
        self.store.offer(CAM, T0, self.frame, [self.look(1)])
        self.store.offer(CAM, T0 + 1, self.frame, [self.look(2, first=T0 + 1)])
        self.store.step(T0)
        self.store.step(T0 + 1)
        app = self.store.appearance(CAM)
        self.assertEqual(app.mode, "on")
        entity = {"tracks": [reid.track_key(1, T0)]}
        self.assertAlmostEqual(app.score({"id": 2, "first_seen": T0 + 1}, entity), 1.0)
        self.assertIsNone(app.score({"id": 9, "first_seen": T0 + 1}, entity))
        self.assertIsNone(reid.Reid(reid.ReidSettings(mode="off")).appearance(CAM))


class DownloadTest(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.dir, ignore_errors=True)

    def test_a_mismatch_leaves_nothing_behind(self):
        with self.assertRaises(ValueError):
            reid.download(self.dir, opener=lambda url: io.BytesIO(b"not the model"))
        self.assertEqual(os.listdir(self.dir), [])

    def test_a_verified_file_is_kept(self):
        body = b"x" * 10
        files = {"FP16": {"a.xml": (10, hashlib.sha384(body).hexdigest())}}
        seen = []
        with mock.patch.object(reid, "OMZ_FILES", files):
            out = reid.download(self.dir, opener=lambda url: seen.append(url) or io.BytesIO(body))
        self.assertEqual([os.path.basename(p) for p in out], ["a.xml"])
        self.assertTrue(seen[0].startswith("https://storage.openvinotoolkit.org/") and seen[0].endswith("FP16/a.xml"))


class PersonLooksTest(unittest.TestCase):
    def test_the_tracker_hands_over_this_look_s_people_with_their_score(self):
        t = CameraTracker(CAM)
        t.update(T0, [(0, 0.7, 0.4, 0.2, 0.5, 0.9)])
        self.assertEqual([x["confirmed"] for x in t.person_looks(T0)], [False])
        t.update(T0 + 0.5, [(0, 0.9, 0.4, 0.2, 0.5, 0.9), (2, 0.9, 0.0, 0.5, 0.2, 0.7)])
        rows = t.person_looks(T0 + 0.5)
        self.assertEqual(len(rows), 1)
        self.assertTrue(rows[0]["confirmed"])
        self.assertAlmostEqual(rows[0]["conf"], 0.9)
        self.assertEqual(t.person_looks(T0 + 0.6), [])


class EntitiesAppearanceTest(unittest.TestCase):
    """What the clothes do to who is who (entities._attach)."""

    def gone_then_back(self, vectors, mode="on", back_start=(0.9, 0.5), back_entry="right", gap=400.0):
        app = ent.Appearance(score=looks(vectors), mode=mode)
        rows = []
        tracks = [trk(1, T0, T0 + 30, active=False, exit_="left"),
                  trk(2, T0 + 30 + gap, T0 + 40 + gap, start=back_start, entry=back_entry)]
        ent.ingest(rows, tracks, T0 + 40 + gap, appearance=app)
        return rows, app

    def test_beyond_reattach_the_clothes_relink_p1(self):
        rows, app = self.gone_then_back({1: RED, 2: [0.95, 0.1, 0.0]})
        self.assertEqual([e["id"] for e in rows], ["P1"])
        self.assertEqual(rows[0]["track_ids"], [1, 2])
        self.assertEqual((rows[0]["linked_by"], rows[0]["link_score"]), ("appearance", 0.995))
        self.assertEqual(app.lines(), ["reid: linked track 2->P1 (0.99)"])

    def test_shadow_only_says_what_it_would_do(self):
        rows, app = self.gone_then_back({1: RED, 2: [0.95, 0.1, 0.0]}, mode="shadow")
        self.assertEqual([e["id"] for e in rows], ["P1", "P2"])
        self.assertEqual(rows[1]["reid_shadow"], {"would": "link", "to": "P1", "score": 0.995})
        self.assertEqual(app.lines(), ["reid: would link P2->P1 (0.99)"])
        self.assertNotIn("linked_by", rows[0])

    def test_different_clothes_are_a_new_person(self):
        rows, app = self.gone_then_back({1: RED, 2: BLUE})
        self.assertEqual([e["id"] for e in rows], ["P1", "P2"])
        self.assertEqual(app.said, [])

    def test_no_embedding_yet_is_geometry_as_before(self):
        rows, _ = self.gone_then_back({1: RED})
        self.assertEqual([e["id"] for e in rows], ["P1", "P2"])

    def test_implausible_place_or_time_never_links(self):
        rows, _ = self.gone_then_back({1: RED, 2: RED}, back_start=(0.9, 0.2), back_entry="")   # mid-picture, far
        self.assertEqual([e["id"] for e in rows], ["P1", "P2"])
        rows, _ = self.gone_then_back({1: RED, 2: RED}, gap=ent.REID_MAX_GAP_SEC + 10)
        self.assertEqual([e["id"] for e in rows], ["P1", "P2"])

    def test_two_lookalikes_need_a_clear_margin(self):
        app = ent.Appearance(score=looks({1: RED, 2: [0.98, 0.2, 0.0], 3: [0.99, 0.1, 0.0]}), mode="on")
        rows = []
        tracks = [trk(1, T0, T0 + 10, active=False, start=(0.2, 0.8)), trk(2, T0, T0 + 12, active=False, start=(0.8, 0.8)),
                  trk(3, T0 + 500, T0 + 510, start=(0.5, 0.5), entry="top")]
        ent.ingest(rows, tracks, T0 + 510, appearance=app)
        self.assertEqual([e["id"] for e in rows], ["P1", "P2", "P3"])

    def test_ambiguous_geometry_resolved_by_a_clear_appearance_winner(self):
        app = ent.Appearance(score=looks({1: RED, 2: BLUE, 3: RED}), mode="on")
        rows = []
        tracks = [trk(1, T0, T0 + 5, start=(0.5, 0.8), active=False), trk(2, T0, T0 + 6, start=(0.52, 0.8), active=False),
                  trk(3, T0 + 20, T0 + 30, start=(0.51, 0.81))]
        ent.ingest(rows, tracks, T0 + 30, appearance=app)
        self.assertEqual([e["track_ids"] for e in rows], [[1, 3], [2]])

    def test_veto_a_geometric_reattach_with_different_clothes(self):
        app = ent.Appearance(score=looks({1: RED, 2: BLUE}), mode="on")
        rows = []
        tracks = [trk(1, T0, T0 + 5, start=(0.5, 0.8), active=False), trk(2, T0 + 60, T0 + 70, start=(0.55, 0.82))]
        ent.ingest(rows, tracks, T0 + 70, appearance=app)
        self.assertEqual([e["id"] for e in rows], ["P1", "P2"])
        self.assertEqual((rows[1]["not_of"], rows[1]["veto_score"]), (["P1"], 0.0))
        self.assertEqual(app.lines(), ["reid: kept P2 apart from P1 (0.00)"])

    def test_veto_a_tracker_return_too(self):
        app = ent.Appearance(score=looks({1: RED, 2: BLUE}), mode="on")
        rows = []
        ent.ingest(rows, [trk(1, T0, T0 + 5, active=False), trk(2, T0 + 300, T0 + 310, start=(0.9, 0.9), prev=1)],
                   T0 + 310, appearance=app)
        self.assertEqual([e["id"] for e in rows], ["P1", "P2"])

    def test_shadow_veto_keeps_geometry(self):
        app = ent.Appearance(score=looks({1: RED, 2: BLUE}), mode="shadow")
        rows = []
        tracks = [trk(1, T0, T0 + 5, start=(0.5, 0.8), active=False), trk(2, T0 + 60, T0 + 70, start=(0.55, 0.82))]
        ent.ingest(rows, tracks, T0 + 70, appearance=app)
        self.assertEqual([e["id"] for e in rows], ["P1"])
        self.assertEqual(rows[0]["reid_shadow"], {"would": "split", "from": "P1", "score": 0.0})
        self.assertEqual(app.lines(), ["reid: would keep track 2 apart from P1 (0.00)"])

    def test_a_weak_match_between_veto_and_link_changes_nothing(self):
        mid = [0.6, 0.8, 0.0]                      # cosine 0.6 with RED: not a link, not a veto
        app = ent.Appearance(score=looks({1: RED, 2: mid}), mode="on")
        rows = []
        tracks = [trk(1, T0, T0 + 5, start=(0.5, 0.8), active=False), trk(2, T0 + 60, T0 + 70, start=(0.55, 0.82))]
        ent.ingest(rows, tracks, T0 + 70, appearance=app)
        self.assertEqual([e["id"] for e in rows], ["P1"])          # geometry re-attaches as before
        rows, _ = self.gone_then_back({1: RED, 2: mid})
        self.assertEqual([e["id"] for e in rows], ["P1", "P2"])    # and no appearance link beyond REATTACH_SEC


if __name__ == "__main__":
    unittest.main()
