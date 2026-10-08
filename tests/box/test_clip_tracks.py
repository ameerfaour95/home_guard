"""The alert clip's tracks file (box/clip_tracks.py): the tracker's boxes per look, frame numbers, entity ids."""
from __future__ import annotations

import json
import os
import tempfile
import unittest
from unittest import mock

import numpy as np

from home_guard_project.box import clip_tracks
from home_guard_project.box import inference as inf
from home_guard_project.box import tracker as tr
from home_guard_project.box.alert_clips import encode_frame
from home_guard_project.box.entities import track_key
from home_guard_project.box.outbox import move_clip

T0 = 1_791_000_000.0
# The keys the offline replay (analysis/replay_tracks.py) writes; its readers rely on them.
REPLAY_TRACK_KEYS = {"id", "kind", "cls", "first_seen", "last_seen", "hits", "confirmed", "shown", "max_conf",
                     "prev_id", "returns", "boxes"}
REPLAY_DOC_KEYS = {"clip", "camera", "start_local", "params", "frames", "looks", "tracks", "stats"}


def person(fx: float, fy: float, conf: float = 0.8):
    return (0, conf, round(fx - 0.03, 4), round(fy - 0.2, 4), round(fx + 0.03, 4), round(fy, 4))


def car(fx: float, fy: float):
    return (2, 0.9, round(fx - 0.1, 4), round(fy - 0.12, 4), round(fx + 0.1, 4), round(fy, 4))


def walked_tracker(start: float = T0, looks: int = 8, step: float = 0.5) -> tr.CameraTracker:
    """One person walking right, and a car that never moves."""
    t = tr.CameraTracker("door")
    for i in range(looks):
        t.update(start + i * step, [person(0.2 + 0.02 * i, 0.8), car(0.7, 0.5)])
    return t


class TrackerBoxesTest(unittest.TestCase):
    def test_each_confirmed_track_has_its_box_at_every_look_in_the_window(self) -> None:
        t = walked_tracker()
        tracks = t.tracks_with_boxes(T0 + 1.0, T0 + 2.0)
        walker = next(x for x in tracks if x["kind"] == "person")
        self.assertEqual([b["ts"] for b in walker["boxes"]], [T0 + 1.0, T0 + 1.5, T0 + 2.0])
        self.assertEqual(walker["boxes"][0]["box"], list(person(0.24, 0.8)[2:]))
        self.assertEqual(walker["first_seen"], T0)                 # the whole visit, not the window
        self.assertEqual(walker["hits"], 8)
        self.assertTrue(walker["confirmed"] and walker["shown"])
        self.assertEqual(walker["max_conf"], 0.8)
        parked = next(x for x in tracks if x["kind"] == "vehicle")
        self.assertFalse(parked["shown"])                          # kept for labeling, flagged as parked
        self.assertEqual(parked["cls"], 2)

    def test_a_one_look_track_is_not_handed_out(self) -> None:
        t = tr.CameraTracker("door")
        t.update(T0, [person(0.5, 0.8)])
        self.assertEqual(t.tracks_with_boxes(T0 - 1, T0 + 1), [])

    def test_boxes_are_capped_like_the_foot_points(self) -> None:
        t = tr.CameraTracker("door")
        for i in range(tr.MAX_POINTS * 2 + 5):
            t.update(T0 + i * 0.1, [person(0.5, 0.8)])
        (track,) = t.tracks_with_boxes(T0, T0 + 1000)
        with t._lock:
            raw = t._active[0]
        self.assertLessEqual(len(raw.boxes), tr.MAX_POINTS)
        self.assertEqual(len(raw.boxes), len(raw.points))
        self.assertEqual([ts for ts, _ in raw.boxes], [p[0] for p in raw.points])
        self.assertEqual(track["boxes"][0]["ts"], T0)                # the first look survives thinning

    def test_looks_between_lists_the_ids_seen(self) -> None:
        t = walked_tracker(looks=3)
        looks = t.looks_between(T0 + 0.5, T0 + 1.0)
        self.assertEqual([ts for ts, _ in looks], [T0 + 0.5, T0 + 1.0])
        self.assertEqual(len(looks[0][1]), 2)

    def test_the_registry_passes_both_through(self) -> None:
        reg = tr.TrackerRegistry(scene_map_loader=lambda cam: None)
        for i in range(3):
            reg.update("door", T0 + i, [person(0.5, 0.8)])
        self.assertEqual(len(reg.tracks_with_boxes("door", T0, T0 + 5)), 1)
        self.assertEqual(len(reg.looks_between("door", T0, T0 + 5)), 3)


class BuildTest(unittest.TestCase):
    META = {"camera_name": "door", "clip_path": "clips\\door\\2026-10-08\\door_1_alert.mp4",
            "clip_start_ts": T0, "clip_end_ts": T0 + 10.0, "frames_written": 71}

    def test_frame_is_the_nearest_clip_frame(self) -> None:
        self.assertEqual(clip_tracks.frame_index(T0, T0, T0 + 10, 71), 0)
        self.assertEqual(clip_tracks.frame_index(T0 + 10, T0, T0 + 10, 71), 70)
        self.assertEqual(clip_tracks.frame_index(T0 + 5.04, T0, T0 + 10, 71), 35)
        self.assertEqual(clip_tracks.frame_index(T0 + 5.08, T0, T0 + 10, 71), 36)
        self.assertEqual(clip_tracks.frame_index(T0 + 99, T0, T0 + 10, 71), 70)    # clamped
        self.assertEqual(clip_tracks.frame_index(T0 + 3, T0, T0, 1), 0)             # one frame

    def test_the_document_has_the_replay_format_plus_source_and_entities(self) -> None:
        t = walked_tracker(start=T0 + 1.0)
        tracks = t.tracks_with_boxes(T0, T0 + 10)
        walker = next(x for x in tracks if x["kind"] == "person")
        session = {"entities": [{"id": "P1", "kind": "person", "tracks": [track_key(walker)], "track_ids": [walker["id"]]},
                                {"id": "P2", "kind": "person", "tracks": ["99@1.000"], "track_ids": [99]}]}
        doc = clip_tracks.build(self.META, tracks, t.looks_between(T0, T0 + 10),
                                {"person_conf": 0.5, "tracker": tr.TRACKS_VERSION}, clip_tracks.entity_ids(session))
        self.assertTrue(REPLAY_DOC_KEYS <= set(doc))
        self.assertEqual(doc["source"], "live")
        self.assertEqual(doc["clip"], "clips/door/2026-10-08/door_1_alert.mp4")
        self.assertEqual(doc["frames"], 71)
        self.assertEqual(doc["params"], {"person_conf": 0.5, "tracker": "tb1"})
        rows = {r["kind"]: r for r in doc["tracks"]}
        self.assertTrue(REPLAY_TRACK_KEYS <= set(rows["person"]))
        self.assertEqual(rows["person"]["entity"], "P1")
        self.assertNotIn("entity", rows["vehicle"])                  # the book never mapped the parked car
        self.assertEqual([b["frame"] for b in rows["person"]["boxes"]], [7, 11, 14, 18, 21, 25, 28, 32])
        self.assertEqual(len(doc["looks"]), 8)
        self.assertEqual(doc["looks"][0]["frame"], 7)
        self.assertEqual(sorted(doc["looks"][0]["tracks"]), sorted(r["id"] for r in doc["tracks"]))
        self.assertEqual(doc["stats"]["person_tracks"], 1)
        self.assertEqual(doc["stats"]["parked_vehicles"], 1)
        self.assertEqual(doc["stats"]["entities"], ["P1"])
        json.dumps(doc)

    def test_boxes_outside_the_clip_are_dropped(self) -> None:
        t = walked_tracker(start=T0 - 2.0)
        doc = clip_tracks.build(self.META, t.tracks_with_boxes(T0 - 10, T0 + 10), t.looks_between(T0 - 10, T0 + 10))
        walker = next(r for r in doc["tracks"] if r["kind"] == "person")
        self.assertTrue(all(b["ts"] >= T0 for b in walker["boxes"]))
        self.assertTrue(all(lk["ts"] >= T0 for lk in doc["looks"]))


class _Assistant:
    def __init__(self) -> None:
        self.clips: list = []

    def send_clip(self, alert_id, clip_path, silent=False):
        self.clips.append(alert_id)
        return {"sent": True}


class _Book:
    def __init__(self, session):
        self.session = session

    def session_of_alert(self, alert_id):
        return self.session


DELIVERED = {"channel": "telegram", "telegram": {"telegram": {"sent": True, "results": [{"ok": True}]}}}


class SaveClipTest(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.production = os.path.join(tmp.name, "production_multi")
        self.training = os.path.join(tmp.name, "dataset_multi")
        self.frames = [(T0 + i * 0.5, encode_frame(np.zeros((48, 64, 3), dtype=np.uint8))) for i in range(9)]
        self.tracker = walked_tracker(start=T0 + 0.5, looks=6)
        self.job = inf.AlertJob(camera="door", stem="door_1791000000_alert", ts=T0 + 1, labels=["person"],
                                alert={"summary": "a person", "alert_command": "[send_message]",
                                       "labels": ["person"], "dispatch": DELIVERED},
                                person_conf=0.8)
        self.job.ready.set()

    def _attach(self) -> None:
        settings_conf = 0.5
        inf._attach_track_boxes(self.job, self.tracker_registry(), self.frames, settings_conf)

    def tracker_registry(self):
        reg = mock.Mock()
        reg.tracks_with_boxes = lambda cam, t0, t1: self.tracker.tracks_with_boxes(t0, t1)
        reg.looks_between = lambda cam, t0, t1: self.tracker.looks_between(t0, t1)
        return reg

    def _save(self, assistant=None) -> None:
        with mock.patch("home_guard_project.box.alert_clips._to_h264", return_value=False):
            inf._save_clip(self.job, self.frames, self.production, self.training, assistant)

    def _metas(self, root):
        out = []
        for dirpath, _, names in os.walk(os.path.join(root, "meta")):
            out += [os.path.join(dirpath, n) for n in names if n.endswith(".meta.json")]
        return out

    def test_both_copies_of_an_alert_get_their_tracks_file_with_entity_ids(self) -> None:
        self._attach()
        self.assertEqual(self.job.track_boxes["params"],
                         {"person_conf": 0.5, "alert_person_conf": 0.8, "tracker": tr.TRACKS_VERSION})
        walker = next(x for x in self.job.track_boxes["tracks"] if x["kind"] == "person")
        book = _Book({"entities": [{"id": "P1", "tracks": [track_key(walker)]}]})
        with mock.patch.object(inf, "EVENTS", book):
            self._save(_Assistant())
        for root in (self.production, self.training):
            (meta,) = self._metas(root)
            path = clip_tracks.path_for(root, meta)
            self.assertTrue(path.endswith(os.path.join("responses", "door", os.path.basename(os.path.dirname(meta)),
                                                       "door_1791000000_alert.tracks.json")))
            with open(path, encoding="utf-8") as f:
                doc = json.load(f)
            person_row = next(r for r in doc["tracks"] if r["kind"] == "person")
            self.assertEqual(person_row["entity"], "P1")
            self.assertEqual(doc["frames"], 9)
            self.assertEqual([b["frame"] for b in person_row["boxes"]], [1, 2, 3, 4, 5, 6])
            self.assertEqual(doc["camera"], "door")

    def test_a_false_positive_gets_its_tracks_file_in_the_training_set(self) -> None:
        self._attach()
        self.job.false_positive = True
        self._save()
        (meta,) = self._metas(self.training)
        self.assertTrue(os.path.isfile(clip_tracks.path_for(self.training, meta)))
        self.assertEqual(self._metas(self.production), [])

    def test_no_tracker_data_means_no_file_and_the_clip_as_before(self) -> None:
        self._save(_Assistant())
        (meta,) = self._metas(self.production)
        self.assertFalse(os.path.exists(clip_tracks.path_for(self.production, meta)))

    def test_a_failing_tracks_file_never_stops_the_clip_or_its_video(self) -> None:
        self._attach()
        assistant = _Assistant()
        with mock.patch.object(clip_tracks, "write", side_effect=OSError("disk full")), \
                self.assertLogs("box.inference", level="WARNING") as logs:
            self._save(assistant)
        self.assertEqual(len(self._metas(self.production)), 1)
        self.assertEqual(len(self._metas(self.training)), 1)
        self.assertEqual(assistant.clips, ["door_1791000000_alert"])
        self.assertTrue(any("tracks file" in line for line in logs.output))

    def test_a_failing_event_book_still_writes_the_tracks_without_entities(self) -> None:
        self._attach()
        book = mock.Mock()
        book.session_of_alert.side_effect = RuntimeError("book broken")
        with mock.patch.object(inf, "EVENTS", book):
            self._save()
        (meta,) = self._metas(self.production)
        with open(clip_tracks.path_for(self.production, meta), encoding="utf-8") as f:
            doc = json.load(f)
        self.assertFalse(any("entity" in r for r in doc["tracks"]))

    def test_a_failing_tracker_leaves_the_job_without_boxes(self) -> None:
        reg = mock.Mock()
        reg.tracks_with_boxes.side_effect = RuntimeError("tracker broken")
        inf._attach_track_boxes(self.job, reg, self.frames, 0.5)
        self.assertIsNone(self.job.track_boxes)
        inf._attach_track_boxes(self.job, None, self.frames, 0.5)
        inf._attach_track_boxes(self.job, reg, [], 0.5)
        self.assertIsNone(self.job.track_boxes)

    def test_the_outbox_moves_the_tracks_file_with_its_clip(self) -> None:
        self._attach()
        self._save()
        (meta,) = self._metas(self.training)
        tracks_path = clip_tracks.path_for(self.training, meta)
        outbox = os.path.join(os.path.dirname(self.training), "outbox", "house2")
        move_clip(meta, self.training, outbox)
        self.assertFalse(os.path.exists(tracks_path))
        self.assertTrue(os.path.isfile(os.path.join(outbox, os.path.relpath(tracks_path, self.training))))


if __name__ == "__main__":
    unittest.main()
