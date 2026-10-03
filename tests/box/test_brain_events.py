# tests/box/test_brain_events.py
from __future__ import annotations

import datetime as dt
import json
import os
import tempfile
import unittest
import shutil
from dataclasses import replace

from test_archive import FakeEmbedder

from home_guard_project.box.archive import load_records
from home_guard_project.box.brain.events import (
    class_words,
    coverage,
    event_doc,
    filter_events,
    load_events,
    order_for_mode,
    rank_events,
    read_desc,
    write_desc,
)
from home_guard_project.box.brain.registry import CameraState, HouseSnapshot
from home_guard_project.box.brain.events import local, latest_desc

NOW = dt.datetime(2026, 10, 3, 23, 0).timestamp()
HOUR = 3600.0


def meta(root, camera, stem, ts, kind="alert", summary="", label="", labels=("person",), people=None,
         with_clip=True, date="2026-10-03"):
    if with_clip:
        clip = os.path.join(root, "clips", camera, date, f"{stem}.mp4")
        os.makedirs(os.path.dirname(clip), exist_ok=True)
        with open(clip, "wb") as f:
            f.write(b"mp4")
    path = os.path.join(root, "meta", camera, date, f"{stem}.meta.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    alert = {"summary": summary, "alert_command": "[send_message]" if kind == "alert" else "[none]",
             "labels": list(labels)}
    if label:
        alert["label"] = label
    if people is not None:
        alert["people"] = people
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"camera_name": camera, "kind": kind, "clip_path": f"clips\\{camera}\\{date}\\{stem}.mp4",
                   "clip_start_ts": ts - 10, "clip_end_ts": ts, "trigger_ts": ts - 6,
                   "mode": "assistant" if kind == "quiet" else "guard",
                   "yolo": {"trigger_classes": list(labels)}, "alert": alert}, f)


class EventsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.root = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.root)
        self.desc = os.path.join(self.root, ".desc")
        meta(self.root, "main_entrance", "main_entrance_1_alert", NOW - 5 * HOUR,
             summary="A man stands at the door looking around.", label="suspicious", people=1)
        meta(self.root, "back_door", "back_door_2_alert", NOW - 4 * HOUR,
             summary="A woman carries bags into the house.", label="normal", people=1)
        meta(self.root, "main_entrance", "main_entrance_3_quiet", NOW - 14 * HOUR, kind="quiet")
        meta(self.root, "front_side", "front_side_4_quiet", NOW - 13 * HOUR, kind="quiet", labels=("car",))

    def test_archive_reads_the_new_fields(self) -> None:
        recs = {r.alert_id: r for r in load_records([self.root])}
        a = recs["main_entrance_1_alert"]
        self.assertEqual((a.kind, a.label, a.people, a.mode, a.described), ("alert", "suspicious", 1, "guard", True))
        self.assertEqual(a.detector_labels, ("person",))
        self.assertEqual((a.clip_start_ts, a.trigger_ts), (NOW - 5 * HOUR - 10, NOW - 5 * HOUR - 6))
        q = recs["main_entrance_3_quiet"]
        self.assertEqual((q.kind, q.described, q.summary), ("quiet", False, ""))

    def test_desc_cache_fills_the_summary_of_a_quiet_event(self) -> None:
        write_desc(self.desc, "main_entrance_3_quiet", "assistant:v1:", {"text": "A courier leaves a parcel.",
                                                                           "ts": NOW})
        self.assertIn("assistant:v1:", read_desc(self.desc, "main_entrance_3_quiet"))
        recs = {r.alert_id: r for r in load_events([self.root], self.desc)}
        self.assertEqual(recs["main_entrance_3_quiet"].summary, "A courier leaves a parcel.")
        self.assertTrue(recs["main_entrance_3_quiet"].described)

    def test_filter_and_order(self) -> None:
        recs = load_events([self.root], self.desc)
        day = filter_events(recs, NOW - 6 * HOUR, NOW)
        self.assertEqual([r.alert_id for r in day], ["main_entrance_1_alert", "back_door_2_alert"])
        self.assertEqual([r.alert_id for r in filter_events(recs, 0, NOW, kinds={"quiet"})],
                         ["main_entrance_3_quiet", "front_side_4_quiet"])
        self.assertEqual([r.alert_id for r in filter_events(recs, 0, NOW, cameras={"back_door"})],
                         ["back_door_2_alert"])
        self.assertEqual([r.alert_id for r in order_for_mode(day, "guard")],
                         ["main_entrance_1_alert", "back_door_2_alert"])

    def test_class_words(self) -> None:
        self.assertEqual(class_words("was anyone near the house"), {"person"})
        self.assertIn("car", class_words("a vehicle in the driveway"))
        self.assertEqual(class_words("anything at all"), set())

    def test_rank_keeps_only_matches_and_undescribed_detector_hits(self) -> None:
        recs = load_events([self.root], self.desc)
        ids = [r.alert_id for r in rank_events(recs, "someone at the door", embedder=None)]
        self.assertIn("main_entrance_1_alert", ids)          # keyword "door"
        self.assertIn("main_entrance_3_quiet", ids)          # detector saw a person, not described
        self.assertNotIn("front_side_4_quiet", ids)          # a car, not a person
        self.assertEqual(rank_events(recs, "giraffe", embedder=None), [])
        ranked = rank_events(recs, "a person walking to the door", embedder=FakeEmbedder())
        self.assertEqual(ranked[0].alert_id, "main_entrance_1_alert")

    def test_event_doc_says_when_an_event_is_not_confirmed(self) -> None:
        recs = {r.alert_id: r for r in load_events([self.root], self.desc)}
        doc = event_doc(recs["main_entrance_3_quiet"], "E3")
        self.assertEqual(doc["handle"], "E3")
        self.assertEqual(doc["summary"], "detector saw a person, not confirmed")
        self.assertFalse(doc["described"])
        json.dumps(doc)

    def test_coverage_names_off_cameras_and_where_the_quiet_log_starts(self) -> None:
        cams = (CameraState("main_entrance", True, live=True), CameraState("back_door", False, live=False))
        snap = HouseSnapshot(now=NOW, mode="assistant", mode_ends=None, mode_started=None, start_hour=22,
                             end_hour=6, cameras=cams, quiet_log=True, quiet_since=NOW - 20 * HOUR)
        cov = coverage(snap, load_events([self.root], self.desc), NOW - 24 * HOUR, NOW)
        self.assertEqual(cov["cameras_off"], ["back_door"])
        self.assertTrue(cov["quiet_log_since"].endswith("03:00"))
        self.assertTrue(cov["oldest_quiet_event_kept"].endswith("09:00"))
        off = HouseSnapshot(now=NOW, mode="assistant", mode_ends=None, mode_started=None, start_hour=22,
                            end_hour=6, cameras=cams)
        self.assertEqual(coverage(off, [], NOW - HOUR, NOW)["quiet_log_since"], "off")

    def test_load_records_skips_bad_numbers_and_feedback_shapes(self):
        folder = os.path.join(self.root, "feedback")
        os.makedirs(folder)
        for i, alert in enumerate([[], "bad", {"alert_id": []}]):
            with open(os.path.join(folder, f"{i}.feedback.json"), "w") as f:
                json.dump({"alert": alert, "verdict": {"bad": True}}, f)
        for i, value in enumerate(["bad", float("nan"), float("inf"), []]):
            path = os.path.join(self.root, "meta", f"bad{i}.meta.json")
            with open(path, "w") as f:
                json.dump({"clip_end_ts": value, "alert": {"people": "bad"},
                           "clip_start_ts": "bad", "yolo": {"trigger_classes": 5}}, f)
        with self.assertLogs("box.archive", level="WARNING"):
            records = load_records([self.root])
        self.assertEqual(len(records), 4)

    def test_read_desc_invalid_utf8_and_shapes(self):
        os.makedirs(self.desc)
        with open(os.path.join(self.desc, "bad.json"), "wb") as f:
            f.write(b"\xff")
        with self.assertLogs("box.brain.events", level="WARNING"):
            self.assertEqual(read_desc(self.desc, "bad"), {})
        with open(os.path.join(self.desc, "bad.json"), "w") as f:
            json.dump([], f)
        with self.assertLogs("box.brain.events", level="WARNING"):
            self.assertEqual(read_desc(self.desc, "bad"), {})

    def test_write_desc_nonserializable_preserves_cache(self):
        write_desc(self.desc, "x", "good", {"text": "saved"})
        with self.assertLogs("box.brain.events", level="WARNING") as logs:
            write_desc(self.desc, "x", "bad", {"text": object()})
        self.assertEqual(len(logs.output), 1)
        self.assertEqual(read_desc(self.desc, "x"), {"good": {"text": "saved"}})
        with self.assertLogs("box.brain.events", level="WARNING"):
            write_desc(self.desc, "x", "bad", ["not a description"])
        self.assertEqual(read_desc(self.desc, "x"), {"good": {"text": "saved"}})

    def test_archive_skips_invalid_optional_fields_and_utf8(self):
        for i, fields in enumerate([{"alert": {"people": float("nan")}},
                                    {"trigger_ts": float("inf")},
                                    {"yolo": {"trigger_classes": "person"}}]):
            with open(os.path.join(self.root, "meta", f"extra{i}.meta.json"), "w") as f:
                json.dump(dict(clip_end_ts=NOW, **fields), f)
        with open(os.path.join(self.root, "meta", "utf8.meta.json"), "wb") as f:
            f.write(b"\xff")
        with self.assertLogs("box.archive", level="WARNING"):
            self.assertEqual(len(load_records([self.root])), 4)

    def test_archive_skips_deeply_nested_meta_and_feedback(self):
        feedback = os.path.join(self.root, "feedback")
        os.makedirs(feedback)
        nested = '{"nested":' + '[' * 100000 + '0' + ']' * 100000 + '}'
        for path in [os.path.join(self.root, "meta", "nested.meta.json"),
                     os.path.join(feedback, "nested.feedback.json")]:
            with open(path, "w", encoding="utf-8") as f:
                f.write(nested)
        with self.assertLogs("box.archive", level="WARNING") as logs:
            records = load_records([self.root])
        self.assertEqual(len(records), 4)
        self.assertEqual(len(logs.output), 2)

    def test_latest_desc_bad_timestamps(self):
        with self.assertLogs("box.brain.events", level="WARNING"):
            result = latest_desc({"bad": {"text": "bad", "ts": "no"},
                                  "good": {"text": "good", "ts": NOW}})
        self.assertEqual(result["text"], "good")

    def test_load_events_skips_bad_cached_description(self):
        write_desc(self.desc, "main_entrance_3_quiet", "bad", {"text": [1], "ts": "bad"})
        with self.assertLogs("box.brain.events", level="WARNING"):
            records = load_events([self.root], self.desc)
        self.assertFalse(next(r for r in records if r.kind == "quiet").described)

    def test_filter_events_bad_time(self):
        for value in ["bad", float("nan"), float("inf")]:
            with self.assertLogs("box.brain.events", level="WARNING"):
                self.assertEqual(filter_events(load_records([self.root]), value, NOW), [])

    def test_class_words_bad_input(self):
        with self.assertLogs("box.brain.events", level="WARNING"):
            self.assertEqual(class_words([]), set())

    def test_rank_events_bad_embedding_falls_back(self):
        class BadEmbedder:
            def embed_one(self, text): return ["bad"]
            def embed(self, texts): return [["bad"] for text in texts]
        records = load_records([self.root])
        with self.assertLogs("box.brain.events", level="WARNING") as logs:
            result = rank_events(records, "door", BadEmbedder())
        self.assertEqual(len(logs.output), 1)
        self.assertEqual([r.alert_id for r in result], ["main_entrance_1_alert"])

    def test_order_for_mode_bad_record(self):
        with self.assertLogs("box.brain.events", level="WARNING"):
            self.assertEqual(order_for_mode([None], "guard"), [])
        for value in [float("nan"), float("inf")]:
            record = replace(load_records([self.root])[0], ts=value)
            with self.assertLogs("box.brain.events", level="WARNING"):
                self.assertEqual(order_for_mode([record], "assistant"), [])

    def test_local_invalid_timestamps(self):
        for value in ["bad", float("nan"), float("inf")]:
            with self.assertLogs("box.brain.events", level="WARNING"):
                self.assertEqual(local(value), "unknown")

    def test_event_doc_bad_record(self):
        with self.assertLogs("box.brain.events", level="WARNING"):
            self.assertEqual(event_doc(None, "E1"), {})

    def test_coverage_bad_retention(self):
        snapshot = HouseSnapshot(NOW, "assistant", None, None, 22, 6, (), retention_days="bad")
        with self.assertLogs("box.brain.events", level="WARNING"):
            self.assertEqual(coverage(snapshot, [], NOW-HOUR, NOW), {})


if __name__ == "__main__":
    unittest.main()
