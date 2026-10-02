from __future__ import annotations

import json
import os
import tempfile
import unittest
from typing import Optional

from home_guard_project.box.archive import expire_old_files, load_records, record_doc, search, window
from home_guard_project.box.feedback import Feedback, Query, save_feedback

NOW = 1_800_000_000.0
HOUR = 3600.0
DAY = 86400.0


def make_alert(root: str, camera: str, stem: str, ts: float, summary: str,
               date: str = "2026-10-02", with_clip: bool = True) -> None:
    """An alert clip as inference mode saves it: the mp4, then the meta with the alert in it."""
    if with_clip:
        clip = os.path.join(root, "clips", camera, date, f"{stem}.mp4")
        os.makedirs(os.path.dirname(clip), exist_ok=True)
        with open(clip, "wb") as f:
            f.write(b"mp4")
    meta = os.path.join(root, "meta", camera, date, f"{stem}.meta.json")
    os.makedirs(os.path.dirname(meta), exist_ok=True)
    with open(meta, "w", encoding="utf-8") as f:
        json.dump({
            "camera_name": camera,
            "clip_path": f"clips\\{camera}\\{date}\\{stem}.mp4",
            "clip_end_ts": ts,
            "alert": {"summary": summary, "alert_command": "[send_message]"},
        }, f)


def query(start: float, end: float, camera: Optional[str] = None, what: str = "", latest: bool = False) -> Query:
    return Query(start_ts=start, end_ts=end, camera=camera, what=what, latest=latest)


class FakeEmbedder:
    """Stands in for the embedding model: 'car' and 'person/door' are the only axes, no network."""

    @staticmethod
    def _vec(text: str):
        t = text.lower()
        vehicle = any(w in t for w in ("car", "vehicle", "driveway"))
        person = any(w in t for w in ("person", "door", "walk"))
        return [1.0 if vehicle else 0.0, 1.0 if person else 0.0, 0.1]

    def embed_one(self, text: str):
        return self._vec(text)

    def embed(self, texts):
        return [self._vec(t) for t in texts]


class DeadEmbedder:
    """An embedder that can never be reached, so the caller must fall back to keywords."""

    def embed_one(self, text: str):
        return None

    def embed(self, texts):
        return None


class ArchiveTest(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.live = os.path.join(tmp.name, "production_multi")
        self.site = os.path.join(tmp.name, "production_archive", "house2")
        make_alert(self.site, "front_door", "front_door_1_alert", NOW - 5 * HOUR, "a person is walking to the door")
        make_alert(self.site, "back_yard", "back_yard_2_alert", NOW - 3 * HOUR, "a car is parked in the yard")
        make_alert(self.live, "front_door", "front_door_3_alert", NOW - 60, "a person is standing at the door")

    def _ids(self, q: Query, **kwargs) -> list[str]:
        return [r.alert_id for r in search(load_records([self.live, self.site]), q, **kwargs)]

    def test_records_come_from_every_folder_with_their_clip(self) -> None:
        records = {r.alert_id: r for r in load_records([self.live, self.site])}
        self.assertEqual(len(records), 3)
        first = records["front_door_1_alert"]
        self.assertEqual((first.camera, first.ts, first.summary),
                         ("front_door", NOW - 5 * HOUR, "a person is walking to the door"))
        self.assertTrue(os.path.isfile(first.clip_path))

    def test_a_record_whose_clip_is_gone_has_no_clip_path(self) -> None:
        make_alert(self.site, "front_door", "front_door_9_alert", NOW - HOUR, "x", with_clip=False)
        record = next(r for r in load_records([self.site]) if r.alert_id == "front_door_9_alert")
        self.assertIsNone(record.clip_path)

    def test_missing_folders_and_damaged_metas_are_skipped(self) -> None:
        bad = os.path.join(self.site, "meta", "front_door", "2026-10-02", "broken.meta.json")
        with open(bad, "w", encoding="utf-8") as f:
            f.write("{not json")
        self.assertEqual(len(load_records([self.live, self.site, os.path.join(self.site, "nope")])), 3)

    def test_search_by_time_range_is_in_time_order(self) -> None:
        self.assertEqual(self._ids(query(NOW - 6 * HOUR, NOW)),
                         ["front_door_1_alert", "back_yard_2_alert", "front_door_3_alert"])
        self.assertEqual(self._ids(query(NOW - 4 * HOUR, NOW - 2 * HOUR)), ["back_yard_2_alert"])
        self.assertEqual(self._ids(query(NOW - 30 * DAY, NOW - 20 * DAY)), [])

    def test_search_by_camera(self) -> None:
        self.assertEqual(self._ids(query(NOW - 6 * HOUR, NOW, camera="front_door")),
                         ["front_door_1_alert", "front_door_3_alert"])

    def test_latest_is_newest_first_and_limited(self) -> None:
        self.assertEqual(self._ids(query(NOW - DAY, NOW, latest=True), limit=2),
                         ["front_door_3_alert", "back_yard_2_alert"])

    def test_words_narrow_the_result_but_never_empty_it(self) -> None:
        self.assertEqual(self._ids(query(NOW - 6 * HOUR, NOW, what="the car")), ["back_yard_2_alert"])
        # No summary mentions a giraffe: show what there is in the range, not nothing.
        self.assertEqual(len(self._ids(query(NOW - 6 * HOUR, NOW, what="giraffe"))), 3)

    def test_semantic_search_ranks_by_meaning_when_an_embedder_is_given(self) -> None:
        ids = self._ids(query(NOW - 6 * HOUR, NOW, what="a vehicle on the property"), embedder=FakeEmbedder())
        self.assertEqual(ids[0], "back_yard_2_alert")  # the car summary is closest in meaning
        self.assertEqual(set(ids), {"front_door_1_alert", "back_yard_2_alert", "front_door_3_alert"})

    def test_semantic_search_falls_back_to_keywords_when_the_embedder_is_down(self) -> None:
        self.assertEqual(self._ids(query(NOW - 6 * HOUR, NOW, what="the car"), embedder=DeadEmbedder()),
                         ["back_yard_2_alert"])

    def test_window_returns_everything_in_range_uncapped_in_time_order(self) -> None:
        recs = load_records([self.live, self.site])
        self.assertEqual([r.alert_id for r in window(recs, query(NOW - 6 * HOUR, NOW))],
                         ["front_door_1_alert", "back_yard_2_alert", "front_door_3_alert"])
        self.assertEqual([r.alert_id for r in window(recs, query(NOW - 6 * HOUR, NOW, camera="back_yard"))],
                         ["back_yard_2_alert"])
        self.assertEqual(window(recs, query(NOW - 30 * DAY, NOW - 20 * DAY)), [])

    def test_record_doc_is_a_json_friendly_summary(self) -> None:
        records = {r.alert_id: r for r in load_records([self.live, self.site])}
        doc = record_doc(records["back_yard_2_alert"], score=0.812345)
        self.assertEqual((doc["id"], doc["camera"], doc["has_video"]), ("back_yard_2_alert", "back_yard", True))
        self.assertEqual(doc["match"], 0.812)
        json.dumps(doc)  # must be serialisable for a tool result

    def test_feedback_is_attached_to_its_alert(self) -> None:
        alert = {"alert_id": "back_yard_2_alert", "camera": "back_yard", "summary": "", "ts": NOW - 3 * HOUR}
        save_feedback(self.live, alert, Feedback(verdict="false_alarm"), "nothing there", {}, "1", NOW)
        records = {r.alert_id: r for r in load_records([self.live, self.site])}
        self.assertEqual(records["back_yard_2_alert"].verdicts, ("false_alarm",))
        self.assertEqual(records["front_door_1_alert"].verdicts, ())

    def test_expire_deletes_only_old_files_and_empty_folders(self) -> None:
        old_clip = os.path.join(self.site, "clips", "front_door", "2026-10-02", "front_door_1_alert.mp4")
        old_meta = os.path.join(self.site, "meta", "front_door", "2026-10-02", "front_door_1_alert.meta.json")
        new_clip = os.path.join(self.site, "clips", "back_yard", "2026-10-02", "back_yard_2_alert.mp4")
        new_meta = os.path.join(self.site, "meta", "back_yard", "2026-10-02", "back_yard_2_alert.meta.json")
        for path in (old_clip, old_meta):
            os.utime(path, (NOW - 15 * DAY, NOW - 15 * DAY))
        for path in (new_clip, new_meta):
            os.utime(path, (NOW - 13 * DAY, NOW - 13 * DAY))

        self.assertEqual(expire_old_files(os.path.dirname(self.site), 14, now=NOW), 2)

        self.assertFalse(os.path.exists(old_clip))
        self.assertFalse(os.path.exists(os.path.dirname(old_clip)))  # the emptied folder went too
        self.assertTrue(os.path.isfile(new_clip))
        self.assertEqual(expire_old_files(os.path.join(self.site, "nope"), 14, now=NOW), 0)


if __name__ == "__main__":
    unittest.main()
