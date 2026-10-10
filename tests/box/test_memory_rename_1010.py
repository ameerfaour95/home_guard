# tests/box/test_memory_rename_1010.py
"""2026-10-10 12:31 the site was renamed (ameer_week_0_1_* -> ameer_v2_*): the pergola workers' marks, the
electricians' activity fact, camera facts and precedents stayed under the old ids. memory_rename carries them once,
at start, with a backup."""

from __future__ import annotations

import glob
import json
import os
import shutil
import tempfile
import unittest

from home_guard_project.box import activity_memory as am
from home_guard_project.box.camera_profiles import CameraProfiles
from home_guard_project.box.events import EventBook
from home_guard_project.box.memory_rename import carry_memory

CAMERAS = ["ameer_v2_ch2", "ameer_v2_ch3", "ameer_v2_ch5", "ameer_v2_ch6", "ameer_v2_ch8", "ameer_v2_ch1"]
WED = 1791527678.3658016            # 2026-10-09 09:34, when the pergola mark was said
FRI_14 = 1791637200.0               # 2026-10-10 ~16:00


class MemoryRenameTests(unittest.TestCase):
    def setUp(self) -> None:
        self.dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.dir, True)
        os.makedirs(os.path.join(self.dir, "events"))
        # The box's own files of 2026-10-10 (known.json and activity.json, as copied from the box).
        known = [{"text": "העובדים של הפרגולה", "by": "Hello_24", "at": WED, "until": 1792076400.0, "people": 0,
                  "camera": cam, "id": kid, "daily_from": "07:00", "daily_to": "18:00"}
                 for cam, kid in (("ameer_week_0_1_ch3", "9b48e65daf"), ("ameer_week_0_1_ch6", "4510fbf0a9"))]
        with open(os.path.join(self.dir, "events", "known.json"), "w", encoding="utf-8") as f:
            json.dump(known, f, ensure_ascii=False)
        activity = [{"id": "act_c86b29f1", "cameras": ["ameer_week_0_1_ch6"], "actions": ["kneeling", "working_ground"],
                     "cause": "שני אנשים שמרכיבים את הלדים במדרגות (חשמלאים)", "place_words": "במדרגות",
                     "owner_words": "אלה מקרים תקינים", "known_id": "4510fbf0a9", "daily_from": "07:00",
                     "daily_to": "18:00", "until": 1792076400.0, "by": "Hello_24", "at": 1791563384.5,
                     "alert_id": "ameer_week_0_1_ch6_1791543787_alert", "cancelled_at": 0.0, "history": []}]
        with open(os.path.join(self.dir, "activity.json"), "w", encoding="utf-8") as f:
            json.dump(activity, f, ensure_ascii=False)
        CameraProfiles(os.path.join(self.dir, "events", "camera_profiles.json")).add_fact(
            "ameer_week_0_1_ch6", "שליחים מגיעים רק לכניסה הראשית", now=WED)
        with open(os.path.join(self.dir, "cases.jsonl"), "w", encoding="utf-8") as f:
            f.write(json.dumps({"event": "create", "case": {"id": "C1", "camera": "ameer_week_0_1_ch6"}}) + "\n")

    def test_old_marks_apply_to_the_renamed_cameras(self) -> None:
        before = EventBook(os.path.join(self.dir, "events"))
        self.assertIsNone(before.known_for("ameer_v2_ch3", FRI_14))         # the bug: the mark no longer applied
        moved = carry_memory(CAMERAS, self.dir, now=FRI_14)
        self.assertEqual(moved[os.path.join("events", "known.json")],
                         {"ameer_week_0_1_ch3": "ameer_v2_ch3", "ameer_week_0_1_ch6": "ameer_v2_ch6"})
        book = EventBook(os.path.join(self.dir, "events"))
        self.assertEqual(book.known_for("ameer_v2_ch3", FRI_14).text, "העובדים של הפרגולה")
        self.assertEqual(book.known_for("ameer_v2_ch6", FRI_14).id, "4510fbf0a9")
        self.assertIsNone(book.known_for("ameer_v2_ch2", FRI_14))
        self.assertTrue(glob.glob(os.path.join(self.dir, "events", "known.json.bak-rename-*")))

    def test_activity_fact_profile_and_precedent_follow(self) -> None:
        carry_memory(CAMERAS, self.dir, now=FRI_14)
        facts = am.ActivityBook(os.path.join(self.dir, "activity.json")).live(FRI_14)
        self.assertEqual(facts[0].cameras, ["ameer_v2_ch6"])
        with open(os.path.join(self.dir, "activity.json"), encoding="utf-8") as f:
            self.assertEqual(json.load(f)[0]["alert_id"], "ameer_week_0_1_ch6_1791543787_alert")   # history stays
        profiles = CameraProfiles(os.path.join(self.dir, "events", "camera_profiles.json"))
        self.assertEqual(list(profiles.data()["cameras"]), ["ameer_v2_ch6"])
        with open(os.path.join(self.dir, "cases.jsonl"), encoding="utf-8") as f:
            self.assertEqual(json.loads(f.readline())["case"]["camera"], "ameer_v2_ch6")

    def test_second_start_moves_nothing(self) -> None:
        carry_memory(CAMERAS, self.dir, now=FRI_14)
        self.assertEqual(carry_memory(CAMERAS, self.dir, now=FRI_14 + 60), {})

    def test_two_old_sites_on_one_channel_move_nothing(self) -> None:
        path = os.path.join(self.dir, "events", "known.json")
        with open(path, encoding="utf-8") as f:
            rows = json.load(f)
        rows.append(dict(rows[0], camera="ameer_tes2_ch3", id="x"))
        with open(path, "w", encoding="utf-8") as f:
            json.dump(rows, f)
        moved = carry_memory(CAMERAS, self.dir, now=FRI_14)
        self.assertEqual(moved[os.path.join("events", "known.json")], {"ameer_week_0_1_ch6": "ameer_v2_ch6"})

    def test_the_box_carries_them_when_its_event_book_starts(self) -> None:
        from unittest import mock  # noqa: PLC0415

        from home_guard_project.box import events  # noqa: PLC0415
        from home_guard_project.box import inference as inf  # noqa: PLC0415
        from home_guard_project.box import memory_rename  # noqa: PLC0415

        with mock.patch.object(events, "_BOOK", None), mock.patch.object(inf, "EVENTS", None),                 mock.patch.object(inf.paths, "state_dir", return_value=self.dir),                 mock.patch.object(inf, "KNOWN_CAMERAS", tuple(CAMERAS)),                 mock.patch.object(memory_rename, "box_cameras", return_value=[]):
            book = inf.start_events({})
            self.assertEqual(book.known_for("ameer_v2_ch3", FRI_14).text, "העובדים של הפרגולה")


if __name__ == "__main__":
    unittest.main()
