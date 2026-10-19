"""Car plates on the box (car_plates.py, known_vehicles.py, events.EventBook.note_vehicles, the inference hooks).

The real case (2026-10-15 / 10-19): the owner named the electrician's car from a live video of the pergola; days
later the same parked car made three "suspicious" alerts. These tests check that alert reads come only from
main-stream pixels at native size, that night / IR / damaged crops are not read, that the full plate stays in the
local reads log and the store while everything else gets the masked plate, and that a car he named is called by his
words in the event and the alert text.
"""
from __future__ import annotations

import json
import os
import shutil
import tempfile
import time
import unittest
from typing import Any, Dict, List
from unittest import mock

import cv2
import numpy as np

from home_guard_project.box import car_plates as cp
from home_guard_project.box import known_vehicles as kv
from home_guard_project.box import plates as pl

MAIN_W, MAIN_H = 2592, 1520
SUB_W, SUB_H = 704, 576


def textured(h: int, w: int, seed: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    small = rng.integers(0, 255, (max(2, h // 16), max(2, w // 16), 3), dtype=np.uint8)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


def scene(seed: int = 1) -> np.ndarray:
    """A colourful main-stream frame (not IR)."""
    return textured(MAIN_H, MAIN_W, seed)


def sub_of(frame: np.ndarray) -> np.ndarray:
    return cv2.resize(frame, (SUB_W, SUB_H), interpolation=cv2.INTER_AREA)


def track(tid: int, box: List[float], t0: float, t1: float, kind: str = "vehicle", step: float = 0.5) -> Dict[str, Any]:
    n = int((t1 - t0) / step) + 1
    return {"id": tid, "kind": kind, "first_seen": t0, "last_seen": t1, "shown": False,
            "boxes": [{"ts": t0 + i * step, "box": list(box)} for i in range(n)]}


class Reader:
    """A plate reader that answers *plate* for every crop (or nothing)."""

    def __init__(self, plate: str = "1609875", ok: bool = True) -> None:
        self.plate, self.ok, self.crops = plate, ok, []

    def available(self) -> bool:
        return self.ok

    def read(self, crops):
        self.crops.extend(crops)
        return [pl.PlateRead(self.plate, 0.99, 55.0) for _ in crops] if self.plate else []


class Tmp(unittest.TestCase):
    def setUp(self) -> None:
        self.dir = tempfile.mkdtemp(prefix="plates_")
        self.addCleanup(shutil.rmtree, self.dir, True)


class SettingsTests(unittest.TestCase):
    def test_no_camera_by_default_and_the_house_list(self) -> None:
        self.assertEqual(cp.settings_of({}).cameras, ())
        s = cp.settings_of({"plate_cameras": ["ameer_v2_ch3", "ameer_v2_ch6"], "plate_device": "cpu"})
        self.assertTrue(s.wants("ameer_v2_ch6"))
        self.assertFalse(s.wants("ameer_v2_ch1"))
        self.assertEqual(s.device, "CPU")
        self.assertFalse(cp.settings_of({"plates": "off", "plate_cameras": "ameer_v2_ch3"}).wants("ameer_v2_ch3"))
        self.assertEqual(cp.settings_of({"plate_cameras": "ameer_v2_ch3, ameer_v2_ch6"}).cameras,
                         ("ameer_v2_ch3", "ameer_v2_ch6"))

    def test_night_is_not_day(self) -> None:
        import datetime as dt

        self.assertTrue(cp.is_day(dt.datetime(2026, 10, 19, 12, 20).timestamp()))
        self.assertFalse(cp.is_day(dt.datetime(2026, 10, 19, 23, 30).timestamp()))


class AlertJobTests(unittest.TestCase):
    def setUp(self) -> None:
        self.frames = [scene(i) for i in range(6)]
        self.start, self.end = 100.0, 105.0
        self.sub = [(self.start + i, sub_of(f)) for i, f in enumerate(self.frames)]

    def test_vehicles_are_cut_from_main_pixels_at_native_size(self) -> None:
        parked = track(7, [0.10, 0.30, 0.35, 0.60], 90.0, 106.0)
        job = cp.alert_job("a1", "ameer_v2_ch3", 103.0, (self.frames, self.start, self.end), self.sub, [parked])
        self.assertIsNotNone(job)
        shot = job.shots[0]
        self.assertEqual(shot.key, "7@90.000")
        self.assertEqual(len(shot.crops), 6)
        c = shot.crops[0]
        x1, x2 = 0.10 * MAIN_W, 0.35 * MAIN_W
        margin = (x2 - x1) * pl.CROP_MARGIN
        self.assertEqual(c.image.shape[1], int(round(x2 + margin)) - int(round(x1 - margin)))    # never resized
        self.assertFalse(np.shares_memory(c.image, self.frames[0]))                                # a copy
        self.assertEqual(shot.frame_size, (MAIN_W, MAIN_H))
        self.assertTrue(all(s is not None and s.shape[1] < c.image.shape[1] for s in shot.subs))

    def test_people_and_tiny_vehicles_give_nothing(self) -> None:
        person = track(1, [0.4, 0.2, 0.5, 0.8], 90.0, 106.0, kind="person")
        tiny = track(2, [0.50, 0.5, 0.51, 0.52], 90.0, 106.0)                 # 26 px wide on the main stream
        self.assertIsNone(cp.alert_job("a1", "c", 103.0, (self.frames, self.start, self.end), self.sub, [person, tiny]))

    def test_the_biggest_vehicles_first_and_at_most_three(self) -> None:
        tracks = [track(i, [0.05 * i, 0.3, 0.05 * i + 0.04 + 0.01 * i, 0.5], 90.0, 106.0) for i in range(1, 6)]
        job = cp.alert_job("a1", "c", 103.0, (self.frames, self.start, self.end), self.sub, tracks)
        self.assertEqual([s.key.split("@")[0] for s in job.shots], ["5", "4", "3"])

    def test_a_vehicle_only_part_of_the_window_is_cut_where_it_was(self) -> None:
        leaving = track(3, [0.2, 0.3, 0.4, 0.6], 100.0, 101.5)
        job = cp.alert_job("a1", "c", 103.0, (self.frames, self.start, self.end), self.sub, [leaving])
        self.assertEqual(len(job.shots[0].crops), 3)        # 100, 101 and 102 s (box_at: within 1 s of its last look)


class ReadShotTests(unittest.TestCase):
    def shot(self, crops: List[np.ndarray], subs: List[Any]) -> cp.Shot:
        return cp.Shot("7@1.000", [pl.VehicleCrop(c, (10, 10, c.shape[1] - 10, c.shape[0] - 10)) for c in crops],
                       [(0.0, 0.0, float(c.shape[1]), float(c.shape[0])) for c in crops], subs, (MAIN_W, MAIN_H))

    def test_ir_and_damaged_crops_are_not_read(self) -> None:
        good = textured(300, 500, 3)
        ir = cv2.cvtColor(cv2.cvtColor(good, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
        damaged = textured(300, 500, 99)                                    # a ghost: not what the sub stream shows
        sub = cv2.resize(good, (140, 84), interpolation=cv2.INTER_AREA)
        reader = Reader()
        plate, agree, counts = cp.read_shot(reader, self.shot([good, ir, damaged, good.copy()], [sub, None, sub, sub]))
        self.assertEqual(plate, "1609875")
        self.assertEqual(agree, 2)
        self.assertEqual((counts["ir"], counts["damaged"], counts["read"]), (1, 1, 2))
        self.assertEqual(len(reader.crops), 2)

    def test_damage_of_the_same_place_is_small(self) -> None:
        good = textured(300, 500, 4)
        self.assertLess(cp.damage(good, cv2.resize(good, (140, 84))), 0.025)
        self.assertEqual(cp.damage(good, None), 0.0)

    def test_one_read_is_no_plate(self) -> None:
        good = textured(300, 500, 5)
        plate, agree, _ = cp.read_shot(Reader(), self.shot([good], [None]))
        self.assertIsNone(plate)


class ServiceTests(Tmp):
    def service(self, reader: Any = None, detector: Any = None, now: float = 1_792_401_700.0) -> cp.PlateService:
        self.store = kv.store_in(self.dir)
        self.clock = {"now": now}
        return cp.PlateService(cp.PlateSettings(True, ("ameer_v2_ch3",), "CPU", 5), reader=reader or Reader(),
                               detector=detector, store=self.store,
                               reads_path=os.path.join(self.dir, cp.READS_NAME), now=lambda: self.clock["now"],
                               start=False)

    def job(self, key: str = "ameer_v2_ch3_1792401641_alert") -> cp.Job:
        good = textured(300, 500, 6)
        crops = [pl.VehicleCrop(good.copy(), (10, 10, 490, 290)) for _ in range(3)]
        return cp.Job(key, "ameer_v2_ch3", 1_792_401_641.0,
                      shots=[cp.Shot("7@1.000", crops, [(0, 0, 500, 300)] * 3, [None] * 3, (MAIN_W, MAIN_H))])

    def test_a_named_car_is_matched_and_the_full_plate_stays_in_the_local_log(self) -> None:
        svc = self.service()
        self.store.add("1609875", "הרכב של החשמלאי מנסור", "לבן", "ameer_v2_ch3", now=1.0)
        out = svc.run_once(self.job())
        v = out["vehicles"]["7@1.000"]
        self.assertEqual((v["plate"], v["label"]), ("1609875", "הרכב של החשמלאי מנסור"))
        with open(svc.reads_path, encoding="utf-8") as f:
            logged = json.loads(f.readline())
        self.assertEqual(logged["vehicles"]["7@1.000"]["plate"], "1609875")
        meta = cp.meta_record(out)
        self.assertEqual(meta["vehicles"]["7@1.000"]["plate"], "****875")
        self.assertNotIn("1609875", json.dumps(meta))
        fields = cp.entity_fields(out)["7@1.000"]
        self.assertEqual(fields["plate_masked"], "****875")
        self.assertEqual(fields["vehicle_label"], "הרכב של החשמלאי מנסור")
        self.assertNotIn("1609875", json.dumps(fields, ensure_ascii=False))

    def test_a_restart_still_finds_the_alerts_plate_in_the_log(self) -> None:
        svc = self.service()
        svc.run_once(self.job("a1"))
        again = self.service()
        found = again.result("a1")
        self.assertTrue(found["from_log"])
        self.assertEqual(found["vehicles"]["7@1.000"]["plate"], "1609875")

    def test_old_reads_are_pruned(self) -> None:
        svc = self.service()
        svc.run_once(self.job("old"))
        self.clock["now"] += cp.KEEP_DAYS * 86400.0 + 10
        svc.run_once(self.job("new"))
        svc.prune()
        with open(svc.reads_path, encoding="utf-8") as f:
            keys = [json.loads(line)["key"] for line in f]
        self.assertEqual(keys, ["new"])

    def test_without_the_package_nothing_is_read_and_it_says_why_once(self) -> None:
        svc = self.service(reader=Reader(ok=False))
        with self.assertLogs("box.car_plates", level="INFO") as said:
            first = svc.run_once(self.job("a"))
            svc.run_once(self.job("b"))
        self.assertIn("fast-alpr", first["skipped"])
        self.assertEqual(sum("nothing read" in line for line in said.output), 1)
        self.assertFalse(os.path.exists(svc.reads_path))

    def test_a_full_queue_drops_the_oldest_job(self) -> None:
        svc = self.service()
        jobs = [self.job(f"a{i}") for i in range(cp.QUEUE_MAX + 1)]
        for j in jobs:
            svc.submit(j)
        self.assertTrue(jobs[0].done.is_set())
        self.assertEqual(jobs[0].result["skipped"], "queue full")
        self.assertEqual([j.key for j in svc._queue], [f"a{i}" for i in range(1, cp.QUEUE_MAX + 1)])

    def test_the_worker_thread_answers_a_waiting_caller(self) -> None:
        svc = cp.PlateService(cp.PlateSettings(True, ("c",), "CPU", 5), reader=Reader(), store=kv.store_in(self.dir),
                              reads_path=os.path.join(self.dir, cp.READS_NAME))
        svc.submit(self.job("w1"))
        self.assertEqual(svc.result("w1", wait=10.0)["vehicles"]["7@1.000"]["plate"], "1609875")

    def test_a_live_photo_is_read_from_its_main_stream_copy(self) -> None:
        small = os.path.join(self.dir, "ameer_v2_ch3_1792401642_e5b0723f.jpg")
        full = os.path.join(self.dir, "ameer_v2_ch3_1792401642_e5b0723f_full.jpg")
        cv2.imwrite(small, textured(562, 960, 7))
        cv2.imwrite(full, scene(8))
        seen: List[Any] = []

        def detect(frame):
            seen.append(frame.shape)
            return [(0.1, 0.3, 0.35, 0.6)]

        svc = self.service(detector=lambda: detect)
        out = svc.run_once(cp.Job(small, "ameer_v2_ch3", 1.0, kind="media", media=small))
        self.assertEqual(seen, [(MAIN_H, MAIN_W, 3)])
        self.assertEqual(out["vehicles"]["v1"]["plate"], "1609875")
        self.assertEqual(out["vehicles"]["v1"]["crops"], 3)        # one picture: three margins, so a vote can form

    def test_a_live_video_is_read_over_its_frames(self) -> None:
        path = os.path.join(self.dir, "ameer_v2_ch3_1_x_live.mp4")
        writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 5, (640, 376))
        for i in range(12):
            writer.write(textured(376, 640, 20 + i))
        writer.release()
        svc = self.service(detector=lambda: (lambda f: [(0.1, 0.3, 0.5, 0.8)]))
        out = svc.run_once(cp.Job(path, "ameer_v2_ch3", 1.0, kind="media", media=path))
        self.assertEqual(out["vehicles"]["v1"]["crops"], cp.MEDIA_FRAMES)

    def test_read_media_off_when_plates_are_off(self) -> None:
        svc = cp.PlateService(cp.PlateSettings(False), reader=Reader(), start=False)
        self.assertEqual(svc.read_media("x.mp4", "c", wait=0.1)["skipped"], "plates off")


class NamingTests(unittest.TestCase):
    def test_a_named_car_replaces_its_look_in_the_alert_lines(self) -> None:
        result = {"vehicles": {"7@1.000": {"plate": "1609875", "reads": 4, "label": "הרכב של החשמלאי מנסור"},
                               "9@2.000": {"plate": "5654230", "reads": 3}}}
        names = cp.named_ids(result, {"7@1.000": "CAR1", "9@2.000": "CAR2", "3@0.500": "P1"})
        self.assertEqual(names, {"CAR1": "הרכב של החשמלאי מנסור"})
        lines = cp.name_entities([{"id": "P1", "appearance": "אדם בקפוצ'ון", "action": "פותח תא מטען"},
                                  {"id": "CAR1", "appearance": "רכב לבן", "action": "חונה בשביל הכניסה"}], names)
        self.assertEqual(lines[1], {"id": "CAR1", "appearance": "הרכב של החשמלאי מנסור", "action": "חונה בשביל הכניסה"})
        self.assertEqual(lines[0]["appearance"], "אדם בקפוצ'ון")
        from home_guard_project.box.describer import compose

        text = compose("suspicious", "פרגולה", "12:32", "", lines, "", "he")
        self.assertIn("CAR1 · הרכב של החשמלאי מנסור: חונה בשביל הכניסה.", text)

    def test_a_drawn_car_with_no_line_gets_its_name_alone(self) -> None:
        lines = cp.name_entities([], {"CAR1": "הרכב של החשמלאי"}, drawn=["P1", "CAR1"])
        self.assertEqual(lines, [{"id": "CAR1", "appearance": "הרכב של החשמלאי", "action": ""}])
        self.assertEqual(cp.name_entities([], {"CAR2": "x"}, drawn=["CAR1"]), [])


class KnownVehiclesTests(Tmp):
    def test_add_match_within_one_digit_and_mask(self) -> None:
        store = kv.store_in(self.dir)
        rec = store.add("16-098-75", "הרכב של החשמלאי מנסור", "לבן", "ameer_v2_ch3", by="Hello_24",
                        source="ameer_v2_ch3_1792045079_949825b2_live.mp4", now=10.0)
        self.assertEqual(rec["plate"], "1609875")
        self.assertFalse(rec["already"])
        self.assertEqual(store.match("1605875")["id"], rec["id"])                # one digit off
        self.assertIsNone(store.match("1605375"))
        self.assertEqual(kv.masked_line(rec), "הרכב הלבן ••••875 = הרכב של החשמלאי מנסור")
        with open(store.path, encoding="utf-8") as f:
            self.assertEqual(json.load(f)[0]["label"], "הרכב של החשמלאי מנסור")

    def test_the_same_words_again_change_nothing_and_new_words_relabel_with_undo(self) -> None:
        store = kv.store_in(self.dir)
        first = store.add("1609875", "הרכב של החשמלאי", now=1.0)
        self.assertTrue(store.add("1609875", "הרכב של החשמלאי", now=2.0)["already"])
        second = store.add("1609875", "הרכב של מנסור", now=3.0)
        self.assertEqual(second["replaced"]["id"], first["id"])
        self.assertEqual([r["label"] for r in store.all()], ["הרכב של מנסור"])
        self.assertIsNotNone(store.remove(second["id"]))
        self.assertIsNotNone(store.restore(first["id"]))
        self.assertEqual([r["label"] for r in store.all()], ["הרכב של החשמלאי"])

    def test_bad_input_and_a_broken_file(self) -> None:
        store = kv.store_in(self.dir)
        with self.assertRaises(ValueError):
            store.add("12345", "x")
        with self.assertRaises(ValueError):
            store.add("1234567", "  ")
        with open(store.path, "w", encoding="utf-8") as f:
            f.write("{not json")
        self.assertEqual(store.all(), [])
        self.assertTrue(os.path.exists(store.path + ".broken"))


class EventBookTests(Tmp):
    def test_a_named_car_labels_the_events_vehicle_and_never_overrides_his_own_words(self) -> None:
        from home_guard_project.box.events import EventBook, Session

        book = EventBook(self.dir)
        s = Session(id="s1", camera="ameer_v2_ch6", opened=1.0, last_active=1.0)
        s.entities = [{"id": "CAR1", "kind": "vehicle", "tracks": ["7@1.000"], "owner_label": ""},
                      {"id": "CAR2", "kind": "vehicle", "tracks": ["9@2.000"], "owner_label": "הטנדר של הגנן"},
                      {"id": "P1", "kind": "person", "tracks": ["7@1.000"], "owner_label": ""}]
        book._open["ameer_v2_ch6"] = s
        book._by_alert["a1"] = "s1"
        fields = {"7@1.000": {"plate_masked": "****875", "plate_reads": 4, "vehicle_id": "V1",
                              "vehicle_label": "הרכב של החשמלאי מנסור"},
                  "9@2.000": {"plate_masked": "****230", "plate_reads": 2, "vehicle_id": "V2",
                              "vehicle_label": "הרכב של אמא"}}
        self.assertEqual(book.note_vehicles("a1", fields), ["CAR1", "CAR2"])
        car1, car2, p1 = s.entities
        self.assertEqual((car1["owner_label"], car1["plate_masked"]), ("הרכב של החשמלאי מנסור", "****875"))
        self.assertEqual(car2["owner_label"], "הטנדר של הגנן")
        self.assertNotIn("plate_masked", p1)
        self.assertEqual(book.note_vehicles("nope", fields), [])


class InferenceHookTests(Tmp):
    def test_the_window_is_cut_off_the_loop_and_let_go(self) -> None:
        from home_guard_project.box import inference as inf

        svc = cp.PlateService(cp.PlateSettings(True, ("ameer_v2_ch3",), "CPU", 5), reader=Reader(),
                              store=kv.store_in(self.dir), reads_path=os.path.join(self.dir, cp.READS_NAME),
                              start=False)
        frames = [scene(i) for i in range(3)]
        job = inf.AlertJob(camera="ameer_v2_ch3", stem="ameer_v2_ch3_1792401641_alert", ts=1_792_401_641.0)
        job.track_boxes = {"tracks": [track(7, [0.1, 0.3, 0.35, 0.6], 1_792_401_630.0, 1_792_401_650.0)]}
        job.plate_window = (frames, 1_792_401_637.0, 1_792_401_647.0)
        with mock.patch.object(inf, "PLATES", svc):
            inf._queue_plates(job, [])
        self.assertIsNone(job.plate_window)
        self.assertEqual([j.key for j in svc._queue], ["ameer_v2_ch3_1792401641_alert"])

    def test_night_and_unlisted_cameras_are_not_read(self) -> None:
        import datetime as dt

        from home_guard_project.box import inference as inf

        svc = cp.PlateService(cp.PlateSettings(True, ("ameer_v2_ch3",), "CPU", 5), reader=Reader(), start=False)
        day = dt.datetime(2026, 10, 19, 12, 20).timestamp()
        night = dt.datetime(2026, 10, 19, 23, 20).timestamp()
        with mock.patch.object(inf, "PLATES", svc):
            self.assertTrue(inf._plates_wanted(inf.AlertJob(camera="ameer_v2_ch3", stem="a", ts=day)))
            self.assertFalse(inf._plates_wanted(inf.AlertJob(camera="ameer_v2_ch3", stem="a", ts=night)))
            self.assertFalse(inf._plates_wanted(inf.AlertJob(camera="ameer_v2_ch1", stem="a", ts=day)))
        with mock.patch.object(inf, "PLATES", None):
            self.assertFalse(inf._plates_wanted(inf.AlertJob(camera="ameer_v2_ch3", stem="a", ts=day)))

    def test_the_alert_text_names_the_car_by_the_describers_own_ids(self) -> None:
        import datetime as dt

        from home_guard_project.box import inference as inf

        ts = dt.datetime(2026, 10, 19, 12, 32).timestamp()
        svc = cp.PlateService(cp.PlateSettings(True, ("ameer_v2_ch3",), "CPU", 5), reader=Reader(), start=False)
        svc._results["ameer_v2_ch3_1_alert"] = {"key": "ameer_v2_ch3_1_alert", "vehicles": {
            "12@5.000": {"plate": "1609875", "reads": 3, "label": "הרכב של החשמלאי מנסור"}}}
        tracks = [{"id": 3, "kind": "person", "first_seen": 4.0, "shown": True},
                  {"id": 12, "kind": "vehicle", "first_seen": 5.0, "shown": False}]
        job = inf.AlertJob(camera="ameer_v2_ch3", stem="ameer_v2_ch3_1_alert", ts=ts)
        lines = [{"id": "P1", "appearance": "אדם", "action": "עומד"}, {"id": "CAR1", "appearance": "רכב סדאן לבן",
                                                                      "action": ""}]
        with mock.patch.object(inf, "PLATES", svc):
            named = inf._named_vehicles(job, "ameer_v2_ch3_1_alert", tracks, {}, lines, ["P1", "CAR1"])
        self.assertEqual(named[1]["appearance"], "הרכב של החשמלאי מנסור")
        self.assertEqual(named[0], lines[0])


class PlatesNeverFromVlmTests(Tmp):
    """Owner rule 2026-10-19: a plate is read by the plate-recognition tool (plates.PlateReader: detector + OCR,
    digits only) and kept as a number; no VLM reads or guesses one - not the Eye, not the assistant's looks."""

    def test_a_plate_like_number_in_a_vlm_answer_is_blanked_times_and_dates_stay(self) -> None:
        text = "White car, plate 16-098-75 (also 1609875), at 12:20 on 15/10/2026; call 0501234567"
        out = pl.scrub(text)
        self.assertNotIn("16-098-75", out)
        self.assertNotIn("1609875", out)
        for kept in ("12:20", "15/10/2026", "0501234567"):
            self.assertIn(kept, out)
        self.assertEqual(pl.scrub({"a": ["רכב 56-542-30"]}), {"a": ["רכב " + pl.SCRUBBED]})

    def test_the_eye_and_the_assistants_vision_answers_are_scrubbed_at_the_parser(self) -> None:
        from home_guard_project.box import inference as inf

        parsed = inf.parse_vlm_json('noise {"summary": "A white Subaru 16-098-75 parks", "label": "normal"} end')
        self.assertEqual(parsed["summary"], f"A white Subaru {pl.SCRUBBED} parks")
        self.assertNotIn("1609875", inf.owner_guard("רכב 1609875 בחניה", "ameer_v2_ch3", "he"))

    def test_the_vision_model_is_never_asked_for_a_plate(self) -> None:
        from home_guard_project.box.brain.vision import PLATE_NOT_BY_VISION, Vision

        def complete(prompt, images, schema):
            raise AssertionError("the vision model was asked about a plate")

        vision = Vision(complete, "x")
        for question in ("what is the licence plate?", "מה מספר הרכב?", "תקרא את הלוחית", "read the plate"):
            self.assertEqual(vision.ask("ameer_v2_ch3", [b"jpg"], question)["error"], PLATE_NOT_BY_VISION)
            self.assertEqual(vision.look("ameer_v2_ch3", [b"jpg"], False, question)["error"], PLATE_NOT_BY_VISION)
        self.assertFalse(pl.asks_plate("what is in his hand, a plate?"))
        self.assertFalse(pl.asks_plate("מה הוא מחזיק ביד?"))

    def test_no_vlm_prompt_asks_for_a_plate(self) -> None:
        from home_guard_project import prompts

        vlm = [n for n in prompts.names() if n.startswith(("eye_", "brain_vision", "describer_", "brain_look"))]
        self.assertGreater(len(vlm), 20)
        for name in vlm:
            self.assertFalse(pl.asks_plate(prompts.load(name)), name)

    def test_a_plate_is_stored_as_digits_only(self) -> None:
        store = kv.store_in(self.dir)
        rec = store.add("56-542-30", "הרכב של החשמלאי", now=1.0)
        with open(store.path, encoding="utf-8") as f:
            saved = json.load(f)[0]["plate"]
        self.assertEqual(saved, "5654230")
        self.assertTrue(isinstance(saved, str) and saved.isdigit())
        self.assertEqual(rec["plate"], saved)

    def test_service_plates_come_from_the_plate_reader(self) -> None:
        reader = Reader("1609875")
        svc = cp.PlateService(cp.PlateSettings(True, ("c",), "CPU", 5), reader=reader, start=False)
        good = textured(300, 500, 6)
        job = cp.Job("a", "c", 1.0, shots=[cp.Shot("k", [pl.VehicleCrop(good.copy(), (10, 10, 490, 290))] * 3,
                                                   [(0, 0, 500, 300)] * 3, [None] * 3, (MAIN_W, MAIN_H))])
        out = svc.run_once(job)
        self.assertEqual(len(reader.crops), 3)                 # the reader saw the crops ...
        self.assertEqual(out["vehicles"]["k"]["plate"], "1609875")   # ... and its digits are the plate
        self.assertIsInstance(out["vehicles"]["k"]["plate"], str)


if __name__ == "__main__":
    unittest.main()
