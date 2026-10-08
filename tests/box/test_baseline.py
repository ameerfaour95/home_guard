"""Task 2.9 (2026-10-08): the per-camera baseline - what is usual at each camera (baseline.py), its activity tags
(activities.py) and the two memory layers (camera_profiles.py: the house memory and one personality per camera).

Owner: "something usually happens at the main entrance, but if it happens in the backyard it's not usual" and
"knocking on the door is usual at the main door but not at the back door". Local data only; the real-history test
is skipped when the owner's folder is not on this machine."""

from __future__ import annotations

import datetime as dt
import io
import json
import os
import shutil
import tempfile
import unittest
from typing import Any, Dict, List

from home_guard_project.box import activities, baseline as bl
from home_guard_project.box.camera_profiles import CameraProfiles, house_profile, household_of, parse_rule

MAIN, BACK, PERGOLA = "ameer_week_0_1_ch6", "ameer_week_0_1_ch1", "ameer_week_0_1_ch3"
NAMES = {MAIN: {"he": "הכניסה הראשית", "en": "the main entrance"},
         BACK: {"he": "הדלת האחורית", "en": "the back door"},
         PERGOLA: {"he": "הפרגולה", "en": "the pergola"}}
FIRST = dt.datetime(2026, 9, 1)
REAL = r"C:/Users/ameer/Ameer/home_guard_data/raw"


def names(camera: str, lang: str) -> str:
    return NAMES.get(camera, {}).get("he" if lang == "he" else "en", "Camera ?")


def at(day: int, hour: int, minute: int = 0) -> float:
    return (FIRST + dt.timedelta(days=day, hours=hour, minutes=minute)).timestamp()


def record(event_id: str, camera: str, start: float, minutes: float, summary: str, people: int = 1,
           parent: str = "", outcome: str = "left", entities: Any = None) -> Dict[str, Any]:
    r = {"event_id": event_id, "camera": camera, "start": start, "end": start + minutes * 60, "parent": parent,
         "people_max": people, "outcome": outcome, "owner_known": [],
         "observations": [{"ts": start, "label": "normal", "people": people, "summary": summary}]}
    if entities is not None:
        r["entities"] = entities
    return r


def history(days: int = 20) -> List[Dict[str, Any]]:
    """The main entrance: a courier knocks every morning at 08:xx. The back door: someone walks across the yard at
    noon. The pergola: workers 09:00-12:30 every day."""
    rows = []
    for d in range(days):
        rows.append(record(f"m{d}", MAIN, at(d, 8, 10), 4, "A courier knocks on the door and leaves a package"))
        rows.append(record(f"b{d}", BACK, at(d, 12, 5), 3, "A man walks across the yard"))
        rows.append(record(f"p{d}", PERGOLA, at(d, 9, 0), 210, "Workers are building the pergola"))
    return rows


class Fixture(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.events_dir = os.path.join(self.tmp, "events")
        os.makedirs(self.events_dir)
        self.ledger = bl.Baseline(os.path.join(self.events_dir, bl.BASELINE_NAME))
        self.profiles = CameraProfiles(os.path.join(self.events_dir, "camera_profiles.json"))
        self.house: Dict[str, Any] = {"state": "home_awake", "expecting": [], "known": [], "facts": [],
                                      "household": household_of([])}

    def historian(self, **settings: Any) -> bl.Historian:
        return bl.Historian(self.ledger, self.profiles, names=names, house=lambda ts: self.house, settings=settings)

    def build(self, days: int = 20, now_day: int = None) -> Dict[str, Any]:
        now_day = days if now_day is None else now_day
        return self.ledger.build_from_archive(history(days), [MAIN, BACK, PERGOLA], now=at(now_day, 3, 30))


class ActivityTagsTest(unittest.TestCase):
    def test_english_summaries(self) -> None:
        self.assertEqual(activities.tags_of("A person knocks on the door"), ["knock"])
        self.assertIn("delivery", activities.tags_of("A courier leaves a package by the gate"))
        self.assertEqual(activities.tags_of("A white car pulls into the driveway and parks"), ["vehicle_move"])
        self.assertIn("at_vehicle", activities.tags_of("A man stands next to the parked car"))
        self.assertIn("working", activities.tags_of("A man mops the floor"))
        self.assertIn("walk_past", activities.tags_of("A person walks past the camera"))

    def test_hebrew_and_prefixes(self) -> None:
        self.assertEqual(activities.tags_of("דופקים בדלת האחורית"), ["knock"])
        self.assertIn("delivery", activities.tags_of("והשליח השאיר חבילה"))
        self.assertIn("kids", activities.tags_of("ילדים משחקים בחצר"))

    def test_negated_mentions_are_not_tags(self) -> None:
        self.assertEqual(activities.tags_of("No animals are present"), [])
        self.assertEqual(activities.tags_of("A car is parked in the driveway"), [])
        self.assertEqual(activities.tags_of("אין שליחים בחצר"), [])
        self.assertEqual(activities.tags_of("אין שליחים בחצר", keep_negated=True), ["delivery"])

    def test_deterministic(self) -> None:
        text = "Two men stand and talk near a pickup truck while a dog walks by"
        self.assertEqual(activities.tags_of(text), activities.tags_of(text))


class BuildTest(Fixture):
    def test_days_of_data_and_hours(self) -> None:
        result = self.build(20)
        self.assertEqual(result["source"], "archive")
        key, days = self.ledger.camera_days(MAIN)
        self.assertEqual(key, MAIN)
        self.assertEqual(len(days), 20)                          # through yesterday, today is not complete
        first = days[FIRST.strftime("%Y-%m-%d")]
        self.assertEqual(first["hours"][8], 1)
        self.assertEqual(sum(first["hours"]), 1)
        self.assertEqual(first["tags"]["knock"], {"8": 1})
        self.assertEqual(first["tags"]["delivery"], {"8": 1})
        self.assertEqual(first["dwell"], [240.0])

    def test_an_event_counts_in_every_hour_it_spans(self) -> None:
        self.build(20)
        _, days = self.ledger.camera_days(PERGOLA)
        hours = days[FIRST.strftime("%Y-%m-%d")]["hours"]
        self.assertEqual(hours[9:13], [1, 1, 1, 1])
        self.assertEqual(sum(hours), 4)

    def test_a_rolled_over_chain_is_one_event(self) -> None:
        rows = [record("a", MAIN, at(0, 10, 0), 30, "Workers clean"),
                record("b", MAIN, at(0, 10, 30), 20, "Workers clean", parent="a")]
        self.ledger.build_from_archive(rows, [MAIN], now=at(1, 3))
        _, days = self.ledger.camera_days(MAIN)
        day = days[FIRST.strftime("%Y-%m-%d")]
        self.assertEqual(day["hours"][10], 1)
        self.assertEqual(day["dwell"], [3000.0])

    def test_no_person_is_not_counted(self) -> None:
        rows = [record("v", MAIN, at(0, 10), 2, "A car drives in", people=0),
                record("p", MAIN, at(0, 11), 2, "A man walks past")]
        self.ledger.build_from_archive(rows, [MAIN], now=at(1, 3))
        _, days = self.ledger.camera_days(MAIN)
        self.assertEqual(sum(days[FIRST.strftime("%Y-%m-%d")]["hours"]), 1)

    def test_areas_from_mapped_entity_paths(self) -> None:
        ent = [{"id": "P1", "kind": "person", "mapped": True, "path": ["gate", "pergola"]}]
        self.ledger.build_from_archive([record("a", MAIN, at(0, 10), 2, "x", entities=ent)], [MAIN], now=at(1, 3))
        _, days = self.ledger.camera_days(MAIN)
        self.assertEqual(days[FIRST.strftime("%Y-%m-%d")]["areas"], {"gate": 1, "pergola": 1})

    def test_rebuild_keeps_days_the_archive_no_longer_has(self) -> None:
        self.build(20)
        later = [r for r in history(20) if r["start"] >= at(15, 0)]           # the archive pruned the first 15
        self.ledger.build_from_archive(later, [MAIN, BACK, PERGOLA], now=at(20, 3, 30))
        _, days = self.ledger.camera_days(MAIN)
        self.assertEqual(len(days), 20)
        self.assertEqual(days[FIRST.strftime("%Y-%m-%d")]["hours"][8], 1)


class RemapTest(Fixture):
    def test_old_site_prefix_maps_by_channel(self) -> None:
        self.assertEqual(bl.remap("ameer_tes2_ch6", [MAIN, BACK]), MAIN)
        self.assertIsNone(bl.remap("ameer_tes2_ch6", [MAIN, "other_site_ch6"]))     # two ch6: not guessed
        self.assertIsNone(bl.remap("back_door", [MAIN, BACK]))                     # no _chN
        self.assertEqual(bl.remap("back_door", []), "back_door")                   # no list: kept as it is

    def test_ledger_moves_to_the_renamed_camera(self) -> None:
        old = [record(f"o{d}", "ameer_tes2_ch6", at(d, 8), 3, "A courier knocks") for d in range(5)]
        self.ledger.build_from_archive(old, [], now=at(5, 3))
        self.assertEqual(self.ledger.cameras(), ["ameer_tes2_ch6"])
        key, days = self.ledger.camera_days(MAIN)          # found by the channel before any rebuild
        self.assertEqual((key, len(days)), ("ameer_tes2_ch6", 5))
        self.ledger.build_from_archive([], [MAIN, BACK], now=at(6, 3))
        self.assertIn(MAIN, self.ledger.cameras())
        self.assertNotIn("ameer_tes2_ch6", self.ledger.cameras())
        self.assertEqual(len(self.ledger.camera_days(MAIN)[1]), 5)


def write_meta(folder: str, camera: str, ts: float, **alert: Any) -> None:
    day = dt.datetime.fromtimestamp(ts).strftime("%Y-%m-%d")
    path = os.path.join(folder, "meta", camera, day, f"{camera}_{int(ts)}_alert.meta.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    meta = {"camera_name": camera, "kind": "alert", "trigger_ts": ts, "clip_start_ts": ts - 3, "clip_end_ts": ts + 7,
            "yolo": {"class_counts": {"person": 1}},
            "alert": dict({"label": "normal", "people": 1, "summary": "A man walks past the gate"}, **alert)}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(meta, f)


class ImportTest(Fixture):
    def test_production_meta_sessions_and_false_positives(self) -> None:
        raw = os.path.join(self.tmp, "production_old")
        write_meta(raw, "ameer_tes2_ch6", at(0, 9, 0))
        write_meta(raw, "ameer_tes2_ch6", at(0, 9, 2))                      # 2 min later: the same event
        write_meta(raw, "ameer_tes2_ch6", at(0, 15, 0))
        write_meta(raw, "ameer_tes2_ch6", at(0, 16, 0), false_positive=True)
        write_meta(raw, "ameer_tes2_ch6", at(0, 17, 0), people=0)           # the VLM saw nobody
        write_meta(raw, "ameer_tes2_ch6", at(2, 9, 0))
        result = self.ledger.import_raw(raw, [MAIN, BACK], now=at(3, 3))
        self.assertEqual(result["production"]["clips"], 6)
        _, days = self.ledger.camera_days(MAIN)
        self.assertEqual(sorted(days), [FIRST.strftime("%Y-%m-%d"), (FIRST + dt.timedelta(days=1)).strftime("%Y-%m-%d"),
                                        (FIRST + dt.timedelta(days=2)).strftime("%Y-%m-%d")])   # day 1: watched, empty
        first = days[FIRST.strftime("%Y-%m-%d")]
        self.assertEqual(first["hours"][9], 1)
        self.assertEqual(first["hours"][15], 1)
        self.assertEqual(sum(first["hours"]), 2)
        self.assertEqual(first["dwell"], [127.0])                   # first trigger to the last clip's end

    def test_an_import_never_overwrites_the_archive(self) -> None:
        self.build(5)
        raw = os.path.join(self.tmp, "production_old")
        write_meta(raw, MAIN, at(0, 20, 0))
        self.ledger.import_raw(raw, [MAIN], now=at(6, 3))
        _, days = self.ledger.camera_days(MAIN)
        self.assertEqual(days[FIRST.strftime("%Y-%m-%d")]["src"], "archive")
        self.assertEqual(days[FIRST.strftime("%Y-%m-%d")]["hours"][20], 0)

    def test_collector_meta(self) -> None:
        raw = os.path.join(self.tmp, "dataset_old", "meta", "ameer_tes2_ch1", "2026-09-01")
        os.makedirs(raw)
        for i, (ts, person) in enumerate(((at(0, 19, 0), 2), (at(0, 19, 1), 1), (at(0, 22, 0), 0))):
            with open(os.path.join(raw, f"c{i}.meta.json"), "w", encoding="utf-8") as f:
                json.dump({"camera_name": "ameer_tes2_ch1", "kind": "trigger", "clip_start_ts": ts,
                           "clip_end_ts": ts + 10, "yolo": {"class_counts": {"person": person}},
                           "model_response": "A dog walks by"}, f)
        self.ledger.import_raw(os.path.join(self.tmp, "dataset_old"), [MAIN, BACK], now=at(1, 3))
        _, days = self.ledger.camera_days(BACK)
        day = days[FIRST.strftime("%Y-%m-%d")]
        self.assertEqual((day["hours"][19], day["hours"][22], day["src"]), (1, 0, "collector"))
        self.assertIn("animal", day["tags"])


class SurpriseTest(Fixture):
    def test_rare_hour_with_enough_history(self) -> None:
        self.build(20)
        out = self.historian().surprise(BACK, at(20, 3, 10))
        self.assertEqual(out["rarity"], "rare")
        self.assertEqual(out["said_by"], "time")
        self.assertEqual(out["seen_in_last_30_days_same_bucket"], 0)
        self.assertGreaterEqual(out["days_of_data"], 14)
        self.assertTrue(out["raise"])
        self.assertIn("לא רגיל למצלמה הזו בשעה הזו", out["text_he"])
        self.assertIn("אף פעם ב-14 הלילות האחרונים", out["text_he"])
        self.assertIn("Not usual for this camera at this hour", out["text_en"])

    def test_once_in_the_window_is_still_rare(self) -> None:
        rows = history(20) + [record("odd", BACK, at(5, 3, 0), 2, "A man walks across the yard")]
        self.ledger.build_from_archive(rows, [MAIN, BACK, PERGOLA], now=at(20, 3, 30))
        out = self.historian().surprise(BACK, at(20, 3, 10))
        self.assertEqual((out["rarity"], out["seen_in_last_30_days_same_bucket"]), ("rare", 1))
        self.assertIn("פעם אחת ב-14 הלילות האחרונים", out["text_he"])

    def test_common_hour(self) -> None:
        self.build(20)
        out = self.historian().surprise(MAIN, at(20, 8, 20))
        self.assertEqual(out["rarity"], "common")
        self.assertFalse(out["raise"])
        self.assertGreater(out["expected_per_hour"], 0.3)
        self.assertIn("כמעט כל יום בשעה הזו", out["text_he"])

    def test_not_enough_history_is_unknown(self) -> None:
        self.build(10)
        out = self.historian().surprise(BACK, at(10, 3, 10))
        self.assertEqual(out["rarity"], "unknown")
        self.assertFalse(out["raise"])
        self.assertIn("עדיין אין מספיק היסטוריה", out["text_he"])
        self.assertEqual(self.historian(baseline_min_days=7).surprise(BACK, at(10, 3, 10))["rarity"], "rare")

    def test_thresholds(self) -> None:
        # Two similar weekday nights (Monday 7.9, Tuesday 8.9) in the window: uncommon; one: rare. 21.9 is a Monday:
        # 14 weekday windows (Friday and Saturday are the weekend).
        rows = history(20) + [record("x1", BACK, at(6, 3), 2, "walks"), record("x2", BACK, at(7, 3), 2, "walks")]
        self.ledger.build_from_archive(rows, [MAIN, BACK, PERGOLA], now=at(20, 3, 30))
        self.assertEqual(self.historian().surprise(BACK, at(20, 3, 10))["rarity"], "uncommon")

    def test_the_owners_knock_example(self) -> None:
        """A knock at the main entrance is common; the same knock at the back door is rare and says where it is
        usual - even at an hour the back door often sees people."""
        self.build(20)
        h = self.historian()
        main = h.surprise(MAIN, at(20, 8, 15), ["knock"])
        self.assertEqual(main["layers"]["activity"]["rarity"], "common")
        self.assertEqual(main["rarity"], "common")
        back = h.surprise(BACK, at(20, 12, 10), ["knock"])
        self.assertEqual(back["layers"]["time"]["rarity"], "common")
        self.assertEqual(back["layers"]["activity"]["rarity"], "rare")
        self.assertEqual((back["rarity"], back["said_by"]), ("rare", "activity"))
        self.assertTrue(back["raise"])
        self.assertEqual(back["text_he"], "דפיקה בדלת האחורית - לא רגיל למצלמה הזו (בדרך כלל רק בכניסה הראשית)")
        self.assertIn("usually only at the main entrance", back["text_en"])

    def test_taught_facts_make_it_rare_without_history(self) -> None:
        self.build(3)
        self.profiles.add_fact(BACK, "בחצר האחורית אין אף אחד בלילה")
        h = self.historian()
        night = h.surprise(BACK, at(3, 23, 30), ["walk_past"])
        self.assertEqual((night["rarity"], night["said_by"], night["raise"]), ("rare", "facts", True))
        self.assertIn("אין אף אחד בלילה", night["text_he"])
        self.assertEqual(h.surprise(BACK, at(3, 12, 30), ["walk_past"])["layers"]["facts"].get("rarity"), "")

    def test_only_here_elsewhere_and_family_only(self) -> None:
        self.build(3)
        self.profiles.add_fact(MAIN, "שליחים מגיעים רק לכניסה הראשית")
        h = self.historian()
        back = h.surprise(BACK, at(3, 12), ["delivery"])
        self.assertEqual(back["said_by"], "facts")
        self.assertIn("(בדרך כלל רק בכניסה הראשית)", back["text_he"])
        self.assertEqual(h.surprise(MAIN, at(3, 8), ["delivery"])["rarity"], "unknown")   # it IS usual there
        self.profiles.add_fact(BACK, "בדלת האחורית משתמשים רק אנחנו")
        self.assertEqual(h.surprise(BACK, at(3, 12), ["knock"])["said_by"], "facts")
        self.assertNotEqual(h.surprise(BACK, at(3, 12), ["walk_past"])["said_by"], "facts")
        self.house["state"] = "away"
        self.assertEqual(h.surprise(BACK, at(3, 12), ["walk_past"])["said_by"], "facts")

    def test_the_house_memory_explains(self) -> None:
        self.build(20)
        h = self.historian()
        self.house["known"] = [{"who": "הגנן", "camera": "", "until": at(21, 0)}]
        out = h.surprise(BACK, at(20, 3, 10))
        self.assertEqual(out["rarity"], "rare")
        self.assertFalse(out["raise"])
        self.assertEqual(out["explained_by"], ["known: הגנן"])
        self.house["known"] = []
        self.house["household"] = household_of(["יש לנו כלב"])
        self.assertFalse(h.surprise(BACK, at(20, 3, 10), ["animal"])["raise"])
        self.assertTrue(h.surprise(BACK, at(20, 3, 10), ["animal", "at_window"])["raise"])
        self.house["expecting"] = [{"text": "plumber", "camera": BACK}]
        self.assertFalse(h.surprise(BACK, at(20, 3, 10))["raise"])

    def test_renamed_camera_is_found(self) -> None:
        self.build(20)
        out = self.historian().surprise("ameer_new_site_ch1", at(20, 3, 10))
        self.assertEqual(out["rarity"], "rare")

    def test_never_raises_on_a_broken_ledger(self) -> None:
        with open(self.ledger.path, "w", encoding="utf-8") as f:
            f.write("{broken")
        self.assertEqual(self.historian().surprise(BACK, at(20, 3))["rarity"], "unknown")


class DescribeTest(Fixture):
    def test_what_is_usual_here(self) -> None:
        self.build(20)
        h = self.historian()
        main = h.describe(MAIN, None, tag="knock", lang="he")
        self.assertEqual(main["camera"], "הכניסה הראשית")
        self.assertEqual(main["busiest_hours"][0], "08:00-09:00")
        self.assertEqual(main["activities"]["knock"]["often"], "בערך פעם ביום")
        self.assertEqual(main["asked_activity"]["elsewhere"]["הדלת האחורית"], "אף פעם ב-20 הימים האחרונים")
        self.assertEqual(main["typical_stay_min"], 4.0)
        back = h.describe(BACK, None, phase="night", lang="he")
        self.assertEqual(back["asked_part_of_day"]["days_with_people"], 0)
        dumped = json.dumps([main, back], ensure_ascii=False)
        self.assertNotIn("ameer_", dumped)
        self.assertNotIn("_ch", dumped)

    def test_role_from_setup_owner_or_statistics(self) -> None:
        self.build(20)
        self.assertEqual(self.historian().describe(MAIN, lang="en")["role"], "an entrance")
        self.assertEqual(self.historian().describe(MAIN, lang="en")["role_from"],
                         "proposed from the statistics (not confirmed)")
        self.assertEqual(self.historian().describe(PERGOLA, lang="en")["role"], "a work area")
        self.profiles.set_role(PERGOLA, "private")
        self.assertEqual(self.historian().describe(PERGOLA, lang="en")["role_from"], "the owner")
        self.assertEqual(self.historian(camera_roles={PERGOLA: "street"}).describe(PERGOLA, lang="en")["role"],
                         "street-facing")


class ModeSettingsTest(unittest.TestCase):
    def test_box_yaml_keys(self) -> None:
        self.assertEqual(bl.mode_of({}), "shadow")
        self.assertEqual(bl.mode_of({"baseline_alerts": "on"}), "on")
        self.assertEqual(bl.mode_of({"baseline_alerts": False}), "off")
        self.assertEqual(bl.mode_of({"baseline_alerts": "loud"}), "shadow")
        self.assertEqual(bl.min_days_of({}), 14)
        self.assertEqual(bl.min_days_of({"baseline_min_days": "7"}), 7)
        self.assertEqual(bl.weekend_of({}), (4, 5))
        self.assertEqual(bl.weekend_of({"baseline_weekend_days": "sat,sun"}), (5, 6))

    def test_next_build_time(self) -> None:
        now = dt.datetime(2026, 10, 8, 23, 0).timestamp()
        self.assertAlmostEqual(bl._seconds_until("03:30", now), 4.5 * 3600, delta=2)


class ProfilesTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.store = CameraProfiles(os.path.join(self.tmp, "camera_profiles.json"))

    def test_rules_from_the_owners_words(self) -> None:
        self.assertEqual(parse_rule("בדלת האחורית משתמשים רק אנחנו")["kind"], "family_only")
        self.assertEqual(parse_rule("שליחים מגיעים רק לכניסה הראשית"),
                         {"kind": "only_here", "tags": ["delivery"], "phases": []})
        self.assertEqual(parse_rule("בחצר האחורית אין אף אחד בלילה"),
                         {"kind": "nobody", "tags": [], "phases": ["night", "late_night"]})
        self.assertEqual(parse_rule("Deliveries only come to the main entrance")["kind"], "only_here")
        self.assertEqual(parse_rule("המצלמה מכוונת לשער")["kind"], "note")

    def test_facts_never_expire_and_survive_a_rename(self) -> None:
        fact = self.store.add_fact("ameer_tes2_ch1", "בחצר האחורית אין אף אחד בלילה", by="Ameer", now=1.0)
        self.assertTrue(fact["id"].startswith("CF"))
        self.assertTrue(self.store.add_fact("ameer_tes2_ch1", "בחצר  האחורית אין אף אחד בלילה")["already"])
        self.assertEqual([f["text"] for f in self.store.facts(BACK)], ["בחצר האחורית אין אף אחד בלילה"])
        cam, removed = self.store.remove_fact(fact["id"])
        self.assertEqual((cam, self.store.facts(BACK)), ("ameer_tes2_ch1", []))
        self.assertTrue(self.store.restore_fact(cam, removed))
        self.assertEqual(self.store.facts(BACK)[0]["id"], fact["id"])
        self.assertIsNone(self.store.remove_fact("CFnope"))

    def test_house_profile_reads_the_existing_stores(self) -> None:
        events = os.path.join(self.tmp, "events")
        os.makedirs(events)
        with open(os.path.join(events, "known.json"), "w", encoding="utf-8") as f:
            json.dump([{"text": "העובדים", "by": "o", "at": 0, "until": 9e9, "people": 3, "camera": PERGOLA},
                       {"text": "old", "by": "o", "at": 0, "until": 5, "people": 0, "camera": ""}], f)
        store = CameraProfiles(os.path.join(events, "camera_profiles.json"))
        store.add_fact("", "יש לנו כלב ושני ילדים")

        class HouseNow:
            state = "home_asleep"
            expecting = [{"text": "plumber", "camera": BACK}]

        out = house_profile(now=100.0, events_dir=events, profiles=store, house_now=HouseNow())
        self.assertEqual(out["state"], "home_asleep")
        self.assertEqual(out["known"][0]["who"], "העובדים")
        self.assertEqual(len(out["known"]), 1)
        self.assertEqual(out["expecting"], [{"text": "plumber", "camera": BACK}])
        self.assertTrue(out["household"]["dog"] and out["household"]["kids"])


class ReportTest(unittest.TestCase):
    def test_report_on_a_folder(self) -> None:
        tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmp, True)
        for d in range(3):
            write_meta(tmp, MAIN, at(d, 8, 0), summary="A courier knocks on the door")
        buf = io.StringIO()
        out = bl.report(tmp, out=buf)
        self.assertIn(MAIN, buf.getvalue())
        self.assertIn("busiest hours: 08:00-09:00", buf.getvalue())
        self.assertEqual(out["cameras"][MAIN]["days_of_data"], 3)
        self.assertEqual(out["cameras"][MAIN]["top_tags"][0], ("knock", 3))


@unittest.skipUnless(os.path.isdir(os.path.join(REAL, "production_ameer_week_0_1")), "the owner's history is not here")
class RealHistoryTest(unittest.TestCase):
    def test_pergola_and_main_entrance_are_busy_backyards_quiet_at_night(self) -> None:
        out = bl.report(os.path.join(REAL, "production_ameer_week_0_1"), out=io.StringIO())["cameras"]
        busy = [out[f"ameer_week_0_1_ch{n}"]["event_hours"] for n in (3, 6)]
        quiet = [out[f"ameer_week_0_1_ch{n}"]["event_hours"] for n in (1, 5, 8)]
        self.assertGreater(min(busy), 3 * max(quiet))
        for n in (1, 5, 8):
            phases = out[f"ameer_week_0_1_ch{n}"]["phase_event_hours"]
            self.assertEqual(phases["night"] + phases["late_night"], 0)

    @unittest.skipUnless(os.path.isdir(os.path.join(REAL, "dataset_multi")), "no collector history here")
    def test_collector_history_without_channels_is_not_mapped(self) -> None:
        ledger = bl.Baseline.in_memory()
        result = ledger.import_raw(os.path.join(REAL, "dataset_multi"), [MAIN, BACK])
        self.assertGreater(sum(result["collector"]["unmapped"].values()), 0)
        self.assertEqual(ledger.cameras(), [])


if __name__ == "__main__":
    unittest.main()
