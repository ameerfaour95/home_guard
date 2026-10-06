from __future__ import annotations

import unittest
from datetime import datetime

from home_guard_project.box import house_state as hs
from home_guard_project.box import situation as st
from home_guard_project.box import taxonomy as tx


def at(month: int, hour: int, minute: int = 0, day: int = 15) -> float:
    return datetime(2026, month, day, hour, minute).timestamp()


def asleep(ts: float) -> hs.HouseNow:
    return hs.scheduled(ts)


class PhaseTest(unittest.TestCase):
    def test_phases_follow_the_sun_in_israel(self) -> None:
        # October: sunrise about 06:35, sunset about 18:10.
        self.assertEqual(st.phase_of(at(10, 2, 14)), ("late_night", True))
        self.assertEqual(st.phase_of(at(10, 6, 15)), ("dawn", False))
        self.assertEqual(st.phase_of(at(10, 12)), ("day", False))
        self.assertEqual(st.phase_of(at(10, 18, 20)), ("evening", False))
        self.assertEqual(st.phase_of(at(10, 21)), ("evening", True))

    def test_summer_days_are_longer_than_winter_days(self) -> None:
        self.assertEqual(st.phase_of(at(6, 19, 15))[0], "day")
        self.assertEqual(st.phase_of(at(12, 19, 15)), ("evening", True))
        self.assertEqual(st.phase_of(at(6, 5, 15))[0], "dawn")
        self.assertEqual(st.phase_of(at(12, 5, 0)), ("late_night", True))

    def test_every_month_has_sunrise_before_sunset(self) -> None:
        for month, (rise, sset) in st.SUN_TABLE.items():
            self.assertLess(rise, sset, month)
        self.assertEqual(sorted(st.SUN_TABLE), list(range(1, 13)))


class RoleTest(unittest.TestCase):
    def test_guessed_from_real_camera_names(self) -> None:
        cases = {"main_door": "entrance", "front_door": "entrance", "back_door": "entrance", "gate": "entrance",
                 "main_entrance": "entrance", "street": "street", "front": "street", "front_side": "street",
                 "left_side_1": "private", "right_side": "private", "back_yard": "private", "pergola": "private",
                 "left_back": "private", "garden": "private", "parking": "parking", "driveway": "parking",
                 "Security": "entrance"}
        for name, role in cases.items():
            self.assertEqual(st.guess_role(name), role, name)

    def test_setting_wins_and_bad_values_are_ignored(self) -> None:
        settings = {"camera_roles": {"front_side": "parking", "gate": "garage"}}
        self.assertEqual(st.camera_role("front_side", settings), "parking")
        self.assertEqual(st.camera_role("gate", settings), "entrance")
        self.assertEqual(st.camera_role("gate", None), "entrance")


class SituationTest(unittest.TestCase):
    def test_header_is_exactly_the_structured_line(self) -> None:
        ts = at(10, 2, 14)
        sit = st.build_situation("back_yard", ts, "alert_triage", house=asleep(ts))
        self.assertEqual(sit.header(), "SITUATION: time 02:14, late_night, dark; house: home_asleep; "
                                       "camera: back_yard (private); intent: alert_triage; expecting: none")

    def test_header_by_day_with_expecting(self) -> None:
        ts = at(10, 14, 5)
        house = hs.HouseNow(state="home_awake", source="owner", set_at=None, expires_at=None,
                            expecting=[{"text": "a package today", "camera": None},
                                       {"text": 'the "plumber" at 10', "camera": "front_door"},
                                       {"text": "the gardener", "camera": "yard"}])
        sit = st.build_situation("front_door", ts, "snapshot", house=house)
        self.assertEqual(sit.header(), "SITUATION: time 14:05, day, light; house: home_awake; "
                                       "camera: front_door (entrance); intent: snapshot; "
                                       "expecting: 'a package today', 'the plumber at 10'")
        self.assertTrue(sit.expecting)

    def test_vacation_is_away(self) -> None:
        ts = at(10, 12)
        house = hs.HouseNow(state="vacation", source="owner", set_at=None, expires_at=None)
        sit = st.build_situation("gate", ts, house=house)
        self.assertEqual(sit.house_state, "away")
        self.assertIn("house: away;", sit.header())

    def test_taxonomy_context(self) -> None:
        ts = at(10, 2, 14)
        lower = {"id": "F1", "effect": "lower", "kind": "people"}
        sit = st.build_situation("back_yard", ts, house=asleep(ts), facts=[lower])
        ctx = sit.to_taxonomy_context(movement="approaching", zone="entrance", flags=["flashlight"])
        self.assertEqual(ctx, tx.Context(phase="late_night", house_state="home_asleep", dark=True,
                                         camera_role="private", expecting=False, fact_covers=True,
                                         movement="approaching", zone="entrance", flags=("flashlight",)))
        raise_only = st.build_situation("back_yard", ts, house=asleep(ts),
                                        facts=[{"id": "F2", "effect": "raise", "kind": "people"}])
        self.assertFalse(raise_only.fact_covers)

    def test_record_for_meta(self) -> None:
        ts = at(10, 12)
        sit = st.build_situation("gate", ts, "alert_triage", house=asleep(ts))
        self.assertEqual(sit.record(), {"phase": "day", "dark": False, "house_state": "home_awake",
                                        "intent": "alert_triage", "camera_role": "entrance"})

    def test_bad_intent_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            st.build_situation("gate", at(10, 12), "chat", house=asleep(at(10, 12)))

    def test_house_state_is_read_when_not_given(self) -> None:
        import os
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "house_state.jsonl")
            hs.set_house_state("away", "owner", path=path, now=at(10, 11))
            sit = st.build_situation("gate", at(10, 12), state_path=path, mute_path=os.path.join(tmp, "m.json"))
            self.assertEqual(sit.house_state, "away")

    def test_zones_from_settings(self) -> None:
        ts = at(10, 12)
        sit = st.build_situation("gate", ts, settings={"camera_zones": {"gate": ["gate", "fence"]}},
                                 house=asleep(ts))
        self.assertEqual(sit.zones, ("gate", "fence"))


if __name__ == "__main__":
    unittest.main()
