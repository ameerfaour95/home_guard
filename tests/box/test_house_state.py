from __future__ import annotations

import json
import os
import tempfile
import threading
import unittest
from datetime import datetime

from home_guard_project.box import house_state as hs


def at(day: int, hour: int, minute: int = 0) -> float:
    return datetime(2026, 10, day, hour, minute).timestamp()


class _Store(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = os.path.join(self.tmp.name, ".registry", "house_state.jsonl")
        self.mute_path = os.path.join(self.tmp.name, "alert_mute.json")
        self.clock = at(6, 14)
        self.store = hs.HouseStateStore(self.path, mute_path=self.mute_path, now=lambda: self.clock)


class ScheduleTest(_Store):
    def test_default_schedule_is_asleep_after_midnight_and_awake_by_day(self) -> None:
        self.assertEqual(self.store.current(at(6, 2)).state, "home_asleep")
        self.assertEqual(self.store.current(at(6, 2)).source, "schedule")
        self.assertEqual(self.store.current(at(6, 6)).state, "home_awake")
        self.assertEqual(self.store.current(at(6, 23, 59)).state, "home_awake")
        self.assertEqual(self.store.current(at(6, 5, 59)).expires_at, datetime(2026, 10, 6, 6).isoformat())

    def test_the_schedule_alone_needs_no_file(self) -> None:
        self.assertEqual(hs.scheduled(at(6, 3)).state, "home_asleep")
        self.assertEqual(hs.scheduled(at(6, 12)).state, "home_awake")
        self.assertFalse(os.path.exists(self.path))

    def test_owner_changes_the_schedule(self) -> None:
        self.store.set_schedule("23:00", "07:00", by="owner")
        self.assertEqual(self.store.current(at(6, 23, 30)).state, "home_asleep")
        self.assertEqual(self.store.current(at(7, 6, 30)).state, "home_asleep")
        self.assertEqual(self.store.current(at(7, 7)).state, "home_awake")

    def test_only_the_owner_changes_the_schedule(self) -> None:
        with self.assertRaises(ValueError):
            self.store.set_schedule("01:00", "05:00", source="system")


class OwnerCommandsTest(_Store):
    def test_going_to_sleep_lasts_until_the_morning(self) -> None:
        self.clock = at(6, 22)
        self.store.set_house_state("home_asleep", "owner", by="dad")
        now = self.store.current(at(6, 22, 30))
        self.assertEqual((now.state, now.source, now.by), ("home_asleep", "owner", "dad"))
        self.assertEqual(self.store.current(at(7, 3)).state, "home_asleep")
        self.assertEqual(self.store.current(at(7, 7)).state, "home_awake")

    def test_woke_up_early_overrides_tonight_only(self) -> None:
        self.clock = at(7, 4)
        self.store.set_house_state("home_awake", "owner")
        self.assertEqual(self.store.current(at(7, 5)).state, "home_awake")
        self.assertEqual(self.store.current(at(8, 2)).state, "home_asleep")

    def test_away_lasts_until_the_owner_says_back(self) -> None:
        self.store.set_house_state("away", "owner")
        self.assertEqual(self.store.current(at(9, 2)).state, "away")
        self.clock = at(9, 12)
        self.store.set_house_state("home_awake", "owner")
        self.assertEqual(self.store.current(at(9, 13)).state, "home_awake")
        self.assertEqual(self.store.current(at(10, 1)).state, "home_asleep")

    def test_until_is_kept(self) -> None:
        self.store.set_house_state("away", "owner", until=at(6, 18))
        self.assertEqual(self.store.current(at(6, 17)).state, "away")
        self.assertEqual(self.store.current(at(6, 17)).expires_at, datetime(2026, 10, 6, 18).isoformat())
        self.assertEqual(self.store.current(at(6, 18)).state, "home_awake")

    def test_vacation_is_away_with_an_end_date(self) -> None:
        with self.assertRaises(ValueError):
            self.store.set_house_state("vacation", "owner")
        self.store.set_house_state("vacation", "owner", until=at(12, 12), since=at(8, 6))
        self.assertEqual(self.store.current(at(7, 12)).state, "home_awake")
        later = self.store.current(at(10, 3))
        self.assertEqual((later.state, later.taxonomy_state), ("vacation", "away"))
        self.assertEqual(self.store.current(at(12, 13)).state, "home_awake")

    def test_bad_values_are_refused(self) -> None:
        for args in (("sleeping", "owner"), ("away", "neighbour")):
            with self.assertRaises(ValueError):
                self.store.set_house_state(*args)
        with self.assertRaises(ValueError):
            self.store.set_house_state("away", "owner", until=at(6, 13))     # in the past


class ProposalTest(_Store):
    def test_the_system_may_make_things_stricter_by_itself(self) -> None:
        entry = self.store.set_house_state("away", "system", by="phones-left")
        self.assertEqual(entry["status"], "applied")
        self.assertEqual(self.store.current(at(6, 15)).state, "away")
        self.assertEqual(self.store.current(at(6, 15)).source, "system")

    def test_relaxing_from_a_non_owner_source_waits_for_the_owner(self) -> None:
        self.store.set_house_state("away", "owner")
        proposal = self.store.set_house_state("home_awake", "proposal", by="historian")
        self.assertEqual(proposal["status"], "pending")
        self.assertEqual(self.store.current(at(6, 15)).state, "away")
        self.assertEqual([p["id"] for p in self.store.pending_proposals()], [proposal["id"]])
        self.clock = at(6, 16)
        self.assertTrue(self.store.approve(proposal["id"], by="mom"))
        self.assertEqual(self.store.current(at(6, 16, 1)).state, "home_awake")
        self.assertEqual(self.store.current(at(6, 16, 1)).source, "proposal")
        self.assertEqual(self.store.pending_proposals(), [])
        self.assertFalse(self.store.approve(proposal["id"]))      # answered once

    def test_waking_early_from_the_system_is_a_proposal(self) -> None:
        self.clock = at(7, 4)
        proposal = self.store.set_house_state("home_awake", "system")
        self.assertEqual(proposal["status"], "pending")
        self.assertEqual(self.store.current(at(7, 4, 30)).state, "home_asleep")

    def test_cancelling_a_vacation_early_from_the_system_is_a_proposal(self) -> None:
        self.store.set_house_state("vacation", "owner", until=at(12, 12))
        proposal = self.store.set_house_state("home_awake", "schedule")
        self.assertEqual(proposal["status"], "pending")
        self.assertTrue(self.store.reject(proposal["id"]))
        self.assertEqual(self.store.current(at(8, 12)).state, "vacation")
        self.assertEqual(self.store.pending_proposals(), [])

    def test_owner_expecting_applies_system_expecting_waits(self) -> None:
        x = self.store.add_expecting("a package today", until=at(6, 23, 59), by="dad")
        self.assertEqual(x["status"], "applied")
        p = self.store.add_expecting("the plumber at 10", camera="front_door", until=at(7, 12), source="system")
        self.assertEqual(p["status"], "pending")
        texts = [e["text"] for e in self.store.current(at(6, 15)).expecting]
        self.assertEqual(texts, ["a package today"])
        self.store.approve(p["id"])
        now = self.store.current(at(7, 10))
        self.assertEqual([(e["text"], e["camera"]) for e in now.expecting], [("the plumber at 10", "front_door")])
        self.assertEqual(now.expecting[0]["source"], "proposal")

    def test_expecting_needs_an_expiry_and_ends(self) -> None:
        with self.assertRaises(ValueError):
            self.store.add_expecting("a package", until=None)
        x = self.store.add_expecting("a package", until=at(6, 18))
        self.assertEqual(len(self.store.current(at(6, 17)).expecting), 1)
        self.assertEqual(self.store.current(at(6, 18)).expecting, [])
        self.store.cancel_expecting(x["id"])
        self.assertEqual(self.store.current(at(6, 17)).expecting, [])

    def test_expecting_for_a_camera(self) -> None:
        self.store.add_expecting("the gardener", camera="yard", until=at(6, 20))
        self.store.add_expecting("a package", until=at(6, 20))
        now = self.store.current(at(6, 15))
        self.assertEqual([e["text"] for e in now.expecting_for("yard")], ["the gardener", "a package"])
        self.assertEqual([e["text"] for e in now.expecting_for("gate")], ["a package"])


class LogTest(_Store):
    def test_every_field_carries_source_set_at_and_expires_at(self) -> None:
        self.store.set_house_state("away", "owner", until=at(6, 20))
        self.store.add_expecting("a package", until=at(6, 20))
        d = self.store.current(at(6, 15)).to_dict()
        for part in (d["state"], *d["expecting"]):
            self.assertEqual({"source", "set_at", "expires_at"} - set(part), set(), part)
        json.dumps(d)

    def test_damaged_lines_are_skipped_with_one_warning(self) -> None:
        self.store.set_house_state("away", "owner")
        with open(self.path, "a", encoding="utf-8") as f:
            f.write("{not json\n[1, 2]\n")
        self.store.add_expecting("a package", until=at(6, 20))
        with self.assertLogs("box.house_state", level="WARNING") as logs:
            now = self.store.current(at(6, 15))
        self.assertEqual(len(logs.records), 1)
        self.assertEqual(now.state, "away")
        self.assertEqual(len(now.expecting), 1)

    def test_a_cut_last_line_does_not_glue_onto_the_next_append(self) -> None:
        self.store.set_house_state("away", "owner")
        with open(self.path, "a", encoding="utf-8") as f:
            f.write('{"event": "state", "id": "H9"')
        self.store.add_expecting("a package", until=at(6, 20))
        with self.assertLogs("box.house_state", level="WARNING"):
            self.assertEqual(len(self.store.current(at(6, 15)).expecting), 1)

    def test_history_is_newest_first(self) -> None:
        self.store.set_house_state("away", "owner")
        self.store.add_expecting("a package", until=at(6, 20))
        self.assertEqual([e["event"] for e in self.store.history(5)], ["expect", "state"])
        self.assertEqual(len(self.store.history(1)), 1)

    def test_writers_from_many_threads_lose_nothing(self) -> None:
        def write(i: int) -> None:
            self.store.add_expecting(f"note {i}", until=at(6, 20))

        threads = [threading.Thread(target=write, args=(i,)) for i in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        now = self.store.current(at(6, 15))
        self.assertEqual(len(now.expecting), 20)
        self.assertEqual(len({e["id"] for e in now.expecting}), 20)

    def test_module_api_uses_the_given_path(self) -> None:
        hs.set_house_state("away", "owner", path=self.path, now=self.clock)
        self.assertEqual(hs.current(at(6, 15), path=self.path, mute_path=self.mute_path).state, "away")
        self.assertEqual(len(hs.history(5, path=self.path)), 1)

    def test_current_never_raises(self) -> None:
        os.makedirs(self.path)       # a folder where the file should be
        self.assertEqual(hs.current(at(6, 3), path=self.path, mute_path=self.mute_path).state, "home_asleep")


class MutesTest(_Store):
    def test_todays_mutes_are_read_from_the_pause_file(self) -> None:
        with open(self.mute_path, "w", encoding="utf-8") as f:
            json.dump({"all": at(6, 13), "cameras": {"gate": at(6, 16), "yard": at(6, 12)}}, f)
        mutes = self.store.current(at(6, 14)).mutes
        self.assertEqual([(m["camera"], m["source"]) for m in mutes], [("gate", "owner")])
        self.assertEqual(mutes[0]["expires_at"], datetime(2026, 10, 6, 16).isoformat())

    def test_a_house_pause_is_listed_as_all(self) -> None:
        with open(self.mute_path, "w", encoding="utf-8") as f:
            json.dump({"all": at(6, 15), "cameras": {}}, f)
        self.assertEqual([m["camera"] for m in self.store.current(at(6, 14)).mutes], [None])

    def test_no_pause_file(self) -> None:
        self.assertEqual(self.store.current(at(6, 14)).mutes, [])
        self.assertFalse(os.path.exists(self.mute_path))


if __name__ == "__main__":
    unittest.main()
