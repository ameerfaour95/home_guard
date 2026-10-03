# tests/box/test_brain_mode.py
from __future__ import annotations

import datetime as dt
import unittest

from home_guard_project.box.brain.mode import (
    ASSISTANT,
    GUARD,
    mode_ends_at,
    mode_started_at,
    resolve_mode,
    status_line,
    switch_announcement,
)


def at(hour: int, minute: int = 0, day: int = 3) -> float:
    return dt.datetime(2026, 10, day, hour, minute).timestamp()


class ModeTest(unittest.TestCase):
    def test_overnight_window(self) -> None:
        self.assertEqual(resolve_mode(at(23), 22, 6), GUARD)
        self.assertEqual(resolve_mode(at(5, 59), 22, 6), GUARD)
        self.assertEqual(resolve_mode(at(6, 0), 22, 6), ASSISTANT)
        self.assertEqual(resolve_mode(at(21, 59), 22, 6), ASSISTANT)
        self.assertEqual(resolve_mode(at(22, 0), 22, 6), GUARD)

    def test_start_equal_end_guards_all_day(self) -> None:
        self.assertEqual(resolve_mode(at(14), 0, 0), GUARD)
        self.assertIsNone(mode_ends_at(at(14), 0, 0))
        self.assertIsNone(mode_started_at(at(14), 0, 0))

    def test_when_the_mode_ends_and_started(self) -> None:
        self.assertEqual(mode_ends_at(at(23), 22, 6), at(6, day=4))
        self.assertEqual(mode_ends_at(at(10), 22, 6), at(22))
        self.assertEqual(mode_started_at(at(23), 22, 6), at(22))
        self.assertEqual(mode_started_at(at(2, day=4), 22, 6), at(22))
        self.assertEqual(mode_started_at(at(10), 22, 6), at(6))

    def test_status_line(self) -> None:
        self.assertEqual(status_line(GUARD, at(23), 22, 6), "🛡️ Guarding until 06:00")
        self.assertEqual(status_line(GUARD, at(23), 0, 0), "🛡️ Guarding all day")
        self.assertEqual(
            status_line(GUARD, at(23), 22, 6, paused=[("main_entrance", at(8, day=4))], offline=["back_door"]),
            "🛡️ Guarding until 06:00 · main_entrance alerts paused until 08:00 · ⚠️ back_door offline")
        self.assertEqual(status_line(ASSISTANT, at(10), 22, 6), "💬 Assistant · quiet logging · guarding from 22:00")
        self.assertEqual(status_line(ASSISTANT, at(10), 22, 6, quiet_log=False),
                         "💬 Assistant · not recording · guarding from 22:00")

    def test_switch_announcement(self) -> None:
        self.assertEqual(switch_announcement(GUARD, at(22), 22, 6, live=5, total=6, lang="en"),
                         "🛡️ Guarding started, until 06:00. 5 of 6 cameras live.")
        self.assertEqual(switch_announcement(ASSISTANT, at(6), 22, 6, live=6, total=6, lang="en"),
                         "💬 Guarding ended. The box keeps a quiet log until 22:00.")
        self.assertIn("06:00", switch_announcement(GUARD, at(22), 22, 6, live=5, total=6, lang="he"))
        self.assertEqual(switch_announcement(ASSISTANT, at(6), 22, 6, live=6, total=6, lang="en", quiet_log=False),
                         "💬 Guarding ended. Nothing is recorded until 22:00.")


if __name__ == "__main__":
    unittest.main()
