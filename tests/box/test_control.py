from __future__ import annotations

import os
import tempfile
import unittest

from home_guard_project.box import control


class ControlTest(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.logs = os.path.join(tmp.name, "logs")   # not created yet: the flags must create it

    def _flags(self) -> list[str]:
        return sorted(os.listdir(self.logs)) if os.path.isdir(self.logs) else []

    def test_stop_then_start(self) -> None:
        self.assertFalse(control.is_stopped(self.logs))
        control.stop(self.logs)
        self.assertTrue(control.is_stopped(self.logs))
        self.assertEqual(self._flags(), ["collector.stop"])
        control.start(self.logs)
        self.assertFalse(control.is_stopped(self.logs))
        self.assertEqual(self._flags(), [])

    def test_start_when_not_stopped_is_harmless(self) -> None:
        control.start(self.logs)
        self.assertFalse(control.is_stopped(self.logs))

    def test_restart_request_leaves_a_flag_for_the_runner(self) -> None:
        control.request_restart(self.logs)
        self.assertEqual(self._flags(), ["collector.restart"])

    def test_a_stopped_box_is_not_restarted_and_start_clears_an_old_request(self) -> None:
        control.stop(self.logs)
        control.request_restart(self.logs)
        self.assertEqual(self._flags(), ["collector.stop"])

        control.start(self.logs)
        control.request_restart(self.logs)
        control.stop(self.logs)
        control.start(self.logs)
        self.assertEqual(self._flags(), [])


if __name__ == "__main__":
    unittest.main()
