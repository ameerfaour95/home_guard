from __future__ import annotations

import os
import tempfile
import unittest

from home_guard_project.box.inference import AlertSettings, LiveSettings, apply_live_settings

NOW = 1_800_000_000.0


class LiveSettingsTest(unittest.TestCase):
    """A change the owner makes in box.yaml applies while the program runs, without a restart."""

    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.path = os.path.join(tmp.name, "box.yaml")
        self._write('site: "test"\nmode: inference\ninference_conf: 0.4\nalert_cooldown_sec: 120\n', NOW - 100)
        self.settings = AlertSettings(conf=0.4, cooldown_sec=120.0)
        self.live = LiveSettings(self.settings, self.path, poll_sec=2.0, now=NOW)

    def _write(self, text: str, mtime: float) -> None:
        with open(self.path, "w", encoding="utf-8") as f:
            f.write(text)
        os.utime(self.path, (mtime, mtime))

    def test_a_new_threshold_and_cooldown_are_taken_over(self) -> None:
        self._write('site: "test"\nmode: inference\ninference_conf: 0.25\nalert_cooldown_sec: 60\n', NOW)
        self.assertEqual(self.live.check(NOW + 2.5), ["cooldown_sec", "conf"])
        self.assertEqual((self.settings.conf, self.settings.cooldown_sec), (0.25, 60.0))

    def test_the_file_is_not_read_more_often_than_the_poll_interval(self) -> None:
        self._write('inference_conf: 0.25\n', NOW)
        self.assertEqual(self.live.check(NOW + 1.0), [])          # too soon to look
        self.assertEqual(self.settings.conf, 0.4)
        self.assertEqual(self.live.check(NOW + 2.0), ["conf"])

    def test_an_untouched_file_changes_nothing(self) -> None:
        self.assertEqual(self.live.check(NOW + 5), [])
        self.assertEqual(self.live.check(NOW + 10), [])

    def test_the_alert_hours_are_live_but_the_mode_and_channel_are_not(self) -> None:
        settings = AlertSettings(alert_start_hour=0, alert_end_hour=0, alert_channel="telegram")
        changed = apply_live_settings(settings, {"alert_start_hour": 22, "alert_end_hour": 6,
                                                 "alert_channel": "twilio", "inference_conf": 0.4})
        self.assertEqual(changed, ["alert_start_hour", "alert_end_hour"])
        self.assertEqual((settings.alert_start_hour, settings.alert_end_hour, settings.alert_channel),
                         (22, 6, "telegram"))

    def test_a_half_written_file_keeps_the_current_values(self) -> None:
        self._write('inference_conf: [oops\n', NOW)
        self.assertEqual(self.live.check(NOW + 3), [])
        self.assertEqual(self.settings.conf, 0.4)

    def test_a_missing_file_is_harmless(self) -> None:
        os.remove(self.path)
        self.assertEqual(self.live.check(NOW + 3), [])


if __name__ == "__main__":
    unittest.main()
