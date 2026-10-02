import unittest
import tempfile
import os
from pathlib import Path
from dataclasses import replace
from unittest import mock
from home_guard_project.box.app.box_controls import BoxControls, Settings, minutes_to_seconds


class StopStartTest(unittest.TestCase):
    def test_demo_stop_and_start_never_touch_real_box(self):
        with mock.patch('home_guard_project.box.app.box_controls.control.stop') as stop, mock.patch('home_guard_project.box.app.box_controls.control.start') as start:
            box = BoxControls(demo=True)
            self.assertFalse(box.is_stopped())
            box.stop()
            self.assertTrue(box.is_stopped())
            box.start()
            self.assertFalse(box.is_stopped())
            stop.assert_not_called(); start.assert_not_called()

    def test_local_adapter_uses_existing_controls(self):
        box = BoxControls()
        with mock.patch('home_guard_project.box.app.box_controls.control.is_stopped', return_value=True) as query, mock.patch('home_guard_project.box.app.box_controls.control.stop') as stop, mock.patch('home_guard_project.box.app.box_controls.control.start') as start:
            self.assertTrue(box.is_stopped())
            box.stop(); box.start()
            query.assert_called_once(); stop.assert_called_once(); start.assert_called_once()


class SettingsTest(unittest.TestCase):
    def test_boundaries_and_overnight_hours_are_valid(self):
        for seconds in (10, 86400):
            Settings(alert_start_hour=22, alert_end_hour=6, alert_cooldown_sec=seconds)
        Settings(alert_start_hour=0, alert_end_hour=0)
        for kwargs in ({"alert_start_hour": 24}, {"alert_end_hour": -1}, {"alert_cooldown_sec": 9}, {"alert_cooldown_sec": 86401}, {"mode": "other"}):
            with self.assertRaises(ValueError): Settings(**kwargs)

    def test_minutes_round_to_valid_whole_seconds(self):
        self.assertEqual(minutes_to_seconds(.17), 10)
        self.assertEqual(minutes_to_seconds(2), 120)
        self.assertEqual(minutes_to_seconds(1440), 86400)
        for value in (0, 1441, float('nan'), float('inf')):
            with self.assertRaises(ValueError): minutes_to_seconds(value)
        for seconds in (10, 11, 59, 61, 86399):
            self.assertEqual(minutes_to_seconds(round(seconds/60, 2)), seconds)

    def test_demo_settings_apply_after_a_restart_and_persist(self):
        now = [100]
        box = BoxControls(demo=True, clock=lambda: now[0])
        new = Settings(mode='inference', alert_start_hour=22, alert_end_hour=6)
        with mock.patch('home_guard_project.box.app.box_controls.boxconfig.set_option') as write, mock.patch('home_guard_project.box.app.box_controls.control.request_restart') as restart:
            box.save_settings(new)
            self.assertEqual(box.load_settings(), new)
            self.assertEqual(box.phase(), 'restarting')
            now[0] += 2
            self.assertEqual(box.phase(), 'running')
            write.assert_not_called(); restart.assert_not_called()

    def test_picture_only_change_needs_no_restart(self):
        box = BoxControls(demo=True)
        box.save_settings(replace(box.load_settings(), show_cameras=False))
        self.assertIsNone(box.pending_at)
        self.assertFalse(box.load_settings().show_cameras)

    def test_local_save_uses_option_api_and_requests_one_restart(self):
        box = BoxControls()
        initial = Settings()
        wanted = replace(initial, mode='inference', alert_cooldown_sec=300, show_cameras=True)
        with mock.patch.object(box, 'load_settings', return_value=initial), mock.patch.object(box, 'is_stopped', return_value=False), mock.patch('home_guard_project.box.app.box_controls.boxconfig.set_option') as write, mock.patch('home_guard_project.box.app.box_controls.control.request_restart') as restart:
            box.save_settings(wanted)
            self.assertEqual(write.call_args_list, [mock.call('mode', 'inference'), mock.call('alert_cooldown_sec', '300'), mock.call('show_cameras', 'true')])
            restart.assert_called_once()
            self.assertIsNotNone(box.pending_at)

    def test_unchanged_settings_do_not_write_or_restart(self):
        box = BoxControls()
        with mock.patch.object(box, 'load_settings', return_value=Settings()), mock.patch('home_guard_project.box.app.box_controls.boxconfig.set_option') as write, mock.patch('home_guard_project.box.app.box_controls.control.request_restart') as restart:
            self.assertFalse(box.save_settings(Settings()))
            write.assert_not_called(); restart.assert_not_called()

    def test_saved_settings_never_start_a_stopped_demo_box(self):
        box = BoxControls(demo=True, stopped=True)
        box.save_settings(Settings(mode='inference'))
        self.assertTrue(box.is_stopped())
        self.assertEqual(box.phase(), 'stopped')
        self.assertIsNone(box.pending_at)

    def test_restart_confirmation_requires_new_pid_and_fresh_alive(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            box = BoxControls(clock=lambda: 100)
            box.pending_at = 100
            alive = root/'collector.alive'
            pid = root/'collector.winpid'
            alive.touch(); pid.touch()
            os.utime(alive, (90,90)); os.utime(pid, (90,90))
            with mock.patch.object(box, 'is_stopped', return_value=False), mock.patch('home_guard_project.box.app.box_controls.boxconfig.LOG_DIR', directory), mock.patch('home_guard_project.box.app.box_controls.boxconfig.ALIVE_FILE', str(alive)):
                self.assertEqual(box.phase(), 'restarting')
                os.utime(alive, (101,101)); os.utime(pid, (101,101))
                flag = root/'collector.restart'; flag.touch()
                self.assertEqual(box.phase(), 'restarting')
                flag.unlink()
                self.assertEqual(box.phase(), 'running')
