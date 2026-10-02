import unittest
from unittest import mock
from home_guard_project.box.app.box_controls import BoxControls


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
