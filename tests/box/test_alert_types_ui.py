import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import json
import time
import unittest
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock, patch
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QWidget

from home_guard_project.box.app.alert_types_ui import AlertTiles, CameraAlertsDialog, CameraAlertButton, ElidedLabel, alert_summary
from home_guard_project.box.app.box_controls import BoxControls, Settings
from home_guard_project.box.app.camera_controls import CameraControls
from home_guard_project.box.app.remote_cameras import RemoteCameras
from home_guard_project.box.app.camera_ui import CameraPage
from home_guard_project.box.app.settings_ui import SettingsPage
from home_guard_project.box.app.strings import tr


class RecordedCameraCommands:
    """One fake command boundary for the local subprocess and the laptop SSH path."""
    def __init__(self):
        self.calls = []; self.own = None; self.house = ['person']; self.error = None

    def run(self, args, **kwargs):
        self.calls.append(args)
        text = ' '.join(args)
        if self.error and 'set-camera-alerts' in text:
            return SimpleNamespace(returncode=1, stdout=json.dumps({'error': self.error}))
        if 'set-option alert_on=' in text:
            self.house = text.split('alert_on=')[-1].split(',')
            data = {}
        elif 'set-camera-alerts' in text:
            self.own = None if '--default' in text else text.split('--on ')[-1].split(',')
            data = {'camera': 'driveway', 'alert_on': self.own}
        elif 'ai_status.json' in text:
            data = self.status()
        else:
            data = {'house': self.house, 'cameras': [{'name': 'driveway', 'alert_on': self.own}]}
        return SimpleNamespace(returncode=0, stdout=json.dumps(data))

    def status(self):
        return {'updated': time.time(), 'settings': {'alert_on': self.house, 'camera_alert_on': {} if self.own is None else {'driveway': self.own}}}


class AlertUiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def settings(self, **values):
        box = BoxControls(demo=True, settings=Settings(mode='inference', **values))
        page = SettingsPage(box, lambda: None); page.reload()
        self.addCleanup(page.widget.close); self.addCleanup(page.timer.stop)
        return box, page

    def finish_command(self, dialog):
        if dialog.future:
            dialog.future.result(timeout=3)
            dialog.poll()

    def dialog(self, backend, own=None):
        dialog = CameraAlertsDialog(backend, 'driveway', ['person'], own)
        dialog.timer.stop(); dialog.show(); self.app.processEvents()
        self.addCleanup(dialog.close)
        self.finish_command(dialog)
        return dialog

    def test_loaded_tiles_and_live_vehicle_write_no_restart(self):
        box, page = self.settings()
        self.assertEqual([k for k, t in page.alert_types.tiles.items() if t.isChecked()], ['person'])
        with patch.object(box, 'save_settings', wraps=box.save_settings) as save:
            page.alert_types.tiles['vehicle'].click()
            self.assertEqual(save.call_args.args[0].alert_on, 'person,vehicle')
            self.assertIsNone(box.pending_at)
            self.assertEqual(page.alert_note.text(), tr('applying'))
            page.check_applied()
            self.assertEqual(page.alert_note.text(), 'Applied')

    def test_last_tile_retains_selection_and_hint(self):
        box, page = self.settings()
        with patch.object(box, 'save_settings', wraps=box.save_settings) as save:
            page.alert_types.tiles['person'].click()
            self.assertEqual(page.alert_types.value(), 'person')
            self.assertEqual(page.alert_types.hint.text(), tr('alert_keep_one'))
            self.assertFalse(page.alert_types.hint.isHidden())
            self.assertEqual(page.alert_types.tiles['person'].animation.state().name, 'Running')
            save.assert_not_called()

    def test_tiles_disabled_when_security_alerts_off(self):
        box, page = self.settings(alert_on='animal')
        page.alerts.setChecked(False)
        self.assertTrue(all(not t.isEnabled() for t in page.alert_types.tiles.values()))
        page.alert_types.tiles['vehicle'].click()
        self.assertEqual(box.load_settings().alert_on, 'animal')
        page.alerts.setChecked(True)
        self.assertTrue(all(t.isEnabled() for t in page.alert_types.tiles.values()))

    def test_reload_and_save_preserve_alert_choice(self):
        box, page = self.settings(alert_on='animal,person')
        self.assertEqual(page.alert_types.value(), 'person,animal')
        page.save_clicked()
        self.assertEqual(box.load_settings().alert_on, 'person,animal')
        self.assertIsNone(box.pending_at)

    def test_failure_restores_saved_house_choice(self):
        box, page = self.settings()
        with patch.object(box, 'save_settings', side_effect=ValueError('Disk full')):
            page.alert_types.tiles['vehicle'].click()
        self.assertEqual(page.alert_types.value(), 'person')
        self.assertEqual(page.alert_note.text(), tr('control_error'))

    def test_keyboard_space_and_enter_and_narrow_layout(self):
        tiles = AlertTiles(); tiles.resize(520, 360); tiles.show(); self.addCleanup(tiles.close)
        self.app.processEvents()
        vehicle = tiles.tiles['vehicle']; vehicle.setFocus(Qt.FocusReason.TabFocusReason)
        QTest.keyClick(vehicle, Qt.Key.Key_Space)
        self.assertEqual(tiles.value(), 'person,vehicle')
        QTest.keyClick(vehicle, Qt.Key.Key_Return)
        self.assertEqual(tiles.value(), 'person')
        self.assertTrue(vehicle.hasFocus())
        self.assertGreater(vehicle.y(), tiles.tiles['person'].y())
        tiles.resize(800, 200); self.app.processEvents()
        self.assertEqual(vehicle.y(), tiles.tiles['person'].y())
        self.assertGreater(vehicle.x(), tiles.tiles['person'].x())

    def test_popover_local_and_remote_exact_commands_and_applied(self):
        for remote in (False, True):
            with self.subTest(remote=remote):
                runner = RecordedCameraCommands()
                backend = RemoteCameras('user@box', runner=runner) if remote else CameraControls(BoxControls(), runner=runner.run)
                with patch.object(backend, 'alert_reported_status', side_effect=runner.status):
                    dialog = self.dialog(backend)
                    self.assertEqual(dialog.house_note.text(), 'House default · People')
                    self.assertTrue(dialog.default.isChecked()); self.assertTrue(dialog.tiles.isHidden())
                    dialog.custom.click(); self.finish_command(dialog)
                    self.finish_command(dialog)
                    dialog.tiles.tiles['vehicle'].click(); self.finish_command(dialog)
                    dialog.last_status = 0; dialog.poll(); self.finish_command(dialog)
                    self.assertEqual(dialog.note.text(), 'Applied')
                    self.assertEqual(runner.own, ['person', 'vehicle'])
                    dialog.default.click(); self.finish_command(dialog)
                    self.assertIsNone(runner.own)
                    commands = [' '.join(args) for args in runner.calls]
                    self.assertTrue(any(c.endswith('set-camera-alerts --camera driveway --on person,vehicle') for c in commands))
                    self.assertTrue(any(c.endswith('set-camera-alerts --camera driveway --default') for c in commands))
                    dialog.close()

    def test_command_error_is_plain_in_popover_and_retry_works(self):
        for remote in (False, True):
            runner = RecordedCameraCommands()
            backend = RemoteCameras('user@box', runner=runner) if remote else CameraControls(BoxControls(), runner=runner.run)
            dialog = self.dialog(backend)
            runner.error = 'Camera driveway no longer exists.'
            dialog.custom.click()
            # Let the GUI collect the error rather than propagating the worker error here.
            try: dialog.future.result(timeout=3)
            except ValueError: pass
            dialog.poll()
            self.assertEqual(dialog.note.text(), runner.error)
            self.assertFalse(dialog.retry.isHidden())
            runner.error = None; dialog.retry.click(); self.finish_command(dialog)
            self.assertEqual(runner.own, ['person']); dialog.close()

    def test_laptop_house_control_writes_live_and_confirms_reported_status(self):
        runner = RecordedCameraCommands(); backend = RemoteCameras('user@box', runner=runner)
        dialog = CameraAlertsDialog(backend, None, ['person'], None)
        dialog.timer.stop(); self.addCleanup(dialog.close); self.finish_command(dialog)
        dialog.tiles.tiles['vehicle'].click(); self.finish_command(dialog)
        self.finish_command(dialog)
        self.assertEqual(runner.house, ['person', 'vehicle'])
        self.assertEqual(dialog.note.text(), 'Applied')
        self.assertTrue(any(args[-1].endswith('set-option alert_on=person,vehicle') for args in runner.calls))
        # The fake box does not answer paths --json, like an old one: its logs are in its code folder.
        self.assertTrue(any(args[-1] == r'type "C:\home_guard\logs\ai_status.json"' for args in runner.calls))

    def test_read_failure_prevents_overwriting_unknown_choice(self):
        controls = CameraControls(BoxControls(demo=True), ['driveway'])
        with patch.object(controls, 'camera_alerts', side_effect=ValueError('Could not read camera choices.')):
            dialog = CameraAlertsDialog(controls, 'driveway', ['person'], None); dialog.timer.stop(); self.addCleanup(dialog.close)
            try: dialog.future.result(timeout=3)
            except ValueError: pass
            dialog.poll()
            self.assertFalse(dialog.custom.isEnabled()); self.assertFalse(dialog.tiles.isEnabled())
            self.assertEqual(dialog.note.text(), 'Could not read camera choices.')

    def test_old_engine_keeps_camera_change_unconfirmed_after_ten_seconds(self):
        controls = CameraControls(BoxControls(demo=True), ['driveway'])
        dialog = self.dialog(controls)
        with patch.object(controls, 'alert_reported_status', return_value={'updated': time.time(), 'settings': {}}):
            dialog.custom.click(); self.finish_command(dialog)
            dialog.ack.requested = time.time() - 11
            dialog.last_status = 0; dialog.poll(); self.finish_command(dialog)
            self.assertEqual(dialog.note.text(), tr('live_not_picked_up'))
            self.assertTrue(dialog.ack.expected)

    def test_explicit_camera_choice_equal_to_house_still_shows_custom_dot(self):
        button = CameraAlertButton(['person'], ['person'])
        self.assertTrue(button.custom)
        self.assertEqual(button.text(), 'Alerts: People')
        button.refresh(['person', 'vehicle'], None)
        self.assertFalse(button.custom)
        self.assertEqual(button.text(), 'Alerts: House default · People, Vehicles')

    def test_camera_card_label_dot_and_geometry_and_house_refresh(self):
        box = BoxControls(demo=True); controls = CameraControls(box, ['driveway', 'yard'])
        page = CameraPage(controls, lambda: None); self.addCleanup(page.close)
        page.render(page.read_zones(controls.load())); page.loaded = True
        page.widget.resize(1000, 600); page.widget.show(); self.addCleanup(page.widget.close); self.app.processEvents()
        button = page.alert_widgets['driveway']; geometry = button.geometry()
        self.assertEqual(button.text(), 'Alerts: House default · People'); self.assertFalse(button.custom)
        page.alert_saved('driveway', ['person', 'vehicle']); self.app.processEvents()
        self.assertEqual(button.text(), 'Alerts: People, Vehicles'); self.assertTrue(button.custom)
        self.assertEqual(button.geometry(), geometry)
        box.save_settings(replace(box.load_settings(), alert_on='animal')); page.open()
        self.assertEqual(page.alert_widgets['yard'].text(), 'Alerts: House default · Animals')
        self.assertEqual(button.text(), 'Alerts: People, Vehicles')
        slots = page.widget.findChildren(QWidget, 'cameraActionSlot')
        self.assertTrue(all(slot.height() == 36 for slot in slots))

    def test_status_combinations_fallback_and_elision_geometry(self):
        cases = [({}, 'People'), ({'alert_on': ['vehicle', 'animal']}, 'Vehicles, Animals'),
                 ({'alert_on': ['animal'], 'camera_alert_on': {'yard': ['person']}}, 'Animals · 1 camera custom'),
                 ({'camera_alert_on': {'a': ['person'], 'b': ['animal']}}, 'People · 2 cameras custom')]
        for settings, expected in cases:
            self.assertEqual(alert_summary({'settings': settings}, 'person'), expected)
        self.assertEqual(alert_summary({}, 'person,vehicle'), 'People, Vehicles')
        self.assertEqual(alert_summary({'settings': {'alert_on': []}}, 'animal'), 'Animals')
        label = ElidedLabel(); label.resize(120, 30); label.setText('People, Vehicles, Animals · 2 cameras custom')
        self.assertFalse(label.wordWrap()); self.assertEqual(label.toolTip(), label.text())
        self.assertTrue(label.grab().width() == 120)

    def test_dashboard_reads_reported_types_and_custom_count(self):
        from home_guard_project.box.app.ui import Window
        args = SimpleNamespace(demo=True,setup=False,theme='dark',panel=None,fail=None,wifi=False,skip_cameras=False,alerts=False,details=False,state='inference',cameras=3,page=None,size='1366x768',screenshot=None,detections=False)
        window = Window(args); self.addCleanup(window.close)
        window.box_controls.camera_alert_on = {'driveway': ['vehicle']}
        window.box_controls.save_settings(replace(window.box_controls.load_settings(), alert_on='person,animal'))
        window.settings_changed()
        self.assertIn('People, Animals · 1 camera custom', window.alert_status.text())
