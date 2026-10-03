import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import json
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch
from PySide6.QtCore import Qt
from PySide6.QtGui import QFontDatabase
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QPushButton
from home_guard_project.box.app.alert_types_ui import CameraAlertsDialog, CameraAlertButton
from home_guard_project.box.app.box_controls import BoxControls, Settings
from home_guard_project.box.app.camera_controls import CameraControls
from home_guard_project.box.app.remote_cameras import RemoteCameras
from home_guard_project.box.app.settings_ui import SettingsPage
from home_guard_project.box.app.camera_ui import CameraPage
from home_guard_project.box.app.strings import tr


class RecordedSensitivityCommands:
    """No processes or network: capture the actual local/SSH command boundary."""
    def __init__(self):
        self.calls = []; self.own = None; self.error = None
        self.house = dict(person=.5, vehicle=.7, animal=.6)

    def run(self, args, **kwargs):
        self.calls.append(args); text = ' '.join(args)
        if 'set-camera-sensitivity' in text:
            if self.error:
                return SimpleNamespace(returncode=1, stdout=json.dumps({'error': self.error}))
            self.own = None if '--default' in text else {k: float(v) for k, v in (pair.split('=') for pair in text.split('--values ')[-1].split(','))}
            data = dict(camera='driveway', sensitivity=self.own)
        elif 'set-option conf_' in text:
            kind, value = text.split('set-option conf_')[-1].split('=')
            self.house[kind] = float(value); data = {}
        elif 'ai_status.json' in text:
            data = self.status()
        else:
            data = dict(house=['person'], house_sensitivity=self.house, cameras=[dict(name='driveway', alert_on=None, sensitivity=self.own)])
        return SimpleNamespace(returncode=0, stdout=json.dumps(data))

    def status(self):
        return dict(updated=time.time(), settings=dict(sensitivity=self.house, camera_sensitivity={} if self.own is None else dict(driveway=self.own)))


class SensitivityUiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])
        QFontDatabase.addApplicationFont('C:/Windows/Fonts/segoeui.ttf')

    def settings(self, **values):
        box = BoxControls(demo=True, settings=Settings(mode='inference', **values))
        page = SettingsPage(box, lambda: None); page.reload()
        self.addCleanup(page.widget.close); self.addCleanup(page.timer.stop)
        return box, page

    def finish(self, owner):
        if owner.future:
            try: owner.future.result(timeout=3)
            except ValueError: pass
            owner.poll()

    def dialog(self, backend, name='driveway'):
        dialog = CameraAlertsDialog(backend, name, ['person'], None)
        dialog.timer.stop(); dialog.sensitivity_panel.timer.stop()
        self.addCleanup(dialog.close); self.finish(dialog)
        return dialog, dialog.sensitivity_panel

    def test_loaded_and_unset_sliders_and_save_preserves_inheritance(self):
        box, page = self.settings(inference_conf=.7, conf_person=.5, conf_animal=.6)
        self.assertEqual({k: s.value() for k, s in page.sensitivities.sliders.items()}, dict(person=10, vehicle=14, animal=12))
        self.assertEqual(page.sensitivities.readouts['person'].text(), 'Needs to be 50% sure')
        page.save_clicked()
        self.assertIsNone(box.load_settings().conf_vehicle)
        self.assertEqual(box.load_settings().inference_conf, .7)

    def test_people_release_writes_only_conf_person_without_restart(self):
        box, page = self.settings(inference_conf=.7)
        with patch.object(box, 'save_settings', wraps=box.save_settings) as save:
            slider = page.sensitivities.sliders['person']; slider.setValue(10)
            save.assert_not_called(); slider.sliderReleased.emit()
            self.assertEqual(save.call_args.args[0].conf_person, .5)
            self.assertIsNone(save.call_args.args[0].conf_vehicle)
            self.assertEqual(box.load_settings().inference_conf, .7)
            self.assertIsNone(box.pending_at)
            self.assertEqual(page.sensitivity_note.text(), tr('applying'))
            page.check_applied(); self.assertEqual(page.sensitivity_note.text(), 'Applied')

    def test_reset_action_saves_all_three_and_keeps_old_value(self):
        box, page = self.settings(inference_conf=.85)
        with patch.object(box, 'save_settings', wraps=box.save_settings) as save:
            next(b for b in page.widget.findChildren(QPushButton) if b.text() == 'Reset to recommended').click()
            save.assert_called_once()
            self.assertEqual(box.load_settings().effective_sensitivity(), dict(person=.5, vehicle=.7, animal=.6))
            self.assertEqual(box.load_settings().inference_conf, .85)
            self.assertIsNone(box.pending_at)

    def test_keyboard_arrows_save_and_focus_is_visible(self):
        box, page = self.settings(conf_person=.5)
        page.widget.resize(1200, 800); page.widget.show(); self.app.processEvents()
        slider = page.sensitivities.sliders['person']; slider.setFocus(Qt.FocusReason.TabFocusReason)
        self.app.processEvents(); unfocused = None
        self.assertTrue(slider.hasFocus())
        focused = slider.grab().toImage()
        QTest.keyClick(slider, Qt.Key.Key_Right)
        self.assertEqual(box.load_settings().conf_person, .55)
        self.assertEqual(page.sensitivities.readouts['person'].text(), 'Needs to be 55% sure')
        QTest.keyClick(slider, Qt.Key.Key_Left)
        slider.clearFocus(); self.app.processEvents(); unfocused = slider.grab().toImage()
        self.assertNotEqual(focused, unfocused)

    def test_house_error_restores_saved_value(self):
        box, page = self.settings(conf_person=.5)
        with patch.object(box, 'save_settings', side_effect=ValueError('Disk full')):
            page.sensitivities.sliders['person'].setValue(8)
            page.sensitivities.sliders['person'].sliderReleased.emit()
        self.assertEqual(page.sensitivities.sliders['person'].value(), 10)
        self.assertEqual(page.sensitivity_note.text(), tr('control_error'))

    def test_camera_popover_exact_values_default_and_applied_local_remote(self):
        for remote in (False, True):
            with self.subTest(remote=remote):
                runner = RecordedSensitivityCommands()
                backend = RemoteCameras('user@box', runner=runner) if remote else CameraControls(BoxControls(), runner=runner.run)
                with patch.object(backend, 'alert_reported_status', side_effect=runner.status):
                    dialog, panel = self.dialog(backend)
                    self.assertEqual(panel.house_note.text(), 'People 50% · Vehicles 70% · Animals 60%')
                    self.assertTrue(panel.default.isChecked()); self.assertTrue(panel.sliders.isHidden())
                    panel.custom.click()
                    self.assertFalse(any('set-camera-sensitivity' in ' '.join(c) for c in runner.calls))
                    for kind, step in [('person', 10), ('vehicle', 16)]:
                        panel.sliders.sliders[kind].setValue(step); panel.sliders.sliders[kind].sliderReleased.emit()
                        self.finish(panel); self.finish(panel)
                    panel.last_status = 0; panel.poll(); self.finish(panel)
                    self.assertEqual(panel.note.text(), 'Applied')
                    self.assertEqual(runner.own, dict(person=.5, vehicle=.8))
                    self.assertTrue(dialog.default.isChecked())  # Independent radio groups.
                    panel.default.click(); self.finish(panel); self.finish(panel)
                    self.assertIsNone(runner.own); self.assertTrue(panel.sliders.isHidden())
                    commands = [' '.join(c) for c in runner.calls]
                    self.assertTrue(any(c.endswith('set-camera-sensitivity --camera driveway --values person=0.5,vehicle=0.8') for c in commands))
                    self.assertTrue(any(c.endswith('set-camera-sensitivity --camera driveway --default') for c in commands))
                    dialog.close()

    def test_camera_loaded_partial_value_preserves_other_types(self):
        box = BoxControls(demo=True, settings=Settings(inference_conf=.7, conf_person=.5))
        backend = CameraControls(box, ['driveway']); backend.set_camera_sensitivity('driveway', dict(vehicle=.8))
        dialog, panel = self.dialog(backend)
        self.assertEqual({k: s.value() for k, s in panel.sliders.sliders.items()}, dict(person=10, vehicle=16, animal=14))
        panel.sliders.sliders['vehicle'].setValue(17); panel.sliders.sliders['vehicle'].sliderReleased.emit(); self.finish(panel)
        self.assertEqual(box.camera_sensitivity, dict(driveway=dict(vehicle=.85)))

    def test_camera_command_error_and_retry_local_remote(self):
        for remote in (False, True):
            runner = RecordedSensitivityCommands()
            backend = RemoteCameras('user@box', runner=runner) if remote else CameraControls(BoxControls(), runner=runner.run)
            with patch.object(backend, 'alert_reported_status', side_effect=runner.status):
                dialog, panel = self.dialog(backend)
                runner.error = 'Camera driveway no longer exists.'
                panel.custom.click(); panel.sliders.sliders['person'].sliderReleased.emit(); self.finish(panel)
                self.assertEqual(panel.note.text(), runner.error); self.assertFalse(panel.retry.isHidden())
                self.assertTrue(panel.default.isChecked())
                runner.error = None; panel.retry.click(); self.finish(panel); self.finish(panel)
                self.assertEqual(runner.own, dict(person=.5)); dialog.close()

    def test_camera_old_engine_confirmation_times_out(self):
        box = BoxControls(demo=True); backend = CameraControls(box, ['driveway'])
        dialog, panel = self.dialog(backend)
        with patch.object(backend, 'alert_reported_status', return_value=dict(updated=time.time(), settings={})):
            panel.custom.click(); panel.sliders.sliders['person'].sliderReleased.emit(); self.finish(panel); self.finish(panel)
            panel.ack.requested = time.time()-11; panel.last_status = 0
            panel.poll(); self.finish(panel)
            self.assertEqual(panel.note.text(), tr('live_not_picked_up'))
            self.assertTrue(panel.ack.expected)

    def test_missing_house_values_disable_controls(self):
        backend = CameraControls(BoxControls(demo=True), ['driveway'])
        with patch.object(backend, 'camera_alerts', return_value=dict(house=['person'], cameras=[])):
            dialog, panel = self.dialog(backend)
        self.assertFalse(panel.custom.isEnabled()); self.assertFalse(panel.sliders.isEnabled())
        self.assertEqual(panel.house_note.text(), tr('sensitivity_unavailable'))

    def test_laptop_house_sensitivity_changes_and_confirmation(self):
        runner = RecordedSensitivityCommands(); backend = RemoteCameras('user@box', runner=runner)
        dialog, panel = self.dialog(backend, name=None)
        self.assertFalse(panel.sliders.isHidden())
        panel.sliders.sliders['person'].setValue(9); panel.sliders.sliders['person'].sliderReleased.emit()
        self.finish(panel); self.finish(panel)
        self.assertEqual(panel.note.text(), 'Applied')
        self.assertTrue(any(c[-1].endswith('set-option conf_person=0.45') for c in runner.calls))
        panel.reset.click(); self.finish(panel); self.finish(panel)
        self.assertEqual(runner.house, dict(person=.5, vehicle=.7, animal=.6))

    def test_card_dot_for_either_override_and_both_keeps_geometry(self):
        button = CameraAlertButton(['person'], None); self.addCleanup(button.close)
        button.resize(300, 32); button.show(); self.app.processEvents()
        geometry = button.geometry()
        for alerts, sensitivity in [(None, dict(person=.5)), (['person'], None), (['vehicle'], dict(animal=.6)), (None, None)]:
            button.refresh(['person'], alerts, sensitivity)
            self.assertEqual(button.custom, alerts is not None or sensitivity is not None)
            self.assertEqual(button.geometry(), geometry)
            self.assertNotIn('\n', button.text())
            if sensitivity: self.assertIn('Own sensitivity', button.toolTip())
            if alerts: self.assertIn('Own alert types', button.toolTip())

    def test_camera_page_load_and_signal_update_sensitivity_dot(self):
        box = BoxControls(demo=True); backend = CameraControls(box, ['driveway'])
        backend.set_camera_sensitivity('driveway', dict(person=.5))
        page = CameraPage(backend, lambda: None); self.addCleanup(page.close); self.addCleanup(page.widget.close)
        page.render(page.read_zones(backend.load()))
        self.assertTrue(page.alert_widgets['driveway'].custom)
        page.open_alerts('driveway'); dialog = page.alert_dialog
        dialog.timer.stop(); dialog.sensitivity_panel.timer.stop(); self.finish(dialog)
        panel = dialog.sensitivity_panel
        panel.default.click(); self.finish(panel); self.finish(panel)
        self.assertFalse(page.alert_widgets['driveway'].custom)
        dialog.close()
