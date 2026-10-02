import os
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
import unittest
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock, patch
from home_guard_project.box.app.box_controls import Settings, BoxControls, RemoteSettingsBackend
from home_guard_project.box.app.live_settings import AppliedState, slider_conf, conf_slider

class LiveSettingsTests(unittest.TestCase):
    def test_slider_and_validation(self):
        for step in range(1,20):
            self.assertEqual(conf_slider(slider_conf(step)),step)
            Settings(inference_conf=slider_conf(step))
        for invalid in (0,.049,.951,1,float('nan'),True,'0.4'):
            with self.assertRaises(ValueError): Settings(inference_conf=invalid)

    def test_runtime_tables_choose_restart(self):
        original=Settings()
        with patch('home_guard_project.box.app.box_controls.boxconfig.get_option',side_effect=lambda key:original.options()[key]), patch('home_guard_project.box.app.box_controls.boxconfig.set_option') as write, patch('home_guard_project.box.app.box_controls.control.request_restart') as restart:
            box=BoxControls()
            self.assertFalse(box.save_settings(replace(original,inference_conf=.35,alert_start_hour=22,alert_end_hour=6,alert_cooldown_sec=300)))
            restart.assert_not_called()
            self.assertIn(unittest.mock.call('inference_conf','0.35'),write.call_args_list)
            with patch.object(box,'is_stopped',return_value=False):
                self.assertTrue(box.save_settings(replace(original,mode='inference')))
            restart.assert_called_once()

    def test_recorded_applied_status(self):
        before=Settings(); after=replace(before,inference_conf=.35,alert_start_hour=22,alert_end_hour=6,alert_cooldown_sec=300)
        ack=AppliedState();ack.request(before,after,100)
        data={'updated':102,'settings':{'conf':.35,'alert_start_hour':22,'alert_end_hour':6,'cooldown_sec':300}}
        self.assertEqual(ack.status({},102),'applying')
        self.assertEqual(ack.status(dict(data,updated=1),110),'live_not_picked_up')
        self.assertEqual(ack.status(data,103),'applied')
        self.assertFalse(ack.expected)
        ack.request(before,after,100)
        data['settings']['cooldown_sec']=120
        self.assertEqual(ack.status(data,110),'live_not_picked_up')

    def test_remote_decimal_passes_unchanged(self):
        runner=Mock();runner.run.return_value=SimpleNamespace(returncode=0)
        backend=RemoteSettingsBackend('installer@box',Settings(),runner,lambda:{},key='test-key')
        self.assertFalse(backend.save_settings(replace(Settings(),inference_conf=.35)))
        self.assertTrue(runner.run.call_args.args[0][-1].endswith('set-option inference_conf=0.35'))
        self.assertEqual(backend.load_settings().inference_conf,.35)

    def test_slider_only_writes_on_release(self):
        from PySide6.QtWidgets import QApplication
        from home_guard_project.box.app.settings_ui import SettingsPage
        app=QApplication.instance() or QApplication([])
        box=BoxControls(demo=True); page=SettingsPage(box,lambda:None);page.reload()
        with patch.object(box,'save_settings',wraps=box.save_settings) as save:
            page.sensitivity.setValue(7)
            save.assert_not_called()
            page.sensitivity.sliderReleased.emit()
            save.assert_called_once()
            self.assertEqual(box.load_settings().inference_conf,.35)
            page.check_applied()
            self.assertEqual(page.note.text(),'Applied')
            page.sensitivity.sliderReleased.emit()
            save.assert_called_once()
        page.widget.close();page.timer.stop()
