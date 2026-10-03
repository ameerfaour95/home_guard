import json
import unittest
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock, patch
from home_guard_project.box.app.box_controls import Settings, BoxControls, RemoteSettingsBackend
from home_guard_project.box.app.camera_controls import CameraControls
from home_guard_project.box.app.remote_cameras import RemoteCameras
from home_guard_project.box.app.live_settings import AppliedState


class SensitivityDataTests(unittest.TestCase):
    def test_optional_values_and_validation(self):
        settings = Settings.from_options({'inference_conf': .7, 'conf_person': .5})
        self.assertEqual(settings.effective_sensitivity(), dict(person=.5, vehicle=.7, animal=.7))
        self.assertIsNone(settings.conf_vehicle)
        self.assertEqual(Settings.from_options(settings.options()), settings)
        for kind in ('person', 'vehicle', 'animal'):
            for value in (True, '0.5', .04, .96, float('nan'), float('inf')):
                with self.subTest(kind=kind, value=value), self.assertRaises(ValueError):
                    Settings(**{'conf_' + kind: value})

    def test_local_house_reads_and_writes_live(self):
        original = Settings(inference_conf=.7, conf_animal=.6)
        with patch('home_guard_project.box.app.box_controls.boxconfig.get_option', side_effect=original.options().get), patch('home_guard_project.box.app.box_controls.boxconfig.set_option') as write, patch('home_guard_project.box.app.box_controls.control.request_restart') as restart:
            box = BoxControls()
            self.assertEqual(box.load_settings(), original)
            self.assertFalse(box.save_settings(replace(original, conf_person=.5, conf_vehicle=.8)))
            self.assertEqual([c.args for c in write.call_args_list], [('conf_person', '0.5'), ('conf_vehicle', '0.8')])
            restart.assert_not_called()

    def test_remote_house_save_and_readback(self):
        runner = Mock(); runner.run.return_value = SimpleNamespace(returncode=0)
        box = RemoteSettingsBackend('user@box', Settings.from_options({'conf_animal': .6}), runner, lambda: {})
        self.assertFalse(box.save_settings(replace(box.load_settings(), conf_person=.5, conf_vehicle=.7)))
        self.assertEqual([c.args[0][-1].split('set-option ')[1] for c in runner.run.call_args_list], ['conf_person=0.5', 'conf_vehicle=0.7'])
        self.assertEqual(box.load_settings().effective_sensitivity(), dict(person=.5, vehicle=.7, animal=.6))

    def test_camera_commands_and_reads_local_remote(self):
        for remote in (False, True):
            runner = Mock()
            backend = RemoteCameras('user@box', runner=runner) if remote else CameraControls(BoxControls(), runner=runner)
            run = runner.run if remote else runner
            rows = dict(house=['person'], house_sensitivity=dict(person=.5, vehicle=.7, animal=.6), cameras=[dict(name='yard', alert_on=None, sensitivity=dict(person=.45))])
            run.return_value = SimpleNamespace(returncode=0, stdout=json.dumps(rows))
            self.assertEqual(backend.camera_alerts(), rows)
            for value, suffix in [(dict(vehicle=.8, person=.5), '--values person=0.5,vehicle=0.8'), (None, '--default')]:
                run.return_value = SimpleNamespace(returncode=0, stdout=json.dumps(dict(camera='yard', sensitivity=value)))
                self.assertEqual(backend.set_camera_sensitivity('yard', value), value)
                self.assertTrue(' '.join(run.call_args.args[0]).endswith('set-camera-sensitivity --camera yard ' + suffix))
            run.return_value = SimpleNamespace(returncode=1, stdout='{"error":"Camera yard no longer exists."}')
            with self.assertRaisesRegex(ValueError, 'Camera yard no longer exists'):
                backend.set_camera_sensitivity('yard', dict(person=.5))
            with self.assertRaises(ValueError): backend.set_camera_sensitivity('yard & exit', dict(person=.5))

    def test_demo_report_partial_override_reset_and_rename(self):
        box = BoxControls(demo=True, settings=Settings(inference_conf=.7, conf_person=.5))
        cameras = CameraControls(box, ['yard'])
        cameras.set_camera_sensitivity('yard', dict(vehicle=.8))
        self.assertEqual(cameras.camera_alerts()['cameras'][0]['sensitivity'], dict(vehicle=.8))
        self.assertEqual(box.reported_status()['settings']['sensitivity'], dict(person=.5, vehicle=.7, animal=.7))
        self.assertIsNone(box.pending_at)
        cameras.save([('yard', 'garden', True)])
        self.assertEqual(box.reported_status()['settings']['camera_sensitivity'], dict(garden=dict(vehicle=.8)))
        cameras.set_camera_sensitivity('garden', None)
        self.assertEqual(box.reported_status()['settings']['camera_sensitivity'], {})

    def test_house_confirmation_requires_new_keys_and_fresh_status(self):
        ack = AppliedState(); ack.request(Settings(), Settings(conf_person=.5, conf_vehicle=.7), 100)
        self.assertEqual(ack.status({'updated': 101, 'settings': {'conf': .5}}, 101), 'applying')
        self.assertEqual(ack.status({'updated': 110, 'settings': {'conf': .5}}, 110), 'live_not_picked_up')
        data = dict(updated=80, settings=dict(sensitivity=dict(person=.5, vehicle=.7)))
        self.assertEqual(ack.status(data, 110), 'live_not_picked_up')
        data['updated'] = 110
        self.assertEqual(ack.status(data, 110), 'applied')

    def test_camera_confirmation_default_partial_and_old_engine(self):
        for value in (None, dict(person=.5, vehicle=.8)):
            ack = AppliedState(); ack.request_camera_sensitivity('yard', value, 100)
            self.assertEqual(ack.status(dict(updated=110, settings={}), 110), 'live_not_picked_up')
            wrong = dict(yard=dict(person=.5, vehicle=.8, animal=.6))
            self.assertEqual(ack.status(dict(updated=110, settings=dict(camera_sensitivity=wrong)), 110), 'live_not_picked_up')
            own = {} if value is None else dict(yard=value)
            self.assertEqual(ack.status(dict(updated=110, settings=dict(camera_sensitivity=own)), 110), 'applied')
