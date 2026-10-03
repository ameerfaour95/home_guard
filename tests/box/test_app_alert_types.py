import json
import unittest
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock, patch

from home_guard_project.box.app.box_controls import BoxControls, Settings, RemoteSettingsBackend
from home_guard_project.box.app.camera_controls import CameraControls
from home_guard_project.box.app.remote_cameras import RemoteCameras
from home_guard_project.box.app.live_settings import AppliedState


class AlertDataTests(unittest.TestCase):
    def test_settings_normalize_and_default(self):
        self.assertEqual(Settings(alert_on='vehicle, person,person').alert_on, 'person,vehicle')
        self.assertEqual(Settings.from_options({'alert_on': None}).alert_on, 'person')
        self.assertEqual(Settings.from_options(Settings(alert_on='animal').options()).alert_on, 'animal')
        for value in ('', 'bird', 'person,bird', None, True):
            with self.subTest(value=value), self.assertRaises(ValueError): Settings(alert_on=value)

    def test_local_house_reads_and_writes_without_restart(self):
        original = Settings()
        with patch('home_guard_project.box.app.box_controls.boxconfig.get_option', side_effect=original.options().get), patch('home_guard_project.box.app.box_controls.boxconfig.set_option') as write, patch('home_guard_project.box.app.box_controls.control.request_restart') as restart:
            box = BoxControls()
            self.assertFalse(box.save_settings(replace(box.load_settings(), alert_on='vehicle,person')))
            write.assert_called_once_with('alert_on', 'person,vehicle')
            restart.assert_not_called()

    def test_remote_house_command_and_readback(self):
        runner = Mock(); runner.run.return_value = SimpleNamespace(returncode=0)
        box = RemoteSettingsBackend('user@box', Settings(), runner, lambda: {})
        self.assertFalse(box.save_settings(replace(box.load_settings(), alert_on='vehicle,person')))
        self.assertTrue(runner.run.call_args.args[0][-1].endswith('set-option alert_on=person,vehicle'))
        self.assertEqual(box.load_settings().alert_on, 'person,vehicle')

    def test_demo_camera_choices_and_report(self):
        box = BoxControls(demo=True)
        cameras = CameraControls(box, ['driveway', 'yard'])
        cameras.set_camera_alerts('driveway', 'vehicle,person')
        self.assertEqual(cameras.camera_alerts()['cameras'][0]['alert_on'], ['person', 'vehicle'])
        self.assertIsNone(cameras.camera_alerts()['cameras'][1]['alert_on'])
        self.assertEqual(box.reported_status()['settings']['camera_alert_on'], {'driveway': ['person', 'vehicle']})
        box.save_settings(replace(box.load_settings(), alert_on='animal'))
        self.assertEqual(cameras.camera_alerts()['house'], ['animal'])
        cameras.set_camera_alerts('driveway', None)
        self.assertEqual(box.reported_status()['settings']['camera_alert_on'], {})
        self.assertIsNone(box.pending_at)

    def test_camera_commands_local_and_remote(self):
        for remote in (False, True):
            with self.subTest(remote=remote):
                runner = Mock()
                backend = RemoteCameras('user@box', runner=runner) if remote else CameraControls(BoxControls(), runner=runner)
                run = runner.run if remote else runner
                rows = {'house': ['person'], 'cameras': [{'name': 'yard', 'alert_on': None}]}
                run.return_value = SimpleNamespace(returncode=0, stdout=json.dumps(rows))
                self.assertEqual(backend.camera_alerts(), rows)
                for value, suffix in [('vehicle,person', '--on person,vehicle'), (None, '--default')]:
                    run.return_value = SimpleNamespace(returncode=0, stdout=json.dumps({'camera': 'yard', 'alert_on': ['person', 'vehicle'] if value else None}))
                    backend.set_camera_alerts('yard', value)
                    args = run.call_args.args[0]
                    self.assertTrue((args[-1] if remote else ' '.join(args)).endswith('set-camera-alerts --camera yard ' + suffix))
                run.return_value = SimpleNamespace(returncode=1, stdout='{"error":"Camera yard no longer exists."}')
                with self.assertRaisesRegex(ValueError, 'Camera yard no longer exists'): backend.set_camera_alerts('yard', 'person')
                with self.assertRaises(ValueError): backend.set_camera_alerts('yard & exit', 'person')

    def test_live_house_confirmation_and_old_engine_timeout(self):
        ack = AppliedState(); ack.request(Settings(), Settings(alert_on='person,vehicle'), 100)
        self.assertEqual(ack.status({'updated': 101, 'settings': {}}, 101), 'applying')
        self.assertEqual(ack.status({'updated': 110, 'settings': {}}, 110), 'live_not_picked_up')
        self.assertEqual(ack.status({'updated': 110, 'settings': {'alert_on': ['vehicle', 'person']}}, 110), 'applied')

    def test_camera_confirmation_distinguishes_old_engine_and_default(self):
        for value in (None, ['person', 'vehicle']):
            ack = AppliedState(); ack.request_camera('yard', value, 100)
            self.assertEqual(ack.status({'updated': 110, 'settings': {}}, 110), 'live_not_picked_up')
            own = {} if value is None else {'yard': ['vehicle', 'person']}
            data = {'updated': 1, 'settings': {'camera_alert_on': own}}
            self.assertEqual(ack.status(data, 110), 'live_not_picked_up')
            data['updated'] = 110
            self.assertEqual(ack.status(data, 110), 'applied')
