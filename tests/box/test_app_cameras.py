import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch
from home_guard_project.box.app.box_controls import BoxControls
from home_guard_project.box.app.camera_controls import Camera, CameraControls, changes_payload

class CameraControlsTests(unittest.TestCase):
    def test_changes_names_and_enabled(self):
        self.assertEqual(changes_payload([('old', 'front_door', False)]), {'cameras': [{'name': 'old', 'new_name': 'front_door', 'enabled': False}]})
        for changes in ([('a', '', True)], [('a', 'Front door', True)], [('a','same',True),('b','same',False)], [('a','ok',1)], [('a','one',True),('a','two',True)]):
            with self.assertRaises(ValueError):
                changes_payload(changes)

    def test_demo_changes_persist_and_disabled_can_be_enabled(self):
        box = BoxControls(demo=True, clock=lambda: 100)
        api = CameraControls(box, ['Front Door', 'Garden'])
        with patch('home_guard_project.box.control.request_restart') as restart:
            api.save([('front_door','entrance',False), ('garden','garden',True)])
            self.assertEqual(api.load()[0], Camera('entrance', False, ok=True))
            api.save([('entrance','entrance',True), ('garden','garden',True)])
            self.assertTrue(api.snapshots()[0].enabled)
            self.assertEqual(box.pending_at,100)
            restart.assert_not_called()

    def test_disabled_metadata_has_no_addresses(self):
        api = CameraControls(BoxControls())
        with patch('home_guard_project.box.find_cameras._read_cameras_raw',return_value={'cameras':{'door':'secret'},'disabled':{'garden':'secret'}}):
            records=api.load()
        self.assertEqual(records,[Camera('door'),Camera('garden',False)])
        self.assertNotIn('secret',repr(records))

    def test_failed_snapshot_is_valid_and_environment_isolated(self):
        runner=Mock(return_value=Mock(returncode=1,stdout=json.dumps({'snapshots':[{'name':'door','file':'','ok':False}]})))
        api=CameraControls(BoxControls(),runner=runner)
        api.records=[Camera('door')]
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ,{'VIRTUAL_ENV':'wrong-checkout'}):
            api.out=Path(directory)
            self.assertFalse(api.snapshots()[0].ok)
        args, kwargs=runner.call_args
        self.assertNotIn('VIRTUAL_ENV',kwargs['env'])
        self.assertEqual(args[0][1:5],['-m','home_guard_project.box.find_cameras','--json','snapshots'])

    def test_apply_writes_names_only_and_cleans_up(self):
        captured=[]
        def runner(args,**kwargs):
            path=Path(args[-1]); captured.append(path)
            self.assertEqual(json.loads(path.read_text()),changes_payload([('door','entrance',False)]))
            return Mock(returncode=0,stdout='{"active": [], "disabled": ["entrance"]}')
        box=BoxControls(clock=lambda:100)
        api=CameraControls(box,runner=runner); api.records=[Camera('door')]
        with tempfile.TemporaryDirectory() as directory, patch('home_guard_project.box.control.is_stopped',return_value=False):
            api.out=Path(directory)
            api.save([('door','entrance',False)])
            self.assertFalse(captured[0].exists())
            self.assertEqual(api.records,[Camera('entrance',False)])
            self.assertEqual(box.pending_at,100)

    def test_apply_failure_preserves_records(self):
        api=CameraControls(BoxControls(),runner=Mock(return_value=Mock(returncode=1,stdout='{"error":"private diagnostic"}')))
        api.records=[Camera('door')]
        with tempfile.TemporaryDirectory() as directory:
            api.out=Path(directory)
            with self.assertRaisesRegex(ValueError,'Camera command failed'):
                api.save([('door','entrance',False)])
        self.assertEqual(api.records,[Camera('door')])
        self.assertIsNone(api.box.pending_at)

    def test_snapshot_path_must_stay_in_photo_directory(self):
        api=CameraControls(BoxControls(),runner=Mock(return_value=Mock(returncode=0,stdout=json.dumps({'snapshots':[{'name':'door','ok':True,'file':'C:/outside.jpg'}]}))))
        api.records=[Camera('door')]
        with tempfile.TemporaryDirectory() as directory:
            api.out=Path(directory)
            with self.assertRaises(ValueError):
                api.snapshots()
