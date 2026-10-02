import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock,patch
from types import SimpleNamespace
from home_guard_project.box.app.alert_hours import hours_description
from home_guard_project.box.app.alert_pause import AlertPause
from home_guard_project.box.app.backend import Answers

class TruthTests(unittest.TestCase):
    def test_hours_default_and_cross_midnight(self):
        self.assertEqual(Answers().start_hour,Answers().end_hour)
        self.assertIn('all day',hours_description(0,0))
        self.assertIn('22:00 tonight until 06:00 tomorrow',hours_description(22,6))
        self.assertIn('08:00 until 17:00',hours_description(8,17))
    def test_pause_global_camera_expired_and_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'mute.json'
            path.write_text(json.dumps({'all':200,'cameras':{'door':300}}))
            pause=AlertPause(path=path,clock=lambda:100)
            self.assertEqual(pause.status(['door']),(200,False))
            path.write_text(json.dumps({'all':0,'cameras':{'door':300,'garden':50}}))
            self.assertEqual(pause.status(['door','garden']),(300,True))
            pause.resume()
            self.assertEqual(pause.status(['door']),(None,True))
    def test_demo_pause_never_writes_file(self):
        pause=AlertPause(demo=True,clock=lambda:100)
        pause.demo_until=200
        with patch('home_guard_project.box.app.alert_pause.MuteState') as mute:
            self.assertEqual(pause.status(['door']),(200,False))
            pause.resume()
            self.assertEqual(pause.status(['door']),(None,False))
            mute.assert_not_called()
    def test_fetch_uses_command_line_folder_choice(self):
        from home_guard_project.box.app.ui import Window
        from home_guard_project.box import boxconfig as bc
        fake=SimpleNamespace(activity_feed=Mock(),reader=Mock())
        fake.activity_feed.read.return_value=([],'')
        fake.reader.names.return_value=[]
        for mode, dirs in [('inference',(bc.PRODUCTION_LIVE_DIR,bc.PRODUCTION_ARCHIVE_DIR)),('data_collection',(bc.LIVE_DIR,bc.OUTBOX_DIR))]:
            with patch.object(bc,'load_box_settings',return_value={'site':'home'}),patch.object(bc,'get_option',side_effect=lambda key:mode if key=='mode' else False),patch('home_guard_project.box.heartbeat.build_heartbeat',return_value={}) as build:
                Window.fetch(fake)
                self.assertEqual(build.call_args.args[1:3],dirs)
