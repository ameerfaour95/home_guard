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

class GuidanceTests(unittest.TestCase):
    def test_field_guidance_and_valid_values(self):
        from home_guard_project.box.app.guidance import field_errors
        self.assertEqual(field_errors(0,{'address':'wrong'}),{'address':'address_error'})
        self.assertEqual(field_errors(0,{'address':'installer@box.example'}),{})
        self.assertEqual(field_errors(1,{},wifi=True),{'ssid':'ssid_error','wifi_password':'wifi_password_error'})
        self.assertEqual(field_errors(1,{},wifi=False),{})
        self.assertEqual(field_errors(2,{'house':'Cedar House'}),{'house':'house_error'})
        self.assertEqual(field_errors(2,{'house':'cedar_house'}),{})
        self.assertEqual(field_errors(3,{},find=False),{})
        self.assertEqual(field_errors(3,{},find=True),{'camera_user':'camera_user_error','camera_password':'camera_password_error'})
    def test_retry_routes_to_step_owner(self):
        from home_guard_project.box.app.guidance import retry_page
        self.assertEqual([retry_page(step) for step in ('update','name_step','network_step','camera_step','readiness')],[0,2,1,3,0])
    def test_activity_clock_and_plain_words(self):
        from home_guard_project.box.app.model import parse_activity, Activity
        event=parse_activity('2026-10-02 14:31:02 INFO Done. Uploaded: 5 | Failed: 0')
        self.assertEqual(event.time,'14:31')
        self.assertEqual(event.text,'5 files sent to the online folder')
        self.assertEqual(parse_activity('14:29:00 Starting inference').text,'Home Guard restarted')
        self.assertEqual(Activity('event','undated').time,'Time unavailable')
