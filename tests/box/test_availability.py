import json
import unittest
from types import SimpleNamespace
from unittest.mock import Mock,patch
from home_guard_project.box.app.availability import stage_state
from home_guard_project.box.app.box_controls import BoxControls
from home_guard_project.box.app.camera_controls import CameraControls
from home_guard_project.box.app.engine_backend import DemoEngine
from home_guard_project.box.app.backend import Answers

class AvailabilityTests(unittest.TestCase):
    def test_empty_and_unreachable_take_priority_over_hidden_pictures(self):
        self.assertEqual(stage_state(reachable=False,cameras=['front'],pictures=False),'box_unreachable')
        self.assertEqual(stage_state(reachable=True,cameras=[],pictures=False),'no_cameras')
        self.assertEqual(stage_state(reachable=True,cameras=['front'],pictures=False),'pictures_off')
        self.assertIsNone(stage_state(reachable=True,cameras=['front']))

    def test_finish_without_cameras_completes_alerts_and_readiness(self):
        events=[]
        self.assertTrue(DemoEngine('cameras').run(Answers(find_cameras=False),events.append,instant=True))
        self.assertFalse(any(e.kind=='camera' for e in events))
        self.assertTrue(any(e.step=='alerts' and e.status=='ok' for e in events))
        self.assertTrue(any(e.step=='readiness' and e.status=='ok' for e in events))

    def test_demo_search_adds_cameras(self):
        controls=CameraControls(BoxControls(demo=True))
        self.assertEqual(controls.load(),[])
        self.assertEqual(len(controls.search('admin','')),3)

    def test_search_password_never_in_command_and_empty_result_does_not_restart(self):
        runner=Mock(return_value=SimpleNamespace(returncode=1,stdout=json.dumps({'cameras':[],'saved':False})))
        controls=CameraControls(BoxControls(),runner=runner)
        with patch('home_guard_project.box.app.camera_controls.boxconfig.load_box_settings',return_value={'site':'house'}),patch('home_guard_project.box.control.request_restart') as restart:
            self.assertEqual(controls.search('admin','test-private'),[])
            restart.assert_not_called()
        argv=runner.call_args.args[0];env=runner.call_args.kwargs['env']
        self.assertNotIn('test-private',argv)
        self.assertEqual(env['HG_CAMERA_PASSWORD'],'test-private')
        self.assertNotIn('VIRTUAL_ENV',env)

if __name__=='__main__': unittest.main()
