import unittest
from home_guard_project.box.app.engine_backend import OutputParser
from home_guard_project.box.app.setup_details import DetailsModel,DetailsSelection

class TimelineTests(unittest.TestCase):
    def test_grouped_recording_retains_time_roles_and_scoped_technical_log(self):
        model=DetailsModel();parser=OutputParser(('secret-test-value',))
        for line in ('@@step update start','22:19:00 INFO Updating the box software','@@step update ok Software is ready','@@step cameras start','22:20:01 WARNING 2 of 4 device(s) refused this login: 192.168.68.101','22:20:03 INFO Channel 2: OK (2688x1520)','@@step cameras fail This camera login was refused.'):
            model.feed(parser.parse(line),now=0)
        self.assertTrue(any(stamp=='22:20:01' and role=='error' for stamp,role,text in model.groups['cameras'].entries))
        self.assertTrue(any(role=='ok' for stamp,role,text in model.groups['cameras'].entries))
        self.assertNotIn('Updating',model.technical_log('cameras'))
        self.assertNotIn('192.168.68.101',model.technical_log('cameras'))
        self.assertIn('[address hidden]',model.technical_log('cameras'))
    def test_following_stops_on_user_selection_and_can_resume(self):
        parser=OutputParser();selection=DetailsSelection()
        selection.feed(parser.parse('@@step update start'),'update');self.assertEqual(selection.selected,'update')
        selection.choose('connect');selection.feed(parser.parse('@@step network start'),'network')
        self.assertEqual(selection.selected,'connect');self.assertFalse(selection.following)
        selection.resume('network');self.assertTrue(selection.following);self.assertEqual(selection.selected,'network')
    def test_failure_selects_the_failed_step(self):
        selection=DetailsSelection();selection.choose('connect')
        selection.feed(OutputParser().parse('@@step cameras fail No cameras found'),'cameras')
        self.assertEqual(selection.selected,'cameras')
    def test_rescue_secrets_never_enter_any_step_transcript(self):
        model=DetailsModel();model.feed(OutputParser().parse('@@rescue rescue private-value'))
        self.assertEqual(model.technical_log(),'')
        self.assertFalse(any(group.raw for group in model.groups.values()))
