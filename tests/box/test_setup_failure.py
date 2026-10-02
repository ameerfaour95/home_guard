import unittest
from home_guard_project.box.app.engine_backend import OutputParser, DemoEngine
from home_guard_project.box.app.backend import Answers
from home_guard_project.box.app.setup_failure import FailureFacts

class FailureTests(unittest.TestCase):
    def test_recorded_network_and_no_devices(self):
        parser=OutputParser(('private-password',))
        facts=FailureFacts()
        for line in ("@@step network start", "The box is on Wi-Fi 'ameer2' (192.168.68.120)", "@@step network ok Connected", "@@step cameras start", "WARNING No device answers on the camera port. Is the box on the cameras' network?", "@@step cameras fail no cameras were found. Check the camera login"):
            event=parser.parse(line);facts.feed(event)
            self.assertNotIn('192.168.68.120',event.text)
        title,body=facts.content('cameras')
        self.assertEqual(title,'No cameras found')
        self.assertIn('ameer2',body);self.assertIn('192.168.68.120',body)
        self.assertTrue(facts.no_devices)

    def test_recorded_login_counts(self):
        facts=FailureFacts();parser=OutputParser()
        for line in ('@@step cameras start','WARNING 2 of 4 device(s) refused this login: 192.168.1.2','WARNING 1 of 4 device(s) did not answer: 192.168.1.3','@@step cameras fail 2 of 4 cameras answered but refused this login.'):
            facts.feed(parser.parse(line))
        title,body=facts.content('cameras')
        self.assertEqual(title,'This login was refused by 2 of 4 devices')
        self.assertIn('admin',body)
        self.assertEqual(facts.address,'')

    def test_failure_headlines_are_not_log_lines(self):
        facts=FailureFacts(network='ameer2',message='raw command error')
        for step,expected in [('connect','The box did not answer'),('update','The box software could not be updated'),('network','The box did not come back on ameer2')]:
            self.assertEqual(facts.content(step)[0],expected)

    def test_demo_failure_at_each_step(self):
        for step in ('connect','update','network','cameras'):
            with self.subTest(step=step):
                events=[]
                self.assertFalse(DemoEngine(step).run(Answers(),events.append,instant=True))
                self.assertEqual(events[-1].step,step)
                self.assertEqual(events[-1].status,'fail')

if __name__=='__main__': unittest.main()
