import unittest
from home_guard_project.box.app.camera_check_state import CameraCheckState

class CameraCheckTests(unittest.TestCase):
    def test_failure_does_not_undo_success_and_details_hide_addresses(self):
        state=CameraCheckState(True,'cedar_house',('front_door','garden'))
        heading=state.heading()
        state.failure(RuntimeError('Invalid camera result'),'ssh: connect to host 192.0.2.10: Connection timed out','installer@box.example')
        self.assertTrue(state.setup_finished)
        self.assertTrue(state.photo_failed)
        self.assertEqual(state.heading(),heading)
        self.assertIn('Setup finished',heading);self.assertIn('cedar_house',heading);self.assertIn('2 cameras found',heading)
        self.assertIn('Connection timed out',state.details);self.assertNotIn('192.0.2.10',state.details)
        state.clear_failure();self.assertFalse(state.photo_failed);self.assertEqual(state.details,'')
    def test_check_only_does_not_claim_a_successful_setup(self):
        state=CameraCheckState()
        self.assertNotIn('Setup finished',state.heading())
        state.failure(RuntimeError('Command timed out'))
        self.assertIn('Command timed out',state.details)
