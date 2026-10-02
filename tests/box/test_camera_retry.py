import unittest
from home_guard_project.box.app.backend import Answers
from home_guard_project.box.app.camera_retry import CameraRetry

class CameraRetryTests(unittest.TestCase):
    def test_retry_keeps_answers_despite_engine_cleanup(self):
        import secrets
        password=secrets.token_hex(8)
        answers=Answers(address='installer@box.example', network='wifi', ssid='Home', wifi_password=password, house='cedar_house', cooldown_sec=180)
        retry=CameraRetry();retry.capture(answers);retry.started();retry.failed('4 cameras refused this login')
        answers.wifi_password=''
        result=retry.retry('admin',secrets.token_hex(8))
        self.assertEqual(result.wifi_password,password)
        self.assertEqual((result.address,result.house,result.cooldown_sec),('installer@box.example','cedar_house',180))
        self.assertFalse(retry.lock_warning)
        retry.started();retry.failed('Login refused')
        self.assertTrue(retry.lock_warning)
        retained=retry.answers;retry.clear()
        self.assertEqual(retained.wifi_password,'')
        self.assertFalse(retry.pending)
