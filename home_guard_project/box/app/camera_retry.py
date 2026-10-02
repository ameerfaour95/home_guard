"""Keep a retry in memory without exposing passwords in representations."""
from dataclasses import replace

class CameraRetry:
    def __init__(self):
        self.answers = None
        self.attempts = 0
        self.message = ''
        self.pending = False
    def capture(self, answers):
        if not self.pending:
            self.clear()
            self.answers = replace(answers, camera_password='')
    def started(self):
        self.attempts += 1
    def failed(self, message):
        self.message = message
        self.pending = True
    @property
    def lock_warning(self):
        return self.attempts >= 2
    def retry(self, user, password):
        return replace(self.answers, camera_user=user, camera_password=password)
    def clear(self):
        if self.answers:
            self.answers.wifi_password = ''
            self.answers.camera_password = ''
        self.answers = None
        self.attempts = 0
        self.message = ''
        self.pending = False
