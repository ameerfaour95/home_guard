from PySide6.QtCore import Signal
from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QFrame, QLineEdit
from .widgets.common import label, button
from .widgets.totp import TotpInput
from .workers import TaskRunner
from .prefs import Preferences


class SignIn(QWidget):
    authenticated = Signal(object)
    demo_requested = Signal()

    def __init__(self, backend, prefs=None, environment='LOCAL'):
        super().__init__()
        self.backend = backend
        self.prefs = prefs or Preferences()
        self.runner = TaskRunner(self)
        self.runner.finished.connect(self.completed)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(48, 32, 48, 24)
        outer.setSpacing(16)
        brand = QHBoxLayout()
        brand.addWidget(label('HOME GUARD', 'section'))
        brand.addWidget(label('ADMIN CENTER', 'muted'))
        brand.addStretch()
        brand.addWidget(label(environment, 'badge'))
        outer.addLayout(brand)
        body = QHBoxLayout()
        body.setSpacing(96)
        story = QVBoxLayout()
        story.setSpacing(8)
        story.addStretch()
        story.addWidget(label('OPERATIONS, WITH CONTEXT', 'eyebrow'))
        story.addSpacing(8)
        story.addWidget(label('Every home.\nOne clear view.', 'hero'))
        story.addSpacing(16)
        story.addWidget(label('Keep an eye on the fleet. Understand every alert.\nHelp customers feel at home.', 'muted', True))
        story.addSpacing(48)
        story.addWidget(label('01   Monitor your fleet', 'section'))
        story.addWidget(label('Know which boxes need attention, and why.', 'muted'))
        story.addSpacing(16)
        story.addWidget(label('02   Review with confidence', 'section'))
        story.addWidget(label('Camera context, AI answers and owner feedback.', 'muted'))
        story.addStretch()
        body.addLayout(story, 1)
        card = QFrame()
        card.setObjectName('card')
        card.setFixedWidth(440)
        form = QVBoxLayout(card)
        form.setContentsMargins(40, 32, 40, 32)
        form.setSpacing(8)
        form.addWidget(label('Staff sign-in', 'title'))
        form.addWidget(label('Connect to Home Guard Cloud.', 'muted'))
        form.addSpacing(16)
        self.email = QLineEdit(self.prefs.email())
        self.email.setPlaceholderText('you@homeguard.com')
        self.email.setAccessibleName('Email')
        self.password = QLineEdit()
        self.password.setEchoMode(QLineEdit.EchoMode.Password)
        self.password.setAccessibleName('Password')
        for title, widget in [('Email', self.email), ('Password', self.password)]:
            form.addWidget(label(title))
            form.addWidget(widget)
            form.addSpacing(8)
        form.addWidget(label('6-digit authentication code'))
        self.totp = TotpInput()
        form.addWidget(self.totp)
        form.addWidget(label('From your authenticator app.', 'muted'))
        self.error = label('', 'error', True)
        self.error.setMinimumHeight(48)
        form.addWidget(self.error)
        self.submit = button('Sign in', self.authenticate, 'primary')
        form.addWidget(self.submit)
        self.demo = button('Use demo data', self.demo_requested.emit, 'link')
        form.addWidget(self.demo)
        body.addWidget(card)
        outer.addStretch()
        outer.addLayout(body)
        outer.addStretch()
        outer.addWidget(label('STAFF ACCESS ONLY    ·    Customer access is role-controlled and audited by Home Guard Cloud.', 'muted'))
        self.password.returnPressed.connect(lambda: self.totp.digits[0].setFocus())
        self.email.returnPressed.connect(self.password.setFocus)
        self.totp.submitted.connect(self.authenticate)

    def authenticate(self):
        if self.runner.busy:
            return
        email, password, code = self.email.text().strip(), self.password.text(), self.totp.code()
        if '@' not in email or not password or len(code) != 6:
            self.error.setText('Enter your email, password and 6-digit code.')
            return
        self.prefs.save_email(email)
        self.error.setText('')
        self.submit.setText('Signing in…')
        self.submit.setEnabled(False)
        self.demo.setEnabled(False)
        self.runner.start(lambda: self.backend.login(email, password, code))

    def completed(self, result, error):
        self.submit.setEnabled(True)
        self.demo.setEnabled(True)
        self.submit.setText('Sign in')
        self.password.clear()
        self.totp.clear()
        if error:
            self.error.setText(str(error))
        else:
            self.authenticated.emit(result.staff)
