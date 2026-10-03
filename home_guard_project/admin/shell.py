from urllib.parse import urlsplit
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QShortcut, QKeySequence
from PySide6.QtWidgets import QWidget, QFrame, QVBoxLayout, QHBoxLayout, QStackedWidget, QLineEdit, QMainWindow
from .demo_backend import DemoBackend
from .signin import SignIn
from .fleet import FleetScreen
from .customer import CustomerScreen
from .widgets.common import label, button, EmptyState
from .widgets.palette import CommandPalette


class SearchField(QLineEdit):
    requested = Signal()

    def focusInEvent(self, event):
        super().focusInEvent(event)
        if event.reason() != Qt.FocusReason.ActiveWindowFocusReason:
            self.requested.emit()

    def mousePressEvent(self, event):
        super().mousePressEvent(event)
        self.requested.emit()


class Shell(QWidget):
    signed_out = Signal()
    session_expired = Signal()

    def __init__(self, backend, staff, environment='DEMO', theme='dark'):
        super().__init__()
        self.backend, self.staff = backend, staff
        self.devices, self.customers = [], []
        self.palette_dialog = None
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        rail = QFrame()
        rail.setObjectName('rail')
        rail.setFixedWidth(184)
        nav_layout = QVBoxLayout(rail)
        nav_layout.setContentsMargins(16, 24, 16, 16)
        nav_layout.setSpacing(8)
        nav_layout.addWidget(label('HOME GUARD', 'section'))
        nav_layout.addWidget(label('ADMIN CENTER', 'eyebrow'))
        nav_layout.addSpacing(40)
        self.navigation = {}
        self.pages = QStackedWidget()
        self.screens = {}
        allowed = ['Review', 'Studio'] if staff.role == 'labeler' else ['Fleet', 'Review', 'Studio'] + (['Audit'] if staff.role == 'admin' else [])
        for title in allowed:
            nav = button(title, lambda checked=False, name=title: self.navigate(name), 'nav')
            nav.setCheckable(True)
            nav_layout.addWidget(nav)
            self.navigation[title] = nav
            if title == 'Fleet':
                page = FleetScreen(backend, theme)
                self.fleet = page
                page.customer_requested.connect(self.open_customer)
                page.loaded.connect(self.fleet_loaded)
                page.session_expired.connect(self.session_expired)
            else:
                descriptions = {'Review': 'Review AI decisions alongside the camera footage and owner feedback.',
                                'Studio': 'Turn reviewed events into consented, versioned training datasets.',
                                'Audit': 'See who accessed a customer recording or changed a setting.'}
                page = EmptyState(f'{title} is coming in this release', descriptions[title], eyebrow=title.upper())
            self.screens[title] = page
            self.pages.addWidget(page)
        nav_layout.addStretch()
        nav_layout.addWidget(label('Home Guard Cloud\nStaff workspace', 'muted'))
        nav_layout.addWidget(button('Sign out', self.signed_out.emit, 'link'))
        layout.addWidget(rail)
        main = QVBoxLayout()
        main.setContentsMargins(0, 0, 0, 0)
        main.setSpacing(0)
        top = QFrame()
        top.setObjectName('topbar')
        top.setFixedHeight(80)
        bar = QHBoxLayout(top)
        bar.setContentsMargins(32, 16, 32, 16)
        self.search = SearchField()
        self.search.setReadOnly(True)
        self.search.setPlaceholderText('Search customers and devices…                         Ctrl+K')
        self.search.setAccessibleName('Open command palette')
        self.search.setMaximumWidth(480)
        self.search.requested.connect(self.open_palette)
        self.search.setEnabled(staff.role != 'labeler')
        if staff.role == 'labeler':
            self.search.setPlaceholderText('Review and training workspace')
        bar.addWidget(self.search, 1)
        bar.addStretch()
        bar.addWidget(label(environment, 'badge'))
        bar.addSpacing(16)
        bar.addWidget(label(staff.name + '   /   ' + staff.role.title()))
        main.addWidget(top)
        main.addWidget(self.pages, 1)
        layout.addLayout(main, 1)
        self.customer_page = None
        if staff.role != 'labeler':
            self.customer_page = CustomerScreen(backend, theme)
            self.customer_page.back.connect(lambda: self.navigate('Fleet'))
            self.customer_page.session_expired.connect(self.session_expired)
            self.pages.addWidget(self.customer_page)
        shortcut = QShortcut(QKeySequence('Ctrl+K'), self)
        shortcut.activated.connect(self.open_palette)
        self.navigate('Studio' if staff.role == 'labeler' else 'Fleet')

    def navigate(self, title):
        if title not in self.screens:
            return
        self.pages.setCurrentWidget(self.screens[title])
        for name, nav in self.navigation.items():
            nav.setChecked(name == title)

    def fleet_loaded(self, snapshot, customers):
        self.devices, self.customers = snapshot.devices, customers

    def open_customer(self, customer_id, device_id=''):
        if self.customer_page is None:
            return
        self.pages.setCurrentWidget(self.customer_page)
        self.customer_page.open(customer_id, device_id)

    def open_palette(self):
        if self.staff.role == 'labeler':
            return
        if self.palette_dialog and self.palette_dialog.isVisible():
            return
        self.palette_dialog = CommandPalette(self, self.devices, self.customers, self.staff.role)
        self.palette_dialog.jump.connect(self.open_customer)
        self.palette_dialog.move(self.mapToGlobal(self.rect().center()) - self.palette_dialog.rect().center())
        self.palette_dialog.show()
        self.palette_dialog.search.setFocus()


class AdminWindow(QMainWindow):
    def __init__(self, backend, *, demo=False, theme='dark', prefs=None):
        super().__init__()
        self.backend, self.theme, self.prefs = backend, theme, prefs
        self.configured_backend = backend
        self.setWindowTitle('Home Guard · Admin Center')
        self.resize(1366, 768)
        self.setMinimumSize(1200, 720)
        self.session = QStackedWidget()
        self.setCentralWidget(self.session)
        self.signin = None
        self.shell = None
        self.show_signin()
        if demo:
            self.use_demo()

    def environment(self):
        if isinstance(self.backend, DemoBackend):
            return 'DEMO'
        return 'LOCAL' if urlsplit(self.backend.base_url).hostname in ('127.0.0.1', 'localhost', '::1') else 'CLOUD'

    def show_signin(self):
        if self.shell:
            if self.shell.palette_dialog:
                self.shell.palette_dialog.close()
            if hasattr(self.shell, 'fleet'):
                self.shell.fleet.timer.stop()
            self.session.removeWidget(self.shell)
            self.shell.deleteLater()
            self.shell = None
        self.backend = self.configured_backend
        if self.signin:
            self.session.removeWidget(self.signin)
            self.signin.deleteLater()
        if hasattr(self.backend, 'tokens'):
            self.backend.tokens = None
        self.signin = SignIn(self.backend, self.prefs, self.environment())
        self.signin.authenticated.connect(self.enter)
        self.signin.demo_requested.connect(self.use_demo)
        self.session.addWidget(self.signin)
        self.session.setCurrentWidget(self.signin)

    def use_demo(self):
        self.backend = DemoBackend()
        self.enter(self.backend.me())

    def enter(self, staff):
        self.shell = Shell(self.backend, staff, self.environment(), self.theme)
        self.shell.signed_out.connect(self.show_signin)
        self.shell.session_expired.connect(self.expired)
        self.session.addWidget(self.shell)
        self.session.setCurrentWidget(self.shell)

    def expired(self):
        self.show_signin()
        self.signin.error.setText('Your session expired. Sign in again.')
