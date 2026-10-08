from urllib.parse import urlsplit
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QShortcut, QKeySequence
from PySide6.QtWidgets import QWidget, QFrame, QVBoxLayout, QHBoxLayout, QStackedWidget, QLineEdit, QMainWindow, QLabel
from .demo_backend import DemoBackend
from .signin import SignIn
from .fleet import FleetScreen
from .customer import CustomerScreen
from .review import ReviewScreen
from .studio import StudioScreen
from .label_view import LabelView
from .tag_view import TagView
from .audit import AuditScreen
from .widgets.common import label, button, EmptyState
from .widgets.palette import CommandPalette
from .widgets.icons import icon
from .workers import TaskRunner
from .backend import AuthError
from .nav import NAV, ACCESS, SCREEN_OF


def inbox_screen(backend, role, theme):
    """The Studio's InboxScreen (admin/inbox.py) once it exists; until then a placeholder in its slot."""
    try:
        from .inbox import InboxScreen
    except ModuleNotFoundError as error:
        if error.name != f'{__package__}.inbox':
            raise
        return EmptyState('Inbox arrives with the Studio update',
                          'Owner answers from Telegram that wait for confirmation will be listed here.', eyebrow='INBOX')
    return InboxScreen(backend, role, theme)


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
        nav_layout.addSpacing(16)
        self.navigation = {}
        self.pages = QStackedWidget()
        self.screens = {}
        allowed = ACCESS[staff.role]
        for section, items in NAV:
            items = [(name, text) for name, text in items if name in allowed]
            if not items:
                continue
            nav_layout.addSpacing(8)
            nav_layout.addWidget(label(section, 'eyebrowMuted'))
            for title, text in items:
                nav = button(text, lambda checked=False, name=title: self.navigate(name), 'nav')
                nav.setIcon(icon(title, theme))
                nav.setAccessibleName(text)
                if title == 'Review':
                    self.review_badge = label('…', 'countBadge')
                    self.review_badge.setParent(nav)
                    self.review_badge.setGeometry(112, 12, 28, 22)
                    self.review_badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
                    self.review_badge.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
                    nav.setToolTip('Unreviewed events in the last 24 hours')
                nav.setCheckable(True)
                nav_layout.addWidget(nav)
                self.navigation[title] = nav
                page = self.make_screen(title, backend, staff, theme)
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
        self.search.setPlaceholderText('Search or run a command…                          Ctrl+K')
        bar.addWidget(self.search, 1)
        self.search.setVisible(staff.role != 'labeler')
        if staff.role == 'labeler':
            bar.addWidget(button('Commands · Ctrl+K', self.open_palette))
        bar.addStretch()
        bar.addWidget(label(environment, 'badge'))
        bar.addSpacing(16)
        bar.addWidget(label(staff.name + '   /   ' + staff.role.title()))
        main.addWidget(top)
        main.addWidget(self.pages, 1)
        layout.addLayout(main, 1)
        self.customer_page = None
        if staff.role != 'labeler':
            self.customer_page = CustomerScreen(backend, theme, staff.role)
            self.customer_page.back.connect(lambda: self.navigate('Fleet'))
            self.customer_page.session_expired.connect(self.session_expired)
            self.pages.addWidget(self.customer_page)
        shortcut = QShortcut(QKeySequence('Ctrl+K'), self)
        shortcut.activated.connect(self.open_palette)
        if staff.role in ('admin', 'labeler'):
            self.review_page.event_view.label_requested.connect(self.open_label)
            if self.customer_page: self.customer_page.event_view.label_requested.connect(self.open_label)
        self.navigate('Label' if staff.role == 'labeler' else 'Fleet')
        self.badge_runner = TaskRunner(self)
        self.badge_runner.finished.connect(self.badge_loaded)
        self.update_badge()
        self.review_page.review_changed.connect(self.update_badge)
        self.review_page.event_view.review_changed.connect(lambda *_: self.update_badge())
        if self.customer_page:
            self.customer_page.timeline.review_runner.finished.connect(lambda *_: self.update_badge())
            self.customer_page.event_view.review_changed.connect(lambda *_: self.update_badge())

    def make_screen(self, title, backend, staff, theme):
        if title == 'Fleet':
            page = FleetScreen(backend, theme)
            self.fleet = page
            page.customer_requested.connect(self.open_customer)
            page.loaded.connect(self.fleet_loaded)
        elif title == 'Tag':
            page = TagView(backend, staff.role, theme)
            self.tag_page = page
            page.customer_requested.connect(self.open_customer)
            page.label_requested.connect(self.open_label)
        elif title == 'Label':
            page = LabelView(backend, staff.role, theme)
            self.label_page = page
            page.annotation_saved.connect(self.annotation_saved)
        elif title == 'Review':
            page = ReviewScreen(backend, theme, staff.role)
            self.review_page = page
        elif title == 'Studio':
            page = StudioScreen(backend, staff.role, theme)
            page.filter_requested.connect(self.open_filter)
            page.event_requested.connect(self.open_event)
            page.tagging_requested.connect(lambda: self.navigate('Tag'))
        elif title == 'Inbox':
            page = inbox_screen(backend, staff.role, theme)
        elif title == 'Index problems':
            from .index_problems import IndexProblems
            page = IndexProblems(backend)
        else:
            page = AuditScreen(backend, theme)
        if hasattr(page, 'session_expired'):
            page.session_expired.connect(self.session_expired)
        return page

    def update_badge(self):
        if self.badge_runner.busy:
            self.badge_dirty = True
            return
        self.badge_dirty = False
        self.badge_runner.start(self.backend.review_count)

    def badge_loaded(self, events, error):
        self.review_badge.setText('—' if error else str(events.unreviewed_24h))
        if isinstance(error,AuthError): self.session_expired.emit()
        elif self.badge_dirty: self.update_badge()

    def navigate(self, title):
        if title not in self.screens:
            return
        self.pages.setCurrentWidget(self.screens[title])
        if title == 'Label' and not self.label_page.doc and not self.label_page.loader.busy:
            self.label_page.open_queue()
        if title == 'Tag':
            self.tag_page.open()
        for name, nav in self.navigation.items():
            nav.setChecked(name == title)

    def fleet_loaded(self, snapshot, customers):
        self.devices, self.customers = snapshot.devices, customers

    def open_customer(self, customer_id, device_id=''):
        if self.customer_page is None:
            return
        self.pages.setCurrentWidget(self.customer_page)
        for name, nav in self.navigation.items(): nav.setChecked(name == 'Fleet')  # a house page belongs to Fleet
        self.customer_page.open(customer_id, device_id)

    def open_palette(self):
        if self.palette_dialog and self.palette_dialog.isVisible():
            return
        cameras = {(e.camera, e.customer_id) for e in self.review_page.timeline.model.rows}
        if self.customer_page:
            cameras.update((e.camera, e.customer_id) for e in self.customer_page.timeline.model.rows)
            if self.customer_page.customer_id:
                cameras.update((name,self.customer_page.customer_id) for name in self.customer_page.timeline.density.rows)
        self.palette_dialog = CommandPalette(self, self.devices, self.customers, self.staff.role,
                                             self.review_page.saved_filters, sorted(cameras))
        self.palette_dialog.execute.connect(self.execute_command)
        self.palette_dialog.jump.connect(self.open_customer)
        self.palette_dialog.move(self.mapToGlobal(self.rect().center()) - self.palette_dialog.rect().center())
        self.palette_dialog.show()
        self.palette_dialog.search.setFocus()

    def open_filter(self, key):
        if key in ('needs_labeling', 'to_review', 'rejected') and hasattr(self, 'label_page'):
            self.pages.setCurrentWidget(self.label_page)
            for name, nav in self.navigation.items(): nav.setChecked(name == 'Label')
            self.label_page.open_queue(key); return
        self.navigate('Review'); self.review_page.open_filter(key)

    def open_label(self, eid):
        self.pages.setCurrentWidget(self.label_page)
        for name, nav in self.navigation.items(): nav.setChecked(name == 'Label')
        if isinstance(eid, str): self.label_page.open_clip(eid)   # a dataset clip's key: no event behind it
        else: self.label_page.open_event(eid)

    def annotation_saved(self, annotation):
        timelines = [self.review_page.timeline]
        if self.customer_page: timelines.append(self.customer_page.timeline)
        for timeline in timelines:
            for row, event in enumerate(timeline.model.rows):
                if event.id == annotation.event_id:
                    event.annotation_status = annotation.status
                    timeline.model.dataChanged.emit(timeline.model.index(row, 0), timeline.model.index(row, 6))
        studio = self.screens['Studio']
        for event in studio.grid.model.items:
            if event.id == annotation.event_id: event.annotation_status = annotation.status
        studio.grid.grid.viewport().update()

    def open_event(self, eid):
        self.navigate('Review'); self.review_page.open_event(eid)

    def execute_command(self, kind, value):
        if kind == 'event': self.open_event(value)
        elif kind == 'filter': self.open_filter(value)
        elif kind == 'camera':
            self.navigate('Review')
            timeline = self.review_page.timeline
            customer_id,value = value
            timeline.customer_id = None if self.staff.role == 'labeler' else customer_id
            timeline.filters['camera'].blockSignals(True)
            if timeline.filters['camera'].findData(value) < 0: timeline.filters['camera'].addItem(value,value)
            timeline.filters['camera'].setCurrentIndex(timeline.filters['camera'].findData(value))
            timeline.filters['camera'].blockSignals(False); timeline.reload()
        elif value == 'Sign out': self.signed_out.emit()
        elif value == 'Export collection…':
            self.navigate('Studio'); self.screens['Studio'].open_export()
        elif value.startswith('Go to '): self.navigate(SCREEN_OF.get(value[6:], value[6:]))


class AdminWindow(QMainWindow):
    def __init__(self, backend, *, demo=False, theme='dark', prefs=None, local=False):
        super().__init__()
        from .prefs import Preferences
        prefs = prefs or Preferences()
        self.backend, self.theme, self.prefs = backend, theme, prefs
        self.retired_backends = []
        self.configured_backend = backend
        self.setWindowTitle('Home Guard · Admin Center')
        self.resize(1366, 768)
        self.setMinimumSize(1200, 720)
        self.session = QStackedWidget()
        self.setCentralWidget(self.session)
        self.signin = None
        self.shell = None
        self.connecting = None
        self.local_relogin = False
        self.local = local
        self.local_runner = TaskRunner(self)
        self.local_runner.finished.connect(self.local_done)
        self.menuBar().addAction('Settings', self.open_settings)
        self.show_signin()
        if demo:
            self.use_demo()
        elif local:
            self.start_local()

    def start_local(self, relogin=False):
        backend = self.configured_backend
        if not hasattr(backend, 'local_login') or not self.local_runner.start(backend.local_login):
            return False
        self.local_relogin = relogin
        if not relogin and self.signin:
            self.connecting = QLabel('Connecting to the local service…', alignment=Qt.AlignmentFlag.AlignCenter)
            self.session.addWidget(self.connecting); self.session.setCurrentWidget(self.connecting)
        return True

    def local_done(self, result, error):
        connecting, self.connecting = self.connecting, None
        if connecting:
            self.session.removeWidget(connecting); connecting.deleteLater()
        if error:
            if self.local_relogin:
                self.show_signin()
                self.signin.error.setText('Your session needs a new sign-in')
            elif self.signin:
                self.session.setCurrentWidget(self.signin)
            return
        if self.shell:
            self.session.removeWidget(self.shell); self.shell.hide(); self.shell.deleteLater(); self.shell = None
        self.backend = self.configured_backend
        self.enter(result.staff)

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
        if hasattr(self.backend, 'clear_session'):
            self.backend.clear_session()
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
        if self.local and not self.local_runner.busy and self.start_local(relogin=True):
            return
        self.show_signin()
        self.signin.error.setText('Your session needs a new sign-in')

    def open_settings(self):
        from .settings import SettingsDialog
        server = getattr(self.configured_backend, 'base_url', self.prefs.get('server') or 'http://127.0.0.1:8610')
        self.settings_dialog = SettingsDialog(server, self.theme, self)
        self.settings_dialog.saved.connect(self.save_settings); self.settings_dialog.show()

    def save_settings(self, server, theme):
        from .http_backend import HttpBackend
        from .theme import apply_theme
        from .backend import BackendError
        from PySide6.QtWidgets import QApplication
        changed = server != getattr(self.configured_backend, 'base_url', self.prefs.get('server') or 'http://127.0.0.1:8610')
        if changed:
            try: backend = HttpBackend(server)
            except BackendError as error:
                self.statusBar().showMessage(str(error)); return
            if hasattr(self.configured_backend, 'clear_session'): self.configured_backend.clear_session()
            self.retired_backends.append(self.configured_backend)
            self.configured_backend = backend
        self.prefs.save(server=server, theme=theme)
        old_theme, self.theme = self.theme, theme
        apply_theme(QApplication.instance(), theme)
        if changed: self.show_signin()
        elif self.shell and old_theme != theme:
            staff = self.shell.staff
            old = self.shell; self.session.removeWidget(old); old.hide(); old.deleteLater()
            self.enter(staff)
