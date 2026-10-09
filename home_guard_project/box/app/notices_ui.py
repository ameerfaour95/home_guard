"""Access notices in the owner's app: what Home Guard support looked at, told here and never over Telegram.

Owner decision (2026-10-03). The box keeps the notices (box/notices.py: the cloud pushes them over ssh); the app
only runs the box's command, on this box or over ssh like the scene map editor (scene_backend.SceneBackend):
``python -m home_guard_project.box notices list|read-all --json``. It never talks to the cloud.

Each line is worded here from the notice's kind, cameras and time; the cloud's English ``message`` is shown only
for a kind this app does not know. The bell in the top bar carries the unread count; opening the panel marks them
all read (on the box: read/unread never goes back to the cloud), and what was new keeps a "New" mark while it is
open.
"""
import datetime as dt
from concurrent.futures import ThreadPoolExecutor

from PySide6.QtCore import QRectF, Qt, QTimer
from PySide6.QtGui import QColor, QPainter
from PySide6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QScrollArea, QWidget

from . import strings
from .strings import isolate, tr

MODULE = "home_guard_project.box"
POLL_SECONDS = 180            # the list is read on open and again every few minutes
KINDS = ("recording", "chat")


# ----------------------------------------------------------------------------
# Words
# ----------------------------------------------------------------------------
def _hebrew(c):
    return "א" <= c <= "ת"


def _when(text):
    try:
        return dt.datetime.strptime(str(text or "")[:16], "%Y-%m-%dT%H:%M")
    except ValueError:
        return None


def time_text(from_local, to_local=None):
    """"09.10 13:15" for one moment, "09.10 13:15–13:40" for a range in one day, else both with their dates."""
    start, end = _when(from_local), _when(to_local) or _when(from_local)
    if start is None:
        return ""
    if end is None or end <= start:
        return f"{start:%d.%m %H:%M}"
    if end.date() == start.date():
        return f"{start:%d.%m %H:%M}–{end:%H:%M}"
    return f"{start:%d.%m %H:%M} – {end:%d.%m %H:%M}"


def cameras_text(names, lang=None):
    """"Front door, Garden and Driveway" ("דלת הכניסה, הגינה והחניה"). In Hebrew each name with Latin letters or
    digits inside a list is isolated; one name alone is left to tr(), which isolates the whole value."""
    lang = lang or strings.LANG
    names = [str(n).strip() for n in names if str(n).strip()]
    if len(names) <= 1:
        return names[0] if names else ""
    key = "notice_and_latin" if lang == "he" and not _hebrew(names[-1][:1]) else "notice_and"
    return strings.TEXT[key].format(rest=", ".join(isolate(n, lang) for n in names[:-1]), last=isolate(names[-1], lang))


def notice_line(notice):
    """The sentence the owner reads for *notice* (a row of ``notices list --json``)."""
    when = time_text(notice.get("from_local"), notice.get("to_local"))
    kind = notice.get("kind")
    if kind == "chat":
        return tr("notice_chat", time=when)
    if kind == "recording":
        names = [str(c) for c in notice.get("cameras") or () if str(c).strip()]
        if not names:
            return tr("notice_recording_any", time=when)
        key = "notice_recording_latin" if strings.LANG == "he" and not _hebrew(names[0][:1]) else "notice_recording"
        return tr(key, cameras=cameras_text(names), time=when)
    message = str(notice.get("message") or "").strip()      # a kind this app does not know: the cloud's words
    return message or tr("notice_other", time=when)


def secondary_line(notice):
    name = str(notice.get("staff_name") or "").strip()
    return tr("notice_by", name=name) if name else ""


def _signature(notice):
    """A notice as it was read: the cloud may rewrite it later (more cameras, a later end), and then it is new."""
    return (str(notice.get("id")), str(notice.get("to_local")), tuple(str(c) for c in notice.get("cameras") or ()))


def unread_count(notices):
    return sum(1 for n in notices if not n.get("read"))


def notices_from(data):
    rows = data.get("notices")
    if not isinstance(rows, list):
        raise ValueError("bad answer")
    return [dict(r) for r in rows if isinstance(r, dict)]


# ----------------------------------------------------------------------------
# The box
# ----------------------------------------------------------------------------
class NoticesBackend:
    """The box's notices command, on this box (no target) or at ``user@address`` over ssh."""
    def __init__(self, target=None, key=None, runner=None):
        from .scene_backend import SceneBackend
        self.box = SceneBackend(target, key, runner)

    def run(self, *args):
        return self.box.run(["notices", *args, "--json"], MODULE)

    def list(self, lang="he"):
        return notices_from(self.run("list", "--lang", "en" if lang == "en" else "he"))

    def read_all(self):
        return self.run("read-all")


def _demo_notices(lang):
    from .scene_backend import DEMO_NAMES
    name = lambda camera: DEMO_NAMES[camera]["en" if lang == "en" else "he"]
    staff = "Noa Cohen"
    return [
        {"id": "41", "kind": "recording", "cameras": [name("front_door"), name("driveway")], "staff_name": staff,
         "from_local": "2026-10-09T13:15", "to_local": "2026-10-09T13:40", "message": "", "read": False},
        {"id": "40", "kind": "chat", "cameras": [], "staff_name": staff,
         "from_local": "2026-10-09T09:05", "to_local": "2026-10-09T09:05", "message": "", "read": False},
        {"id": "38", "kind": "recording", "cameras": [name("garden")], "staff_name": "Amir Haddad",
         "from_local": "2026-10-07T21:30", "to_local": "2026-10-07T21:30", "message": "", "read": True},
        {"id": "35", "kind": "recording", "cameras": [], "staff_name": "",
         "from_local": "2026-10-05T18:02", "to_local": "2026-10-05T18:10", "message": "", "read": True},
    ]


class DemoNoticesBackend:
    """No box: a few notices (two unread, two read), or none."""
    def __init__(self, empty=False):
        self.empty = empty
        self.read = set()
        self.calls = []

    def list(self, lang="he"):
        self.calls.append("list")
        rows = [] if self.empty else _demo_notices(lang)
        return [dict(r, read=r["read"] or r["id"] in self.read) for r in rows]

    def read_all(self):
        self.calls.append("read-all")
        self.read.update(r["id"] for r in _demo_notices("en"))
        return {"marked": 0, "unread": 0}


def notices_backend_for(args):
    """The demo, this box, or the remote box (``--remote-box``)."""
    if getattr(args, "demo", False):
        return DemoNoticesBackend(empty=getattr(args, "notices", None) == "empty")
    target = getattr(args, "remote_box", None)
    return NoticesBackend(target) if target else NoticesBackend()


# ----------------------------------------------------------------------------
# The screen
# ----------------------------------------------------------------------------
class BadgeButton(QPushButton):
    """An icon button with the unread count in a small circle on its corner (the start corner right to left)."""
    def __init__(self):
        super().__init__()
        self.count = 0

    def set_count(self, count):
        self.count = max(0, int(count or 0))
        self.update()

    def paintEvent(self, event):
        super().paintEvent(event)
        if not self.count:
            return
        from .theme import ERROR
        text = "9+" if self.count > 9 else str(self.count)
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        font = p.font(); font.setPixelSize(11); font.setBold(True); p.setFont(font)
        width = max(18, p.fontMetrics().horizontalAdvance(text) + 8)
        x = 2 if self.layoutDirection() == Qt.LayoutDirection.RightToLeft else self.width() - width - 2
        rect = QRectF(x, 2, width, 18)
        p.setPen(Qt.PenStyle.NoPen); p.setBrush(QColor(ERROR)); p.drawRoundedRect(rect, 9, 9)
        p.setPen(QColor("#0c1218")); p.drawText(rect, Qt.AlignmentFlag.AlignCenter, text)


class NoticesPage:
    """The panel (a page of the dashboard) and the unread count. *on_count(n)* is told every new count."""
    def __init__(self, backend, on_count=lambda n: None, lang=None):
        from .ui import card, label, layout_for
        from .ai_activity_ui import icon
        self._card, self._label, self._icon = card, label, icon
        self.backend, self.on_count = backend, on_count
        self.lang = lang or strings.LANG
        self.notices, self.fresh, self.failed, self.opened = [], set(), False, False
        self.seen = set()      # notices read here: a list read before the box marked them must not bring them back
        self.pool = ThreadPoolExecutor(max_workers=1)
        self.future = self.read_future = None
        self.widget = QWidget()
        layout = layout_for(self.widget, 0)
        head = QHBoxLayout()
        titles = QWidget(); titles.setStyleSheet("background: transparent;")
        title_layout = layout_for(titles, 0); title_layout.setSpacing(6)
        title_layout.addWidget(label(tr("notices"), "title"))
        title_layout.addWidget(label(tr("notices_hint"), "muted"))
        head.addWidget(titles, 1)
        layout.addLayout(head)
        self.note = label("", "warning"); self.note.hide(); layout.addWidget(self.note)
        self.list_widget = QWidget(); self.list_widget.setStyleSheet("background: transparent;")
        self.rows = layout_for(self.list_widget, 0); self.rows.setSpacing(12)
        scroll = QScrollArea(); scroll.setWidgetResizable(True); scroll.setWidget(self.list_widget)
        layout.addWidget(scroll, 1)
        self.timer = QTimer(self.widget); self.timer.timeout.connect(self.poll); self.timer.start(200)
        self.refresh_timer = QTimer(self.widget); self.refresh_timer.timeout.connect(self.refresh)
        self.render()

    # -- the box ------------------------------------------------------------------------------------
    def start(self):
        """When the app opens: the list is read now and every few minutes."""
        self.refresh()
        self.refresh_timer.start(POLL_SECONDS * 1000)

    def refresh(self):
        if self.future is None or self.future.done():
            self.future = self.pool.submit(self.backend.list, self.lang)

    def poll(self):
        if self.future is not None and self.future.done():
            future, self.future = self.future, None
            try:
                self.loaded(future.result())
            except Exception:  # noqa: BLE001
                self.failed = True
                self.render()
        if self.read_future is not None and self.read_future.done():
            self.read_future = None

    def loaded(self, notices):
        self.failed = False
        for n in notices:
            if _signature(n) in self.seen:
                n["read"] = True
        self.notices = sorted(notices, key=lambda n: (str(n.get("from_utc") or n.get("from_local") or ""),
                                                      str(n.get("id"))), reverse=True)
        if self.opened and self.widget.isVisible() and unread_count(self.notices):
            self.read_all()
            return
        self.render()
        self.on_count(unread_count(self.notices))

    # -- the owner ----------------------------------------------------------------------------------
    def open(self):
        """The panel is shown: what was new stays marked "New" while it is open; on the box it is read now."""
        self.opened = True
        self.fresh = {n.get("id") for n in self.notices if not n.get("read")}
        self.read_all()

    def read_all(self):
        unread = [n for n in self.notices if not n.get("read")]
        self.fresh |= {n.get("id") for n in unread}
        for n in self.notices:
            n["read"] = True
            self.seen.add(_signature(n))
        if unread:
            self.read_future = self.pool.submit(self.backend.read_all)
        self.render()
        self.on_count(0)

    def close(self):
        self.timer.stop(); self.refresh_timer.stop()
        self.pool.shutdown(wait=False, cancel_futures=True)

    # -- drawing ------------------------------------------------------------------------------------
    def render(self):
        while self.rows.count():
            item = self.rows.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self.note.setText(tr("notices_unavailable")); self.note.setVisible(self.failed)
        if not self.notices:
            self.rows.addWidget(self.empty_state())
        for notice in self.notices:
            self.rows.addWidget(self.row(notice))
        self.rows.addStretch()

    def empty_state(self):
        box = self._card(); lay = QHBoxLayout(box); lay.setContentsMargins(28, 28, 28, 28); lay.setSpacing(16)
        picture = QLabel(); picture.setPixmap(self._icon("shield").pixmap(40, 40)); lay.addWidget(picture)
        lay.addWidget(self._label(tr("notices_empty"), "section"), 1)
        box.setProperty("notices_empty", True)
        return box

    def row(self, notice):
        box = self._card(); box.setProperty("notice_id", str(notice.get("id")))
        lay = QHBoxLayout(box); lay.setContentsMargins(20, 16, 20, 16); lay.setSpacing(16)
        picture = QLabel(); picture.setAlignment(Qt.AlignmentFlag.AlignTop)
        picture.setPixmap(self._icon({"recording": "video", "chat": "bot"}.get(notice.get("kind"), "eye")).pixmap(28, 28))
        lay.addWidget(picture, 0, Qt.AlignmentFlag.AlignTop)
        text = QWidget(); text.setStyleSheet("background: transparent;")
        from .ui import layout_for
        text_layout = layout_for(text, 0); text_layout.setSpacing(4)
        line = self._label(notice_line(notice)); line.setObjectName("noticeLine")
        text_layout.addWidget(line)
        secondary = secondary_line(notice)
        if secondary:
            text_layout.addWidget(self._label(secondary, "muted"))
        lay.addWidget(text, 1)
        if notice.get("id") in self.fresh:
            new = self._label(tr("notice_new"), "accent"); new.setWordWrap(False)
            new.setStyleSheet("font-weight: 600;")
            lay.addWidget(new, 0, Qt.AlignmentFlag.AlignTop)
        return box
