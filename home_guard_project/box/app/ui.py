from pathlib import Path
import math
import re
import time
from concurrent.futures import ThreadPoolExecutor

from PySide6.QtCore import Qt, QTimer, QSize
from PySide6.QtGui import QColor, QPainter, QPen, QPixmap, QFont, QLinearGradient
from PySide6.QtWidgets import (
    QMainWindow,
    QWidget,
    QLabel,
    QPushButton,
    QCheckBox,
    QLineEdit,
    QComboBox,
    QSpinBox,
    QVBoxLayout,
    QHBoxLayout,
    QGridLayout,
    QFrame,
    QStackedWidget,
    QScrollArea,
    QSizePolicy,
)

from .strings import tr, TEXT
from .model import State, Activity, ActivityFeed
from .backend import Answers, SimulatedBackend, Sequence, STEPS
from ..preview import PreviewReader
from .. import boxconfig as bc

ACCENT = "#39c6e5"
STYLE = """
QWidget { background: #10151d; color: #edf1f7; font-family: 'Segoe UI'; font-size: 15px; }
QLabel { background: transparent; }
QLabel#title { font-size: 29px; font-weight: 600; }
QLabel#section { font-size: 20px; font-weight: 600; }
QLabel#muted { color: #98a6ba; }
QLabel#accent { color: #39c6e5; font-weight: 600; }
QFrame#card { background: #1a222e; border: 1px solid #2c3745; border-radius: 12px; }
QPushButton { background: #39c6e5; color: #111722; border: none; border-radius: 7px; padding: 12px 24px; font-weight: 600; }
QPushButton#secondary { background: #263344; color: #edf1f7; }
QPushButton:disabled { background: #263344; color: #8793a4; }
QPushButton:hover { background: #77ddf2; }
QLineEdit, QComboBox, QSpinBox { background: #151e2a; border: 1px solid #435063; border-radius: 7px; padding: 10px; min-height: 23px; }
QLineEdit:focus { border: 1px solid #39c6e5; }
QCheckBox { spacing: 12px; background: transparent; padding: 6px 0; }
QCheckBox::indicator { width: 21px; height: 21px; border: 1px solid #526179; border-radius: 4px; background: #151e2a; }
QCheckBox::indicator:checked { background: #39c6e5; border: 1px solid #39c6e5; }
QScrollArea { border: none; background: transparent; }
"""


def label(text, role=None):
    widget = QLabel(text)
    widget.setTextFormat(Qt.TextFormat.PlainText)
    if role:
        widget.setObjectName(role)
    widget.setWordWrap(True)
    return widget


def card():
    widget = QFrame()
    widget.setObjectName("card")
    return widget


def layout_for(widget, margin=22):
    layout = QVBoxLayout(widget)
    layout.setContentsMargins(margin, margin, margin, margin)
    layout.setSpacing(14)
    return layout


def demo_picture(number):
    # Deliberately synthetic architectural views, no network or private imagery.
    pix = QPixmap(960, 540)
    painter = QPainter(pix)
    gradient = QLinearGradient(0, 0, 0, 540)
    gradient.setColorAt(0, QColor("#647a8d"))
    gradient.setColorAt(1, QColor("#1c2935"))
    painter.fillRect(pix.rect(), gradient)
    painter.fillRect(0, 255, 960, 285, QColor("#43524b"))
    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(QColor("#c3b8a1"))
    painter.drawRect(180 + number * 15, 130, 490, 230)
    painter.setBrush(QColor("#38434b"))
    painter.drawPolygon(
        [
            __import__("PySide6.QtCore", fromlist=["QPoint"]).QPoint(x, y)
            for x, y in [
                (140 + number * 15, 130),
                (420 + number * 15, 30),
                (710 + number * 15, 130),
            ]
        ]
    )
    painter.setBrush(QColor("#253b49"))
    for x in (220, 520):
        painter.drawRect(x + number * 15, 175, 100, 86)
    painter.setBrush(QColor("#574f45"))
    painter.drawRect(385 + number * 15, 218, 90, 142)
    painter.setBrush(QColor("#8a8b83"))
    painter.drawPolygon(
        [
            __import__("PySide6.QtCore", fromlist=["QPoint"]).QPoint(x, y)
            for x, y in [(380, 360), (490, 360), (660, 540), (240, 540)]
        ]
    )
    for x in (70, 800, 890):
        painter.setBrush(QColor("#263c32"))
        painter.drawEllipse(x - 40, 120 + number * 4, 110, 190)
    painter.setPen(QPen(QColor(ACCENT), 3))
    painter.setBrush(Qt.BrushStyle.NoBrush)
    painter.drawRect(455, 280, 42, 110)
    painter.end()
    return pix


class CameraTile(QFrame):
    def __init__(self, name):
        super().__init__()
        self.setObjectName("card")
        self.name = name
        self.picture = None
        self.status = label(tr("offline"), "muted")
        self.caption = label(name.replace("_", " "))
        layout = QHBoxLayout(self)
        layout.setContentsMargins(16, 8, 16, 8)
        layout.addWidget(self.caption)
        layout.addStretch()
        layout.addWidget(self.status)
        layout.setAlignment(Qt.AlignmentFlag.AlignBottom)
        self.setMinimumSize(180, 140)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    def update_picture(self, pix):
        self.picture = pix if pix and not pix.isNull() else None
        self.status.setText(tr("live") if self.picture else tr("offline"))
        self.status.setObjectName("accent" if self.picture else "muted")
        self.status.setStyleSheet("color: " + (ACCENT if self.picture else "#98a6ba"))
        self.update()

    def paintEvent(self, event):
        super().paintEvent(event)
        p = QPainter(self)
        area = self.rect().adjusted(2, 2, -2, -42)
        if self.picture:
            scaled = self.picture.scaled(
                area.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            p.drawPixmap(
                area.center().x() - scaled.width() // 2,
                area.center().y() - scaled.height() // 2,
                scaled,
            )
        else:
            p.setPen(QColor("#98a6ba"))
            p.setFont(QFont("Segoe UI", 14))
            p.drawText(area, Qt.AlignmentFlag.AlignCenter, tr("offline_hint"))
        p.end()


class Window(QMainWindow):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.setWindowTitle(tr("brand"))
        self.resize(*map(int, args.size.split("x")))
        self.setMinimumSize(1000, 650)
        self.setStyleSheet(STYLE)
        self.root = QWidget()
        self.setCentralWidget(self.root)
        self.outer = layout_for(self.root, 28)
        header = QHBoxLayout()
        logo = QLabel()
        pix = QPixmap(str(Path(__file__).parents[1] / "assets" / "logo.png"))
        logo.setPixmap(
            pix.scaled(
                130,
                60,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )
        header.addWidget(logo)
        header.addSpacing(20)
        titles = QVBoxLayout()
        self.house_label = label(tr("setup") if args.setup else tr("home"), "title")
        titles.addWidget(self.house_label)
        titles.addWidget(
            label(tr("simulation") if args.setup else tr("close_hint"), "muted")
        )
        header.addLayout(titles, 1)
        header.addStretch()
        if args.demo or args.setup:
            header.addWidget(label(tr("demo"), "accent"))
        self.outer.addLayout(header)
        if args.setup:
            self.build_setup()
        else:
            self.build_dashboard()

    def build_dashboard(self):
        stats = QHBoxLayout()
        self.stats = []
        for key in ("collecting", "upload", "waiting", "disk"):
            panel = card()
            lay = layout_for(panel, 16)
            value = label(tr("loading"), "section")
            hint = label(tr(key), "muted")
            lay.addWidget(value)
            lay.addWidget(hint)
            stats.addWidget(panel)
            self.stats.append((value, hint))
        self.outer.addLayout(stats)
        body = QHBoxLayout()
        left = QWidget()
        leftlay = layout_for(left, 0)
        self.camera_heading = label(tr("cameras"), "section")
        leftlay.addWidget(self.camera_heading)
        self.camera_stack = QStackedWidget()
        self.grid_widget = QWidget()
        self.grid = QGridLayout(self.grid_widget)
        self.grid.setContentsMargins(0, 0, 0, 0)
        self.grid.setSpacing(14)
        self.camera_stack.addWidget(self.grid_widget)
        self.message = card()
        ml = layout_for(self.message, 32)
        ml.addStretch()
        self.message_title = label(tr("loading"), "title")
        self.message_hint = label("", "muted")
        ml.addWidget(self.message_title)
        ml.addWidget(self.message_hint)
        ml.addStretch()
        self.camera_stack.addWidget(self.message)
        leftlay.addWidget(self.camera_stack, 1)
        body.addWidget(left, 4)
        panel = card()
        panel.setMinimumWidth(275)
        panel.setMaximumWidth(360)
        al = layout_for(panel, 20)
        al.addWidget(label(tr("activity"), "section"))
        self.details = QCheckBox(tr("details"))
        self.details.toggled.connect(self.render_activity)
        al.addWidget(self.details)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        self.activity_widget = QWidget()
        self.activity_widget.setStyleSheet("background: transparent;")
        self.activity_layout = layout_for(self.activity_widget, 0)
        scroll.setWidget(self.activity_widget)
        al.addWidget(scroll, 1)
        body.addWidget(panel, 1)
        self.outer.addLayout(body, 1)
        self.tiles = []
        self.events = []
        self.render_activity()
        self.names = None
        self.reader = PreviewReader(Path(bc.LOG_DIR) / "preview")
        self.activity_feed = ActivityFeed(bc.LOG_DIR)
        self.pool = ThreadPoolExecutor(max_workers=1)
        self.future = None
        self.last_poll = 0
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.tick)
        self.timer.start(500)
        self.camera_stack.setCurrentIndex(1)
        self.current_state = None
        self.tick()
        self.details.setChecked(self.args.details)

    def fetch(self):
        from ..heartbeat import build_heartbeat

        settings = bc.load_box_settings()
        events, upload = self.activity_feed.read()
        payload = build_heartbeat(
            str(settings.get("site", "")),
            bc.LIVE_DIR,
            bc.OUTBOX_DIR,
            bc.ALIVE_FILE,
            mode=bc.get_option("mode"),
        )
        # Read only names, never retain or display camera URLs.
        names = self.reader.names() or list(payload.get("cameras", {}))
        return (
            State.from_heartbeat(payload, cameras=names, upload=upload),
            bool(bc.get_option("show_cameras")),
            events,
        )

    def tick(self):
        if self.args.demo:
            scenario = self.args.state
            names = TEXT["demo_names"][: self.args.cameras]
            state = State(
                tr("demo_house"),
                scenario != "stopped",
                "inference" if scenario == "inference" else "data_collection",
                7,
                112.8,
                ("" if scenario == "quiet" else tr("demo_time")),
                names,
                error=scenario == "error",
            )
            if scenario == "empty":
                state.cameras = []
            self.apply_state(
                state,
                scenario != "hidden",
                (
                    []
                    if scenario in ("quiet", "empty", "loading")
                    else [
                        Activity(e, d)
                        for e, d in zip(TEXT["demo_events"], TEXT["demo_details"])
                    ]
                ),
            )
            for i, tile in enumerate(self.tiles):
                tile.update_picture(
                    None
                    if scenario in ("offline", "stopped")
                    or (scenario == "mixed" and i == len(self.tiles) - 1)
                    else demo_picture(i)
                )
            return
        now = time.monotonic()
        if self.future and self.future.done():
            try:
                self.apply_state(*self.future.result())
            except Exception:
                self.apply_state(State(error=True), False, [])
            self.future = None
        if not self.future and now - self.last_poll >= 5:
            self.future = self.pool.submit(self.fetch)
            self.last_poll = now
        # Re-read permission at each viewer tick. Never touch the marker when hidden.
        try:
            allowed = bool(bc.get_option("show_cameras"))
        except Exception:
            allowed = False
        if self.current_state is not None and allowed != self.current_show:
            self.apply_state(self.current_state, allowed, self.events)
        if allowed:
            try:
                self.reader.touch()
                for tile in self.tiles:
                    data = self.reader.read(tile.name)
                    pix = QPixmap()
                    if data:
                        pix.loadFromData(data)
                    tile.update_picture(pix)
            except OSError:
                for tile in self.tiles:
                    tile.update_picture(None)

    def apply_state(self, state, show, events):
        self.current_state, self.current_show = state, show
        if self.events != events:
            self.events = events
            self.render_activity()
        self.house_label.setText(state.site or tr("home"))
        values = [
            tr("collecting") if state.collecting else tr("stopped"),
            state.upload or tr("never"),
            str(state.waiting),
            tr("gb", value=state.disk),
        ]
        for (value, hint), text in zip(self.stats, values):
            value.setText(tr("unknown") if state.error else text)
        self.stats[0][1].setText(
            tr("inference") if state.mode == "inference" else tr("collection")
        )
        self.camera_heading.setText(
            tr("cameras")
            + tr("separator")
            + tr("camera_count", count=len(state.cameras))
        )
        scenario = self.args.state if self.args.demo else ""
        if scenario == "loading":
            for value, _ in self.stats:
                value.setText(tr("loading"))
        message = (
            "loading"
            if scenario == "loading"
            else (
                "error"
                if state.error
                else (
                    "pictures_off"
                    if not show
                    else "empty" if not state.cameras else None
                )
            )
        )
        if message:
            self.camera_stack.setCurrentIndex(1)
            self.message_title.setText(tr(message))
            hint = {
                "loading": "",
                "error": "error_hint",
                "pictures_off": "pictures_hint",
                "empty": "empty_hint",
            }[message]
            self.message_hint.setText(tr(hint) if hint else "")
            return
        self.camera_stack.setCurrentIndex(0)
        if self.names != state.cameras:
            self.names = list(state.cameras)
            while self.grid.count():
                self.grid.takeAt(0).widget().deleteLater()
            self.tiles = [CameraTile(name) for name in state.cameras]
            self.arrange_tiles()

    def arrange_tiles(self):
        if not self.tiles:
            return
        columns = 1 if len(self.tiles) == 1 else 2 if len(self.tiles) <= 4 else 3
        for i, tile in enumerate(self.tiles):
            self.grid.addWidget(tile, i // columns, i % columns)
        for i in range(3):
            self.grid.setRowStretch(
                i, 1 if i < math.ceil(len(self.tiles) / columns) else 0
            )
            self.grid.setColumnStretch(i, 1 if i < columns else 0)

    def render_activity(self):
        while self.activity_layout.count():
            item = self.activity_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        for event in self.events:
            self.activity_layout.addWidget(label(tr("bullet") + event.text))
            if self.details.isChecked():
                self.activity_layout.addWidget(label(event.detail, "muted"))
            divider = QFrame()
            divider.setFixedHeight(1)
            divider.setStyleSheet("background: #34404e;")
            self.activity_layout.addWidget(divider)
        if not self.events:
            self.activity_layout.addWidget(label(tr("activity_empty"), "muted"))
        self.activity_layout.addStretch()

    def closeEvent(self, event):
        if hasattr(self, "pool"):
            self.timer.stop()
            self.pool.shutdown(wait=False, cancel_futures=True)
        if hasattr(self, "setup_pool"):
            self.step_timer.stop()
            self.setup_pool.shutdown(wait=False, cancel_futures=True)
        super().closeEvent(event)

    def build_setup(self):
        self.outer.addWidget(label(tr("steps"), "accent"))
        self.pages = QStackedWidget()
        self.outer.addWidget(self.pages, 1)
        self.inputs = {}
        titles = [
            ("address_title", "address_hint"),
            ("network_title", "network_hint"),
            ("house_title", "house_hint"),
            ("camera_title", "camera_hint"),
            ("progress_title", "progress_hint"),
            ("summary_title", "summary_hint"),
        ]
        self.page_layouts = []
        for title, hint in titles:
            panel = card()
            lay = layout_for(panel, 30)
            title_widget = label(tr(title), "title")
            if title == "progress_title":
                self.progress_title = title_widget
            lay.addWidget(title_widget)
            lay.addWidget(label(tr(hint), "muted"))
            self.page_layouts.append(lay)
            self.pages.addWidget(panel)
        self.add_input(0, "address", tr("address"))
        self.network = QComboBox()
        self.network.addItems([tr("ethernet"), tr("wifi")])
        self.page_layouts[1].addWidget(self.network)
        self.add_input(1, "ssid", tr("ssid"))
        self.add_input(1, "wifi_password", tr("wifi_password"), True)
        self.network.currentIndexChanged.connect(self.network_changed)
        self.network_changed(0)
        self.add_input(2, "house", tr("house"))
        self.show_pictures = QCheckBox(tr("show"))
        self.alerts = QCheckBox(tr("alerts"))
        self.page_layouts[2].addWidget(self.show_pictures)
        self.page_layouts[2].addWidget(self.alerts)
        hours = QHBoxLayout()
        hours.addWidget(label(tr("hours")))
        self.hour_start, self.hour_end = QSpinBox(), QSpinBox()
        for spin in (self.hour_start, self.hour_end):
            spin.setRange(0, 23)
            hours.addWidget(spin)
        self.hour_end.setValue(23)
        self.page_layouts[2].addLayout(hours)
        self.alerts.toggled.connect(
            lambda enabled: [
                w.setEnabled(enabled) for w in (self.hour_start, self.hour_end)
            ]
        )
        self.hour_start.setEnabled(False)
        self.hour_end.setEnabled(False)
        self.find = QCheckBox(tr("find"))
        self.find.setChecked(True)
        self.page_layouts[3].addWidget(self.find)
        self.add_input(3, "camera_user", tr("camera_user"))
        self.add_input(3, "camera_password", tr("camera_password"), True)
        self.find.toggled.connect(
            lambda enabled: [
                self.inputs[key].setEnabled(enabled)
                for key in ("camera_user", "camera_password")
            ]
        )
        self.fail = QCheckBox(tr("failure"))
        self.fail.setChecked(self.args.fail)
        self.page_layouts[3].addWidget(self.fail)
        self.step_rows = []
        for step in STEPS:
            row = QHBoxLayout()
            name = label(tr(step))
            name.setMinimumWidth(300)
            status = label(tr("pending"), "muted")
            row.addWidget(name, 2)
            row.addWidget(status, 3)
            self.page_layouts[4].addLayout(row)
            self.step_rows.append(status)
        summary_columns = QHBoxLayout()
        summary_columns.setSpacing(28)
        summary_left = QWidget()
        summary_left.setStyleSheet("background: transparent;")
        sl = layout_for(summary_left, 0)
        sl.addWidget(label(tr("status_rows"), "section"))
        self.check_labels = []
        for _ in range(4):
            item = label("")
            self.check_labels.append(item)
            sl.addWidget(item)
        sl.addWidget(label(tr("manual"), "section"))
        manual = label(tr("manual_items"))
        manual.setMinimumHeight(105)
        sl.addWidget(manual)
        self.later = label(tr("camera_later"), "accent")
        sl.addWidget(self.later)
        sl.addStretch()
        summary_right = QWidget()
        summary_right.setStyleSheet("background: transparent;")
        sr = layout_for(summary_right, 0)
        sr.addWidget(label(tr("rescue"), "section"))
        sr.addWidget(label(tr("rescue_hint"), "muted"))
        sr.addWidget(label(tr("rescue_demo"), "accent"))
        self.rescue_labels = {}
        for key in ("rescue_name", "rescue_password"):
            item = label(tr(key) + ": " + tr("service_pending"), "muted")
            self.rescue_labels[key] = item
            sr.addWidget(item)
        sr.addStretch()
        summary_columns.addWidget(summary_left, 3)
        summary_columns.addWidget(summary_right, 2)
        self.page_layouts[5].addLayout(summary_columns)
        for lay in self.page_layouts:
            lay.addStretch()
        self.validation = label("", "accent")
        self.outer.addWidget(self.validation)
        nav = QHBoxLayout()
        self.back = QPushButton(tr("back"))
        self.back.setObjectName("secondary")
        self.back.clicked.connect(
            lambda: self.set_page(max(0, self.pages.currentIndex() - 1))
        )
        nav.addWidget(self.back)
        nav.addStretch()
        self.next = QPushButton(tr("next"))
        self.next.clicked.connect(self.next_page)
        nav.addWidget(self.next)
        self.outer.addLayout(nav)
        if self.args.demo:
            self.inputs["address"].setText(tr("demo_address"))
            self.inputs["house"].setText(tr("demo_site"))
            self.inputs["ssid"].setText(tr("demo_ssid"))
            self.inputs["camera_user"].setText(tr("demo_user"))
            # No sample passwords, even masked, in screenshots or source.
        self.find.setChecked(not self.args.skip_cameras)
        self.alerts.setChecked(self.args.alerts)
        if self.args.wifi:
            self.network.setCurrentIndex(1)
        self.set_page(0)
        if self.args.page:
            index = {
                "address": 0,
                "network": 1,
                "house": 2,
                "cameras": 3,
                "progress": 4,
                "summary": 5,
                "failure": 4,
                "validation": 0,
            }[self.args.page]
            self.set_page(index)
            if self.args.page == "validation":
                self.inputs["address"].clear()
                self.next_page()
            if index >= 4:
                self.begin_setup()
                if self.args.page in ("summary", "failure"):
                    self.step_timer.stop()
                    if self.args.page == "failure":
                        self.sequence.backend.failure = True
                    while not self.sequence.done:
                        self.advance_setup()
                elif self.args.page == "progress":
                    self.advance_setup()

    def add_input(self, page, key, caption, secret=False):
        self.page_layouts[page].addWidget(label(caption))
        field = QLineEdit()
        field.textChanged.connect(lambda: field.setStyleSheet(""))
        if secret:
            field.setEchoMode(QLineEdit.EchoMode.Password)
        field.setMaximumWidth(700)
        self.page_layouts[page].addWidget(field)
        self.inputs[key] = field

    def network_changed(self, index):
        for key in ("ssid", "wifi_password"):
            self.inputs[key].setEnabled(index == 1)

    def set_page(self, index):
        self.pages.setCurrentIndex(index)
        self.validation.setText("")
        self.back.setVisible(0 < index < 4)
        self.next.setVisible(index != 4)
        self.next.setText(
            tr("start") if index == 3 else tr("finish") if index == 5 else tr("next")
        )

    def valid_page(self, index):
        values = {key: w.text().strip() for key, w in self.inputs.items()}
        if index == 0:
            return bool(re.fullmatch(r"[^@\s]+@[^@\s]+", values["address"]))
        if index == 1 and self.network.currentIndex() == 1:
            return (
                bool(values["ssid"])
                and 8 <= len(self.inputs["wifi_password"].text()) <= 63
            )
        if index == 2:
            return bool(re.fullmatch(r"[a-z0-9_]+", values["house"]))
        if index == 3 and self.find.isChecked():
            return bool(
                re.fullmatch(r"[A-Za-z0-9._@-]+", values["camera_user"])
            ) and bool(self.inputs["camera_password"].text())
        return True

    def next_page(self):
        index = self.pages.currentIndex()
        if index == 5:
            for key in ("wifi_password", "camera_password"):
                self.inputs[key].clear()
            self.set_page(0)
            return
        if not self.valid_page(index):
            keys = {
                0: ("address",),
                1: ("ssid", "wifi_password"),
                2: ("house",),
                3: ("camera_user", "camera_password"),
            }[index]
            for key in keys:
                self.inputs[key].setStyleSheet("border: 1px solid #f27d7d;")
            self.inputs[keys[0]].setFocus()
            self.validation.setText(tr("validation"))
            return
        if index == 3:
            self.begin_setup()
        else:
            self.set_page(index + 1)

    def begin_setup(self):
        self.progress_title.setText(tr("progress_title"))
        answers = Answers(
            **{key: w.text() for key, w in self.inputs.items()},
            network="wifi" if self.network.currentIndex() else "ethernet",
            show_cameras=self.show_pictures.isChecked(),
            alerts=self.alerts.isChecked(),
            start_hour=self.hour_start.value(),
            end_hour=self.hour_end.value(),
            find_cameras=self.find.isChecked(),
        )
        self.sequence = Sequence(SimulatedBackend(self.fail.isChecked()), answers)
        for row in self.step_rows:
            row.setText(tr("pending"))
            row.setStyleSheet("")
        self.set_page(4)
        self.step_rows[0].setText(tr("running"))
        self.step_timer = QTimer(self)
        if not hasattr(self, "setup_pool"):
            self.setup_pool = ThreadPoolExecutor(max_workers=1)
        self.step_timer.setSingleShot(True)
        self.step_timer.timeout.connect(self.dispatch_setup)
        self.step_timer.start(self.sequence.backend.delay_for(STEPS[0]))

    def dispatch_setup(self):
        self.setup_future = self.setup_pool.submit(self.sequence.advance)
        QTimer.singleShot(50, self.check_setup)

    def check_setup(self):
        if not self.setup_future.done():
            QTimer.singleShot(50, self.check_setup)
            return
        self.show_setup_result(self.setup_future.result())
        if not self.sequence.done:
            self.step_timer.start(
                self.sequence.backend.delay_for(STEPS[self.sequence.index])
            )

    def advance_setup(self):
        # Deterministic screenshot path uses the same sequencer and presenter.
        self.show_setup_result(self.sequence.advance())

    def show_setup_result(self, result):
        if result:
            row = self.step_rows[self.sequence.index - 1]
            row.setText(tr(result.status) + tr("separator") + tr(result.message_key))
            row.setStyleSheet(
                "color: "
                + (
                    "#f27d7d"
                    if result.status == "FAIL"
                    else ACCENT if result.status == "WARN" else "#81d3b0"
                )
            )
        if self.sequence.done:
            self.step_timer.stop()
            # Secrets remain only in form memory until completion, then are discarded.
            self.sequence.answers.wifi_password = ""
            self.sequence.answers.camera_password = ""
            for key in ("wifi_password", "camera_password"):
                self.inputs[key].clear()
            if self.sequence.failed:
                self.progress_title.setText(tr("failed_title"))
                self.validation.setText(tr(result.message_key))
                self.next.setText(tr("retry"))
                self.next.setVisible(True)
                try:
                    self.next.clicked.disconnect()
                except RuntimeError:
                    pass
                self.next.clicked.connect(self.retry_setup)
            else:
                for widget, check in zip(self.check_labels, result.checks):
                    widget.setText(
                        tr(check.status) + tr("separator") + tr(check.message_key)
                    )
                    widget.setStyleSheet(
                        "color: " + (ACCENT if check.status == "WARN" else "#81d3b0")
                    )
                for key, widget in self.rescue_labels.items():
                    widget.setText(
                        tr(key) + ": " + (getattr(result, key) or tr("service_pending"))
                    )
                self.later.setVisible(not self.sequence.answers.find_cameras)
                self.set_page(5)
        elif self.sequence.index < len(STEPS):
            self.step_rows[self.sequence.index].setText(tr("running"))

    def retry_setup(self):
        self.next.clicked.disconnect()
        self.next.clicked.connect(self.next_page)
        self.set_page(1)
