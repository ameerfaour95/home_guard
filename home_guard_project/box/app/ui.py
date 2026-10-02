from pathlib import Path
import math
import re
import time
from concurrent.futures import ThreadPoolExecutor

from PySide6.QtCore import Qt, QTimer, QSize, Signal
from PySide6.QtGui import QColor, QPainter, QPen, QPixmap, QFont, QLinearGradient, QIcon, QShortcut, QKeySequence
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
    QDialog,
)

from .strings import tr, TEXT
from .model import State, Activity, ActivityFeed
from .backend import Answers, SimulatedBackend, Sequence, STEPS
from ..preview import PreviewReader
from .. import boxconfig as bc

from .theme import ACTION, OK, WARNING, ERROR, MUTED
ACCENT = ACTION
STYLE = """
QWidget { background: #10151d; color: #edf1f7; font-family: 'Segoe UI'; font-size: 15px; }
QLabel { background: transparent; }
QLabel#title { font-size: 29px; font-weight: 600; }
QLabel#section { font-size: 20px; font-weight: 600; }
QLabel#muted { color: @muted; }
QLabel#accent { color: @action; font-weight: 600; }
QFrame#card { background: #1a222e; border: 1px solid #2c3745; border-radius: 12px; }
QPushButton { background: @action; color: #111722; border: 2px solid transparent; border-radius: 7px; padding: 10px 22px; font-weight: 600; }
QPushButton#secondary { background: #263344; color: #edf1f7; }
QPushButton:disabled { background: #263344; color: #8793a4; }
QPushButton:hover { background: #77ddf2; }
QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox { background: #151e2a; border: 1px solid #435063; border-radius: 7px; padding: 10px; min-height: 23px; }
QLineEdit:focus { border: 1px solid @action; }
QCheckBox { spacing: 12px; background: transparent; padding: 6px 0; }
QCheckBox::indicator { width: 21px; height: 21px; border: 1px solid #526179; border-radius: 4px; background: #151e2a; }
QCheckBox::indicator:checked { background: @action; border: 1px solid @action; }
QScrollArea { border: none; background: transparent; }
"""
for token, colour in (("@action",ACTION),("@muted",MUTED)):
    STYLE = STYLE.replace(token,colour)
STYLE += f"""
QLabel#ok {{ color: {OK}; }}
QLabel#warning {{ color: {WARNING}; }}
QLabel#error {{ color: {ERROR}; }}
QPushButton:focus {{ border: 2px solid #edf1f7; }}
QCheckBox:focus {{ border: 1px solid {ACTION}; border-radius: 4px; }}
QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus {{ border: 2px solid {ACTION}; }}
QCheckBox::indicator:checked {{ background: {OK}; border: 1px solid {OK}; image: url("{(Path(__file__).parent / 'check.svg').as_posix()}"); }}
QScrollBar:vertical {{ background: #10151d; width: 9px; }}
QScrollBar::handle:vertical {{ background: #435063; min-height: 30px; border-radius: 4px; }}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0px; }}
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
    clicked = Signal()
    def __init__(self, name):
        super().__init__()
        self.setObjectName("card")
        self.name = name
        self.picture = None
        self.stopped = False
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

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit()
        super().mousePressEvent(event)

    def update_picture(self, pix):
        self.picture = pix if pix and not pix.isNull() else None
        self.status.setText(tr("stopped") if self.stopped else tr("live") if self.picture else tr("offline"))
        self.status.setObjectName("ok" if self.picture else "muted")
        self.status.setStyleSheet("color: " + (OK if self.picture else MUTED))
        self.update()

    def paintEvent(self, event):
        super().paintEvent(event)
        p = QPainter(self)
        area = self.rect().adjusted(2, 2, -2, -42)
        if self.picture:
            scaled = self.picture.scaled(
                area.size(),
                Qt.AspectRatioMode.KeepAspectRatioByExpanding,
                Qt.TransformationMode.SmoothTransformation,
            )
            p.setClipRect(area)
            p.drawPixmap(
                area.center().x() - scaled.width() // 2,
                area.center().y() - scaled.height() // 2,
                scaled,
            )
        else:
            p.setPen(QColor("#98a6ba"))
            p.setFont(QFont("Segoe UI", 14))
            p.drawText(area, Qt.AlignmentFlag.AlignCenter, tr("stopped") if self.stopped else tr("offline_hint"))
        p.end()


class Window(QMainWindow):
    def __init__(self, args):
        super().__init__()
        self.args = args
        from .box_controls import BoxControls, Settings
        self.box_controls = BoxControls(demo=args.demo, stopped=args.state == "stopped", settings=Settings(mode="inference" if args.state == "inference" else "data_collection", show_cameras=args.state != "hidden"))
        self.start_requested = False
        self.setWindowTitle(tr("brand"))
        self.setWindowIcon(QIcon(str(Path(__file__).parents[1] / "assets" / "logo.ico")))
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
        self.header_hint = label(tr("simulation") if args.setup else tr("close_hint"), "muted")
        titles.addWidget(self.header_hint)
        header.addLayout(titles, 1)
        header.addStretch()
        if args.demo or args.setup:
            header.addWidget(label(tr("demo"), "muted"))
        self.outer.addLayout(header)
        if args.setup:
            self.build_setup()
        else:
            self.build_dashboard()

    def build_dashboard(self):
        control_row = QHBoxLayout()
        self.control_note = label("", "warning")
        overview_button = QPushButton(tr("overview"))
        overview_button.setObjectName("secondary")
        overview_button.clicked.connect(lambda: self.content_stack.setCurrentIndex(0))
        control_row.addWidget(overview_button)
        cameras_button = QPushButton(tr("cameras_page"))
        cameras_button.setObjectName("secondary")
        cameras_button.clicked.connect(self.open_cameras)
        control_row.addWidget(cameras_button)
        settings_button = QPushButton(tr("settings"))
        settings_button.setObjectName("secondary")
        settings_button.clicked.connect(self.open_settings)
        control_row.addWidget(settings_button)
        control_row.addWidget(self.control_note, 1)
        self.run_button = QPushButton(tr("stop_box"))
        self.run_button.clicked.connect(self.toggle_running)
        control_row.addWidget(self.run_button)
        self.outer.addLayout(control_row)
        self.stop_banner = QFrame()
        self.stop_banner.setStyleSheet(f"QFrame {{ background: #493b20; border-radius: 8px; }} QLabel {{ color: {WARNING}; }}")
        banner_layout = QHBoxLayout(self.stop_banner)
        banner_layout.setContentsMargins(18,12,18,12)
        banner_layout.addWidget(label(tr("stopped_banner")),1)
        banner_start = QPushButton(tr("start_box"))
        banner_start.clicked.connect(self.toggle_running)
        banner_layout.addWidget(banner_start)
        self.outer.addWidget(self.stop_banner)
        self.stop_banner.hide()
        from .alert_pause import AlertPause
        self.alert_pause = AlertPause(demo=self.args.demo)
        if self.args.demo and self.args.state == "paused":
            self.box_controls._settings = __import__("dataclasses").replace(self.box_controls._settings,mode="inference")
            self.alert_pause.demo_until = time.time()+3600
        self.content_stack = QStackedWidget()
        overview = QWidget()
        overview_layout = layout_for(overview, 0)
        self.content_stack.addWidget(overview)
        self.outer.addWidget(self.content_stack, 1)
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
        overview_layout.addLayout(stats)
        pause_row = QHBoxLayout()
        self.pause_label = label("", "warning")
        self.resume_button = QPushButton(tr("resume_alerts"))
        self.resume_button.clicked.connect(self.resume_alerts)
        pause_row.addWidget(self.pause_label,1)
        pause_row.addWidget(self.resume_button)
        overview_layout.addLayout(pause_row)
        self.pause_label.hide()
        self.resume_button.hide()
        body = QHBoxLayout()
        left = QWidget()
        leftlay = layout_for(left, 0)
        self.camera_heading = label(tr("cameras"), "section")
        leftlay.addWidget(self.camera_heading)
        leftlay.addWidget(label(tr("enlarge_hint"),"muted"))
        self.camera_stack = QStackedWidget()
        self.grid_widget = QWidget()
        self.grid = QGridLayout(self.grid_widget)
        self.grid.setContentsMargins(0, 0, 0, 0)
        self.grid.setSpacing(14)
        self.grid_scroll = QScrollArea()
        self.grid_scroll.setWidgetResizable(True)
        self.grid_scroll.setWidget(self.grid_widget)
        self.camera_stack.addWidget(self.grid_scroll)
        self.expanded_tile = None
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
        overview_layout.addLayout(body, 1)
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
        from .camera_controls import CameraControls
        self.camera_controls = CameraControls(self.box_controls, TEXT["demo_names"][:self.args.cameras] if self.args.state != "empty" else ())
        self.tick()
        self.details.setChecked(self.args.details)
        from .settings_ui import SettingsPage
        self.settings_page = SettingsPage(self.box_controls, self.settings_changed)
        self.content_stack.addWidget(self.settings_page.widget)
        from .camera_ui import CameraPage
        self.cameras_page = CameraPage(self.camera_controls, self.settings_changed)
        self.content_stack.addWidget(self.cameras_page.widget)
        if getattr(self.args, "panel", None) == "cameras":
            self.open_cameras()
        if getattr(self.args, "panel", None) == "settings":
            self.open_settings()

    def fetch(self):
        from ..heartbeat import build_heartbeat

        settings = bc.load_box_settings()
        events, upload = self.activity_feed.read()
        from ..__main__ import _clip_dirs
        mode = bc.get_option("mode")
        payload = build_heartbeat(str(settings.get("site", "")), *_clip_dirs(mode), bc.ALIVE_FILE, mode=mode)
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
            names = [c.name for c in self.camera_controls.records if c.enabled]
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
            demo_settings = self.box_controls.load_settings()
            state.mode = demo_settings.mode
            state.collecting = self.box_controls.phase() == "running"
            self.apply_state(
                state,
                demo_settings.show_cameras,
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
                    if scenario == "offline" or self.box_controls.is_stopped()
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
        if self.current_state and self.box_controls.pending_at is not None:
            self.apply_state(self.current_state, self.current_show, self.events)
        if self.current_state and self.box_controls.is_stopped() and self.current_state.collecting:
            self.apply_state(self.current_state, self.current_show, self.events)
        # Re-read permission at each viewer tick. Never touch the marker when hidden.
        try:
            allowed = bool(bc.get_option("show_cameras"))
        except Exception:
            allowed = False
        if self.current_state is not None and allowed != self.current_show:
            self.apply_state(self.current_state, allowed, self.events)
        if allowed and not self.box_controls.is_stopped():
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
        stopped = self.box_controls.is_stopped()
        phase = self.box_controls.phase(state.collecting)
        if stopped:
            state.collecting = False
            self.start_requested = False
        if state.collecting:
            self.start_requested = False
        self.stop_banner.setVisible(stopped)
        self.run_button.setVisible(not stopped)
        self.header_hint.setText(tr("close_stopped") if stopped else tr("close_hint") if state.collecting else tr("close_starting"))
        self.run_button.setText(tr("start_box") if stopped else tr("stop_box"))
        self.run_button.setEnabled(not self.start_requested)
        self.control_note.setText("" if stopped else tr("applying") if phase == "restarting" else tr("start_pending") if self.start_requested else "")
        self.run_button.setStyleSheet(f"background: {ERROR};" if not stopped else "")
        if self.events != events:
            self.events = events
            self.render_activity()
        self.house_label.setText(state.site or tr("home"))
        values = [
            tr("restarting") if phase == "restarting" else tr("watching" if state.mode == "inference" else "collecting") if state.collecting else tr("stopped"),
            state.upload or tr("never"),
            str(state.waiting),
            tr("gb", value=state.disk),
        ]
        for (value, hint), text in zip(self.stats, values):
            value.setText(tr("unknown") if state.error else text)
        from .alert_hours import hours_description
        settings = self.box_controls.load_settings()
        until, some = self.alert_pause.status(state.cameras) if state.mode == "inference" else (None,False)
        paused = tr("paused_some" if some else "paused_until", time=time.strftime("%H:%M",time.localtime(until))) if until else ""
        self.pause_label.setText(paused)
        self.pause_label.setVisible(bool(paused) and not stopped)
        self.resume_button.setVisible(bool(paused) and not stopped)
        self.stats[0][1].setText(hours_description(settings.alert_start_hour,settings.alert_end_hour) if state.mode == "inference" else tr("collection"))
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
            "applying" if phase == "restarting" else
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
                "stopped_title": "stopped_hint",
                "applying": "applying_hint",
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
            self.expanded_tile = None
            for tile in self.tiles:
                tile.clicked.connect(lambda tile=tile: self.toggle_tile(tile))
            self.arrange_tiles()
        for tile in self.tiles:
            tile.stopped = stopped
            if stopped: tile.update_picture(None)

    def resume_alerts(self):
        try:
            self.alert_pause.resume()
            self.tick()
            if self.current_state: self.apply_state(self.current_state,self.current_show,self.events)
        except Exception:
            self.control_note.setText(tr("control_error"))

    def open_cameras(self):
        self.content_stack.setCurrentIndex(2)
        self.cameras_page.open()

    def open_settings(self):
        self.settings_page.reload()
        self.content_stack.setCurrentIndex(1)

    def settings_changed(self):
        self.last_poll = 0
        if self.current_state:
            self.apply_state(self.current_state, self.box_controls.load_settings().show_cameras, self.events)
        self.tick()

    def toggle_running(self):
        if self.box_controls.is_stopped():
            try:
                self.box_controls.start()
                self.start_requested = not self.args.demo
                if self.current_state:
                    self.current_state.collecting = False
                    self.apply_state(self.current_state, self.current_show, self.events)
                self.last_poll = 0
                self.tick()
            except OSError:
                self.control_note.setText(tr("control_error"))
            return
        dialog = QDialog(self)
        dialog.setWindowTitle(tr("stop_confirm"))
        dialog.setMinimumWidth(520)
        lay = layout_for(dialog)
        lay.addWidget(label(tr("stop_confirm"), "section"))
        lay.addWidget(label(tr("stop_explanation")))
        row = QHBoxLayout()
        cancel = QPushButton(tr("keep_running"))
        cancel.setObjectName("secondary")
        cancel.clicked.connect(dialog.reject)
        stop = QPushButton(tr("stop_box"))
        stop.setStyleSheet(f"background: {ERROR};")
        stop.clicked.connect(dialog.accept)
        row.addWidget(cancel); row.addWidget(stop)
        lay.addLayout(row)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            try:
                self.box_controls.stop()
                self.last_poll = 0
                if self.current_state:
                    self.apply_state(self.current_state, self.current_show, self.events)
                self.tick()
            except OSError:
                self.control_note.setText(tr("control_error"))

    def toggle_tile(self, tile):
        self.expanded_tile = None if self.expanded_tile is tile else tile
        self.arrange_tiles()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Escape and getattr(self, "expanded_tile", None):
            self.expanded_tile = None
            self.arrange_tiles()
            event.accept()
        else:
            super().keyPressEvent(event)

    def arrange_tiles(self):
        while self.grid.count():
            self.grid.takeAt(0)
        for index in range(3):
            self.grid.setRowStretch(index,0)
            self.grid.setColumnStretch(index,0)
        if self.expanded_tile:
            for tile in self.tiles:
                tile.setVisible(tile is self.expanded_tile)
            self.grid.addWidget(self.expanded_tile,0,0)
            self.grid.setRowStretch(0,1)
            self.grid.setColumnStretch(0,1)
            return
        columns = 1 if len(self.tiles) == 1 else 2 if len(self.tiles) <= 4 else 3
        for i, tile in enumerate(self.tiles):
            tile.show()
            self.grid.addWidget(tile,i//columns,i%columns)
            self.grid.setRowStretch(i//columns,1)
            self.grid.setColumnStretch(i%columns,1)

    def render_activity(self):
        while self.activity_layout.count():
            item = self.activity_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        for event in self.events:
            self.activity_layout.addWidget(label(tr("event_line",time=event.time,text=event.text)))
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
        if hasattr(self, "cameras_page"):
            self.cameras_page.close()
        if hasattr(self, "pool"):
            self.timer.stop()
            self.pool.shutdown(wait=False, cancel_futures=True)
        if hasattr(self, "setup_pool"):
            self.step_timer.stop()
            self.setup_pool.shutdown(wait=False, cancel_futures=True)
        super().closeEvent(event)

    def build_setup(self):
        step_bar = QHBoxLayout()
        self.step_labels = []
        self.step_icons = []
        done_icon = QPixmap(18,18)
        done_icon.fill(Qt.GlobalColor.transparent)
        painter = QPainter(done_icon)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(QPen(QColor(OK),2.5))
        painter.drawLine(2,9,7,14)
        painter.drawLine(7,14,16,3)
        painter.end()
        for name in TEXT["step_names"]:
            item = label(name,"muted")
            step = QHBoxLayout()
            icon = QLabel()
            icon.setPixmap(done_icon)
            icon.setFixedSize(18,18)
            step.addWidget(icon)
            step.addWidget(item,1)
            step_bar.addLayout(step,1)
            self.step_icons.append(icon)
            self.step_labels.append(item)
        self.outer.addLayout(step_bar)
        self.pages = QStackedWidget()
        self.outer.addWidget(self.pages, 1)
        self.inputs = {}
        self.field_guidance = {}
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
            if title == "summary_title":
                summary_scroll = QScrollArea()
                summary_scroll.setWidgetResizable(True)
                summary_scroll.setWidget(panel)
                self.pages.addWidget(summary_scroll)
            else:
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
        from .alert_hours import AlertHours
        self.wizard_hours = AlertHours()
        self.hour_start, self.hour_end = self.wizard_hours.start, self.wizard_hours.end
        self.page_layouts[2].addWidget(self.wizard_hours)
        self.alerts.toggled.connect(self.wizard_hours.setEnabled)
        self.wizard_hours.setEnabled(False)
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
        self.summary_house = label("", "section")
        self.page_layouts[5].addWidget(self.summary_house)
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
        manual.setMinimumHeight(180)
        manual.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        sl.addWidget(manual)
        self.later = label(tr("camera_later"), "warning")
        sl.addWidget(self.later)
        sl.addStretch()
        summary_right = QWidget()
        summary_right.setStyleSheet("background: transparent;")
        sr = layout_for(summary_right, 0)
        sr.addWidget(label(tr("rescue"), "section"))
        sr.addWidget(label(tr("rescue_hint"), "muted"))
        sr.addWidget(label(tr("rescue_demo"), "muted"))
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
        self.validation = label("", "error")
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
        self.next.setDefault(True)
        self.next.clicked.connect(self.next_page)
        nav.addWidget(self.next)
        self.outer.addLayout(nav)
        self.enter_shortcuts = []
        for key in (Qt.Key.Key_Return,Qt.Key.Key_Enter):
            shortcut = QShortcut(QKeySequence(key),self)
            shortcut.activated.connect(lambda: self.next.click() if self.next.isVisible() and self.next.isEnabled() else None)
            self.enter_shortcuts.append(shortcut)
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
        guidance = label("", "error")
        guidance.hide()
        self.page_layouts[page].addWidget(guidance)
        self.field_guidance[key] = guidance
        field.textChanged.connect(lambda: guidance.hide())

    def network_changed(self, index):
        for key in ("ssid", "wifi_password"):
            self.inputs[key].setEnabled(index == 1)

    def set_page(self, index):
        self.pages.setCurrentIndex(index)
        if index == 5:
            self.summary_house.setText(tr("summary_house", house=self.inputs["house"].text()))
        self.validation.setText("")
        self.update_step_bar(index)
        self.back.setVisible(0 < index < 4)
        self.next.setVisible(index != 4)
        self.next.setText(
            tr("start") if index == 3 else tr("finish") if index == 5 else tr("next")
        )

    def update_step_bar(self,index):
        current = min(index,4)
        for i,(item,name) in enumerate(zip(self.step_labels,TEXT["step_names"])):
            done = i < current or index == 5
            self.step_icons[i].setVisible(done)
            item.setText(name if done else tr("step_number",number=i+1,name=name))
            item.setStyleSheet(f"color: {OK if done else '#edf1f7' if i == current else MUTED}; padding: 8px; border-bottom: 2px solid {'#edf1f7' if i == current and index != 5 else 'transparent'};")

    def errors_for_page(self,index):
        from .guidance import field_errors
        return field_errors(index,{key:w.text() for key,w in self.inputs.items()},wifi=self.network.currentIndex()==1,find=self.find.isChecked())

    def valid_page(self,index):
        return not self.errors_for_page(index)

    def next_page(self):
        index = self.pages.currentIndex()
        if index == 5:
            self.close()
            return
        errors = self.errors_for_page(index)
        if errors:
            for key,message in errors.items():
                self.inputs[key].setStyleSheet(f"border: 1px solid {ERROR};")
                self.field_guidance[key].setText(tr(message))
                self.field_guidance[key].show()
            self.inputs[next(iter(errors))].setFocus()
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
            start_hour=self.wizard_hours.values()[0],
            end_hour=self.wizard_hours.values()[1],
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
                    ERROR
                    if result.status == "FAIL"
                    else WARNING if result.status == "WARN" else OK
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
                from .guidance import retry_page
                self.update_step_bar(4 if result.step == "readiness" else retry_page(result.step))
                self.validation.setText("")
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
                        "color: " + (WARNING if check.status == "WARN" else ERROR if check.status == "FAIL" else OK)
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
        from .guidance import retry_page
        self.set_page(retry_page(self.sequence.results[-1].step))
