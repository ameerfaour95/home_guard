from pathlib import Path
import math
import re
import time
from concurrent.futures import ThreadPoolExecutor

from PySide6.QtCore import Qt, QTimer, QSize, Signal, QRectF, QEvent, QVariantAnimation
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
    QDoubleSpinBox,
    QTextEdit,
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
from .setup_pages import Page
from .model import State, Activity, ActivityFeed
from .backend import Answers, SimulatedBackend, Sequence, STEPS
from ..preview import PreviewReader
from .. import boxconfig as bc

from .theme import ACTION, OK, WARNING, ERROR, MUTED
ACCENT = ACTION
from .theme import stylesheet
STYLE = stylesheet()


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
    layout.setSpacing(16)
    return layout


def demo_picture(number):
    from .demo_media import picture
    return picture(number)


class CameraTile(QFrame):
    clicked = Signal()
    double_clicked = Signal()
    frame_changed = Signal()
    def __init__(self, name):
        super().__init__()
        self.setObjectName("card")
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.name = name
        self.picture = None
        self.frame_stamp = None
        self.live_started = time.time()
        self.stopped = False
        self.off = False
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.hero = False
        self.box_opacity = 1
        self.detections = ()
        self.detector_enabled = False
        self.show_detections=False
        self.detection_labels="confidence"
        from . import motion
        self.caption_position=0.
        self.caption_corner=0.
        self.caption_animation=QVariantAnimation(self)
        self.caption_animation.setDuration(motion.PANE_MS)
        self.caption_animation.setEasingCurve(motion.EASING)
        self.caption_animation.valueChanged.connect(self.move_caption)
        self.detector_note = label("", "muted")
        self.detector_note.setObjectName("cameraDetection")
        self.detector_note.hide()
        self.status = label(tr("offline"), "muted")
        self.status.hide()
        self.caption = label(name.replace("_", " ").title())
        self.caption.setObjectName("cameraCaption");self.caption.hide()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 8, 16, 8)
        layout.addStretch()
        self.turn_on=QPushButton('Turn on');self.turn_on.setMaximumWidth(140);self.turn_on.hide();layout.addWidget(self.turn_on,0,Qt.AlignmentFlag.AlignCenter)
        footer=QHBoxLayout()
        footer.addWidget(self.caption)
        footer.addStretch()
        footer.addWidget(self.status)
        layout.addLayout(footer)
        self.detector_note.setContentsMargins(16,0,0,8)
        layout.addWidget(self.detector_note)
        layout.setSpacing(3)
        self.skeleton_timer=QTimer(self);self.skeleton_timer.setInterval(50);self.skeleton_timer.timeout.connect(self.update)
        self.skeleton_timer.start()
        self.heartbeat = QTimer(self)
        self.heartbeat.timeout.connect(self.live_tick)
        self.heartbeat.start(1000)
        self.setMinimumSize(180, 140)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.setFocus();self.clicked.emit()
        super().mousePressEvent(event)

    def mouseDoubleClickEvent(self,event):
        self.double_clicked.emit()

    def update_picture(self, pix, stamp=None):
        if stamp is not None:
            self.frame_stamp = stamp
        elif pix and not pix.isNull() and pix is not self.picture:
            self.frame_stamp = time.time()
        self.picture = pix if pix and not pix.isNull() else None
        if self.picture or self.stopped or self.off: self.skeleton_timer.stop()
        elif not self.skeleton_timer.isActive(): self.skeleton_timer.start()
        self.status.setText("Off" if self.off else tr("stopped") if self.stopped else tr("premium_live") if self.picture else tr("offline"))
        self.status.setObjectName("ok" if self.picture else "muted")
        self.status.setStyleSheet("")
        self.update()
        self.frame_changed.emit()

    def live_tick(self):
        if self.isVisible(): self.update()

    def showEvent(self, event):
        super().showEvent(event)
        self.heartbeat.start(1000)

    def hideEvent(self, event):
        self.heartbeat.stop()
        self.skeleton_timer.stop()
        super().hideEvent(event)

    def thumbnail_labels(self, badge_width):
        """Shared paint/test geometry, including the 80 px thumbnail rail floor."""
        area = QRectF(self.rect().adjusted(1, 1, -1, -1))
        width = max(0, area.width() - 16)
        return {
            "name": QRectF(area.left()+8, area.bottom()-62, width, 24),
            "caption": QRectF(area.left()+8, area.bottom()-34, width, 24),
            "badge": QRectF(area.left()+8, area.top()+8, min(width, badge_width), 28),
        }

    def thumbnail_scrim(self):
        area=QRectF(self.rect().adjusted(1,1,-1,-1))
        return QRectF(area.left(),area.bottom()-area.height()*.45,area.width(),area.height()*.45)

    def move_caption(self, position):
        self.caption_position=float(position)
        self.update()

    def hero_caption_rect(self, visible, obstacles):
        """Keep the chosen corner until obstructed; never paint over a box."""
        width=min(visible.width()-32,340)
        left=QRectF(visible.left()+16,visible.bottom()-76,width,60)
        travel=max(0,visible.width()-32-width)
        def at(position): return left.translated(travel*position,0)
        def blocked(rect): return any(rect.intersects(box.adjusted(-4,-4,4,4)) for box in obstacles)
        other=1-self.caption_corner
        if blocked(at(self.caption_corner)) and not blocked(at(other)):
            self.caption_corner=other
            self.caption_animation.stop()
            self.caption_animation.setStartValue(self.caption_position)
            self.caption_animation.setEndValue(other)
            self.caption_animation.start()
        rect=at(self.caption_position)
        # During travel (or if both corners are occupied), conceal the caption
        # until it has a clear rectangle instead of obscuring a subject.
        return rect,not blocked(rect)

    def paintEvent(self, event):
        super().paintEvent(event)
        p = QPainter(self)
        area = self.rect().adjusted(1,1,-1,-1)
        from .liveness import frame_health
        live_text, reconnecting = frame_health(self.frame_stamp, time.time(), self.live_started)
        self.painted_label_rects = {}
        self.painted_detection_rects = []
        if self.picture:
            from .detector_view import box_rect,picture_rect
            from .camera_presentation import image_rect
            x,y,w,h=image_rect((area.x(),area.y(),area.width(),area.height()),(self.picture.width(),self.picture.height()),self.hero,self.devicePixelRatioF())
            p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
            from PySide6.QtGui import QPainterPath
            clip=QPainterPath();clip.addRoundedRect(QRectF(area),16,16);p.setClipPath(clip)
            if self.hero:
                from .camera_presentation import ambient_picture
                key=(int(time.monotonic()),area.width(),area.height())
                if key!=getattr(self,'_ambient_key',None):
                    self._ambient=ambient_picture(self.picture);self._ambient_key=key
                p.drawPixmap(QRectF(area),self._ambient,QRectF(self._ambient.rect()))
                p.fillRect(area,QColor(0,0,0,165))
            p.drawPixmap(QRectF(x,y,w,h),self.picture,QRectF(self.picture.rect()))
            if reconnecting: p.fillRect(area, QColor(0,0,0,110))
            if not self.off and not self.stopped and not reconnecting and self.show_detections:
                p.save()
                detection_area=QRectF(area)
                if not self.hero:
                    detection_area.setBottom(self.thumbnail_scrim().top())
                    p.setClipRect(detection_area,Qt.ClipOperation.IntersectClip)
                p.setOpacity(self.box_opacity)
                p.setFont(QFont("Segoe UI",11))
                for detection in self.detections:
                    rect=QRectF(*box_rect(detection.box,(x,y,w,h)))
                    painted=rect.intersected(detection_area)
                    if painted.isEmpty(): continue
                    self.painted_detection_rects.append(painted)
                    p.setPen(QPen(QColor(detection.color),(3 if self.hero else 2) if detection.width>1 else 1))
                    length=min(24,rect.width()/3,rect.height()/3)
                    for cx,cy,sx,sy in ((rect.left(),rect.top(),1,1),(rect.right(),rect.top(),-1,1),(rect.left(),rect.bottom(),1,-1),(rect.right(),rect.bottom(),-1,-1)):
                        from PySide6.QtCore import QLineF
                        p.drawLine(QLineF(cx,cy,cx+sx*length,cy));p.drawLine(QLineF(cx,cy,cx,cy+sy*length))
                    if not self.hero or self.detection_labels=="none": continue
                    from .detector_view import object_name
                    text=detection.caption() if self.detection_labels=="confidence" else object_name(detection.label).capitalize();metrics=p.fontMetrics()
                    tx=max(x,min(rect.left(),x+w-metrics.horizontalAdvance(text)-16))
                    ty=max(y,rect.top()-metrics.height()-12)
                    tag=QRectF(tx,ty,metrics.horizontalAdvance(text)+16,metrics.height()+8)
                    self.painted_label_rects[f"chip_{len(self.painted_label_rects)}"] = tag
                    p.fillRect(tag,QColor("#101a21"));p.drawText(tag.adjusted(8,0,-8,0),Qt.AlignmentFlag.AlignVCenter,text)
                p.restore()
            if self.off:
                p.fillRect(area,QColor(0,0,0,155))
                p.save();p.translate(area.right()-35,35);p.rotate(45);p.fillRect(QRectF(-80,-15,160,30),QColor('#293945'));p.setPen(QColor('#edf4f6'));p.drawText(QRectF(-60,-15,120,30),Qt.AlignmentFlag.AlignCenter,'Off');p.restore()
            p.setFont(QFont("Segoe UI",11))
            visible=QRectF(x,y,w,h).intersected(QRectF(area))
            p.save()
            caption_visible=True
            if self.hero:
                obstacles=self.painted_detection_rects+list(self.painted_label_rects.values())
                footer,caption_visible=self.hero_caption_rect(visible,obstacles)
                if not caption_visible: p.setOpacity(0)
                p.setPen(Qt.PenStyle.NoPen);p.setBrush(QColor(12,22,28,220));p.drawRoundedRect(footer,8,8)
            else:
                footer=QRectF(area.left()+12,area.bottom()-66,area.width()-24,56)
                scrim=self.thumbnail_scrim()
                gradient=QLinearGradient(0,scrim.top(),0,scrim.bottom());gradient.setColorAt(0,QColor(0,0,0,0));gradient.setColorAt(1,QColor(0,0,0,225))
                p.fillRect(scrim,gradient)
            labels = self.thumbnail_labels(0) if not self.hero else {
                "name": footer.adjusted(12,0,-12,-28), "caption": footer.adjusted(12,28,-12,0)}
            if caption_visible:
                self.painted_label_rects.update({key: labels[key] for key in ("name", "caption")})
            name = p.fontMetrics().elidedText(self.caption.text(), Qt.TextElideMode.ElideRight, int(labels["name"].width()))
            p.setPen(QColor('#edf4f6'));p.drawText(labels["name"],Qt.AlignmentFlag.AlignVCenter,name)
            p.setFont(QFont('Segoe UI',10));p.setPen(QColor('#bfccd3'))
            note=self.detector_note.text()
            if reconnecting:
                from .liveness import relative_time
                note="Last frame "+relative_time(self.frame_stamp,time.time())
            note=p.fontMetrics().elidedText(note,Qt.TextElideMode.ElideRight,int(labels["caption"].width()))
            p.drawText(labels["caption"],Qt.AlignmentFlag.AlignVCenter,note)
            p.restore()
        else:
            if not self.off and not self.stopped:
                gradient=QLinearGradient(0,0,area.width(),0);phase=(math.sin(time.monotonic()*2)+1)/2
                gradient.setColorAt(0,QColor('#14212b'));gradient.setColorAt(max(.01,min(.99,phase)),QColor('#20333e'));gradient.setColorAt(1,QColor('#14212b'));p.fillRect(area,gradient)
            p.setPen(QColor("#98a6ba"))
            p.setFont(QFont("Segoe UI", 14))
            p.drawText(area, Qt.AlignmentFlag.AlignCenter, "Off" if self.off else tr("stopped") if self.stopped else tr("offline_hint"))
        if not self.off and not self.stopped:
            p.setClipping(False)
            p.setFont(QFont("Segoe UI", 10))
            width = p.fontMetrics().horizontalAdvance(live_text)+38
            badge = QRectF(area.left()+12, area.top()+12, width, 28)
            if not self.hero: badge = self.thumbnail_labels(width)["badge"]
            self.painted_label_rects["badge"] = badge
            live_text = p.fontMetrics().elidedText(live_text, Qt.TextElideMode.ElideRight, max(0, int(badge.width()-30)))
            p.setPen(Qt.PenStyle.NoPen);p.setBrush(QColor(12,22,28,225));p.drawRoundedRect(badge,7,7)
            color = QColor(WARNING if reconnecting else OK)
            color.setAlphaF(1 if reconnecting else .65+.35*(math.sin(time.monotonic()*3)+1)/2)
            p.setBrush(color);p.drawEllipse(QRectF(badge.left()+10,badge.top()+11,6,6))
            p.setPen(QColor('#edf4f6'));p.drawText(badge.adjusted(23,0,-7,0),Qt.AlignmentFlag.AlignVCenter,live_text)
        p.end()


class Window(QMainWindow):
    def __init__(self, args):
        super().__init__()
        from .motion import install
        install()
        self.args = args
        self.remote_target = getattr(args,"remote_box",None)
        from .preferences import ViewerPreference
        from dataclasses import replace
        self.viewer_preference=ViewerPreference()
        self.viewer_settings=self.viewer_preference.load()
        if getattr(args,"detections",False): self.viewer_settings=replace(self.viewer_settings,detections=True)
        from .box_controls import BoxControls, Settings
        self.box_controls = BoxControls(demo=args.demo or bool(self.remote_target), stopped=args.state in ("stopped","ai-stopped"), settings=Settings(mode="inference" if (args.state in ("inference","live-detections","off-camera","no-cameras","box-unreachable") or args.state.startswith("ai-")) else "data_collection", show_cameras=args.state != "hidden"))
        self.start_requested = False
        self.box_unreachable = not args.setup and (args.state=="box-unreachable" if args.demo else not Path(bc.BOX_YAML).is_file())
        if self.remote_target: self.box_unreachable=False
        from .preferences import AddressPreference
        self.last_box_address=tr("demo_address").split("@")[-1] if args.demo else AddressPreference().load().split("@")[-1]
        self.setWindowTitle(tr("setup_window_title") if args.setup else tr("brand"))
        self.setWindowIcon(QIcon(str(Path(__file__).parents[1] / "assets" / "home_guard.ico")))
        self.resize(*map(int, args.size.split("x")))
        self.setMinimumSize(1000, 650)
        self.setStyleSheet(stylesheet(getattr(args,"theme","dark")))
        self.root = QWidget()
        self.setCentralWidget(self.root)
        self.outer = layout_for(self.root, 24)
        self.outer.setSpacing(16)
        header = QHBoxLayout()
        logo = QLabel()
        pix = QPixmap(str(Path(__file__).parents[1] / "assets" / "logo.png"))
        # The logo can be unreadable (an antivirus sandbox hides files): run without it.
        if not pix.isNull():
            logo.setPixmap(
                pix.scaled(
                    130,
                    60,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
            )
        if not args.setup:
            from .ai_activity_ui import icon
            logo.setPixmap(icon("shield").pixmap(40,40))
        header.addWidget(logo)
        header.addSpacing(20)
        titles = QVBoxLayout()
        self.house_label = label(tr("setup_window_title") if args.setup else tr("home"), "title")
        if not args.setup:
            self.house_label.setObjectName("title")

        titles.addWidget(self.house_label)
        self.header_hint = label(tr("simulation") if args.setup and args.demo else tr("setup_live_hint") if args.setup else tr("close_hint"), "muted")
        if not args.setup:
            from .alert_types_ui import ElidedLabel
            self.header_hint.deleteLater()
            self.header_hint = ElidedLabel(tr('close_hint'))
            self.header_hint.setObjectName("headline");self.header_hint.setWordWrap(False)
        if args.setup: titles.addWidget(self.header_hint)
        else:
            self.status_header=QHBoxLayout();self.status_header.setSpacing(16);self.status_header.addWidget(self.header_hint,1);titles.addLayout(self.status_header)
            from .alert_types_ui import ElidedLabel
            self.alert_status=ElidedLabel();self.alert_status.setObjectName("muted");titles.addWidget(self.alert_status)
        header.addLayout(titles, 1)
        if args.setup: header.addStretch()
        if args.demo and args.setup:
            header.addWidget(label(tr("demo"), "muted"))
        self.top_header=header
        self.outer.addLayout(header)
        if args.setup:
            self.build_setup()
        else:
            self.build_dashboard()

    def build_dashboard(self):
        self.delivery_banner=QFrame()
        self.delivery_banner.setObjectName("deliveryBanner")
        delivery_layout=layout_for(self.delivery_banner,16)
        delivery_layout.setSpacing(6)
        delivery_layout.addWidget(label(tr("ai_delivery_banner"),"section"))
        self.delivery_error=label("")
        self.delivery_error.setTextFormat(Qt.TextFormat.PlainText)
        delivery_layout.addWidget(self.delivery_error)
        self.outer.insertWidget(1,self.delivery_banner)
        self.delivery_banner.hide()
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
        self.top_header.addWidget(self.control_note)
        from .ai_activity_ui import icon
        for button,name in ((overview_button,"shield"),(cameras_button,"camera"),(settings_button,"settings")):
            button.setToolTip(button.text());button.setAccessibleName(button.text());button.setText("");button.setIcon(icon(name));button.setIconSize(QSize(22,22));button.setObjectName("iconButton");self.top_header.addWidget(button)
        self.top_header.addWidget(self.run_button)
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
        self.alert_pause = AlertPause(demo=self.args.demo or bool(self.remote_target))
        if self.args.demo and self.args.state == "paused":
            self.box_controls._settings = __import__("dataclasses").replace(self.box_controls._settings,mode="inference")
            self.alert_pause.demo_until = time.time()+3600
        from .motion import AnimatedStack
        self.content_stack = AnimatedStack()
        overview = QWidget()
        overview_layout = layout_for(overview, 0)
        self.content_stack.addWidget(overview)
        self.outer.addWidget(self.content_stack, 1)
        stats = QHBoxLayout()
        self.stats = []
        for key in ("collecting", "upload", "waiting", "disk"):
            panel = card()
            lay = QHBoxLayout(panel)
            lay.setContentsMargins(16,8,16,8)
            value = label(tr("loading"))
            hint = label(tr(key), "muted")
            lay.addWidget(value)
            lay.addWidget(hint)
            stats.addWidget(panel)
            self.stats.append((value, hint))
        self.stats_layout=stats
        pause_row = QHBoxLayout()
        self.pause_label = label("", "warning")
        self.resume_button = QPushButton(tr("resume_alerts"))
        self.resume_button.clicked.connect(self.resume_alerts)
        self.resume_button.setObjectName("secondary");self.status_header.insertWidget(1,self.resume_button)
        pause_row.addWidget(self.pause_label,1)

        self.pause_row=pause_row
        self.pause_label.hide()
        self.resume_button.hide()
        body = QHBoxLayout()
        left = QWidget()
        leftlay = layout_for(left, 0)
        self.camera_heading = label(tr("cameras"), "section")
        stage_header=QHBoxLayout();stage_header.addWidget(self.camera_heading,1)
        from .ai_activity_ui import icon
        self.detection_toggle=QPushButton();self.detection_toggle.setObjectName("iconButton");self.detection_toggle.setIcon(icon("eye"));self.detection_toggle.setCheckable(True);self.detection_toggle.setToolTip(tr("show_detections_tooltip"));self.detection_toggle.setAccessibleName(tr("show_detections_tooltip"));self.detection_toggle.setChecked(self.viewer_settings.detections)
        self.detection_toggle.toggled.connect(lambda checked:self.set_viewer_settings(detections=checked))
        self.top_header.insertWidget(self.top_header.count()-1,self.detection_toggle)
        self.camera_heading.hide()
        self.pause_label.hide()

        self.camera_stack = QStackedWidget()
        self.grid_widget = QWidget()
        self.grid_widget.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.detection_shortcut=QShortcut(QKeySequence("D"),self.grid_widget)
        self.detection_shortcut.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
        self.detection_shortcut.activated.connect(lambda:self.detection_toggle.toggle())
        self.grid = QGridLayout(self.grid_widget)
        self.grid.setContentsMargins(0, 0, 0, 0)
        self.grid.setSpacing(14)
        self.grid_scroll = QScrollArea()
        self.grid_scroll.setWidgetResizable(True)
        self.grid_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.grid_scroll.setWidget(self.grid_widget)
        self.camera_stack.addWidget(self.grid_scroll)
        self.expanded_tile = None
        self.message = card()
        ml = layout_for(self.message, 32)
        ml.addStretch()
        from .ai_activity_ui import icon
        self.message_icon=QLabel()
        self.message_icon.setPixmap(icon("camera").pixmap(56,56))
        ml.addWidget(self.message_icon)
        self.message_title = label(tr("loading"), "title")
        self.message_hint = label("", "muted")
        ml.addWidget(self.message_title)
        ml.addWidget(self.message_hint)
        self.message_action=QPushButton(tr("find_cameras_action"))
        self.message_action.setMaximumWidth(240)
        self.message_action.clicked.connect(self.stage_action)
        ml.addSpacing(16)
        ml.addWidget(self.message_action)
        self.message_action.hide()
        ml.addStretch()
        self.camera_stack.addWidget(self.message)
        leftlay.addWidget(self.camera_stack, 1)
        self.detection_hint=label("", "muted");self.detection_hint.hide();leftlay.addWidget(self.detection_hint)
        self.status_strip=label("","muted")

        body.addWidget(left, 2)
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
        self.log_panel=panel
        body.addWidget(panel, 1)
        from .ai_activity_ui import AiActivity
        self.ai_panel=AiActivity()
        body.addWidget(self.ai_panel,1)
        self.ai_panel.hide()
        overview_layout.addLayout(body, 1)
        overview_layout.addWidget(self.status_strip)
        self.tiles = []
        self.events = []
        self.render_activity()
        self.names = None
        self.reader = PreviewReader(Path(bc.LOG_DIR) / "preview")
        self.activity_feed = ActivityFeed(bc.LOG_DIR)
        self.pool = ThreadPoolExecutor(max_workers=1)
        self.future = None
        self.ai_data={}
        self.last_ai_poll=-10
        self.last_poll = 0
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.tick)
        self.timer.start(1000)
        self.camera_stack.setCurrentIndex(1)
        self.current_state = None
        from .camera_controls import CameraControls
        self.camera_controls = CameraControls(self.box_controls, TEXT["demo_names"][:self.args.cameras] if self.args.state not in ("empty","no-cameras","box-unreachable") else ())
        if self.args.demo and self.args.state=='off-camera' and self.camera_controls.records:
            from dataclasses import replace
            self.camera_controls.records[-1]=replace(self.camera_controls.records[-1],enabled=False)
        self.tick()
        self.details.setChecked(self.args.details)
        from .settings_ui import SettingsPage
        self.settings_page = SettingsPage(self.box_controls, self.settings_changed, self.viewer_settings, self.set_viewer_settings)
        settings_scroll=QScrollArea();settings_scroll.setWidgetResizable(True);settings_scroll.setWidget(self.settings_page.widget)
        self.content_stack.addWidget(settings_scroll)
        from .camera_ui import CameraPage
        self.cameras_page = CameraPage(self.camera_controls, self.settings_changed)
        self.content_stack.addWidget(self.cameras_page.widget)
        if getattr(self.args, "panel", None) == "cameras":
            self.open_cameras()
        if getattr(self.args, "panel", None) == "settings":
            self.open_settings()
        if self.remote_target:
            for button in (cameras_button,settings_button,self.run_button,self.resume_button):
                button.setEnabled(False);button.setToolTip("Change settings on the box")

    def fetch(self):
        from ..heartbeat import build_heartbeat

        settings = bc.load_box_settings()
        events, upload = self.activity_feed.read()
        from ..__main__ import _clip_dirs
        mode = bc.get_option("mode")
        payload = build_heartbeat(str(settings.get("site", "")), *_clip_dirs(mode), bc.ALIVE_FILE, mode=mode)
        # Read only names, never retain or display camera URLs.
        from .camera_presentation import active_names
        records=self.camera_controls.load()
        names = [c.name for c in records]
        return (
            State.from_heartbeat(payload, cameras=names, disabled=[c.name for c in records if not c.enabled], upload=upload),
            bool(bc.get_option("show_cameras")),
            events,
        )

    def tick(self):
        if self.remote_target:
            if not hasattr(self,"live_transport"):
                from .remote_live import RemoteLiveTransport
                self.live_transport=RemoteLiveTransport(self.remote_target,self)
                self.live_transport.frames.connect(self.receive_frames)
                self.live_transport.status.connect(self.receive_status)
                self.apply_state(State(mode="inference",cameras=[],site=self.remote_target.split('@')[-1]),True,[])
            self.update_demand()
            self.update_detector()
            return
        if self.box_unreachable:
            self.apply_state(State(site=tr("demo_house") if self.args.demo else "",mode="inference",error=True),True,[])
            self.update_detector()
            return
        if self.args.demo:
            scenario = self.args.state
            names = [c.name for c in self.camera_controls.records]
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
            state.disabled=[c.name for c in self.camera_controls.records if not c.enabled]
            if scenario == "empty":
                state.cameras = []
            demo_settings = self.box_controls.load_settings()
            state.mode = demo_settings.mode
            state.collecting = self.box_controls.phase() == "running"
            if not state.cameras:
                state.waiting=0;state.upload=''
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
                    else __import__("home_guard_project.box.app.demo_media",fromlist=["picture"]).picture(i,getattr(self.args,"aspect","16:9"))
                )
            self.update_detector()
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
        if not hasattr(self, "live_transport"):
            from .live_transport import LiveTransport
            self.live_transport = LiveTransport(self.reader.directory, self)
            self.live_transport.frames.connect(self.receive_frames)
            self.live_transport.status.connect(self.receive_status)
            self.content_stack.currentChanged.connect(lambda _: self.update_demand())
        self.update_demand()
        self.update_detector()

    def update_demand(self):
        if not hasattr(self, "live_transport"): return
        visible = (self.isVisible() and not self.isMinimized()
                   and self.content_stack.currentIndex() == 0
                   and getattr(self, "current_show", False)
                   and not self.box_controls.is_stopped())
        hero = getattr(self, "expanded_tile", None)
        self.live_transport.demand([t.name for t in self.tiles if not t.off],
                                   hero.name if hero else None, visible, self.viewer_settings.detections)

    def receive_frames(self, frames):
        for tile in self.tiles:
            if tile.name in frames and not tile.off:
                image, stamp, *overlay = frames[tile.name]
                if overlay and overlay[0] is not None:
                    tile.detections=overlay[0]
                    tile.tracked_stamp=self.ai_data.get("cameras",{}).get(tile.name,{}).get("ts")
                tile.update_picture(QPixmap.fromImage(image), stamp)

    def receive_status(self, packet):
        if "images" in packet: self.ai_panel.receive_images(packet["images"])
        overview={key:packet.get("overview",{}).get(key) for key in ("box","settings","cameras")}
        if "overview" in packet and overview!=getattr(self,"remote_overview",None):
            self.remote_overview=overview
            data=packet["overview"];box=data.get("box",{});settings=data.get("settings",{})
            cameras=data.get("cameras",{});active=cameras.get("active",[]);disabled=cameras.get("disabled",[])
            from .box_controls import Settings
            self.box_controls._settings=Settings.from_options(settings)
            self.box_controls._stopped=bool(box.get("stopped",False))
            self.apply_state(State(site=settings.get("site") or box.get("site") or self.remote_target.split('@')[-1],
                mode=settings.get("mode") or "inference",collecting=not self.box_controls._stopped,
                cameras=list(dict.fromkeys(active+disabled)),disabled=disabled),
                settings.get("show_cameras") is not False,[])
        if "status" in packet:
            self.ai_data = packet["status"]
            self.box_controls.live_status = self.ai_data
            if hasattr(self,"settings_page"): self.settings_page.check_applied()
        if "chat" in packet: self.chat_data = packet["chat"]
        self.update_detector()

    def changeEvent(self, event):
        super().changeEvent(event)
        if event.type() == QEvent.Type.WindowStateChange: self.update_demand()

    def showEvent(self, event):
        super().showEvent(event)
        self.update_demand()

    def hideEvent(self, event):
        self.update_demand()
        super().hideEvent(event)

    def update_detector(self):
        from .detector_view import camera_view
        now=time.time()
        stopped=self.box_controls.is_stopped()
        if self.args.demo:
            from .ai_demo import demo_status
            if (not stopped or not self.ai_data) and (not self.ai_data or now-getattr(self,"demo_status_at",0)>=1):
                self.demo_status_at=now
                self.ai_data=demo_status(self.names or [],now,self.args.state)
                if getattr(self,"demo_history_state",None)!=self.args.state:
                    self.demo_history_state=self.args.state
                    self.demo_decisions=self.ai_data['decisions']
                self.ai_data['decisions']=self.demo_decisions
            self.ai_data['settings']=self.box_controls.reported_status()['settings']
        self.update_alert_status()
        inference=self.current_state is not None and self.current_state.mode=="inference"
        from .ai_view import undelivered_alert
        failed=undelivered_alert(self.ai_data) if inference else None
        self.delivery_banner.setVisible(failed is not None)
        self.delivery_error.setText((failed.error or tr("ai_no_error")) if failed else "")
        if failed: self.header_hint.setText(tr("ai_delivery_banner"))
        self.ai_panel.setVisible(inference)
        self.log_panel.setVisible(not inference)
        if not hasattr(self,'chat_data'): self.chat_data=[]
        if self.args.demo:
            if not hasattr(self,'demo_media_dir'):
                import tempfile
                self.demo_media_dir=tempfile.TemporaryDirectory(prefix='homeguard-demo-')
                for i in range(3): demo_picture(i).save(str(Path(self.demo_media_dir.name)/('demo_'+str(i)+'.jpg')),'JPEG',88)
            from .demo_chat import demo_feed
            self.chat_data=demo_feed(self.demo_decisions if hasattr(self,'demo_decisions') else [])
            if self.args.state in ('paused','ai-conversation','ai-group'):
                from .demo_chat import conversation_feed
                self.chat_data=conversation_feed(self.demo_decisions)
                if self.args.state=='ai-group':
                    self.chat_data=self.chat_data[:2]
                    for message in self.chat_data: message['ts']-=100
            image_dir=Path(self.demo_media_dir.name)
        else:
            image_dir=None if self.remote_target else Path(bc.LOG_DIR)/'chat_images'
        if inference:
            until,some=self.alert_pause.status(self.current_state.cameras)
            self.ai_panel.render(self.ai_data,now,stopped,self.chat_data,image_dir,until,sum(not t.off for t in self.tiles),failed is not None)
            if self.box_unreachable: self.ai_panel.note.setText(tr('box_unreachable'))
            latest=next((d for d in reversed(self.ai_data.get('decisions',[])) if isinstance(d,dict) and d.get('camera') in (self.names or []) and d.get('camera') not in self.current_state.disabled),None)
            if latest and latest.get('ts')!=getattr(self,'hero_event_ts',None):
                self.hero_event_ts=latest.get('ts')
                self.expanded_tile=next(t for t in self.tiles if t.name==latest['camera']);self.arrange_tiles()
        for tile in self.tiles:
            tile.show_detections=self.viewer_settings.detections
            tile.detection_labels=self.viewer_settings.detection_labels
            tile.detector_enabled=inference
            tile.detector_note.hide()
            observations,text=camera_view(self.ai_data,tile.name,now,stopped or tile.off)
            if not observations or self.args.demo or not hasattr(tile,"tracked_stamp"):
                tile.detections=observations
            if tile.off: text="Off"
            tile.detector_note.setToolTip(text)
            if not tile.hero: text=tile.detector_note.fontMetrics().elidedText(text,Qt.TextElideMode.ElideRight,max(120,tile.width()-48))
            tile.detector_note.setText(text)
            from .detector_view import entry_opacity
            entries=self.ai_data.get("cameras",{})
            entry=entries.get(tile.name,{}) if isinstance(entries,dict) else {}
            tile.box_opacity=entry_opacity(entry,now) if tile.show_detections and isinstance(entry,dict) else 1
            tile.update()

    def apply_state(self, state, show, events):
        self.current_state, self.current_show = state, show
        stopped = self.box_controls.is_stopped()
        phase = self.box_controls.phase(state.collecting)
        if stopped:
            state.collecting = False
            self.start_requested = False
        if state.collecting:
            self.start_requested = False
        self.stop_banner.hide()
        self.run_button.show()
        self.header_hint.setText(tr("stopped") if stopped else tr("premium_protecting") if state.collecting and state.mode=="inference" else tr("premium_collecting") if state.collecting else tr("close_starting"))
        self.run_button.setText(tr("start_box") if stopped else tr("stop_box"))
        from .motion import busy
        busy(self.run_button,self.start_requested)
        self.run_button.setEnabled(not self.start_requested and not self.box_unreachable)
        if self.remote_target: self.run_button.setEnabled(False)
        self.run_button.setToolTip(tr("offline_controls") if self.box_unreachable else "")
        if self.box_unreachable: self.header_hint.setText(tr("box_unreachable"))
        elif not state.cameras and not stopped: self.header_hint.setText(tr('ready_for_cameras'))
        self.control_note.setText("" if stopped else tr("applying") if phase == "restarting" else tr("start_pending") if self.start_requested else "")
        self.run_button.setObjectName("primary" if stopped else "stopAction")
        self.run_button.setStyleSheet("")
        self.run_button.style().unpolish(self.run_button);self.run_button.style().polish(self.run_button)
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
        from .status_strip import status_text
        status=status_text(state.upload,state.waiting,state.disk,state.error)
        self.status_strip.setText(status)
        if hasattr(self,'settings_page'): self.settings_page.facts.setText(status)
        from .alert_hours import hours_description
        try: settings = self.box_controls.load_settings()
        except Exception:
            from .box_controls import Settings
            settings=Settings()
        self.update_alert_status()
        until, some = self.alert_pause.status(state.cameras) if state.mode == "inference" else (None,False)
        paused = tr("paused_some" if some else "paused_until", time=time.strftime("%H:%M",time.localtime(until))) if until else ""
        self.pause_label.setText(paused)
        self.pause_label.hide()
        if paused and not stopped and not self.box_unreachable: self.header_hint.setText(paused)
        self.resume_button.setVisible(bool(paused) and not stopped)
        if self.remote_target: self.resume_button.setEnabled(False)
        self.stats[0][1].setText(tr("close_stopped") if stopped else hours_description(settings.alert_start_hour,settings.alert_end_hour) if state.mode == "inference" else tr("collection"))
        if self.box_unreachable: self.stats[0][1].setText(tr('offline_controls'))
        self.camera_heading.setText(
            tr("cameras")
            + tr("separator")
            + tr("camera_count", count=len(state.cameras))
        )
        scenario = self.args.state if self.args.demo else ""
        if scenario == "loading":
            for value, _ in self.stats:
                value.setText(tr("loading"))
        from .availability import stage_state
        empty_state=stage_state(reachable=not self.box_unreachable,cameras=state.cameras,pictures=show)
        message = (
            empty_state if empty_state in ("box_unreachable","no_cameras") else
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
                "no_cameras": "no_cameras_hint",
                "box_unreachable": "box_unreachable_hint",
            }[message]
            self.message_hint.setText(tr(hint,address=self.last_box_address or tr("no_known_address")) if hint else "")
            self.message_action.setVisible(message in ("no_cameras","box_unreachable"))
            self.message_action.setEnabled(not self.remote_target)
            self.message_icon.setVisible(message in ("no_cameras","box_unreachable"))
            self.message_action.setText(tr("retry_setup_action" if message=="box_unreachable" else "find_cameras_action"))
            return
        self.camera_stack.setCurrentIndex(0)
        if self.names != state.cameras:
            self.names = list(state.cameras)
            while self.grid.count():
                self.grid.takeAt(0).widget().deleteLater()
            self.tiles = [CameraTile(name) for name in state.cameras]
            self.expanded_tile = None
            for tile in self.tiles:
                tile.off=tile.name in state.disabled
                tile.turn_on.clicked.connect(lambda checked=False,tile=tile:self.cameras_page.set_camera_enabled(tile.name,True))
                tile.clicked.connect(lambda tile=tile: self.toggle_tile(tile))
                tile.double_clicked.connect(lambda tile=tile: self.fullscreen_camera(tile))
            self.arrange_tiles()
        for tile in self.tiles:
            tile.off=tile.name in state.disabled
            tile.turn_on.setVisible(tile.off)
            tile.turn_on.setEnabled(not self.remote_target)
            tile.stopped = stopped
            tile.update_picture(tile.picture)
            if stopped: tile.update_picture(None)
        if self.expanded_tile and self.expanded_tile.off: self.expanded_tile=None;self.arrange_tiles()

    def resume_alerts(self):
        try:
            self.alert_pause.resume()
            self.tick()
            if self.current_state: self.apply_state(self.current_state,self.current_show,self.events)
        except Exception:
            self.control_note.setText(tr("control_error"))

    def set_viewer_settings(self,**changes):
        from dataclasses import replace
        self.viewer_settings=replace(self.viewer_settings,**changes)
        try:
            if not self.args.demo: self.viewer_preference.save(self.viewer_settings)
        except OSError: self.control_note.setText(tr("viewer_save_error"))
        self.detection_toggle.blockSignals(True);self.detection_toggle.setChecked(self.viewer_settings.detections);self.detection_toggle.blockSignals(False)
        if hasattr(self,"settings_page"): self.settings_page.sync_viewer(self.viewer_settings)
        if changes.get('detections') and self.box_controls.is_stopped() and not getattr(self,'detection_hint_seen',False):
            self.detection_hint_seen=True;self.detection_hint.setText('Detections appear when Home Guard is running.');self.detection_hint.show()
        self.update_detector()
        for tile in self.tiles:
            tile.show_detections=self.viewer_settings.detections;tile.detection_labels=self.viewer_settings.detection_labels;tile.update()
        self.update_demand()

    def stage_action(self):
        if self.box_unreachable:
            if self.args.demo:
                self.args.state='no-cameras';self.box_unreachable=False
            else: self.box_unreachable=not Path(bc.BOX_YAML).is_file()
            self.last_poll=0
            self.tick()
        else:
            self.open_cameras()
            self.cameras_page.show_search()

    def open_cameras(self):
        self.content_stack.setCurrentIndex(2)
        if self.box_unreachable:
            self.cameras_page.note.setText(tr("offline_controls"))
            self.cameras_page.refresh.setEnabled(False)
            self.cameras_page.search_button.setEnabled(False)
        else: self.cameras_page.open()

    def open_settings(self):
        self.settings_page.reload()
        self.settings_page.save.setEnabled(not self.box_unreachable)
        if self.box_unreachable: self.settings_page.note.setText(tr("offline_settings"))
        self.content_stack.setCurrentIndex(1)

    def settings_changed(self):
        self.last_poll = 0
        if self.current_state and self.camera_controls.records:
            self.current_state.cameras=[c.name for c in self.camera_controls.records]
            self.current_state.disabled=[c.name for c in self.camera_controls.records if not c.enabled]
        if self.current_state:
            self.apply_state(self.current_state, self.box_controls.load_settings().show_cameras, self.events)
        self.tick()

    def update_alert_status(self):
        if not self.current_state: return
        from .alert_types_ui import alert_summary
        from .alert_hours import hours_description
        from .box_controls import Settings
        try: settings = self.box_controls.load_settings()
        except Exception: settings = Settings()
        self.alert_status.setText(hours_description(settings.alert_start_hour, settings.alert_end_hour) + " · " + alert_summary(self.ai_data, settings.alert_on))
        self.alert_status.setVisible(self.current_state.mode == 'inference')

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
        if tile.off: return
        self.expanded_tile=tile
        self.arrange_tiles()

    def fullscreen_camera(self,tile):
        if tile is not self.expanded_tile: self.toggle_tile(tile);return
        dialog=QDialog(self);dialog.setStyleSheet(self.styleSheet())
        lay=layout_for(dialog,0)
        clone=CameraTile(tile.name);clone.hero=True;clone.picture=tile.picture;clone.detections=tile.detections;clone.box_opacity=tile.box_opacity;clone.show_detections=tile.show_detections;clone.detection_labels=tile.detection_labels
        clone.caption.setText(tile.caption.text());clone.detector_enabled=tile.detector_enabled;clone.detector_note.setText(tile.detector_note.text());clone.detector_note.setVisible(tile.detector_enabled)
        lay.addWidget(clone);clone.clicked.connect(dialog.close)
        def refresh():
            clone.stopped=tile.stopped;clone.detections=tile.detections;clone.box_opacity=tile.box_opacity;clone.show_detections=tile.show_detections;clone.detection_labels=tile.detection_labels;clone.detector_note.setText(tile.detector_note.text());clone.update_picture(tile.picture,getattr(tile,"frame_stamp",None))
        tile.frame_changed.connect(refresh)
        refresh()
        dialog.showFullScreen();dialog.exec()
        tile.frame_changed.disconnect(refresh)
        dialog.deleteLater()

    def keyPressEvent(self, event):
        if event.key()==Qt.Key.Key_F11:
            self.showNormal() if self.isFullScreen() else self.showFullScreen();event.accept()
        elif event.key()==Qt.Key.Key_Escape and self.isFullScreen(): self.showNormal();event.accept()
        else: super().keyPressEvent(event)

    def arrange_tiles(self):
        before=self.grid_widget.grab() if self.grid_widget.isVisible() else None
        while self.grid.count(): self.grid.takeAt(0)
        for tile in self.tiles: tile.setParent(self.grid_widget)
        old=getattr(self,'thumbnail_scroll',None)
        if old:
            old.setParent(None);old.deleteLater()
        if not self.tiles: return
        if self.expanded_tile not in self.tiles or self.expanded_tile.off: self.expanded_tile=next((t for t in self.tiles if not t.off),None)
        if self.expanded_tile is None:
            self.thumbnail_scroll=None
            for i,tile in enumerate(self.tiles):
                tile.hero=False;tile.setMinimumSize(180,144);tile.setMaximumSize(16777215,16777215);self.grid.addWidget(tile,i//2,i%2);tile.show()
            return
        hero=self.expanded_tile;hero.hero=True;hero.setMinimumSize(180,260);hero.setMaximumSize(16777215,16777215)
        self.grid.addWidget(hero,0,0);self.grid.setRowStretch(0,1);self.grid.setColumnStretch(0,1)
        others=[tile for tile in self.tiles if tile is not hero]
        if others:
            self.thumbnail_scroll=QScrollArea();self.thumbnail_scroll.setWidgetResizable(True);self.thumbnail_scroll.setFixedHeight(176);self.thumbnail_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
            holder=QWidget();strip=QGridLayout(holder);strip.setContentsMargins(0,0,0,0);strip.setSpacing(12)
            rows=1 if len(self.tiles)<=6 else 2
            columns=math.ceil(len(others)/rows)
            self.thumbnail_scroll.setFixedHeight(156*rows+12*(rows-1))
            for i,tile in enumerate(others):
                tile.hero=False;tile.setMinimumSize(80,144);tile.setMaximumSize(16777215,144);strip.addWidget(tile,i//columns,i%columns);strip.setColumnStretch(i%columns,1)
            self.thumbnail_scroll.setWidget(holder);self.grid.addWidget(self.thumbnail_scroll,1,0)
        else: self.thumbnail_scroll=None
        for tile in self.tiles: tile.show();tile.update()
        self.update_demand()
        if before is not None:
            self.grid.activate()
            from .motion import Transition
            self.rail_transition=Transition(self.grid_widget,before,self.grid_widget.grab(),self.grid_widget.rect())

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
        if hasattr(self, "live_transport"): self.live_transport.close()
        if hasattr(self, "demo_media_dir"): self.demo_media_dir.cleanup()
        if hasattr(self, "camera_retry"): self.camera_retry.clear()
        if hasattr(self,"saved_run_answers"):
            self.saved_run_answers.wifi_password="";self.saved_run_answers.camera_password="";self.saved_run_answers.box_password=""
        if hasattr(self, "discovery_timer"): self.discovery_timer.stop()
        if hasattr(self, "discovery_pool"): self.discovery_pool.shutdown(wait=False, cancel_futures=True)
        if hasattr(self, "cameras_page"):
            self.cameras_page.close()
        if hasattr(self,"wizard_cameras"):
            self.wizard_cameras.close()
        if hasattr(self, "pool"):
            self.timer.stop()
            self.pool.shutdown(wait=False, cancel_futures=True)
        if hasattr(self, "engine_backend"):
            self.engine_backend.cancel()
        if hasattr(self, "setup_pool"):
            if hasattr(self,"step_timer"): self.step_timer.stop()
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
        from .motion import AnimatedStack
        self.pages = AnimatedStack()
        self.outer.addWidget(self.pages, 1)
        from .camera_retry import CameraRetry
        self.camera_retry = CameraRetry()
        self.inputs = {}
        self.field_guidance = {}
        titles = [
            ("address_title", "address_hint"),
            ("network_title", "network_hint"),
            ("house_title", "house_hint"),
            ("owner_title", "owner_hint"),
            ("camera_title", "camera_hint"),
            ("progress_title", "progress_hint"),
            ("summary_title", "summary_hint"),
        ]
        self.page_layouts = []
        for title, hint in titles:
            panel = card()
            lay = layout_for(panel, 30)
            title_widget = label(tr(title), "title")
            if title=='summary_title': self.summary_title=title_widget
            if title == "progress_title":
                self.progress_title = title_widget
            lay.addWidget(title_widget)
            hint_widget = label(tr("live_summary_hint") if hint == "summary_hint" and not self.args.demo else tr(hint), "muted")
            lay.addWidget(hint_widget)
            if hint=="summary_hint": self.summary_hint=hint_widget
            if hint == "progress_hint": self.progress_hint = hint_widget
            self.page_layouts.append(lay)
            if title in ("summary_title", "owner_title"):
                summary_scroll = QScrollArea()
                summary_scroll.setWidgetResizable(True)
                summary_scroll.setWidget(panel)
                self.pages.addWidget(summary_scroll)
                if title == "owner_title": self.owner_scroll = summary_scroll
            else:
                self.pages.addWidget(panel)
        self.inputs["address"] = QLineEdit()
        self.inputs["address"].hide()
        self.field_guidance["address"] = label("")
        self.field_guidance["address"].setStyleSheet("color: " + ERROR)
        self.page_layouts[Page.ADDRESS].addWidget(label(tr("box_label")))
        self.box_picker = QComboBox()
        self.box_picker.addItem(tr("box_loading"), None)
        self.page_layouts[Page.ADDRESS].addWidget(self.box_picker)
        self.box_address = QLineEdit()
        self.box_address.setPlaceholderText(tr("box_address_placeholder"))
        self.page_layouts[Page.ADDRESS].addWidget(self.box_address)
        self.box_hint = label(tr("box_loading"), "muted")
        self.page_layouts[Page.ADDRESS].addWidget(self.box_hint)
        self.page_layouts[Page.ADDRESS].addWidget(label(tr("box_user_label")))
        self.box_user = QLineEdit()
        self.page_layouts[Page.ADDRESS].addWidget(self.box_user)
        self.page_layouts[Page.ADDRESS].addWidget(label(tr("box_user_help"), "muted"))
        self.add_input(Page.ADDRESS, "box_password", tr("box_password"), True)
        self.page_layouts[Page.ADDRESS].addWidget(label(tr("box_password_help"), "muted"))
        self.page_layouts[Page.ADDRESS].addWidget(self.field_guidance["address"])
        self.only_cameras = QPushButton(tr("only_check_cameras"))
        self.page_layouts[Page.ADDRESS].setSpacing(8)
        self.only_cameras.setMinimumHeight(44)
        self.only_cameras.setObjectName("secondary")
        self.only_cameras.clicked.connect(self.check_only)
        self.page_layouts[Page.ADDRESS].addWidget(self.only_cameras)
        self.box_picker.currentIndexChanged.connect(self.choose_box)
        self.box_address.textChanged.connect(self.compose_box_target)
        self.box_user.textChanged.connect(self.compose_box_target)
        self.network = QComboBox()
        self.network.addItems([tr("ethernet"), tr("wifi")])
        self.page_layouts[Page.NETWORK].addWidget(self.network)
        self.add_input(Page.NETWORK, "ssid", tr("ssid"))
        self.add_input(Page.NETWORK, "wifi_password", tr("wifi_password"), True)
        self.network.currentIndexChanged.connect(self.network_changed)
        self.network_changed(0)
        self.add_input(Page.HOUSE, "house", tr("house"))
        self.show_pictures = QCheckBox(tr("show"))
        self.alerts = QCheckBox(tr("alerts"))
        self.page_layouts[Page.HOUSE].addWidget(self.show_pictures)
        self.page_layouts[Page.HOUSE].addWidget(self.alerts)
        from .alert_hours import AlertHours
        self.wizard_hours = AlertHours()
        self.hour_start, self.hour_end = self.wizard_hours.start, self.wizard_hours.end
        self.page_layouts[Page.HOUSE].setSpacing(10)
        self.page_layouts[Page.HOUSE].addWidget(self.wizard_hours)
        self.page_layouts[Page.HOUSE].addWidget(label(tr("cooldown")))
        self.wizard_cooldown = QDoubleSpinBox()
        self.wizard_cooldown.setDecimals(2)
        self.wizard_cooldown.setRange(.17,1440)
        self.wizard_cooldown.setValue(2)
        self.wizard_cooldown.setSuffix(tr("minutes_suffix"))
        self.page_layouts[Page.HOUSE].addWidget(self.wizard_cooldown)
        self.alerts.toggled.connect(self.wizard_hours.setEnabled)
        self.wizard_hours.setEnabled(False)
        from .owner_consent import build_owner_page
        build_owner_page(self)
        self.find = QCheckBox(tr("find"))
        self.find.setChecked(True)
        self.page_layouts[Page.CAMERAS].addWidget(self.find)
        self.add_input(Page.CAMERAS, "camera_user", tr("camera_user"))
        self.add_input(Page.CAMERAS, "camera_password", tr("camera_password"), True)
        self.camera_retry_message = label("")
        self.camera_retry_message.setStyleSheet("color: " + ERROR)
        self.page_layouts[Page.CAMERAS].addWidget(self.camera_retry_message)
        self.page_layouts[Page.CAMERAS].addWidget(label(tr("camera_login_help"), "muted"))
        self.camera_lock_warning = label(tr("camera_lock_warning"))
        self.camera_lock_warning.setStyleSheet("color: " + WARNING)
        self.camera_lock_warning.hide()
        self.page_layouts[Page.CAMERAS].addWidget(self.camera_lock_warning)
        self.find.toggled.connect(
            lambda enabled: [
                self.inputs[key].setEnabled(enabled)
                for key in ("camera_user", "camera_password")
            ]
        )
        self.fail = QCheckBox(tr("failure"))
        self.fail.setChecked(bool(self.args.fail))
        self.fail.setVisible(self.args.demo)
        self.page_layouts[Page.CAMERAS].addWidget(self.fail)
        from .setup_details_ui import DetailsPanel
        from .setup_workspace import SetupWorkspace
        self.setup_details=DetailsPanel()
        self.setup_workspace=SetupWorkspace(self.setup_details)
        self.step_rows=[row.status for row in self.setup_workspace.rows]
        self.engine_step_names=[row.title for row in self.setup_workspace.rows]
        self.camera_search_note=next(row.note for row in self.setup_workspace.rows if row.step=="cameras")
        self.network_progress_note=next(row.note for row in self.setup_workspace.rows if row.step=="network")
        self.setup_details_toggle=self.setup_workspace.toggle
        self.page_layouts[Page.PROGRESS].setContentsMargins(12,12,12,12)
        self.page_layouts[Page.PROGRESS].setSpacing(8)
        self.page_layouts[Page.PROGRESS].addWidget(self.setup_workspace,1)
        self.setup_cancel = QPushButton(tr("cancel_setup"))
        self.setup_cancel.setObjectName("secondary")
        self.setup_cancel.clicked.connect(self.cancel_setup)
        self.page_layouts[Page.PROGRESS].addWidget(self.setup_cancel)
        self.failure_actions = QWidget()
        self.failure_actions.setObjectName('failureActions')
        self.failure_actions.setStyleSheet('QWidget#failureActions { background: transparent; }')
        actions = QHBoxLayout(self.failure_actions)
        actions.setContentsMargins(0,0,0,0)
        self.failure_retry = QPushButton(tr("search_again"))
        self.failure_retry.clicked.connect(self.replay_setup)
        actions.addWidget(self.failure_retry)
        self.failure_login = QPushButton(tr("change_camera_login"))
        self.failure_login.setObjectName("secondary")
        self.failure_login.clicked.connect(self.retry_setup)
        actions.addWidget(self.failure_login)
        self.failure_finish = QPushButton(tr("finish_without_cameras"))
        self.failure_finish.setObjectName("secondary")
        self.failure_finish.clicked.connect(self.finish_without_cameras)
        actions.addWidget(self.failure_finish)
        actions.addStretch()
        self.failure_actions.hide()
        self.failure_panel=QWidget()
        self.failure_panel.setObjectName('failurePanel')
        self.failure_panel.setStyleSheet('QWidget#failurePanel { background: transparent; }')
        failure_layout=layout_for(self.failure_panel,24)
        self.failure_layout=failure_layout
        failure_layout.setSpacing(24)
        self.failure_title=label('', 'headline')
        self.failure_explanation=label('')
        self.failure_explanation.setMaximumWidth(850)
        failure_layout.addWidget(self.failure_title)
        failure_layout.addWidget(self.failure_explanation)
        failure_layout.addWidget(self.failure_actions)
        failure_layout.addStretch()
        self.setup_workspace.panes.addWidget(self.failure_panel,2)
        self.setup_workspace.fallback=self.failure_panel
        self.failure_panel.hide()
        self.setup_details_toggle.setText(tr("show_readable_details"))
        self.summary_house = label("", "section")
        summary_header=QHBoxLayout()
        summary_header.addWidget(self.summary_house,1)
        self.summary_details_action=QPushButton(tr("show_readable_details"))
        self.summary_details_action.setObjectName("textAction")
        self.summary_details_action.setCheckable(True)
        self.summary_details_action.toggled.connect(self.toggle_summary_details)
        summary_header.addWidget(self.summary_details_action)
        self.page_layouts[Page.SUMMARY].addLayout(summary_header)
        self.summary_network=label("", "muted")
        self.summary_count=label("", "section")
        self.page_layouts[Page.SUMMARY].addWidget(self.summary_network)
        self.page_layouts[Page.SUMMARY].addWidget(self.summary_count)
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
        if self.args.demo:
            sr.addWidget(label(tr("rescue_demo"), "muted"))
        self.rescue_labels = {}
        for key in ("rescue_name", "rescue_password"):
            item = label(tr(key) + ": " + tr("service_pending"), "muted")
            self.rescue_labels[key] = item
            sr.addWidget(item)
        sr.addStretch()
        summary_columns.addWidget(summary_left, 3)
        summary_columns.addWidget(summary_right, 2)
        self.summary_overview=QWidget()
        self.summary_overview.setObjectName("summaryOverview")
        self.summary_overview.setLayout(summary_columns)
        self.page_layouts[Page.SUMMARY].addWidget(self.summary_overview)
        self.summary_details=DetailsPanel(self.setup_details.model)
        self.summary_workspace=SetupWorkspace(self.summary_details)
        self.summary_workspace.toggle.hide()
        self.summary_workspace.hide()
        self.page_layouts[Page.SUMMARY].addWidget(self.summary_workspace,1)
        self.page_layouts[Page.SUMMARY].setContentsMargins(12,12,12,12)
        self.page_layouts[Page.SUMMARY].setSpacing(8)
        for lay in self.page_layouts:
            lay.addStretch()
        review = card()
        review_layout = layout_for(review,30)
        review_layout.addWidget(label(tr("review_title"),"title"))
        review_layout.addWidget(label(tr("review_hint"),"muted"))
        self.review_text = label("")
        review_layout.addWidget(self.review_text)
        review_layout.addStretch()
        self.pages.addWidget(review)
        self.validation = label("", "error")
        self.outer.addWidget(self.validation)
        nav = QHBoxLayout()
        self.back = QPushButton(tr("back"))
        self.back.setObjectName("secondary")
        self.back.clicked.connect(
            lambda: self.set_page(Page.CAMERAS if self.pages.currentIndex()==Page.REVIEW else max(0, self.pages.currentIndex() - 1))
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
        if not self.args.demo:
            from .preferences import AddressPreference
            self.address_preference=AddressPreference()
            self.inputs["address"].setText(self.address_preference.load())
        saved_target = self.inputs["address"].text()
        if "@" in saved_target:
            saved_user, saved_box = saved_target.split("@", 1)
            if self.args.demo: self.box_user.setText(saved_user)
            else: self.box_user.setText(self.address_preference.load_user())
            self.box_address.setText(saved_box)
        if self.args.demo:
            from .box_discovery import Peer
            self.populate_boxes([Peer(tr("demo_box_name"), "100.100.100.10", True)])
        else:
            from .box_discovery import discover
            self.discovery_pool=ThreadPoolExecutor(max_workers=1)
            self.discovery_future=self.discovery_pool.submit(discover)
            self.discovery_timer=QTimer(self)
            self.discovery_timer.timeout.connect(self.poll_boxes)
            self.discovery_timer.start(100)
        self.find.setChecked(not self.args.skip_cameras)
        self.alerts.setChecked(self.args.alerts)
        if self.args.wifi:
            self.network.setCurrentIndex(1)
        self.set_page(Page.ADDRESS)
        if self.args.page:
            index = {
                "address": Page.ADDRESS,
                "network": Page.NETWORK,
                "house": Page.HOUSE,
                "owner-consent": Page.OWNER,
                "cameras": Page.CAMERAS,
                "progress": Page.PROGRESS,
                "summary": Page.SUMMARY,
                "failure": Page.PROGRESS,
                "validation": Page.ADDRESS,
                "review": Page.REVIEW,
                "camera-check": Page.CAMERA_CHECK,
            }[self.args.page]
            if index == Page.CAMERA_CHECK:
                self.run_answers=self.collect_answers()
                self.engine_cameras=[]
                self.open_camera_check()
            else:
                self.set_page(index)
            if self.args.page == "validation":
                self.inputs["address"].clear()
                self.next_page()
            if index in (Page.PROGRESS,Page.SUMMARY):
                if not self.args.demo:
                    return  # Render flags never launch a real setup process.
                self.begin_setup(instant=True)
                if self.args.page=='summary': self.set_page(Page.SUMMARY)

        if self.args.details:
            if self.pages.currentIndex()==Page.SUMMARY: self.summary_details_action.setChecked(True)
            else: self.setup_details_toggle.setChecked(True)
        if getattr(self.args,"technical_log",False):
            (self.summary_details if self.pages.currentIndex()==Page.SUMMARY else self.setup_details).technical.setChecked(True)

    def add_input(self, page, key, caption, secret=False):
        caption_widget = label(caption)
        self.page_layouts[page].addWidget(caption_widget)
        field = QLineEdit()
        caption_widget.setBuddy(field)
        field.setAccessibleName(caption.replace('&', ''))
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

    def toggle_summary_details(self,opened):
        from .ai_activity_ui import icon
        self.summary_details_action.setText(tr("hide_details" if opened else "show_readable_details"))
        self.summary_details_action.setIcon(icon("chevron-up" if opened else "chevron-down"))
        self.summary_workspace.setVisible(opened)
        self.summary_workspace.toggle.setChecked(opened)
        self.summary_overview.setVisible(not opened)
        self.summary_hint.setVisible(not opened)
        self.summary_title.setVisible(not opened)
        self.summary_house.setText(tr('summary_details_heading' if opened else 'summary_house',house=self.inputs['house'].text()))
        self.summary_network.setVisible(not opened)
        self.summary_count.setVisible(not opened)
        if opened:
            self.summary_details.reset(self.setup_details.model)
            self.summary_workspace.refresh()

    def set_page(self, index):
        self.pages.setCurrentIndex(index)
        if index == Page.SUMMARY:
            self.summary_house.setText(tr("summary_house", house=self.inputs["house"].text()))
            from .availability import network_description
            self.summary_network.setText(tr("summary_network",network=network_description(getattr(self,"run_answers",self.collect_answers()))))
            count=len(getattr(self,"engine_cameras",[]))
            self.summary_count.setText(tr("summary_camera_count",count=count) if count else tr("zero_cameras_summary"))
            self.later.hide()
        self.validation.setText("")
        self.validation.setVisible(index != Page.PROGRESS)
        self.update_step_bar(index)
        if index == Page.REVIEW:
            self.populate_review()
        if index == Page.OWNER:
            self.inputs['owner_name'].setFocus()
        self.back.setVisible(Page.ADDRESS < index < Page.PROGRESS or index == Page.REVIEW)
        self.next.setVisible(index != Page.PROGRESS)
        self.next.setText(
            tr("camera_check_continue") if index == Page.CAMERA_CHECK else tr("start_setup") if index == Page.REVIEW else tr("open_home_guard") if index == Page.SUMMARY else tr("next")
        )

    def update_step_bar(self,index):
        current = Page.CAMERAS if index == Page.REVIEW else Page.PROGRESS if index == Page.CAMERA_CHECK else min(index,Page.PROGRESS)
        for i,(item,name) in enumerate(zip(self.step_labels,TEXT["step_names"])):
            done = i < current or index == Page.SUMMARY
            self.step_icons[i].setVisible(done)
            item.setText(name if done else tr("step_number",number=i+1,name=name))
            item.setStyleSheet(f"color: {OK if done else '#edf1f7' if i == current else MUTED}; padding: 8px; border-bottom: 2px solid {'#edf1f7' if i == current and index != Page.SUMMARY else 'transparent'};")

    def errors_for_page(self,index):
        from .guidance import field_errors
        return field_errors(index,{key:w.text() for key,w in self.inputs.items()},wifi=self.network.currentIndex()==1,find=self.find.isChecked())

    def valid_page(self,index):
        return not self.errors_for_page(index)

    def compose_box_target(self):
        self.inputs["address"].setText(self.box_user.text().strip()+"@"+self.box_address.text().strip())

    def choose_box(self, index):
        address=self.box_picker.itemData(index)
        self.box_address.setVisible(not bool(address))
        if address: self.box_address.setText(address)

    def populate_boxes(self, peers):
        remembered=self.box_address.text()
        self.box_picker.blockSignals(True)
        self.box_picker.clear()
        for peer in peers:
            self.box_picker.addItem(tr("box_peer", name=peer.name, address=peer.address)+("" if peer.online else tr("box_offline")), peer.address)
        self.box_picker.addItem(tr("box_manual"), None)
        selection=next((i for i in range(self.box_picker.count()) if self.box_picker.itemData(i)==remembered), self.box_picker.count()-1 if remembered else 0)
        self.box_picker.setCurrentIndex(selection)
        self.box_picker.blockSignals(False)
        self.box_picker.setVisible(bool(peers))
        self.box_hint.setText(tr("box_pick_hint") if peers else tr("box_address_help"))
        self.choose_box(selection)
        self.compose_box_target()

    def poll_boxes(self):
        if self.discovery_future.done():
            self.discovery_timer.stop()
            try: peers=self.discovery_future.result()
            except Exception: peers=[]
            self.populate_boxes(peers)

    def next_page(self):
        index = self.pages.currentIndex()
        if index == Page.SUMMARY:
            from types import SimpleNamespace
            values=vars(self.args).copy()
            values.update(setup=False,page=None,panel=None,state="no-cameras" if not getattr(self,"engine_cameras",[]) else "inference")
            self.main_window=Window(SimpleNamespace(**values))
            self.main_window.show()
            self.close()
            return
        if index == Page.CAMERA_CHECK:
            if self.wizard_cameras.check_state.photo_failed:
                self.skip_camera_check()
                return
            if not self.wizard_cameras.loaded or self.wizard_cameras.future is not None:
                self.validation.setText(tr("camera_check_pending"))
                return
            if self.wizard_cameras.rows and not self.wizard_cameras.saved:
                self.validation.setText(tr("camera_check_unsaved"))
                return
            if getattr(self,"camera_check_only",False): self.close()
            else: self.set_page(Page.SUMMARY)
            return
        if index == Page.REVIEW:
            self.begin_setup()
            return
        if index == Page.ADDRESS: self.compose_box_target()
        errors = self.errors_for_page(index)
        if errors:
            for key,message in errors.items():
                self.inputs[key].setStyleSheet(f"border: 1px solid {ERROR};")
                self.field_guidance[key].setText(tr("box_target_error") if index == Page.ADDRESS and key == "address" else tr(message))
                self.field_guidance[key].show()
            (self.box_user if not self.box_user.text().strip() else self.box_address).setFocus() if index == Page.ADDRESS else self.inputs[next(iter(errors))].setFocus()
            return
        if index == Page.ADDRESS and not self.args.demo:
            try: self.address_preference.save(self.inputs["address"].text().strip())
            except OSError:
                self.set_page(Page.NETWORK)
                self.validation.setText(tr("address_not_remembered"))
                return
        if index == Page.OWNER and not self.args.demo:
            try: self.installer_preference.save(self.inputs['installer'].text().strip())
            except OSError:
                self.set_page(Page.CAMERAS)
                self.validation.setText(tr('installer_not_remembered'))
                return
        if index == Page.CAMERAS:
            if self.camera_retry.pending:
                self.begin_setup()
            else:
                self.set_page(Page.REVIEW)
        else:
            self.set_page(index + 1)

    def collect_answers(self):
        from .box_controls import minutes_to_seconds
        return Answers(**{key:w.text() if key.endswith("password") else w.text().strip() for key,w in self.inputs.items()},
            **{'consent_' + key: toggle.isChecked() for key, toggle in self.consent_toggles.items()},
            network="wifi" if self.network.currentIndex() else "ethernet",
            show_cameras=self.show_pictures.isChecked(), alerts=self.alerts.isChecked(),
            start_hour=self.wizard_hours.values()[0], end_hour=self.wizard_hours.values()[1],
            cooldown_sec=minutes_to_seconds(self.wizard_cooldown.value()), find_cameras=self.find.isChecked())

    def populate_review(self):
        from .alert_hours import hours_description
        a=self.collect_answers()
        self.review_text.setText(tr("review_text",target=a.address,network=tr("wifi")+" - "+a.ssid if a.network=="wifi" else tr("ethernet"),house=a.house,alerts=tr("alerts_on_review",hours=hours_description(a.start_hour,a.end_hour),minutes=a.cooldown_sec/60) if a.alerts else tr("alerts_off_review"),cameras=tr("cameras_find_review") if a.find_cameras else tr("cameras_later_review")))

    def begin_setup(self,instant=False,answers_override=None):
        import queue
        from .engine_backend import EngineBackend,DemoEngine
        self.camera_check_only=False
        self.setup_completed=False
        self.progress_title.setText(tr("progress_title"))
        self.progress_hint.setStyleSheet("")
        self.run_answers = self.camera_retry.retry(self.inputs["camera_user"].text().strip(), self.inputs["camera_password"].text()) if self.camera_retry.pending else self.collect_answers()
        from dataclasses import replace
        if answers_override is not None: self.run_answers=replace(answers_override)
        self.saved_run_answers=replace(self.run_answers)
        from .setup_failure import FailureFacts
        self.failure_facts=FailureFacts(network=self.run_answers.ssid if self.run_answers.network=="wifi" else "")
        self.failure_actions.hide()
        self.failure_panel.hide()
        self.progress_title.show();self.progress_hint.hide()
        self.setup_details.setMaximumHeight(16777215)
        self.setup_cancel.show()
        self.setup_details_toggle.setChecked(False)
        self.camera_retry.capture(self.run_answers)
        self.progress_hint.setText(tr("retry_confirming") if self.camera_retry.pending else tr("progress_hint"))
        self.engine_backend=DemoEngine(self.args.fail or self.fail.isChecked() or self.args.page=="failure") if self.args.demo else EngineBackend()
        self.engine_events=queue.Queue()
        self.engine_checks=[];self.engine_cameras=[];self.engine_failed_step=None;self.running_step=None
        from .search_progress import CameraSearch
        self.camera_search=CameraSearch()
        self.camera_search_note.hide()
        self.network_progress_note.hide()
        self.setup_details.reset()
        self.setup_workspace.refresh()
        for row in self.step_rows:
            row.setText(tr("pending"));row.setStyleSheet("")
        for name in self.engine_step_names: name.setStyleSheet('color: '+MUTED)
        for item in self.check_labels: item.clear()
        self.setup_cancel.setEnabled(True)
        self.set_page(Page.PROGRESS)
        if not hasattr(self,"setup_pool"): self.setup_pool=ThreadPoolExecutor(max_workers=1)
        if instant:
            events=[]
            self.engine_backend.run(self.run_answers,events.append,instant=True)
            for event in events:
                self.engine_events.put(event)
                if self.args.page=="progress" and event.kind=="detail" and event.step=="update": break
            self.present_engine_events()
            if self.args.page!="progress": self.finish_engine(not self.engine_failed_step)
            return
        self.setup_future=self.setup_pool.submit(self.engine_backend.run,self.run_answers,self.engine_events.put)
        self.step_timer=QTimer(self)
        self.step_timer.timeout.connect(self.poll_engine)
        self.step_timer.start(100)

    def cancel_setup(self):
        self.setup_cancel.setEnabled(False)
        self.engine_backend.cancel()

    def present_engine_events(self):
        import queue
        from .engine_backend import ENGINE_STEPS, is_progress_warning
        while True:
            try: event=self.engine_events.get_nowait()
            except queue.Empty: break
            self.failure_facts.feed(event)
            self.setup_details.feed(event)
            if event.kind == "detail": self.camera_search.feed(event.text)
            if event.kind=="step":
                index=ENGINE_STEPS.index(event.step)
                row=self.step_rows[index]
                if event.status=="start":
                    if event.step == "cameras":
                        self.camera_retry.started()
                        self.camera_search.start(time.monotonic())
                        self.camera_search_note.show()
                    self.running_step=index;self.running_since=time.monotonic()
                    row.setText(tr("setup_elapsed",seconds=0))
                    row.setStyleSheet("")
                    self.engine_step_names[index].setStyleSheet('color: '+ACTION)
                elif is_progress_warning(event):
                    self.running_step=index
                    if not hasattr(self,'running_since'): self.running_since=time.monotonic()
                    self.network_progress_note.setText(event.text)
                    self.network_progress_note.show()
                else:
                    self.running_step=None
                    if event.step=='network': self.network_progress_note.hide()
                    if event.step == "cameras":
                        self.camera_search.finish()
                        self.camera_search_note.hide()
                    status={"ok":"PASS","warn":"WARN","fail":"FAIL","skip":"step_skip"}[event.status]
                    row.setText(tr(status)+tr("separator")+event.text)
                    row.setStyleSheet("color: "+(ERROR if event.status=="fail" else WARNING if event.status in ("warn","skip") else OK))
                    self.engine_step_names[index].setStyleSheet('color: '+(ERROR if event.status=='fail' else WARNING if event.status in ('warn','skip') else OK))
                    if event.status=="fail":
                        self.engine_failed_step=event.step
                        if event.step == "cameras":
                            self.camera_retry.failed(event.text)
                            self.progress_hint.setText(event.text + "\n" + tr("camera_login_help"))
                            self.progress_hint.setStyleSheet("color: " + ERROR)
                            row.setText(tr("FAIL"))
                        self.setup_details_toggle.setChecked(False)
            elif event.kind=="check": self.engine_checks.append(event)
            elif event.kind=="camera": self.engine_cameras.append(event.name)
            elif event.kind=="rescue":
                self.rescue_labels["rescue_name"].setText(tr("rescue_name")+": "+event.name)
                self.rescue_labels["rescue_password"].setText(tr("rescue_password")+": "+event.password)

        self.setup_workspace.refresh(int(time.monotonic()-getattr(self,"running_since",time.monotonic())))
        if self.running_step is not None:
            if self.camera_search.running:
                elapsed=self.camera_search.elapsed(time.monotonic())
                self.step_rows[self.running_step].setText(tr("camera_search_elapsed",minutes=elapsed//60,seconds=elapsed%60))
                self.camera_search_note.setText(self.camera_search.sentence)
                self.camera_search_note.setStyleSheet("color: " + (ERROR if self.camera_search.role == "error" else WARNING if self.camera_search.role == "warning" else MUTED))
            else:
                self.step_rows[self.running_step].setText(tr("setup_elapsed",seconds=int(time.monotonic()-self.running_since)))

        self.setup_workspace.refresh(int(time.monotonic()-getattr(self,"running_since",time.monotonic())))

    def poll_engine(self):
        self.present_engine_events()
        if self.setup_future.done():
            self.present_engine_events()
            self.step_timer.stop()
            try: success=self.setup_future.result()
            except Exception: success=False
            self.finish_engine(success)

    def finish_engine(self,success):
        self.setup_cancel.setEnabled(False)
        for key in ("wifi_password","camera_password","box_password"): self.inputs[key].clear()
        self.run_answers.wifi_password="";self.run_answers.camera_password="";self.run_answers.box_password=""
        if not success:
            from .engine_backend import OWNERS
            self.update_step_bar(OWNERS.get(self.engine_failed_step,0))
            title,explanation=self.failure_facts.content(self.engine_failed_step)
            self.progress_title.setObjectName("headline")
            self.progress_title.style().unpolish(self.progress_title);self.progress_title.style().polish(self.progress_title)
            self.progress_title.setText(title)
            self.progress_hint.setObjectName("")
            self.progress_hint.setStyleSheet("")
            self.progress_hint.setText(explanation)
            self.failure_title.setText(title)
            self.failure_explanation.setText(explanation)
            self.progress_title.hide();self.progress_hint.hide()
            self.failure_panel.show()
            self.setup_cancel.hide()
            self.failure_actions.show()
            cameras=self.engine_failed_step=="cameras"
            self.failure_retry.setText(tr("search_again" if cameras else "retry_setup_action"))
            self.failure_login.setVisible(cameras)
            self.failure_finish.setVisible(cameras)
            self.failure_retry.setFocus()
            self.setup_details.setMaximumHeight(16777215)
            self.setup_details_toggle.setChecked(False)
            from .engine_backend import ENGINE_STEPS
            for step,row in zip(ENGINE_STEPS,self.step_rows):
                group=self.setup_details.model.groups[step]
                row.setText(tr("FAIL") if group.status=="fail" else tr("PASS") if group.status=="ok" else tr("pending") if group.status in ("pending","start") else group.status.upper())
            if self.engine_backend.cancelled.is_set():
                self.validation.setText(tr("setup_cancelled"));self.validation.show()
            self.next.setText(tr("retry"));self.next.hide()
            self.next.clicked.disconnect();self.next.clicked.connect(self.retry_setup)
            self.setup_workspace.refresh()
            return
        for i,event in enumerate(self.engine_checks):
            if i>=len(self.check_labels):
                item=label("");self.check_labels.append(item)
                self.check_labels[0].parentWidget().layout().insertWidget(i+1,item)
            item=self.check_labels[i];item.setText(event.status+tr("separator")+event.text)
            item.setStyleSheet("color: "+(ERROR if event.status=="FAIL" else WARNING if event.status=="WARN" else OK))
        self.camera_retry.clear()
        if not self.args.demo:
            try: self.address_preference.save_success(self.run_answers.address)
            except OSError: pass
        self.later.setVisible(not self.run_answers.find_cameras)
        self.setup_success()

    def setup_success(self):
        self.setup_completed=True
        if hasattr(self,"saved_run_answers"):
            self.saved_run_answers.wifi_password="";self.saved_run_answers.camera_password="";self.saved_run_answers.box_password=""
        if not self.engine_cameras: self.set_page(Page.SUMMARY)
        else: self.open_camera_check()

    def check_only(self):
        self.compose_box_target()
        if self.errors_for_page(0):
            self.field_guidance["address"].setText(tr("box_target_error"))
            self.field_guidance["address"].show()
            return
        self.camera_check_only=True
        self.setup_completed=False
        self.run_answers=Answers(address=self.inputs["address"].text())
        self.engine_cameras=[]
        self.open_camera_check()

    def skip_camera_check(self):
        self.wizard_cameras.close()
        if getattr(self,"camera_check_only",False): self.close()
        else: self.set_page(Page.SUMMARY)

    def open_camera_check(self):
        from .camera_ui import CameraPage
        if hasattr(self,"wizard_cameras"):
            self.wizard_cameras.close()
            old=self.pages.widget(Page.CAMERA_CHECK)
            self.pages.removeWidget(old)
            old.deleteLater()
        if self.args.demo:
            from .camera_controls import CameraControls
            from .box_controls import BoxControls
            controls=CameraControls(BoxControls(demo=True),["front_door","garden","driveway"] if self.run_answers.find_cameras else ())
        else:
            from .remote_cameras import RemoteCameras
            controls=RemoteCameras(self.run_answers.address,self.engine_cameras)
        from .camera_check_state import CameraCheckState
        state=CameraCheckState(setup_finished=getattr(self,"setup_completed",False),house=self.run_answers.house,cameras=tuple(self.engine_cameras))
        self.wizard_cameras=CameraPage(controls,lambda:self.validation.clear(),wizard=True,check_state=state,skip=self.skip_camera_check)
        self.pages.addWidget(self.wizard_cameras.widget)
        self.set_page(Page.CAMERA_CHECK)
        self.wizard_cameras.open()

    def replay_setup(self):
        self.next.clicked.disconnect();self.next.clicked.connect(self.next_page)
        self.begin_setup(answers_override=self.saved_run_answers)
        self.progress_hint.setText(tr("retry_confirming"))

    def finish_without_cameras(self):
        from dataclasses import replace
        answers=replace(self.saved_run_answers,find_cameras=False,camera_password='')
        self.camera_retry.clear()
        self.next.clicked.disconnect();self.next.clicked.connect(self.next_page)
        self.begin_setup(answers_override=answers)
        self.progress_hint.setText(tr('finishing_without_cameras'))

    def retry_setup(self):
        from .engine_backend import OWNERS
        self.next.clicked.disconnect();self.next.clicked.connect(self.next_page)
        self.set_page(OWNERS.get(self.engine_failed_step,0))
        if self.engine_failed_step == "cameras":
            self.camera_retry_message.setText(self.camera_retry.message)
            self.camera_lock_warning.setVisible(self.camera_retry.lock_warning)
            self.inputs["camera_password"].clear()
            self.inputs["camera_password"].setFocus()
