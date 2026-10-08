"""Setup's "the map of each camera" step: the scene map editor, camera by camera, after the camera check.

It never blocks finishing setup: every camera can be skipped ("Skip for now"), and "Finish without maps" leaves
at once. The window keeps one small hook (``Window.open_scene_step``); everything else lives here.
"""
from PySide6.QtCore import Qt, QRectF, Signal
from PySide6.QtGui import QColor, QPainter
from PySide6.QtWidgets import QFrame, QHBoxLayout, QVBoxLayout, QWidget

from .scene_editor import SceneMapEditor, TextAction, drive, words
from .scene_strings import language, st
from .zone_editor import alpha, colors

SAVED, SKIPPED = "saved", "skipped"


class CameraDots(QWidget):
    """One dot per camera: saved (filled), skipped (hollow), current (ringed)."""
    def __init__(self, count, parent=None):
        super().__init__(parent)
        self.states = [None] * count
        self.current = 0
        self.setFixedSize(max(1, count) * 18, 18)

    def paintEvent(self, event):
        t = colors(self)
        p = QPainter(self); p.setRenderHint(QPainter.RenderHint.Antialiasing)
        rtl = self.layoutDirection() == Qt.LayoutDirection.RightToLeft
        for i, state in enumerate(self.states):
            x = (len(self.states) - 1 - i) * 18 if rtl else i * 18
            box = QRectF(x + 3, 3, 12, 12)
            if i == self.current:
                p.setPen(QColor(t['action'])); p.setBrush(alpha(t['action'], .3)); p.drawEllipse(box)
            elif state == SAVED:
                p.setPen(Qt.PenStyle.NoPen); p.setBrush(QColor(t['ok'])); p.drawEllipse(box)
            else:
                p.setPen(QColor(t['muted'] if state == SKIPPED else t['border'])); p.setBrush(Qt.BrushStyle.NoBrush)
                p.drawEllipse(box.adjusted(1, 1, -1, -1))


class SceneSetupStep(QFrame):
    """``finished`` carries {camera: saved | skipped} (cameras never reached are missing)."""
    finished = Signal(dict)

    def __init__(self, backend, cameras, lang=None, pictures=None, parent=None):
        super().__init__(parent)
        self.backend = backend
        self.cameras = list(cameras)
        self.pictures = dict(pictures or {})
        self.lang = lang or language()
        self.states = {}
        self.index = 0
        self.editor = None
        self.setObjectName('card')
        self.setLayoutDirection(Qt.LayoutDirection.RightToLeft if self.lang == 'he' else Qt.LayoutDirection.LeftToRight)
        t = colors(self)
        self.setStyleSheet(f'QLabel#scene_heading {{ color: {t["text"]}; font-size: 15pt; font-weight: 600; }}'
                           f'QLabel#scene_body {{ color: {t["secondary"]}; font-size: 10.5pt; }}'
                           f'QLabel#scene_progress {{ color: {t["action"]}; font-size: 10.5pt; font-weight: 600; }}')
        root = QVBoxLayout(self); root.setContentsMargins(26, 16, 26, 18); root.setSpacing(10)
        head = QHBoxLayout(); head.setSpacing(24)
        titles = QVBoxLayout(); titles.setSpacing(2)
        top = QHBoxLayout(); top.setSpacing(14)
        top.addWidget(words(st('setup_title', self.lang), 'heading'))
        self.progress = words('', 'progress'); top.addWidget(self.progress); top.addStretch(1)
        titles.addLayout(top)
        titles.addWidget(words(st('setup_hint', self.lang), 'body'))
        head.addLayout(titles, 1)
        self.dots = CameraDots(len(self.cameras)); head.addWidget(self.dots, 0, Qt.AlignmentFlag.AlignVCenter)
        self.skip_all = TextAction(st('setup_skip_all', self.lang)); head.addWidget(self.skip_all, 0, Qt.AlignmentFlag.AlignVCenter)
        self.skip_all.clicked.connect(self.skip_everything)
        root.addLayout(head)
        self.body = QVBoxLayout(); self.body.setContentsMargins(0, 0, 0, 0)
        root.addLayout(self.body, 1)
        if self.cameras:
            self.show_camera(0)

    def show_camera(self, index, start=True):
        if self.editor is not None:
            # Often called from inside the old editor's own button: it is deleted later, never under its feet.
            old, self.editor = self.editor, None
            old.close_jobs(); old.hide(); self.body.removeWidget(old); old.deleteLater()
        self.index = index
        camera = self.cameras[index]
        self.editor = SceneMapEditor(self.backend, camera, self.lang, setup=(index + 1, len(self.cameras)),
                                     placeholder=self.pictures.get(camera))
        self.progress.setText(st('setup_eyebrow', self.lang, number=index + 1, total=len(self.cameras)))
        self.editor.finished.connect(self.camera_finished)
        self.body.addWidget(self.editor)
        self.dots.states = [self.states.get(c) for c in self.cameras]; self.dots.current = index; self.dots.update()
        if start:
            self.editor.start()

    def camera_finished(self, how, value):
        camera = self.cameras[self.index]
        if how == "back":
            if self.index > 0:
                self.show_camera(self.index - 1)
            return
        self.states[camera] = SAVED if how == "saved" else self.states.get(camera, SKIPPED)
        self.advance()

    def advance(self):
        if self.index + 1 < len(self.cameras):
            self.show_camera(self.index + 1)
        else:
            self.leave()

    def skip_everything(self):
        if self.editor is not None and self.editor.busy():
            return                          # the box is saving this camera's map
        for camera in self.cameras:
            self.states.setdefault(camera, SKIPPED)
        self.leave()

    def leave(self):
        if self.editor is not None:
            self.editor.close_jobs()
        self.finished.emit(dict(self.states))


def setup_cameras(window):
    """The cameras the check left switched on (their ids; the editor shows their names), with the check's photos
    (shown while the box numbers each picture)."""
    from PySide6.QtGui import QPixmap
    controls = window.wizard_cameras.controls
    records = [c for c in getattr(controls, 'records', []) if c.enabled]
    pictures = {}
    for i, camera in enumerate(records):
        if window.args.demo:
            from .ui import demo_picture
            pictures[camera.name] = demo_picture(i)
        elif camera.ok and camera.file:
            pictures[camera.name] = QPixmap(camera.file)
    return [c.name for c in records], pictures


def open_scene_step(window, demo_state=None):
    """Show the step after the camera check; with no cameras, go straight on to the summary."""
    from .setup_pages import Page
    cameras, pictures = setup_cameras(window)
    if not cameras:
        window.set_page(Page.SUMMARY)
        return None
    if window.args.demo:
        from .scene_backend import DemoSceneBackend
        backend = DemoSceneBackend()
    else:
        from .scene_backend import SceneBackend
        backend = SceneBackend(window.run_answers.address)
    old = getattr(window, 'scene_step', None)
    if old is not None:
        window.pages.removeWidget(old); old.deleteLater()
    step = SceneSetupStep(backend, cameras, pictures=pictures)
    if demo_state:
        step.show_camera(min(1, len(cameras) - 1), start=False)
        step.states[cameras[0]] = SAVED; step.dots.states[0] = SAVED; step.dots.update()
        drive(step.editor, demo_state)
    window.scene_step = step
    window.pages.addWidget(step)
    window.pages.setCurrentWidget(step)
    window.validation.setText('')
    window.back.hide(); window.next.hide()
    window.update_step_bar(Page.CAMERA_CHECK)

    def leave(states):
        window.scene_states = states
        window.set_page(Page.SUMMARY)
        window.pages.removeWidget(step); step.deleteLater()
        window.scene_step = None
    step.finished.connect(leave)
    return step
