from concurrent.futures import ThreadPoolExecutor
from PySide6.QtCore import QTimer, QRegularExpression, Qt
from PySide6.QtGui import QPixmap, QRegularExpressionValidator
from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QScrollArea, QPushButton, QLineEdit, QCheckBox, QLabel, QProgressBar, QTextEdit
from .strings import tr
from .theme import OK, ERROR, MUTED
from .camera_controls import changes_payload

class CameraPage:
    def __init__(self, controls, changed, wizard=False, check_state=None, skip=None):
        from .ui import card, label, layout_for
        self.controls, self.changed = controls, changed
        from .camera_check_state import CameraCheckState
        self.check_state=check_state or CameraCheckState()
        self.wizard = wizard
        self.saved = False
        self.widget = QWidget()
        outer = layout_for(self.widget, 0)
        if wizard:
            self.completion = label(self.check_state.heading(), "section")
            self.completion.setStyleSheet("color: " + (OK if self.check_state.setup_finished else MUTED))
            outer.addWidget(self.completion)
        heading = QHBoxLayout()
        heading.addWidget(label(tr("check_cameras_title" if wizard else "cameras_title"), "section"), 1)
        self.refresh = QPushButton(tr("refresh_photos"))
        self.refresh.clicked.connect(self.refresh_clicked)
        heading.addWidget(self.refresh)
        outer.addLayout(heading)
        outer.addWidget(label(tr("check_cameras_hint" if wizard else "cameras_hint"), "muted"))
        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.hide()
        outer.addWidget(self.progress)
        self.details_toggle=QCheckBox(tr("details"))
        self.details_toggle.hide()
        outer.addWidget(self.details_toggle)
        self.error_details=QTextEdit()
        self.error_details.setReadOnly(True)
        self.error_details.setMaximumHeight(145)
        self.error_details.setStyleSheet("font-family: Consolas; font-size: 12px;")
        self.error_details.hide()
        self.details_toggle.toggled.connect(self.error_details.setVisible)
        outer.addWidget(self.error_details)
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)

        outer.addWidget(self.scroll, 1)
        bottom = QHBoxLayout()
        self.note = label("", "muted")
        bottom.addWidget(self.note, 1)
        self.save = QPushButton(tr("save_cameras"))
        self.save.setEnabled(False)
        self.save.clicked.connect(self.save_clicked)
        bottom.addWidget(self.save)
        if wizard and skip:
            self.skip=QPushButton(tr("skip_camera_check"))
            self.skip.setObjectName("secondary")
            self.skip.clicked.connect(skip)
            bottom.addWidget(self.skip)
        outer.addLayout(bottom)
        self.pool = ThreadPoolExecutor(max_workers=1)
        self.future = None
        self.rows = []
        self.loaded = False
        self.timer = QTimer(self.widget)
        self.timer.timeout.connect(self.poll)
        self.timer.start(100)

    def open(self):
        if not self.loaded and self.future is None:
            self.begin(self.load_photos)

    def load_photos(self):
        self.controls.load()
        return self.controls.snapshots()

    def begin(self, task):
        self.check_state.clear_failure()
        self.details_toggle.setChecked(False)
        self.details_toggle.hide()
        self.refresh.setText(tr("refresh_photos"))
        self.refresh.setEnabled(False)
        self.save.setEnabled(False)
        for _, field, enabled in self.rows:
            field.setEnabled(False)
            enabled.setEnabled(False)
        self.progress.show()
        self.note.setStyleSheet(f"color: {MUTED};")
        self.note.setText(tr("camera_working"))
        self.future = self.pool.submit(task)

    def refresh_clicked(self):
        if self.future is None:
            self.begin(self.controls.snapshots if self.loaded else self.load_photos)

    def poll(self):
        if self.future is None or not self.future.done():
            return
        future, self.future = self.future, None
        self.progress.hide()
        self.refresh.setEnabled(True)
        try:
            records = future.result()
            self.render(records)
            self.loaded = True
            self.note.setStyleSheet(f"color: {OK if getattr(self, 'saving', False) else MUTED};")
            self.note.setText((tr("saved_stopped") if self.controls.box.is_stopped() else tr("camera_saved")) if getattr(self, "saving", False) else tr("camera_ready"))
            if getattr(self, "saving", False):
                self.saved = True
                if self.wizard: self.note.setText(tr("remote_cameras_saved"))
                self.changed()
        except Exception as exc:
            for _, field, enabled in self.rows:
                field.setEnabled(True)
                enabled.setEnabled(True)
            self.validate()
            self.note.setStyleSheet(f"color: {ERROR};")
            if self.wizard and not getattr(self,"saving",False):
                self.check_state.failure(exc,getattr(self.controls,"diagnostic_output", ""),getattr(self.controls,"target", ""))
                self.error_details.setPlainText(self.check_state.details)
                self.details_toggle.show()
                self.refresh.setText(tr("retry"))
                self.note.setText(tr("camera_photos_load_failed"))
            else:
                self.note.setText(tr("camera_save_failed") if self.wizard and self.check_state.setup_finished else tr("camera_changes_save_failed") if self.wizard else tr("camera_error"))
        self.saving = False

    def render(self, records):
        from .ui import card, label, layout_for, demo_picture
        content = QWidget()
        grid = QGridLayout(content)
        grid.setContentsMargins(0, 0, 12, 0)
        grid.setSpacing(16)
        self.rows = []
        if not records:
            grid.addWidget(label(tr("camera_empty"), "muted"), 0, 0)
        for i, camera in enumerate(records):
            tile = card()
            layout = layout_for(tile, 16)
            from .ai_activity_ui import AlertPicture
            photo=AlertPicture(camera.file if camera.ok else None)
            if self.controls.box.demo and camera.ok: photo.pix=demo_picture(i)
            if photo.pix.isNull():
                photo=label(tr("camera_snapshot_failed" if self.wizard else "camera_no_photo"),"muted")
                photo.setMinimumHeight(180)
            layout.addWidget(photo)
            line = QHBoxLayout()
            name = QLineEdit(camera.name)
            name.setAccessibleName(tr("camera_name"))
            name.setValidator(QRegularExpressionValidator(QRegularExpression("[a-z0-9_]+"), name))
            name.textChanged.connect(self.validate)
            enabled = QCheckBox(tr("camera_enabled"))
            enabled.setChecked(camera.enabled)
            enabled.toggled.connect(self.validate)
            line.addWidget(name, 1)
            line.addWidget(enabled)
            layout.addLayout(line)
            grid.addWidget(tile, i // 2, i % 2)
            self.rows.append((camera.name, name, enabled))
        grid.setRowStretch((len(records)+1)//2, 1)
        self.scroll.setWidget(content)
        self.validate()

    def validate(self):
        self.saved = False
        try:
            changes_payload(self.changes())
            valid = bool(self.rows) and all(field.hasAcceptableInput() for _, field, _ in self.rows)
        except ValueError:
            valid = False
        self.save.setEnabled(valid and self.future is None)
        if self.rows and not valid:
            self.note.setStyleSheet(f"color: {ERROR};")
            self.note.setText(tr("camera_names_invalid"))

    def changes(self):
        return [(old, field.text(), enabled.isChecked()) for old, field, enabled in self.rows]

    def save_clicked(self):
        changes = self.changes()
        self.saving = True
        self.begin(lambda: self.controls.save(changes))

    def close(self):
        self.timer.stop()
        if hasattr(self.controls,"cancel"):
            self.controls.cancel()
            if self.future is None or self.future.done(): self.controls.cleanup()
            else: self.future.add_done_callback(lambda future:self.controls.cleanup())
        self.pool.shutdown(wait=False, cancel_futures=True)
