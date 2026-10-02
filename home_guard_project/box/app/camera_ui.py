from concurrent.futures import ThreadPoolExecutor
from PySide6.QtCore import QTimer, QRegularExpression, Qt
from PySide6.QtGui import QPixmap, QRegularExpressionValidator
from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QScrollArea, QPushButton, QLineEdit, QCheckBox, QLabel, QProgressBar
from .strings import tr
from .camera_controls import changes_payload

class CameraPage:
    def __init__(self, controls, changed):
        from .ui import card, label, layout_for
        self.controls, self.changed = controls, changed
        self.widget = QWidget()
        outer = layout_for(self.widget, 0)
        heading = QHBoxLayout()
        heading.addWidget(label(tr("cameras_title"), "section"), 1)
        self.refresh = QPushButton(tr("refresh_photos"))
        self.refresh.clicked.connect(self.refresh_clicked)
        heading.addWidget(self.refresh)
        outer.addLayout(heading)
        outer.addWidget(label(tr("cameras_hint"), "muted"))
        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.hide()
        outer.addWidget(self.progress)
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setStyleSheet("QScrollArea { border: none; background: transparent; } QScrollBar:vertical { background: #10151d; width: 8px; } QScrollBar::handle:vertical { background: #334354; min-height: 30px; border-radius: 4px; } QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0px; }")
        outer.addWidget(self.scroll, 1)
        bottom = QHBoxLayout()
        self.note = label("", "muted")
        bottom.addWidget(self.note, 1)
        self.save = QPushButton(tr("save_cameras"))
        self.save.setEnabled(False)
        self.save.clicked.connect(self.save_clicked)
        bottom.addWidget(self.save)
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
        self.refresh.setEnabled(False)
        self.save.setEnabled(False)
        for _, field, enabled in self.rows:
            field.setEnabled(False)
            enabled.setEnabled(False)
        self.progress.show()
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
            self.note.setText((tr("saved_stopped") if self.controls.box.is_stopped() else tr("camera_saved")) if getattr(self, "saving", False) else tr("camera_ready"))
            if getattr(self, "saving", False):
                self.changed()
        except Exception:
            for _, field, enabled in self.rows:
                field.setEnabled(True)
                enabled.setEnabled(True)
            self.validate()
            self.note.setText(tr("camera_error"))
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
            photo = QLabel()
            photo.setAlignment(Qt.AlignmentFlag.AlignCenter)
            photo.setMinimumHeight(100)
            photo.setMaximumHeight(120)
            pix = demo_picture(i) if self.controls.box.demo and camera.ok else QPixmap(camera.file) if camera.ok else QPixmap()
            if pix.isNull():
                photo.setText(tr("camera_no_photo"))
            else:
                photo.setPixmap(pix.scaled(540, 120, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
            layout.addWidget(photo)
            line = QHBoxLayout()
            name = QLineEdit(camera.name)
            name.setAccessibleName(tr("camera_name"))
            name.setValidator(QRegularExpressionValidator(QRegularExpression("[a-z0-9_]+"), name))
            name.textChanged.connect(self.validate)
            enabled = QCheckBox(tr("camera_enabled"))
            enabled.setChecked(camera.enabled)
            line.addWidget(name, 1)
            line.addWidget(enabled)
            layout.addLayout(line)
            grid.addWidget(tile, i // 2, i % 2)
            self.rows.append((camera.name, name, enabled))
        grid.setRowStretch((len(records)+1)//2, 1)
        self.scroll.setWidget(content)
        self.validate()

    def validate(self):
        try:
            changes_payload(self.changes())
            valid = bool(self.rows) and all(field.hasAcceptableInput() for _, field, _ in self.rows)
        except ValueError:
            valid = False
        self.save.setEnabled(valid and self.future is None)
        if self.rows and not valid:
            self.note.setText(tr("camera_names_invalid"))

    def changes(self):
        return [(old, field.text(), enabled.isChecked()) for old, field, enabled in self.rows]

    def save_clicked(self):
        changes = self.changes()
        self.saving = True
        self.begin(lambda: self.controls.save(changes))

    def close(self):
        self.timer.stop()
        self.pool.shutdown(wait=False, cancel_futures=True)
