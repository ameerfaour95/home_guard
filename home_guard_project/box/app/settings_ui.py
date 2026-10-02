from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QCheckBox, QSpinBox, QDoubleSpinBox, QPushButton, QHBoxLayout, QGridLayout
from .strings import tr
from .box_controls import Settings, minutes_to_seconds
from .ui import card, label, layout_for


class SettingsPage:
    def __init__(self, box, changed):
        self.box, self.changed = box, changed
        self.widget = card()
        layout = layout_for(self.widget, 30)
        layout.addWidget(label(tr("settings_title"), "title"))
        layout.addWidget(label(tr("settings_hint"), "muted"))
        self.alerts = QCheckBox(tr("security_alerts"))
        layout.addWidget(self.alerts)
        layout.addWidget(label(tr("security_alerts_hint"), "muted"))
        grid = QGridLayout()
        self.start, self.end = QSpinBox(), QSpinBox()
        for spin in (self.start, self.end):
            spin.setRange(0, 23)
            spin.setMaximumWidth(300)
            spin.valueChanged.connect(self.hours_changed)
        grid.addWidget(label(tr("alert_from")), 0, 0)
        grid.addWidget(label(tr("alert_until")), 0, 1)
        grid.addWidget(self.start, 1, 0)
        grid.addWidget(self.end, 1, 1)
        layout.addLayout(grid)
        self.hours_note = label("", "accent")
        layout.addWidget(self.hours_note)
        layout.addWidget(label(tr("cooldown")))
        self.cooldown = QDoubleSpinBox()
        self.cooldown.setDecimals(2)
        self.cooldown.setRange(0.17, 1440)
        self.cooldown.setSingleStep(1)
        self.cooldown.setSuffix(tr("minutes_suffix"))
        self.cooldown.setMaximumWidth(500)
        layout.addWidget(self.cooldown)
        layout.addWidget(label(tr("cooldown_hint"), "muted"))
        self.pictures = QCheckBox(tr("show"))
        layout.addWidget(self.pictures)
        self.note = label("", "accent")
        layout.addWidget(self.note)
        layout.addStretch()
        row = QHBoxLayout()
        row.addStretch()
        self.save = QPushButton(tr("save_settings"))
        self.save.clicked.connect(self.save_clicked)
        row.addWidget(self.save)
        layout.addLayout(row)
        self.waiting = False
        self.timer = QTimer(self.widget)
        self.timer.timeout.connect(self.check_applied)
        self.timer.start(250)

    def reload(self):
        try:
            settings = self.box.load_settings()
            self.alerts.setChecked(settings.mode == "inference")
            self.start.setValue(settings.alert_start_hour)
            self.end.setValue(settings.alert_end_hour)
            self.cooldown.setValue(settings.alert_cooldown_sec/60)
            self.pictures.setChecked(settings.show_cameras)
            self.note.setText("")
        except Exception:
            self.note.setText(tr("control_error"))
        self.hours_changed()

    def hours_changed(self):
        if not hasattr(self, "hours_note"):
            return
        start, end = self.start.value(), self.end.value()
        self.hours_note.setText(tr("all_day") if start == end else tr("hours_overnight" if end < start else "hours_window", start=start, end=end))

    def save_clicked(self):
        settings = Settings("inference" if self.alerts.isChecked() else "data_collection", self.start.value(), self.end.value(), minutes_to_seconds(self.cooldown.value()), self.pictures.isChecked())
        try:
            self.box.save_settings(settings)
            self.waiting = self.box.phase() == "restarting"
            self.save.setEnabled(not self.waiting)
            self.note.setText(tr("applying") if self.waiting else tr("saved_stopped") if self.box.is_stopped() else tr("saved"))
            self.changed()
        except Exception:
            self.note.setText(tr("control_error"))

    def check_applied(self):
        if not self.waiting:
            return
        phase = self.box.phase()
        if phase != "restarting":
            self.waiting = False
            self.save.setEnabled(True)
            self.note.setText(tr("saved_stopped") if phase == "stopped" else tr("applied"))
        elif self.box.clock() - self.box.pending_at > 30:
            self.note.setText(tr("apply_slow"))
