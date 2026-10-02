from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QCheckBox, QSpinBox, QDoubleSpinBox, QPushButton, QHBoxLayout, QGridLayout
from .strings import tr
from .theme import OK, ERROR, WARNING
from .box_controls import Settings, minutes_to_seconds
from .ui import card, label, layout_for


class SettingsPage:
    def __init__(self, box, changed):
        from PySide6.QtWidgets import QWidget
        self.box,self.changed=box,changed
        self.widget=QWidget();layout=layout_for(self.widget,0);layout.setSpacing(24)
        layout.addWidget(label(tr("settings_title"),"title"));layout.addWidget(label(tr("settings_hint"),"muted"))
        columns=QHBoxLayout();columns.setSpacing(24)
        security=card();form=layout_for(security,24);form.setSpacing(16)
        form.addWidget(label(tr("premium_security"),"section"))
        self.alerts=QCheckBox(tr("security_alerts"));form.addWidget(self.alerts)
        form.addWidget(label(tr("security_alerts_hint"),"muted"))
        from .alert_hours import AlertHours
        self.hours=AlertHours();self.start,self.end=self.hours.start,self.hours.end;form.addWidget(self.hours)
        form.addWidget(label(tr("cooldown")))
        self.cooldown=QDoubleSpinBox();self.cooldown.setDecimals(2);self.cooldown.setRange(.17,1440);self.cooldown.setSingleStep(1);self.cooldown.setSuffix(tr("minutes_suffix"));self.cooldown.setMaximumWidth(500);form.addWidget(self.cooldown)
        form.addWidget(label(tr("cooldown_hint"),"muted"));form.addStretch()
        viewing=card();view=layout_for(viewing,24);view.setSpacing(16)
        view.addWidget(label(tr("premium_viewing"),"section"));self.pictures=QCheckBox(tr("show"));view.addWidget(self.pictures);view.addWidget(label(tr("premium_view_hint"),"muted"));view.addStretch()
        columns.addWidget(security,2);columns.addWidget(viewing,1);layout.addLayout(columns,1)
        footer=QHBoxLayout();self.note=label("","muted");footer.addWidget(self.note,1)
        self.save=QPushButton(tr("save_settings"));self.save.clicked.connect(self.save_clicked);footer.addWidget(self.save);layout.addLayout(footer)
        self.waiting=False;self.timer=QTimer(self.widget);self.timer.timeout.connect(self.check_applied);self.timer.start(250)

    def reload(self):
        try:
            settings = self.box.load_settings()
            self.alerts.setChecked(settings.mode == "inference")
            self.hours.set_hours(settings.alert_start_hour, settings.alert_end_hour)
            self.cooldown.setValue(settings.alert_cooldown_sec/60)
            self.pictures.setChecked(settings.show_cameras)
            self.note.setText("")
        except Exception:
            self.note.setStyleSheet(f"color: {ERROR};")
            self.note.setText(tr("control_error"))
        self.hours_changed()

    def hours_changed(self):
        if not hasattr(self, "hours_note"):
            return
        start, end = self.start.value(), self.end.value()
        self.hours_note.setText(tr("all_day") if start == end else tr("hours_overnight" if end < start else "hours_window", start=start, end=end))

    def save_clicked(self):
        settings = Settings("inference" if self.alerts.isChecked() else "data_collection", *self.hours.values(), minutes_to_seconds(self.cooldown.value()), self.pictures.isChecked())
        try:
            self.box.save_settings(settings)
            self.waiting = self.box.phase() == "restarting"
            self.save.setEnabled(not self.waiting)
            self.note.setStyleSheet(f"color: {WARNING if self.waiting else OK};")
            self.note.setText(tr("applying") if self.waiting else tr("saved_stopped") if self.box.is_stopped() else tr("settings_saved"))
            self.changed()
        except Exception:
            self.note.setStyleSheet(f"color: {ERROR};")
            self.note.setText(tr("control_error"))

    def check_applied(self):
        if not self.waiting:
            return
        phase = self.box.phase()
        if phase != "restarting":
            self.waiting = False
            self.save.setEnabled(True)
            self.note.setStyleSheet(f"color: {OK};")
            self.note.setText(tr("saved_stopped") if phase == "stopped" else tr("applied"))
        elif self.box.clock() - self.box.pending_at > 30:
            self.note.setText(tr("apply_slow"))
