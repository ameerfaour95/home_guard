from PySide6.QtCore import QTimer, Qt
from dataclasses import replace
from .live_settings import AppliedState,slider_conf,conf_slider
from PySide6.QtWidgets import QCheckBox, QSpinBox, QDoubleSpinBox, QPushButton, QHBoxLayout, QGridLayout, QComboBox, QSlider
from .motion import Switch, busy
from .strings import tr
from .theme import OK, ERROR, WARNING
from .box_controls import Settings, minutes_to_seconds
from .ui import card, label, layout_for


class SettingsPage:
    def __init__(self, box, changed, viewer=None, viewer_changed=None):
        from PySide6.QtWidgets import QWidget
        self.box,self.changed=box,changed
        self.loading=False;self.ack=AppliedState()
        self.widget=QWidget();layout=layout_for(self.widget,0);layout.setSpacing(16)
        layout.addWidget(label(tr("settings_title"),"title"));layout.addWidget(label(tr("settings_hint"),"muted"))
        columns=QHBoxLayout();columns.setSpacing(16)
        security=card();form=layout_for(security,24);form.setSpacing(16)
        form.addWidget(label(tr("premium_security"),"section"))
        self.alerts=Switch(tr("security_alerts"));form.addWidget(self.alerts)
        form.addWidget(label(tr("security_alerts_hint"),"muted"))
        from .alert_hours import AlertHours
        self.hours=AlertHours();self.start,self.end=self.hours.start,self.hours.end;form.addWidget(self.hours)
        form.addWidget(label(tr("cooldown")))
        self.cooldown=QDoubleSpinBox();self.cooldown.setDecimals(2);self.cooldown.setRange(.17,1440);self.cooldown.setSingleStep(1);self.cooldown.setSuffix(tr("minutes_suffix"));self.cooldown.setMaximumWidth(500);form.addWidget(self.cooldown)
        form.addWidget(label(tr("cooldown_hint"),"muted"));form.addStretch()
        viewing=card();view=layout_for(viewing,24);view.setSpacing(10)
        view.addWidget(label(tr("premium_viewing"),"section"));self.pictures=Switch(tr("show"));view.addWidget(self.pictures);view.addWidget(label(tr("premium_view_hint"),"muted"))
        view.addWidget(label(tr("display_group"),"section"))
        self.detections=Switch(tr("show_detections"));view.addWidget(self.detections)
        view.addWidget(label(tr("detection_labels")))
        self.labels=QComboBox();self.labels.addItems([tr("labels_confidence"),tr("labels_name"),tr("labels_none")]);view.addWidget(self.labels)
        self.viewer_changed=viewer_changed or (lambda **changes:None)
        from .preferences import ViewerSettings
        self.sync_viewer(viewer or ViewerSettings())
        self.detections.toggled.connect(lambda value:self.viewer_changed(detections=value))
        self.labels.currentIndexChanged.connect(lambda index:self.viewer_changed(detection_labels=("confidence","name","none")[index]))
        detection=card();detect=layout_for(detection,20);detect.setSpacing(12)
        detect.addWidget(label(tr("detection_group"),"section"))
        detect.addWidget(label(tr("sensitivity")))
        self.sensitivity=QSlider(Qt.Orientation.Horizontal);self.sensitivity.setRange(1,19);self.sensitivity.setSingleStep(1);self.sensitivity.setPageStep(1);self.sensitivity.setValue(8)
        self.certainty=label(tr("certainty_required",percent=40),"muted");detect.addWidget(self.certainty)
        self.sensitivity.valueChanged.connect(lambda value:self.certainty.setText(tr("certainty_required",percent=round(slider_conf(value)*100))))
        self.sensitivity.sliderReleased.connect(self.sensitivity_released)
        detect.addWidget(self.sensitivity)
        ends=QHBoxLayout();ends.addWidget(label(tr("more_alerts"),"muted"));ends.addStretch();ends.addWidget(label(tr("fewer_false_alarms"),"muted"));detect.addLayout(ends)
        reset=QPushButton(tr("reset_sensitivity"));reset.setObjectName("textAction");reset.clicked.connect(self.reset_sensitivity);detect.addWidget(reset,0,Qt.AlignmentFlag.AlignLeft)
        detect.addStretch()
        self.start.editingFinished.connect(self.live_hours);self.end.editingFinished.connect(self.live_hours);self.hours.all_day.toggled.connect(self.live_hours)
        self.cooldown.editingFinished.connect(self.live_cooldown)
        columns.addWidget(security,8);columns.addWidget(viewing,6);columns.addWidget(detection,5);layout.addLayout(columns,1)
        footer=QHBoxLayout();self.note=label("","muted");footer.addWidget(self.note,1)
        self.save=QPushButton(tr("save_settings"));self.save.clicked.connect(self.save_clicked);footer.addWidget(self.save);layout.addLayout(footer)
        self.facts=label("","muted");layout.addWidget(self.facts)
        self.waiting=False;self.timer=QTimer(self.widget);self.timer.timeout.connect(self.check_applied);self.timer.start(250)

    def sync_viewer(self,viewer):
        self.detections.blockSignals(True);self.labels.blockSignals(True)
        self.detections.setChecked(viewer.detections);self.labels.setCurrentIndex(("confidence","name","none").index(viewer.detection_labels))
        self.labels.setEnabled(viewer.detections)
        self.detections.blockSignals(False);self.labels.blockSignals(False)

    def reload(self):
        self.loading=True
        try:
            settings = self.box.load_settings()
            self.alerts.setChecked(settings.mode == "inference")
            self.hours.set_hours(settings.alert_start_hour, settings.alert_end_hour)
            self.cooldown.setValue(settings.alert_cooldown_sec/60)
            self.pictures.setChecked(settings.show_cameras)
            self.sensitivity.setValue(conf_slider(settings.inference_conf))
            self.note.setText("")
        except Exception:
            self.note.setStyleSheet(f"color: {ERROR};")
            self.note.setText(tr("control_error"))
        self.loading=False
        self.hours_changed()

    def hours_changed(self):
        if not hasattr(self, "hours_note"):
            return
        start, end = self.start.value(), self.end.value()
        self.hours_note.setText(tr("all_day") if start == end else tr("hours_overnight" if end < start else "hours_window", start=start, end=end))

    def sensitivity_released(self):
        self.apply_live(inference_conf=slider_conf(self.sensitivity.value()))

    def reset_sensitivity(self):
        self.sensitivity.setValue(8);self.sensitivity_released()

    def live_hours(self):
        start,end=self.hours.values();self.apply_live(alert_start_hour=start,alert_end_hour=end)

    def live_cooldown(self):
        self.apply_live(alert_cooldown_sec=minutes_to_seconds(self.cooldown.value()))

    def apply_live(self,**changes):
        if self.loading or not self.save.isEnabled(): return
        try:
            before=self.box.load_settings();after=replace(before,**changes)
            if before == after: return
            self.box.save_settings(after);self.ack.request(before,after,self.box.clock())
            self.note.setStyleSheet('color: '+WARNING);self.note.setText(tr('applying'))
            self.changed()
        except Exception:
            self.note.setStyleSheet('color: '+ERROR);self.note.setText(tr('control_error'))

    def save_clicked(self):
        settings = Settings("inference" if self.alerts.isChecked() else "data_collection", *self.hours.values(), minutes_to_seconds(self.cooldown.value()), self.pictures.isChecked(),slider_conf(self.sensitivity.value()))
        try:
            before=self.box.load_settings()
            self.box.save_settings(settings)
            self.ack.request(before,settings,self.box.clock())
            self.waiting = self.box.phase() == "restarting"
            busy(self.save,self.waiting)
            self.note.setStyleSheet(f"color: {WARNING if self.waiting or self.ack.expected else OK};")
            self.note.setText(tr("applying") if self.waiting or self.ack.expected else tr("saved_stopped") if self.box.is_stopped() else tr("settings_saved"))
            self.changed()
        except Exception:
            self.note.setStyleSheet(f"color: {ERROR};")
            self.note.setText(tr("control_error"))

    def check_applied(self):
        if self.ack.expected:
            status=self.ack.status(self.box.reported_status(),self.box.clock())
            self.note.setText(tr("live_applied" if status=="applied" else status));self.note.setStyleSheet('color: '+(OK if status=='applied' else WARNING))
        if not self.waiting:
            return
        phase = self.box.phase()
        if phase != "restarting":
            self.waiting = False
            busy(self.save,False)
            self.note.setStyleSheet(f"color: {OK};")
            if not self.ack.expected:
                self.note.setText(tr("saved_stopped") if phase == "stopped" else tr("applied"))
        elif self.box.clock() - self.box.pending_at > 30:
            self.note.setText(tr("apply_slow"))
