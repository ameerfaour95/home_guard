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
        self.search_button=QPushButton(tr('find_cameras_action'))
        self.search_button.setObjectName('secondary')
        self.search_button.clicked.connect(self.show_search)
        self.search_button.setVisible(not wizard)
        heading.insertWidget(1,self.search_button)
        self.house_alerts=QPushButton(tr("alert_house_title"));self.house_alerts.setObjectName("secondary")
        self.house_alerts.clicked.connect(lambda:self.open_alerts(None));heading.insertWidget(1,self.house_alerts)
        self.house_alerts.setVisible(bool(getattr(controls,"remote",False)))
        self.search_form=card()
        search_layout=layout_for(self.search_form,24)
        search_layout.addWidget(label(tr('camera_search_title'),'section'))
        search_layout.addWidget(label(tr('camera_search_hint'),'muted'))
        fields=QHBoxLayout()
        fields.addWidget(label(tr('camera_search_user')))
        self.search_user=QLineEdit('admin');fields.addWidget(self.search_user)
        fields.addWidget(label(tr('camera_search_password')))
        self.search_password=QLineEdit();self.search_password.setEchoMode(QLineEdit.EchoMode.Password);fields.addWidget(self.search_password)
        self.search_start=QPushButton(tr('find_cameras_action'));self.search_start.clicked.connect(self.search_clicked);fields.addWidget(self.search_start)
        search_layout.addLayout(fields)
        self.search_form.hide();outer.addWidget(self.search_form)
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
        self.error_details.setStyleSheet("font-family: Consolas; font-size: 10.5pt;")
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
        self.row_busy={}
        self.loaded = False
        self.zone_values = {}
        self.zones_loaded = False
        self.zone_load_failed = False
        self.zone_widgets = {}
        self.zone_dialog = None
        self.alert_dialog = None
        self.alert_house = ["person"]
        self.alert_values = {}
        self.alert_widgets = {}
        self.alert_load_failed = False
        self.timer = QTimer(self.widget)
        self.timer.timeout.connect(self.poll)
        self.timer.start(100)

    def open(self):
        if self.controls.box.demo or not getattr(self.controls, 'remote', False):
            self.alert_house=self.controls.box.load_settings().alert_on.split(',')
            self.refresh_alert_labels()
        if not self.loaded and self.future is None:
            self.begin(self.load_photos)

    def show_search(self):
        self.search_form.show()
        self.search_password.setFocus()

    def search_clicked(self):
        if self.future is not None: return
        user=self.search_user.text().strip()
        password=self.search_password.text()
        if not user or (not password and not self.controls.box.demo):
            self.note.setText(tr('camera_login_help'));return
        self.search_password.clear()
        self.searching=True
        self.begin(lambda:self.read_zones(self.controls.search(user,password)))
        self.note.setText(tr('camera_search_working'))

    def load_photos(self):
        self.controls.load()
        return self.read_zones(self.controls.snapshots())

    def read_zones(self, records):
        # One zone request after snapshots, never one request per card or UI tick.
        try:
            self.zone_values = self.controls.zones()
            self.zones_loaded = True
            self.zone_load_failed = False
        except Exception:
            self.zone_load_failed = True
        try:
            alerts = self.controls.camera_alerts()
            self.alert_house = alerts['house']
            self.alert_values = {row['name']: row['alert_on'] for row in alerts['cameras']}
            self.alert_load_failed = False
        except Exception:
            self.alert_load_failed = True
        return records

    def begin(self, task):
        self.check_state.clear_failure()
        self.details_toggle.setChecked(False)
        self.details_toggle.hide()
        self.refresh.setText(tr("refresh_photos"))
        self.refresh.setEnabled(False)
        self.search_start.setEnabled(False)
        self.save.setEnabled(False)
        for _, field, enabled in self.rows:
            field.setEnabled(False)
            enabled.setEnabled(False)
        for button, _, _ in self.zone_widgets.values(): button.setEnabled(False)
        for button in self.alert_widgets.values(): button.setEnabled(False)
        from .motion import busy
        busy(self.search_start if getattr(self,'searching',False) else self.save if getattr(self,'saving',False) else self.refresh,True)
        self.progress.hide()
        self.note.setStyleSheet(f"color: {MUTED};")
        self.note.setText(tr("camera_working"))
        self.future = self.pool.submit(task)

    def refresh_clicked(self):
        if self.future is None:
            self.begin(lambda:self.read_zones(self.controls.snapshots()) if self.loaded else self.load_photos())

    def poll(self):
        if self.future is None and not self.wizard and not self.controls.box.demo and self.widget.isVisible() and not (self.zone_dialog and self.zone_dialog.isVisible()) and not (self.alert_dialog and self.alert_dialog.isVisible()):
            import time
            if time.monotonic()-getattr(self,'last_names_poll',0)>=1:
                self.last_names_poll=time.monotonic()
                try:
                    records=self.controls.load()
                    signature=tuple((r.name,r.enabled) for r in records)
                    if signature!=getattr(self,'record_signature',None): self.render(records)
                except (OSError,ValueError,TypeError): pass
        if self.future is None or not self.future.done():
            return
        if getattr(self,'toggle_before',None) is not None:
            self.finish_toggle();return
        future, self.future = self.future, None
        self.progress.hide()
        from .motion import busy
        for button in (self.refresh,self.search_start,self.save): busy(button,False)
        self.refresh.setEnabled(True)
        self.search_start.setEnabled(True)
        try:
            records = future.result()
            self.render(records)
            self.loaded = True
            self.note.setStyleSheet(f"color: {OK if getattr(self, 'saving', False) else MUTED};")
            self.note.setText((tr("saved_stopped") if self.controls.box.is_stopped() else tr("camera_saved")) if getattr(self, "saving", False) else tr("camera_ready"))
            if getattr(self,'searching',False):
                self.note.setText(tr('camera_search_empty') if not records else tr('camera_ready'))
                if records: self.search_form.hide()
                self.changed()
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
                if getattr(self,'searching',False): self.note.setText(tr('camera_search_error'))
        self.saving = False
        self.searching = False

    def render(self, records):
        from .ui import card, label, layout_for, demo_picture
        content = QWidget()
        grid = QGridLayout(content)
        grid.setContentsMargins(0, 0, 12, 0)
        grid.setSpacing(16)
        self.rows = []
        self.zone_widgets = {}
        self.alert_widgets = {}
        self.record_signature=tuple((r.name,r.enabled) for r in records)
        if not records:
            grid.addWidget(label(tr("camera_empty"), "muted"), 0, 0)
        for i, camera in enumerate(records):
            tile = card()
            layout = layout_for(tile, 16)
            from .zone_picture import ZonePicture
            photo=ZonePicture(camera.file if camera.ok else None,self.zone_values.get(camera.name, []));photo.height_limit=200
            if self.controls.box.demo and camera.ok: photo.pix=demo_picture(i)
            if photo.pix.isNull():
                photo=label(tr("camera_snapshot_failed" if self.wizard else "camera_no_photo"),"muted")
                photo.setMinimumHeight(180)
            layout.addWidget(photo)
            # Keep the reserved row and card geometry stable.
            slot=QWidget();slot.setObjectName('cameraActionSlot');slot.setFixedHeight(36)
            actions=QHBoxLayout(slot);actions.setContentsMargins(0,0,0,0);actions.setSpacing(8);layout.addWidget(slot)
            if self.zones_loaded or self.zone_load_failed:
                from .zone_editor import ZonePill, colors
                from .zone_picture import WatchingStatus
                zone_button=ZonePill(tr('camera_zone_button'),compact=True)
                zone_button.setEnabled(isinstance(photo,ZonePicture) and not photo.pix.isNull() and not self.zone_load_failed)
                zone_button.clicked.connect(lambda checked=False,n=camera.name,p=photo:self.open_zone(n,p))
                status=WatchingStatus(tr('camera_error') if self.zone_load_failed else tr('camera_zone_drawn' if self.zone_values.get(camera.name) else 'camera_zone_whole'), bool(self.zone_values.get(camera.name)))
                status.setStyleSheet(f'color: {colors(self.widget)["secondary"]}; font-size: 13px;')
                status.setAccessibleName(status.text())
                actions.addWidget(zone_button);status.hide();zone_button.setToolTip(status.text())
                self.zone_widgets[camera.name]=(zone_button,status,photo)
            from .alert_types_ui import CameraAlertButton
            alert_button=CameraAlertButton(self.alert_house,self.alert_values.get(camera.name))
            if self.alert_load_failed: alert_button.setToolTip(tr('alert_read_error'))
            alert_button.clicked.connect(lambda checked=False,n=camera.name:self.open_alerts(n))
            actions.addWidget(alert_button,1);self.alert_widgets[camera.name]=alert_button
            line = QHBoxLayout()
            name = QLineEdit(camera.name)
            name.setAccessibleName(tr("camera_name"))
            name.setValidator(QRegularExpressionValidator(QRegularExpression("[a-z0-9_]+"), name))
            name.textChanged.connect(self.validate)
            from .motion import Switch
            enabled = Switch(tr("camera_enabled" if camera.enabled else "camera_off"))
            enabled.setChecked(camera.enabled)
            enabled.toggled.connect(self.validate)
            if not self.wizard: enabled.toggled.connect(lambda checked,n=camera.name:self.set_camera_enabled(n,checked))
            busy=QProgressBar();busy.setRange(0,0);busy.setTextVisible(False);busy.setFixedSize(24,3);busy.hide();line.addWidget(busy);self.row_busy[camera.name]=busy
            line.addWidget(name, 1)
            line.addWidget(enabled)
            layout.addLayout(line)
            columns=2 if self.wizard else 3
            grid.addWidget(tile, i // columns, i % columns)
            self.rows.append((camera.name, name, enabled))
        grid.setRowStretch((len(records)+(1 if self.wizard else 2))//(2 if self.wizard else 3), 1)
        self.scroll.setWidget(content)
        self.validate()

    def open_alerts(self, name):
        if self.future is not None: return
        from .alert_types_ui import CameraAlertsDialog
        # Refresh the house choice if Settings changed since the photos were loaded.
        if self.controls.box.demo or not getattr(self.controls, 'remote', False):
            self.alert_house = self.controls.box.load_settings().alert_on.split(',')
        dialog=CameraAlertsDialog(self.controls,name,self.alert_house,self.alert_values.get(name),self.widget)
        self.alert_dialog=dialog
        dialog.saved.connect(lambda value:self.alert_saved(name,value))
        dialog.house_saved.connect(self.house_alert_saved)
        def finished(result):
            self.alert_house=dialog.house
            self.refresh_alert_labels()
            self.alert_dialog=None
            dialog.deleteLater()
        dialog.finished.connect(finished)
        dialog.open()

    def refresh_alert_labels(self):
        for name,button in self.alert_widgets.items():
            button.refresh(self.alert_house,self.alert_values.get(name))

    def alert_saved(self,name,value):
        self.alert_values[name]=value
        self.refresh_alert_labels();self.changed()

    def house_alert_saved(self,value):
        self.alert_house=value
        self.refresh_alert_labels();self.changed()

    def open_zone(self, name, photo):
        if self.future is not None or self.zone_load_failed or not self.zones_loaded: return
        from .zone_editor import ZoneEditorDialog
        dialog=ZoneEditorDialog(self.controls,name,photo.pix,self.zone_values.get(name,[]),self.widget)
        self.zone_dialog=dialog
        dialog.zone_saved.connect(lambda points:self.zone_saved(name,points))
        def finished(result):
            self.zone_dialog=None
            dialog.deleteLater()
        dialog.finished.connect(finished)
        dialog.open()

    def zone_saved(self, name, points):
        self.zone_values[name]=points
        if name in self.zone_widgets:
            _,status,photo=self.zone_widgets[name]
            status.drawn = bool(points)
            status.change(tr('camera_zone_drawn' if points else 'camera_zone_whole'))
            status.setAccessibleName(status.text())
            photo.set_zone(points)

    def set_camera_enabled(self,name,enabled):
        if self.future is not None: return
        from dataclasses import replace
        self.controls.toggle_pending=True
        self.toggle_before=list(self.controls.records)
        changes=[(c.name,c.name,enabled if c.name==name else c.enabled) for c in self.controls.records]
        self.controls.records=[replace(c,enabled=enabled) if c.name==name else c for c in self.controls.records]
        for old,field,switch in self.rows:
            switch.setEnabled(False)
            if old==name:
                switch.blockSignals(True);switch.setChecked(enabled);switch.setText(tr('camera_enabled' if enabled else 'camera_off'));switch.blockSignals(False)
        if name in self.row_busy: self.row_busy[name].show()
        self.save.setEnabled(False)
        self.future=self.pool.submit(lambda:self.controls.save(changes))
        self.changed()

    def finish_toggle(self):
        future,self.future=self.future,None
        try:
            records=future.result()
            self.note.setText(tr('camera_saved'))
        except Exception:
            self.controls.records=self.toggle_before
            records=self.toggle_before
            self.note.setText(tr('camera_error'))
            from .motion import toast
            toast(self.widget.window(),tr('camera_error'))
        self.controls.toggle_pending=False
        self.toggle_before=None
        edits={old:field.text() for old,field,_ in self.rows}
        self.render(records)
        for old,field,_ in self.rows:
            if old in edits: field.setText(edits[old])
        self.changed()

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
        def save():
            records=self.controls.save(changes)
            self.zone_values={new:self.zone_values.get(old,[]) for old,new,_ in changes}
            self.alert_values={new:self.alert_values.get(old) for old,new,_ in changes}
            return records
        self.begin(save)

    def close(self):
        if self.alert_dialog: self.alert_dialog.shutdown();self.alert_dialog.hide()
        self.timer.stop()
        if hasattr(self.controls,"cancel"):
            self.controls.cancel()
            if self.future is None or self.future.done(): self.controls.cleanup()
            else: self.future.add_done_callback(lambda future:self.controls.cleanup())
        self.pool.shutdown(wait=False, cancel_futures=True)
