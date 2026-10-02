"""Installer-readable timeline with a secondary technical transcript."""
from PySide6.QtCore import Qt,QTimer,Signal
from PySide6.QtWidgets import QWidget,QVBoxLayout,QHBoxLayout,QLabel,QTextEdit,QPushButton,QApplication,QScrollArea,QGraphicsOpacityEffect
from .strings import tr
from .theme import ERROR,WARNING,OK,MUTED
from .setup_details import DetailsModel,DetailsSelection

class DetailsPanel(QWidget):
    selection_changed=Signal()
    def __init__(self,model=None):
        super().__init__();self.setObjectName('setupDetails')
        from .ui import label
        layout=QVBoxLayout(self);layout.setContentsMargins(20,16,20,16);layout.setSpacing(12)
        self.heading=label('', 'section');layout.addWidget(self.heading)
        self.result=label('', 'muted');self.result.setTextFormat(Qt.TextFormat.PlainText);layout.addWidget(self.result)
        self.follow=QPushButton(tr('follow_progress'));self.follow.setObjectName('textAction');self.follow.clicked.connect(self.follow_progress);layout.addWidget(self.follow,0,Qt.AlignmentFlag.AlignLeft)
        self.scroll=QScrollArea();self.scroll.setWidgetResizable(True)
        self.body=QWidget();self.body.setObjectName('timelineBody');self.timeline=QVBoxLayout(self.body);self.timeline.setContentsMargins(0,0,0,0);self.timeline.setSpacing(14)
        self.scroll.setWidget(self.body);layout.addWidget(self.scroll,1)
        footer=QHBoxLayout()
        self.technical=QPushButton(tr('technical_log'));self.technical.setObjectName('textAction');self.technical.setCheckable(True);self.technical.toggled.connect(self.render);footer.addWidget(self.technical)
        footer.addStretch()
        self.copy_button=QPushButton(tr('copy_log'));self.copy_button.setObjectName('textAction');self.copy_button.clicked.connect(self.copy);footer.addWidget(self.copy_button);layout.addLayout(footer)
        self.copy_timer=QTimer(self);self.copy_timer.setSingleShot(True);self.copy_timer.setInterval(2000);self.copy_timer.timeout.connect(lambda:self.copy_button.setText(tr('copy_log')))
        self.pulse=QTimer(self);self.pulse.setInterval(650);self.pulse.timeout.connect(self.pulse_working);self.pulse.start();self.dim=False;self.working_effect=None
        self.reset(model)
    @property
    def selected(self): return self.selection.selected
    def reset(self,model=None):
        self.model=model or DetailsModel();self.selection=DetailsSelection();self.selection.resume(self.model.current)
        self.technical.setChecked(False);self.render()
    def select(self,step):
        self.selection.choose(step);self.technical.setChecked(False);self.render();self.selection_changed.emit()
    def follow_progress(self):
        self.selection.resume(self.model.current);self.render();self.selection_changed.emit()
    def feed(self,event):
        self.model.feed(event);self.selection.feed(event,self.model.current)
        if event.kind=='step' and event.status=='fail': self.technical.setChecked(False)
        self.render();self.selection_changed.emit()
    def render(self):
        if not hasattr(self,'model'): return
        from .ui import label
        from .ai_activity_ui import icon
        group=self.model.groups[self.selected]
        self.heading.setText(tr('step_'+self.selected))
        self.result.setText(group.result if group.status!='start' else tr('details_running_hint'))
        self.result.setStyleSheet('color: '+(ERROR if group.status=='fail' else WARNING if group.status=='warn' else MUTED))
        self.follow.setVisible(not self.selection.following)
        self.working_effect=None
        while self.timeline.count():
            item=self.timeline.takeAt(0)
            if item.widget(): item.widget().deleteLater()
        for stamp,role,text in group.entries:
            row=QWidget();line=QHBoxLayout(row);line.setContentsMargins(0,0,0,0);line.setSpacing(10)
            clock=label(stamp[:5],'muted');clock.setFixedWidth(48);line.addWidget(clock,0,Qt.AlignmentFlag.AlignTop)
            mark=QLabel();mark.setPixmap(icon({'ok':'check','warning':'warning','error':'cross'}.get(role,'bot')).pixmap(16,16));line.addWidget(mark,0,Qt.AlignmentFlag.AlignTop)
            sentence=label(text);sentence.setTextFormat(Qt.TextFormat.PlainText);sentence.setStyleSheet('color: '+{'ok':OK,'warning':WARNING,'error':ERROR}.get(role,'#b5c8d1'));sentence.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse);line.addWidget(sentence,1)
            self.timeline.addWidget(row)
        if group.status=='pending': self.timeline.addWidget(label(tr('details_not_started'),'muted'))
        if group.status=='start':
            working=label(tr('details_working'),'muted');self.working_effect=QGraphicsOpacityEffect(working);working.setGraphicsEffect(self.working_effect);self.timeline.addWidget(working)
        if self.technical.isChecked():
            raw=QTextEdit();raw.setReadOnly(True);raw.setStyleSheet('font-family: Consolas; font-size: 10.5pt;');raw.setPlainText(self.model.technical_log(self.selected));raw.setMinimumHeight(220);self.timeline.addWidget(raw)
        self.timeline.addStretch()
    def pulse_working(self):
        if self.working_effect:
            self.dim=not self.dim;self.working_effect.setOpacity(.55 if self.dim else 1)
    def copy(self):
        group=self.model.groups[self.selected]
        text=self.model.technical_log(self.selected) if self.technical.isChecked() else '\n'.join(stamp+'  '+sentence for stamp,role,sentence in group.entries)
        QApplication.clipboard().setText(text)
        self.copy_button.setText(tr('details_copied'));self.copy_timer.start()
