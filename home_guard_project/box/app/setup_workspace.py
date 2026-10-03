"""Stable setup rows and a responsive, scrollable two-pane workspace."""
from PySide6.QtCore import Qt,Signal
from PySide6.QtWidgets import QWidget,QFrame,QLabel,QPushButton,QVBoxLayout,QHBoxLayout,QBoxLayout,QScrollArea,QProgressBar
from .engine_backend import ENGINE_STEPS
from .strings import tr
from .theme import OK,WARNING,ERROR,MUTED,ACTION

class StepRow(QFrame):
    clicked=Signal(str)
    def __init__(self,step):
        super().__init__();self.step=step;self.setObjectName('setupStep');self.setFixedHeight(50)
        box=QVBoxLayout(self);box.setContentsMargins(8,4,8,4);box.setSpacing(1)
        top=QHBoxLayout();top.setSpacing(8)
        self.symbol=QLabel();self.symbol.setFixedWidth(18);top.addWidget(self.symbol)
        self.title=QLabel(tr('step_'+step));self.title.setObjectName('muted');top.addWidget(self.title,1)
        self.indicator=QProgressBar();self.indicator.setRange(0,0);self.indicator.setTextVisible(False);self.indicator.setFixedSize(24,3)
        self.indicator.setStyleSheet('QProgressBar {border:0;background:transparent} QProgressBar::chunk {background:'+ACTION+';}')
        policy=self.indicator.sizePolicy();policy.setRetainSizeWhenHidden(True);self.indicator.setSizePolicy(policy);self.indicator.hide();top.addWidget(self.indicator)
        self.status=QLabel(tr('pending'));self.status.setObjectName('muted');top.addWidget(self.status)
        box.addLayout(top)
        self.note=QLabel();self.note.setObjectName('muted');box.addWidget(self.note)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
    def mousePressEvent(self,event):
        if event.button()==Qt.MouseButton.LeftButton: self.clicked.emit(self.step)
        super().mousePressEvent(event)
    def refresh(self,group,elapsed=0,selected=False):
        from .ai_activity_ui import icon
        status=group.status
        key={'pending':'pending','start':'details_working_time','ok':'details_finished','warn':'details_finished_warning','fail':'details_failed','skip':'details_skipped'}[status]
        self.status.setText(tr(key,seconds=elapsed))
        colour={'ok':OK,'warn':WARNING,'fail':ERROR,'start':ACTION}.get(status,MUTED)
        self.status.setStyleSheet('color: '+colour)
        self.note.setStyleSheet('color: '+MUTED)
        self.symbol.setPixmap(icon({'ok':'check','warn':'warning','fail':'cross','skip':'dash'}.get(status,'dash')).pixmap(16,16))
        self.indicator.setVisible(status=='start')
        self.note.setText('')
        self.note.setVisible(status=='start')
        if status=='start' and group.messages:
            sentence=group.messages[-1][1]
            self.note.setText(self.note.fontMetrics().elidedText(sentence,Qt.TextElideMode.ElideRight,max(10,self.width()-16)))
            self.note.setToolTip(sentence)
        if self.property('selected')!=selected:
            self.setProperty('selected',selected);self.style().unpolish(self);self.style().polish(self)

class SetupWorkspace(QScrollArea):
    selected=Signal(str)
    def __init__(self,details):
        super().__init__();self.details=details;self.fallback=None;self.setWidgetResizable(True)
        self.content=QWidget();self.setWidget(self.content)
        self.content.setObjectName('setupWorkspaceContent')
        self.panes=QBoxLayout(QBoxLayout.Direction.LeftToRight,self.content);self.panes.setContentsMargins(0,0,0,0);self.panes.setSpacing(24)
        self.step_list=QWidget();self.step_list.setFixedWidth(440)
        self.step_list.setObjectName('setupWorkspaceContent')
        layout=QVBoxLayout(self.step_list);layout.setContentsMargins(0,0,0,0);layout.setSpacing(0)
        header=QHBoxLayout();header.addWidget(QLabel(tr('details_steps')),1)
        self.toggle=QPushButton(tr('show_readable_details'));self.toggle.setObjectName('textAction');self.toggle.setCheckable(True);self.toggle.toggled.connect(self.set_open);header.addWidget(self.toggle)
        layout.addLayout(header)
        self.rows=[]
        for step in ENGINE_STEPS:
            row=StepRow(step);row.clicked.connect(self.select);layout.addWidget(row);self.rows.append(row)
        self.step_list.setFixedHeight(34 + 50 * len(ENGINE_STEPS))
        self.panes.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        self.panes.addWidget(self.step_list,0,Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        self.panes.addWidget(details,1);details.hide()
        if hasattr(details,'selection_changed'): details.selection_changed.connect(self.refresh)
    def select(self,step):
        if hasattr(self.details,'select'): self.details.select(step)
        else: self.details.steps.setCurrentIndex(ENGINE_STEPS.index(step))
        self.toggle.setChecked(True);self.refresh();self.selected.emit(step)
    def set_open(self,opened):
        from .ai_activity_ui import icon
        self.toggle.setText(tr('hide_details' if opened else 'show_readable_details'))
        self.toggle.setIcon(icon('chevron-up' if opened else 'chevron-down'))
        from .motion import reveal
        reveal(self.details,opened)
        if self.fallback: self.fallback.setVisible(not opened)
        self.adapt()
    def adapt(self):
        narrow=self.window().width()<1200
        self.panes.setDirection(QBoxLayout.Direction.TopToBottom if narrow else QBoxLayout.Direction.LeftToRight)
        self.details.setMinimumHeight(320 if narrow and self.toggle.isChecked() else 0)
    def resizeEvent(self,event):
        super().resizeEvent(event);self.adapt();self.refresh()
    def refresh(self,elapsed=0):
        selected=getattr(self.details,'selected',None)
        for row in self.rows: row.refresh(self.details.model.groups[row.step],elapsed,row.step==selected)
