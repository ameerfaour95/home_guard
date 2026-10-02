from html import escape
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QWidget,QVBoxLayout,QHBoxLayout,QComboBox,QLabel,QTextEdit,QPushButton,QCheckBox,QApplication,QScrollArea
from .strings import tr
from .theme import ERROR,WARNING,OK,MUTED
from .engine_backend import ENGINE_STEPS
from .setup_details import DetailsModel

class DetailsPanel(QWidget):
    def __init__(self):
        super().__init__()
        from .ui import label
        layout=QVBoxLayout(self);layout.setContentsMargins(0,0,0,0);layout.setSpacing(6)
        row=QHBoxLayout()
        self.steps=QComboBox();self.steps.currentIndexChanged.connect(self.render)
        row.addWidget(self.steps,1)
        self.technical=QCheckBox(tr('technical_log'));self.technical.toggled.connect(self.render)
        row.addWidget(self.technical)
        copy=QPushButton(tr('copy_log'));copy.setObjectName('secondary');copy.clicked.connect(self.copy)
        row.addWidget(copy);layout.addLayout(row)
        self.attention=label('', 'warning');layout.addWidget(self.attention)
        self.readable=label('');self.readable.setAlignment(__import__('PySide6.QtCore',fromlist=['Qt']).Qt.AlignmentFlag.AlignTop)
        scroll=QScrollArea();scroll.setWidgetResizable(True);scroll.setWidget(self.readable)
        self.readable_scroll=scroll;layout.addWidget(scroll,1)
        self.raw=QTextEdit();self.raw.setReadOnly(True);self.raw.setStyleSheet("font-family: Consolas; font-size: 10.5pt;");layout.addWidget(self.raw,1)
        self.setMinimumHeight(180);self.setMaximumHeight(16777215)
        self.reset()
    def reset(self):
        self.model=DetailsModel();self.technical.setChecked(False);self.steps.blockSignals(True);self.steps.clear()
        for step in ENGINE_STEPS: self.steps.addItem(tr('step_'+step))
        self.steps.blockSignals(False);self.render()
    def feed(self,event):
        self.model.feed(event)
        for i,step in enumerate(ENGINE_STEPS):
            group=self.model.groups[step]
            status={'pending':'pending','start':'running','ok':'PASS','warn':'WARN','fail':'FAIL','skip':'step_skip'}[group.status]
            self.steps.setItemText(i,tr('step_'+step)+tr('separator')+tr(status))
        if event.kind=='step':
            selected=self.model.groups[ENGINE_STEPS[max(0,self.steps.currentIndex())]]
            pinned=selected.status in ('warn','fail') or any(role in ('warning','error') for role,_ in selected.messages)
            if event.status in ('fail','warn') or (event.status=='start' and not self.model.failed and not pinned):
                self.steps.setCurrentIndex(ENGINE_STEPS.index(event.step))
            if event.status=='fail': self.technical.setChecked(False)
        self.render()
    def render(self):
        if not hasattr(self,'model'): return
        step=ENGINE_STEPS[max(0,self.steps.currentIndex())];group=self.model.groups[step]
        warnings=[(role,text) for role,text in group.messages if role in ('warning','error')]
        if group.status in ('fail','warn') and group.result: warnings.insert(0,('error' if group.status=='fail' else 'warning',group.result))
        self.attention.setTextFormat(Qt.TextFormat.RichText)
        self.attention.setText('<br>'.join('<span style="color:'+ (ERROR if role=='error' else WARNING)+'">'+escape(text)+'</span>' for role,text in dict.fromkeys(warnings)))
        self.attention.setVisible(bool(warnings))
        ordinary=[text for role,text in group.messages if role not in ('warning','error')]
        self.readable.setText('\n'.join(ordinary) or tr('details_empty'))
        self.readable_scroll.setVisible(not self.technical.isChecked())
        self.raw.setVisible(self.technical.isChecked())
        if self.technical.isChecked():
            self.raw.setPlainText(self.model.technical_log())
    def copy(self): QApplication.clipboard().setText(self.model.technical_log())
