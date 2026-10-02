from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QCheckBox, QSpinBox, QLabel
from .strings import tr

def hours_description(start, end):
    return tr('all_day') if start == end else tr('hours_overnight' if end < start else 'hours_window', start=start, end=end)

class AlertHours(QWidget):
    def __init__(self):
        super().__init__()
        layout=QVBoxLayout(self)
        layout.setContentsMargins(0,0,0,0)
        self.all_day=QCheckBox(tr('all_day_choice'))
        self.all_day.setChecked(True)
        layout.addWidget(self.all_day)
        row=QHBoxLayout()
        self.start,self.end=QSpinBox(),QSpinBox()
        for key, spin in (('hours_from',self.start),('hours_to',self.end)):
            spin.setRange(0,23)
            spin.setSuffix(':00')
            row.addWidget(QLabel(tr(key)))
            row.addWidget(spin,1)
            spin.valueChanged.connect(self.update_hours)
        layout.addLayout(row)
        self.note=QLabel()
        self.note.setWordWrap(True)
        layout.addWidget(self.note)
        self.all_day.toggled.connect(self.update_hours)
        self.update_hours()
    def values(self):
        return (0,0) if self.all_day.isChecked() else (self.start.value(),self.end.value())
    def set_hours(self,start,end):
        self.start.setValue(start)
        self.end.setValue(end)
        self.all_day.setChecked(start==end)
        self.update_hours()
    def update_hours(self):
        if not hasattr(self,'note'): return
        for spin in (self.start,self.end): spin.setEnabled(not self.all_day.isChecked())
        self.note.setText(hours_description(*self.values()))
