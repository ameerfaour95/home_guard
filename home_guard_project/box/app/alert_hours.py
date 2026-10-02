from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QCheckBox, QSpinBox, QLabel
from .strings import tr

def hours_description(start, end):
    return tr('all_day') if start == end else tr('hours_overnight' if end < start else 'hours_window', start=start, end=end)

class HourBox(QSpinBox):
    def textFromValue(self,value):
        return f"{value:02d}:00"

class AlertHours(QWidget):
    def __init__(self):
        super().__init__()
        layout=QVBoxLayout(self)
        layout.setContentsMargins(0,0,0,0)
        self.all_day=QCheckBox(tr('all_day_choice'))
        self.all_day.setChecked(True)
        layout.addWidget(self.all_day)
        self.fields=QWidget()
        row=QHBoxLayout(self.fields)
        row.setContentsMargins(0,0,0,0)
        self.start,self.end=HourBox(),HourBox()
        for key, spin in (('hours_from',self.start),('hours_to',self.end)):
            spin.setRange(0,23)
            row.addWidget(QLabel(tr(key)))
            row.addWidget(spin,1)
            spin.valueChanged.connect(self.update_hours)
        layout.addWidget(self.fields)
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
        self.fields.setVisible(not self.all_day.isChecked())
        self.setMinimumHeight(64 if self.all_day.isChecked() else 120)
        self.note.setText(hours_description(*self.values()))
