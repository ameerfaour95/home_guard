from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (QDialog, QWidget, QVBoxLayout, QHBoxLayout, QLineEdit,
    QComboBox, QCheckBox, QSlider, QStackedWidget)
from .workers import TaskRunner
from .backend import AuthError
from collections import Counter
from .widgets.common import label, button
from .widgets.data_table import RowsModel, data_table


from .export_logic import validate_export


class ExportWizard(QDialog):
    exported = Signal(object)
    session_expired = Signal()

    def __init__(self, backend, collections, selected=None, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        self.backend, self.collections = backend, collections
        self.preview, self.preview_error, self.step = None, '', 0
        self.setWindowTitle('Export training dataset'); self.setModal(True); self.resize(760, 650)
        box = QVBoxLayout(self); box.setContentsMargins(32,28,32,24); box.setSpacing(18)
        box.addWidget(label('Export training dataset', 'title'))
        self.steps = label('', 'eyebrow'); box.addWidget(self.steps)
        self.pages = QStackedWidget(); box.addWidget(self.pages,1)
        first = QWidget(); form = QVBoxLayout(first); form.setContentsMargins(0,0,0,0); form.setSpacing(12)
        form.addWidget(label('Collection', 'muted')); self.collection = QComboBox()
        for c in collections:
            self.collection.addItem(f'{c.name} · {c.event_count} events', c.id)
        if selected is not None:
            self.collection.setCurrentIndex(max(0,self.collection.findData(selected)))
        form.addWidget(self.collection)
        form.addWidget(label('Export name', 'muted')); self.name = QLineEdit(); self.name.setPlaceholderText('entrance_october'); form.addWidget(self.name)
        form.addWidget(label('Lowercase letters, numbers, underscores and hyphens. Each export gets a new version.', 'muted', True))
        form.addSpacing(12); form.addWidget(label('Include in the dataset', 'section'))
        self.formats = {}
        for key, title, detail in [('yolo','YOLO labels','Detection boxes with saved provenance'),('vlm_jsonl','VLM JSONL','Saved model answers and input frames'),('clips','Video clips','Original moments for review and training')]:
            check = QCheckBox(title); check.setChecked(True); self.formats[key] = check
            form.addWidget(check); form.addWidget(label(detail,'muted'))
        form.addStretch(); self.pages.addWidget(first)
        second = QWidget(); split = QVBoxLayout(second); split.setContentsMargins(0,0,0,0); split.setSpacing(18)
        split.addWidget(label('Keep each household together', 'section'))
        split.addWidget(label('Split is by house and day, never mixed. Related footage stays in one partition to prevent leakage.', 'muted', True))
        self.sliders, self.values = {}, {}
        for key, title, value in [('train','Training',80),('val','Validation',10),('test','Test',10)]:
            row = QHBoxLayout(); row.addWidget(label(title)); row.addStretch()
            self.values[key] = label(f'{value}%', 'badge'); row.addWidget(self.values[key]); split.addLayout(row)
            slider = QSlider(Qt.Orientation.Horizontal); slider.setRange(0,100); slider.setValue(value); slider.setAccessibleName(title+' percentage')
            self.sliders[key] = slider; slider.valueChanged.connect(lambda value,k=key:self.balance(k,value)); split.addWidget(slider)
        self.total = label('Total 100%', 'muted'); split.addWidget(self.total)
        self.fallback = QCheckBox('Include fallback AI'); split.addWidget(self.fallback)
        split.addWidget(label('Off by default. Fallback and failed AI are excluded only from vlm.jsonl. These events still contribute video clips and YOLO labels when those formats are selected.', 'muted', True))
        split.addStretch(); self.pages.addWidget(second)
        third = QWidget(); consent = QVBoxLayout(third); consent.setContentsMargins(0,0,0,0); consent.setSpacing(12)
        consent.addWidget(label('Check consent before exporting', 'section'))
        self.summary = label('Checking collection and consent…', 'muted', True); consent.addWidget(self.summary)
        self.exclusions = RowsModel([('Excluded because',lambda r:r[0]),('Events',lambda r:str(r[1]))])
        self.table = data_table(self.exclusions,0); self.table.setColumnWidth(1,100); consent.addWidget(self.table,1)
        self.table.verticalHeader().setDefaultSectionSize(44)
        self.check = QCheckBox('Export the eligible events shown above'); consent.addWidget(self.check)
        self.check.toggled.connect(self.update_confirm)
        consent.addWidget(label('Consent is checked again by Cloud when the export runs. A versioned manifest records the source events and split.', 'muted', True))
        self.pages.addWidget(third)
        self.error = label('', 'error', True); box.addWidget(self.error)
        actions = QHBoxLayout(); self.back = button('Back',self.previous); actions.addWidget(self.back)
        actions.addStretch(); actions.addWidget(button('Cancel',self.reject)); self.next = button('Continue',self.advance,'primary'); actions.addWidget(self.next); box.addLayout(actions)
        self.runner = TaskRunner(self); self.runner.finished.connect(self.preview_loaded)
        self.writer = TaskRunner(self); self.writer.finished.connect(self.created)
        self.set_step(0)

    def balance(self, key, value):
        # Retain one other partition where possible; the final partition absorbs the remainder.
        others = [k for k in self.sliders if k != key]
        a = min(self.sliders[others[0]].value(),100-value)
        for k,v in [(others[0],a),(others[1],100-value-a)]:
            self.sliders[k].blockSignals(True); self.sliders[k].setValue(v); self.sliders[k].blockSignals(False)
        for k,slider in self.sliders.items():
            self.values[k].setText(f'{slider.value()}%')
        self.total.setText(f'Total {sum(s.value() for s in self.sliders.values())}%')

    def request(self):
        return dict(collection_id=self.collection.currentData(), name=self.name.text().strip(),
                    formats=[k for k,c in self.formats.items() if c.isChecked()],
                    split={k:s.value()/100 for k,s in self.sliders.items()}, include_fallback_ai=self.fallback.isChecked())

    def set_step(self, step):
        self.step = step; self.pages.setCurrentIndex(step)
        self.steps.setText('   /   '.join(('● ' if i == step else '')+s for i,s in enumerate(['1  Dataset','2  Split & evidence','3  Consent & confirm'])))
        self.back.setEnabled(step > 0); self.next.setText('Create export' if step == 2 else 'Continue'); self.next.setEnabled(True)
        self.error.clear()
        if step == 2:
            self.preview = None; self.check.setChecked(False); self.next.setEnabled(False); self.back.setEnabled(False)
            self.summary.setText('Checking collection and consent…'); self.exclusions.replace([])
            request = self.request()
            self.runner.start(lambda: self.backend.export_preview(**request))

    def advance(self):
        request = self.request()
        error = validate_export(request['name'],request['formats'],request['split'])
        if request['collection_id'] is None:
            error = 'Create a collection before exporting.'
        if error:
            self.error.setText(error); return
        if self.step < 2:
            self.set_step(self.step+1); return
        if not self.preview or not self.preview.included_ids or not self.check.isChecked() or self.writer.busy:
            return
        self.next.setEnabled(False); self.back.setEnabled(False); self.check.setEnabled(False)
        self.writer.start(lambda:self.backend.create_export(**request))

    def previous(self):
        self.set_step(max(0,self.step-1))

    def preview_loaded(self, result, error):
        self.back.setEnabled(True); self.preview = result
        if error or result is None:
            if isinstance(error, AuthError): self.session_expired.emit()
            self.summary.setText(str(error) if error else 'Consent preview is unavailable with this server version. Collection membership and per-event training eligibility are required before you can confirm an export.')
            self.check.setEnabled(False); return
        self.check.setEnabled(True)
        eligible, excluded = len(result.included_ids), len(result.excluded)
        splits = ' · '.join(f'{key}: {value}' for key, value in result.split_counts.items())
        self.summary.setText(f'{eligible} included events · {excluded} excluded · {result.groups} house/day groups\n{splits}' + ''.join('\n'+w for w in result.warnings))
        self.exclusions.replace([(key.replace('_', ' ').capitalize(), count) for key, count in Counter(e.reason for e in result.excluded).items()])
        self.check.setText(f'Export the {eligible} eligible events')
        self.update_confirm()

    def update_confirm(self):
        self.next.setEnabled(bool(self.preview and self.preview.included_ids and self.check.isChecked() and not self.writer.busy))

    def created(self, export, error):
        self.back.setEnabled(True); self.check.setEnabled(True)
        if error:
            if isinstance(error, AuthError): self.session_expired.emit()
            self.error.setText(str(error)); self.update_confirm(); return
        self.exported.emit(export); self.accept()
