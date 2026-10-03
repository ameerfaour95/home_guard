import json
from dataclasses import asdict
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLineEdit, QSplitter, QPlainTextEdit, QStackedWidget, QSizePolicy, QComboBox
from .widgets.common import label, button, Skeleton, EmptyState
from .widgets.data_table import RowsModel, data_table
from .workers import TaskRunner
from .backend import AuthError, UnsupportedError
from .formatting import local_time


def action_words(action):
    return {'recording_access':'Viewed a recording','artifact_access':'Opened saved evidence','collection_create':'Created a collection',
            'collection_add':'Added collection events', 'collection_remove':'Removed collection events',
            'export_create':'Created a training export','event_review':'Reviewed an event','auth_login':'Signed in',
            'dev_seed':'Prepared local sample data', 'event_view':'Opened an event', 'customer_update':'Updated a customer',
            'thumbnail_access':'Viewed a preview', 'export_view':'Opened an export'}.get(action.replace('.','_'),action.replace('.',' ').replace('_',' ').capitalize())


class ActionFilter(QComboBox):
    def __init__(self):
        super().__init__(); self.addItem('All actions', '')
        for key in ('recording_access','artifact_access','event_view','event_review','collection_create',
                    'collection_add','collection_remove','export_create','export_view','auth_login','customer_update'):
            self.addItem(action_words(key), key)

    def text(self): return self.currentData() or ''

    def setText(self, value):
        if self.findData(value) < 0: self.addItem(action_words(value), value)
        self.setCurrentIndex(self.findData(value))

    def clear(self): self.setCurrentIndex(0)


class AuditScreen(QWidget):
    session_expired = Signal()

    def __init__(self,backend,theme='dark'):
        super().__init__(); self.backend = backend
        self.cursor,self.generation,self.loaded_once = None,0,False
        box = QVBoxLayout(self); box.setContentsMargins(32,24,32,20); box.setSpacing(16)
        box.addWidget(label('Audit','title')); box.addWidget(label('A record of staff access and changes across Home Guard.','muted'))
        filters = QHBoxLayout(); self.filters = {}
        for key,placeholder in [('staff','Staff ID or email'),('customer_id','Customer ID'),('action','Action')]:
            edit = ActionFilter() if key == 'action' else QLineEdit()
            if key != 'action':
                edit.setPlaceholderText(placeholder); edit.returnPressed.connect(self.reload)
            edit.setAccessibleName(placeholder)
            self.filters[key] = edit; filters.addWidget(edit, 1)
        filters.addWidget(button('Apply filters',self.reload)); filters.addWidget(button('Reset',self.reset,'link')); box.addLayout(filters)
        self.message = label('Loading audit…','muted',True); box.addWidget(self.message)
        self.model = RowsModel([('Time · UTC',lambda a:local_time(a.ts,'UTC')),('Staff',lambda a:a.staff),('Action',lambda a:action_words(a.action)),
                               ('Customer',lambda a:str(a.customer_id) if a.customer_id is not None else '—'),('Target',lambda a:a.target),('Reason',lambda a:a.reason or '—')])
        self.table = data_table(self.model,5)
        for i,w in enumerate([195,155,220,90,160,240]): self.table.setColumnWidth(i,w)
        self.table.selectionModel().currentRowChanged.connect(self.select)
        self.table.verticalScrollBar().valueChanged.connect(self.scroll)
        split = QSplitter(Qt.Orientation.Horizontal); split.addWidget(self.table)
        self.drawer = QWidget(); drawer = QVBoxLayout(self.drawer); drawer.setContentsMargins(20,0,0,0); drawer.setSpacing(14)
        top = QHBoxLayout(); self.title = label('Audit entry','section'); top.addWidget(self.title,1); top.addWidget(button('Close',self.drawer.hide,'link')); drawer.addLayout(top)
        self.details = label('','muted',True); drawer.addWidget(self.details)
        drawer.addWidget(label('ENTRY JSON','eyebrow'))
        self.json = QPlainTextEdit(); self.json.setReadOnly(True); self.json.setStyleSheet('font-family: Consolas; font-size: 9pt;'); drawer.addWidget(self.json,1)
        drawer.addWidget(label('Saved detail is included in the entry JSON.','muted',True))
        self.drawer.setMinimumWidth(320); split.addWidget(self.drawer); self.drawer.hide(); split.setSizes([1000,380])
        self.content = split; self.stack = QStackedWidget(); self.stack.setSizePolicy(QSizePolicy.Policy.Expanding,QSizePolicy.Policy.Ignored)
        self.stack.addWidget(split); self.loading = Skeleton(theme); self.stack.addWidget(self.loading)
        self.failure = EmptyState('Audit could not be loaded','Check your connection, then retry. Your filters are saved.',eyebrow='UNAVAILABLE')
        self.failure.action.show(); self.failure.action.clicked.connect(self.reload); self.stack.addWidget(self.failure)
        self.unsupported = EmptyState('Audit is not available yet', 'This server has not enabled the audit list yet.', eyebrow='AUDIT')
        self.stack.addWidget(self.unsupported)
        self.empty = EmptyState('No audit entries match','Try another staff member, customer or action.',eyebrow='AUDIT')
        self.empty.action.setText('Reset filters'); self.empty.action.show(); self.empty.action.clicked.connect(self.reset); self.stack.addWidget(self.empty)
        box.addWidget(self.stack,1)
        footer = QHBoxLayout(); self.count = label('','muted'); footer.addWidget(self.count); footer.addStretch()
        self.more = button('Load older',self.load_more); footer.addWidget(self.more); box.addLayout(footer)
        self.runner = TaskRunner(self); self.runner.finished.connect(self.loaded)

    def showEvent(self,event):
        super().showEvent(event)
        if not self.loaded_once: self.reload()

    def reset(self):
        for edit in self.filters.values(): edit.clear()
        self.reload()

    def reload(self):
        self.generation += 1; self.cursor = None; self.drawer.hide(); self.request(False)

    def request(self,append):
        if self.runner.busy: return
        query = {k:e.text().strip() or None for k,e in self.filters.items()}
        if query['customer_id']:
            try: query['customer_id'] = int(query['customer_id'])
            except ValueError:
                self.message.setText('Customer ID must be a number.'); return
        query['cursor'] = self.cursor if append else None
        self.pending = self.generation,append
        if not append: self.stack.setCurrentWidget(self.loading)
        self.more.setEnabled(False); self.message.setText('Loading older entries…' if append else 'Loading audit…')
        self.runner.start(lambda:self.backend.audit(**query))

    def loaded(self,page,error):
        generation,append = self.pending
        if generation != self.generation:
            self.request(False); return
        if error:
            self.message.setText(str(error)+' · Apply filters to retry.'); self.more.setEnabled(bool(self.cursor))
            self.stack.setCurrentWidget(self.content if append else self.failure)
            if isinstance(error, UnsupportedError): self.stack.setCurrentWidget(self.unsupported)
            if isinstance(error,AuthError): self.session_expired.emit()
            return
        self.loaded_once = True; self.cursor = page.next_cursor
        existing = {e.id for e in self.model.items} if append else set()
        self.model.replace([e for e in page.items if e.id not in existing],append)
        self.stack.setCurrentWidget(self.content if self.model.items else self.empty)
        self.message.setText('Select a row to inspect the saved entry.' if self.model.items else 'No audit entries match these filters.')
        self.more.setEnabled(bool(self.cursor)); self.more.setText('Load older' if self.cursor else 'End of history')
        self.count.setText(f'{len(self.model.items)} entries loaded · Newest first')

    def load_more(self):
        if self.cursor: self.request(True)

    def scroll(self,value):
        bar = self.table.verticalScrollBar()
        if bar.maximum() > 0 and value >= bar.maximum()-1: self.load_more()

    def select(self,current,previous):
        if not current.isValid(): return
        entry = self.model.items[current.row()]; self.title.setText(f'Entry #{entry.id}')
        self.details.setText(f'{action_words(entry.action)}\n{entry.staff}\n{local_time(entry.ts,"UTC")}\n\n{entry.reason}')
        self.json.setPlainText(json.dumps(asdict(entry),indent=2,default=str,ensure_ascii=False)); self.drawer.show()
