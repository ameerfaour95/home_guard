from PySide6.QtCore import Qt, Signal, QTimer, QUrl
from PySide6.QtGui import QDesktopServices, QColor, QFont
from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QTabWidget, QStackedWidget, QApplication, QStyledItemDelegate, QSizePolicy
from .workers import TaskRunner
from .backend import AuthError, UnsupportedError
from .formatting import local_time
from .theme import PALETTES
from .widgets.common import label, button, Skeleton, EmptyState
from .widgets.data_table import RowsModel, data_table
from .collections import CollectionGrid, CreateCollection
from .export_wizard import ExportWizard


class StateDelegate(QStyledItemDelegate):
    def __init__(self, theme, parent):
        super().__init__(parent); self.tokens = PALETTES[theme]

    def paint(self, p, option, index):
        from PySide6.QtWidgets import QStyle
        from PySide6.QtCore import QRectF
        t, text = self.tokens,index.data()
        p.save(); p.fillRect(option.rect,QColor(t['raised' if option.state & QStyle.StateFlag.State_Selected else 'surface']))
        r = QRectF(option.rect.x()+12,option.rect.center().y()-13,82,26)
        p.setPen(Qt.PenStyle.NoPen); p.setBrush(QColor(t['raised'])); p.drawRoundedRect(r,5,5)
        p.setPen(QColor(t[{'Ready':'action','Running':'action','Queued':'muted','Partial':'warning','Failed':'error'}.get(text,'muted')]))
        p.setFont(QFont('Segoe UI',9)); p.drawText(r,Qt.AlignmentFlag.AlignCenter,text); p.restore()


class StudioScreen(QWidget):
    filter_requested = Signal(str)
    event_requested = Signal(int)
    session_expired = Signal()

    def __init__(self, backend, role='admin', theme='dark'):
        super().__init__(); self.backend,self.role = backend,role
        self.filters,self.collections,self.exports,self.counts = [],[],[],{}
        self.loaded_once = False
        self.pending_export = False
        self.refresh_dirty = False
        box = QVBoxLayout(self); box.setContentsMargins(32,24,32,20); box.setSpacing(16)
        top = QHBoxLayout(); titles = QVBoxLayout(); titles.addWidget(label('Training Studio','title'))
        titles.addWidget(label('From reviewed moments to trusted training data.','muted')); top.addLayout(titles,1)
        self.refresh_button = button('Refresh',self.refresh); top.addWidget(self.refresh_button)
        self.export_button = button('Export collection…',self.open_export,'primary'); self.export_button.setVisible(role != 'support'); top.addWidget(self.export_button)
        box.addLayout(top)
        self.message = label('Loading Studio…','muted',True); box.addWidget(self.message)
        self.tabs = QTabWidget(); self.tabs.tabBar().setDrawBase(False)
        self.content_stack = QStackedWidget(); self.content_stack.setSizePolicy(QSizePolicy.Policy.Expanding,QSizePolicy.Policy.Ignored)
        self.content_stack.addWidget(self.tabs)
        self.loading = Skeleton(theme); self.content_stack.addWidget(self.loading)
        self.failure = EmptyState('Studio could not be loaded','Check your connection and try again.',eyebrow='UNAVAILABLE')
        self.failure.action.show(); self.failure.action.clicked.connect(self.refresh); self.content_stack.addWidget(self.failure)
        self.unsupported = EmptyState('Studio is not available yet', 'This server has not enabled Studio lists and exports yet.', eyebrow='TRAINING STUDIO')
        self.content_stack.addWidget(self.unsupported)
        self.content_stack.setCurrentWidget(self.loading); box.addWidget(self.content_stack,1)
        filters_page = QWidget(); filters = QVBoxLayout(filters_page); filters.setContentsMargins(0,12,0,0); filters.setSpacing(14)
        filters.addWidget(label('A focused place to start','section'))
        filters.addWidget(label('Built-in views from Cloud. Select a view to check for matches; open it to begin reviewing.','muted',True))
        self.filter_model = RowsModel([('Saved view',lambda f:f.title),('What to review',lambda f:f.description),('Live matches',lambda f:self.counts.get(f.key,'Select to check'))])
        self.filter_table = data_table(self.filter_model,1); self.filter_table.setColumnWidth(0,280); self.filter_table.setColumnWidth(2,160)
        self.filter_table.verticalHeader().setDefaultSectionSize(46)
        filters.addWidget(self.filter_table,1)
        self.filter_table.selectionModel().currentRowChanged.connect(self.count_filter)
        self.filter_table.doubleClicked.connect(self.open_filter)
        action = QHBoxLayout(); action.addWidget(label('Counts are checked on demand. “1+” means more matches are available.','muted')); action.addStretch()
        action.addWidget(button('Open in Review',self.open_filter,'primary')); filters.addLayout(action)
        self.tabs.addTab(filters_page,'Saved filters')
        collections_page = QWidget(); collections = QVBoxLayout(collections_page); collections.setContentsMargins(0,12,0,0); collections.setSpacing(12)
        tools = QHBoxLayout(); self.back = button('← All collections',self.close_collection,'link'); self.back.hide(); tools.addWidget(self.back)
        tools.addStretch(); tools.addWidget(button('New collection',self.new_collection)); collections.addLayout(tools)
        self.collection_stack = QStackedWidget(); collections.addWidget(self.collection_stack,1)
        self.collection_model = RowsModel([('Collection',lambda c:c.name),('Description',lambda c:c.description),('Events',lambda c:c.event_count),('Created by',lambda c:c.created_by),('Created',lambda c:local_time(c.created_utc,'UTC'))])
        self.collection_table = data_table(self.collection_model,1)
        for i,w in enumerate([270,360,80,160,190]): self.collection_table.setColumnWidth(i,w)
        self.collection_table.doubleClicked.connect(self.open_collection)
        self.collection_stack.addWidget(self.collection_table)
        self.grid = CollectionGrid(backend,theme); self.grid.changed.connect(self.refresh); self.grid.event_requested.connect(self.event_requested)
        self.collection_stack.addWidget(self.grid)
        self.open_collection_button = button('Open selected collection',self.open_collection,'link')
        collections.addWidget(self.open_collection_button,alignment=Qt.AlignmentFlag.AlignLeft)
        self.tabs.addTab(collections_page,'Collections')
        history = QWidget(); exports = QVBoxLayout(history); exports.setContentsMargins(0,12,0,0); exports.setSpacing(12)
        exports.addWidget(label('Versioned datasets','section'))
        exports.addWidget(label('Each export freezes its source events, saved evidence and household split.','muted'))
        self.export_model = RowsModel([('Export',lambda e:e.name),('State',lambda e:e.state.title()),('Version',lambda e:f'v{e.version}'),('Items',lambda e:e.item_count),('Created by',lambda e:e.created_by),('Created · UTC',lambda e:local_time(e.created_utc,'UTC'))])
        self.export_table = data_table(self.export_model)
        self.export_table.verticalHeader().setDefaultSectionSize(48)
        for i,w in enumerate([280,120,85,80,160,200]): self.export_table.setColumnWidth(i,w)
        self.export_table.setItemDelegateForColumn(1,StateDelegate(theme,self.export_table)); exports.addWidget(self.export_table,1)
        self.export_detail = label('Select an export to see its manifest and storage path.','muted',True); exports.addWidget(self.export_detail)
        actions = QHBoxLayout(); self.copy = button('Copy S3 path',self.copy_path); self.manifest = button('Open manifest',self.open_manifest)
        self.copy.setEnabled(False); self.manifest.setEnabled(False); actions.addWidget(self.copy); actions.addWidget(self.manifest); actions.addStretch(); exports.addLayout(actions)
        self.export_table.selectionModel().currentRowChanged.connect(self.export_selected)
        self.tabs.addTab(history,'Export history'); self.tabs.setTabVisible(2,role != 'support')
        self.runner = TaskRunner(self); self.runner.finished.connect(self.loaded)
        self.count_runner = TaskRunner(self); self.count_runner.finished.connect(self.count_loaded)
        self.poll = QTimer(self); self.poll.setInterval(15000); self.poll.timeout.connect(self.poll_exports)

    def showEvent(self,event):
        super().showEvent(event)
        if not self.loaded_once: self.refresh()
        self.poll.start()

    def hideEvent(self,event):
        self.poll.stop(); super().hideEvent(event)

    def refresh(self):
        if self.runner.busy:
            self.refresh_dirty = True
            return
        self.refresh_dirty = False
        self.counts.clear()
        self.refresh_button.setEnabled(False)
        def fetch():
            return self.backend.saved_filters(),self.backend.collections(),self.backend.exports() if self.role != 'support' else []
        self.runner.start(fetch)

    def loaded(self,result,error):
        self.refresh_button.setEnabled(True)
        if self.refresh_dirty and not isinstance(error, AuthError):
            self.refresh()
            return
        if error:
            self.message.setText(str(error)+' · Select Refresh to retry.'); self.message.show()
            if not self.loaded_once: self.content_stack.setCurrentWidget(self.failure)
            if isinstance(error, UnsupportedError): self.content_stack.setCurrentWidget(self.unsupported)
            if isinstance(error,AuthError): self.session_expired.emit()
            return
        self.loaded_once = True; self.filters,self.collections,self.exports = result
        self.content_stack.setCurrentWidget(self.tabs)
        self.filter_model.replace(self.filters); self.collection_model.replace(self.collections)
        selected = self.current_export()
        self.export_model.replace(self.exports)
        if self.exports:
            row = next((i for i,e in enumerate(self.exports) if selected and e.id == selected.id),0)
            self.export_table.setCurrentIndex(self.export_model.index(row,0))
        self.message.setText(f'{len(self.filters)} saved views   /   {len(self.collections)} collections'+(f'   /   {len(self.exports)} exports' if self.role != 'support' else ''))
        self.export_button.setEnabled(bool(self.collections))
        if self.pending_export:
            self.pending_export = False; self.open_export()

    def count_filter(self,current,previous):
        if not current.isValid(): return
        key = self.filter_model.items[current.row()].key
        self.count_requested = key
        self.request_count()

    def request_count(self):
        if self.count_runner.busy: return
        key = self.count_requested
        if key in self.counts: return
        self.count_pending = key
        self.count_runner.start(lambda:self.backend.events(filter=key,limit=1,with_total=True))

    def count_loaded(self,page,error):
        key = self.count_pending
        self.counts[key] = 'Unavailable' if error or page.total is None else '10,000+' if page.total_capped else f'{page.total:,}'
        if isinstance(error, AuthError): self.session_expired.emit()
        self.filter_model.dataChanged.emit(self.filter_model.index(0,2),self.filter_model.index(max(0,len(self.filters)-1),2))
        if self.count_requested != key: self.request_count()

    def open_filter(self,*_):
        index = self.filter_table.currentIndex()
        if index.isValid(): self.filter_requested.emit(self.filter_model.items[index.row()].key)

    def new_collection(self):
        self.create_dialog = CreateCollection(self.backend,self); self.create_dialog.created.connect(lambda _:self.refresh()); self.create_dialog.show()
        self.create_dialog.session_expired.connect(self.session_expired)

    def open_collection(self,*_):
        index = self.collection_table.currentIndex()
        if index.isValid():
            self.grid.session_expired.connect(self.session_expired, Qt.ConnectionType.UniqueConnection)
            self.grid.open(self.collection_model.items[index.row()]); self.collection_stack.setCurrentWidget(self.grid); self.back.show()
            self.open_collection_button.hide()

    def close_collection(self):
        self.collection_stack.setCurrentWidget(self.collection_table); self.back.hide()
        self.open_collection_button.show()

    def open_export(self):
        if self.role == 'support': return
        if not self.loaded_once and self.runner.busy:
            self.pending_export = True; return
        if not self.collections:
            self.message.setText('Create a collection before exporting.'); return
        selected = self.grid.collection.id if self.collection_stack.currentWidget() is self.grid and self.grid.collection else None
        self.wizard = ExportWizard(self.backend,self.collections,selected,self)
        self.wizard.session_expired.connect(self.session_expired)
        self.wizard.exported.connect(self.export_created); self.wizard.show()

    def export_created(self,export):
        self.tabs.setCurrentIndex(2); self.refresh()

    def current_export(self):
        index = self.export_table.currentIndex()
        return self.export_model.items[index.row()] if index.isValid() else None

    def export_selected(self,*_):
        e = self.current_export()
        self.copy.setEnabled(bool(e and e.s3_prefix)); self.manifest.setEnabled(bool(e and e.manifest_url))
        if e:
            self.export_detail.setText(e.error or e.s3_prefix or 'Cloud is preparing this dataset. This view refreshes every 15 seconds.')

    def copy_path(self):
        e = self.current_export()
        if e and e.s3_prefix: QApplication.clipboard().setText(e.s3_prefix)

    def open_manifest(self):
        e = self.current_export()
        if e and e.manifest_url:
            url = QUrl(e.manifest_url)
            if url.scheme() in ('https','http'): QDesktopServices.openUrl(url)

    def poll_exports(self):
        if self.tabs.currentIndex() == 2 and self.role != 'support' and any(e.state in ('queued','running') for e in self.exports): self.refresh()
