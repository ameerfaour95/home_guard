import time
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QWidget,QVBoxLayout,QScrollArea,QGraphicsOpacityEffect
from .ai_view import decisions,status_note
from .strings import tr
from .theme import OK,ERROR,WARNING,MUTED

class AiActivity(QWidget):
    def __init__(self):
        super().__init__()
        from .ui import label,layout_for
        self.setMinimumWidth(360);self.setMaximumWidth(440)
        outer=layout_for(self,0)
        outer.addWidget(label(tr('ai_activity'),'section'))
        self.note=label('');outer.addWidget(self.note)
        self.scroll=QScrollArea();self.scroll.setWidgetResizable(True)
        outer.addWidget(self.scroll,1)
        self.opacity=QGraphicsOpacityEffect(self.scroll);self.scroll.setGraphicsEffect(self.opacity)
        self.key=None
    def render(self,data,now,stopped):
        from .ui import label,layout_for,card
        records=decisions(data)
        note=status_note(data,now,stopped)
        key=(tuple(records),note,stopped)
        if key==self.key: return
        self.key=key
        self.note.setText(note);self.note.setVisible(bool(note))
        self.note.setStyleSheet('color: '+(WARNING if not stopped else MUTED))
        self.opacity.setOpacity(.45 if stopped else 1)
        content=QWidget();layout=layout_for(content,0);layout.setSpacing(10)
        if not records:
            layout.addWidget(label(tr('ai_empty'),'muted'))
        for record in records:
            item=card();row=layout_for(item,12);row.setSpacing(7)
            def plain(text,style=''):
                widget=label(text,style);widget.setTextFormat(Qt.TextFormat.PlainText)
                return widget
            row.addWidget(plain(tr('ai_record_meta',time=time.strftime('%H:%M:%S',time.localtime(record.ts)),camera=record.camera.replace('_',' '),labels=record.seen()),'muted'))
            main=plain(record.summary or tr('ai_no_summary'));main.setStyleSheet('font-size: 16px;');row.addWidget(main)
            role,text=record.outcome();chip=plain(text)
            color={'ok':OK,'warning':WARNING,'error':ERROR,'muted':MUTED}[role]
            chip.setStyleSheet('color: '+color+'; background: #10151d; border-radius: 5px; padding: 6px; font-weight: 600;')
            row.addWidget(chip)
            if role=='error':
                error=plain(record.error or tr('ai_no_error'));error.setStyleSheet('color: '+ERROR);row.addWidget(error)
            layout.addWidget(item)
        layout.addStretch()
        old=self.scroll.takeWidget()
        if old: old.deleteLater()
        self.scroll.setWidget(content)
