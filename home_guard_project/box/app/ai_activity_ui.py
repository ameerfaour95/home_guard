import time,math
from pathlib import Path
from PySide6.QtCore import Qt,QTimer,QVariantAnimation,QRectF,QSize
from PySide6.QtGui import QPixmap,QPainter,QPainterPath,QIcon
from PySide6.QtWidgets import QWidget,QHBoxLayout,QScrollArea,QGraphicsOpacityEffect,QLabel
from .ai_view import status_note
from .timeline import merge_timeline,thinking_camera,image_path
from .strings import tr

def icon(name): return QIcon(str(Path(__file__).parent/'assets'/'icons'/(name+'.svg')))

class AlertPicture(QWidget):
    def __init__(self,path):
        super().__init__();self.pix=QPixmap(str(path)) if path else QPixmap();self.setMinimumHeight(144);self.setMaximumHeight(208)
    def paintEvent(self,event):
        if self.pix.isNull(): return
        painter=QPainter(self);painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        rect=QRectF(self.rect());clip=QPainterPath();clip.addRoundedRect(rect,10,10);painter.setClipPath(clip)
        scale=max(rect.width()/self.pix.width(),rect.height()/self.pix.height())
        w,h=self.pix.width()*scale,self.pix.height()*scale
        painter.drawPixmap(QRectF((rect.width()-w)/2,(rect.height()-h)/2,w,h),self.pix,QRectF(self.pix.rect()))

class AiActivity(QWidget):
    def __init__(self):
        super().__init__()
        from .ui import label,layout_for
        self.setObjectName('assistantColumn');self.setMinimumWidth(360);self.setMaximumWidth(640)
        outer=layout_for(self,16);outer.setSpacing(16)
        heading=QHBoxLayout();bot=QLabel();bot.setPixmap(icon('bot').pixmap(24,24));heading.addWidget(bot);heading.addWidget(label(tr('ai_assistant'),'section'),1);outer.addLayout(heading)
        self.note=label('','muted');outer.addWidget(self.note)
        self.scroll=QScrollArea();self.scroll.setWidgetResizable(True);outer.addWidget(self.scroll,1)
        self.opacity=QGraphicsOpacityEffect(self.scroll);self.scroll.setGraphicsEffect(self.opacity)
        self.thinking=label('','accent');outer.addWidget(self.thinking);self.thinking.hide()
        self.pulse=QGraphicsOpacityEffect(self.thinking);self.thinking.setGraphicsEffect(self.pulse)
        self.animation=QVariantAnimation(self);self.animation.setStartValue(.4);self.animation.setEndValue(1.0);self.animation.setDuration(1800);self.animation.setLoopCount(-1)
        self.animation.valueChanged.connect(lambda value:self.pulse.setOpacity(.7+.3*math.sin(float(value)*math.pi)))
        self.key=None
    def render(self,data,now,stopped,feed=(),image_dir=None,paused=None,count=0,refused=False):
        from .ui import label,layout_for,card
        rows=merge_timeline(data,list(feed));thinking=thinking_camera(data,now) if not stopped else ''
        self.thinking.setText(tr('ai_thinking',camera=thinking));self.thinking.setVisible(bool(thinking))
        if thinking and self.animation.state()!=QVariantAnimation.State.Running: self.animation.start()
        if not thinking: self.animation.stop()
        note=status_note(data,now,stopped) or (tr('ai_delivery_banner') if refused else tr('paused_until',time=time.strftime('%H:%M',time.localtime(paused))) if paused else tr('ai_watching',count=count))
        self.note.setText(note);self.note.setObjectName('error' if refused else 'muted')
        self.opacity.setOpacity(.45 if stopped else 1)
        key=tuple(rows)
        if key==self.key: return
        stick=self.key is None or self.scroll.verticalScrollBar().value()>=self.scroll.verticalScrollBar().maximum()-24
        value=self.scroll.verticalScrollBar().value();self.key=key
        content=QWidget();layout=layout_for(content,0);layout.setSpacing(16)
        if not rows: layout.addWidget(label(tr('ai_empty'),'muted'))
        day=None
        for record in rows:
            date=time.strftime('%d %b',time.localtime(record.ts))
            if date!=day:
                separator=label(date,'muted');separator.setAlignment(Qt.AlignmentFlag.AlignCenter);layout.addWidget(separator);day=date
            stamp=time.strftime('%H:%M',time.localtime(record.ts))
            if record.kind in ('quiet','button'):
                text=tr('ai_quiet_line',camera=record.camera.replace('_',' ').title(),text=record.text) if record.kind=='quiet' else tr('chat_button',name=record.name,text=record.text)
                note_row=QHBoxLayout();glyph=QLabel();glyph.setPixmap(icon('bot' if record.kind=='quiet' else 'check').pixmap(18,18));note_row.addWidget(glyph);line=label(text,'muted');note_row.addWidget(line,1);layout.addLayout(note_row);continue
            item=card();item.setObjectName('familyBubble' if record.who=='owner' else 'assistantBubble' if record.who=='assistant' else 'alertCard')
            if record.urgent: item.setProperty('urgent',True)
            row=layout_for(item,16);row.setSpacing(8)
            meta=QHBoxLayout();glyph=QLabel();glyph.setPixmap(icon('user' if record.who=='owner' else 'bot' if record.who=='assistant' else 'camera').pixmap(18,18));meta.addWidget(glyph);meta.addWidget(label(tr('chat_meta',name=record.name if record.who=='owner' else tr('ai_assistant') if record.who=='assistant' else record.camera.replace('_',' ').title(),time=stamp),'muted'),1);row.addLayout(meta)
            if record.who=='box' and record.image:
                path=image_path(image_dir,record.image) if image_dir else None
                row.addWidget(AlertPicture(path))
            message=label(record.text,'section' if record.who=='box' else None);row.addWidget(message)
            if record.who=='box':
                mark=label(tr('chat_delivered' if record.delivered else 'chat_refused'),'ok' if record.delivered else 'error');delivery=QHBoxLayout();glyph=QLabel();glyph.setPixmap(icon('check' if record.delivered else 'warning').pixmap(18,18));delivery.addWidget(glyph);delivery.addWidget(mark,1);row.addLayout(delivery)
                if not record.delivered: row.addWidget(label(record.error or tr('ai_no_error'),'error'))
            bubble=QHBoxLayout()
            if record.who=='owner': bubble.addStretch(1)
            bubble.addWidget(item,4)
            if record.who=='assistant': bubble.addStretch(1)
            layout.addLayout(bubble)
        layout.addStretch()
        old=self.scroll.takeWidget()
        if old: old.deleteLater()
        self.scroll.setWidget(content)
        QTimer.singleShot(0,lambda:self.scroll.verticalScrollBar().setValue(self.scroll.verticalScrollBar().maximum() if stick else value))
