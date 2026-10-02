import time,math
from pathlib import Path
from PySide6.QtCore import Qt,QTimer,QVariantAnimation,QRectF,QSize,QEvent
from PySide6.QtGui import QPixmap,QPainter,QPainterPath,QIcon
from PySide6.QtWidgets import QWidget,QHBoxLayout,QScrollArea,QGraphicsOpacityEffect,QLabel,QPushButton,QVBoxLayout
from .ai_view import status_note
from .timeline import merge_timeline,thinking_camera,image_path,group_quiet,QuietGroup,text_direction
from .strings import tr

def icon(name): return QIcon(str(Path(__file__).parent/'assets'/'icons'/(name+'.svg')))

class AlertPicture(QWidget):
    def __init__(self,path):
        super().__init__();self.pix=QPixmap(str(path)) if path else QPixmap();self.setMinimumHeight(120)
    def resizeEvent(self,event):
        height=max(120,round(self.width()*9/16))
        if self.height()!=height: self.setFixedHeight(height)
        super().resizeEvent(event)
    def paintEvent(self,event):
        if self.pix.isNull():
            painter=QPainter(self);painter.drawText(self.rect(),Qt.AlignmentFlag.AlignCenter,tr("chat_image_unavailable"));return
        painter=QPainter(self);painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        rect=QRectF(self.rect());clip=QPainterPath();clip.addRoundedRect(rect,10,10);painter.setClipPath(clip)
        scale=max(rect.width()/self.pix.width(),rect.height()/self.pix.height())
        w,h=self.pix.width()*scale,self.pix.height()*scale
        painter.drawPixmap(QRectF((rect.width()-w)/2,(rect.height()-h)/2,w,h),self.pix,QRectF(self.pix.rect()))

class AiActivity(QWidget):
    def __init__(self):
        super().__init__()
        from .ui import label,layout_for
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground,True);self.setObjectName('assistantColumn');self.setMinimumWidth(360);self.setMaximumWidth(640)
        outer=layout_for(self,16);outer.setSpacing(16)
        heading=QHBoxLayout();bot=QLabel();bot.setPixmap(icon('bot').pixmap(24,24));heading.addWidget(bot);heading.addWidget(label(tr('ai_assistant'),'section'),1);outer.addLayout(heading)
        self.note=label('','muted');outer.addWidget(self.note)
        self.sensitivity=label('','muted');outer.addWidget(self.sensitivity)
        self.scroll=QScrollArea();self.scroll.setWidgetResizable(True);outer.addWidget(self.scroll,1)
        self.opacity=QGraphicsOpacityEffect(self.scroll);self.scroll.setGraphicsEffect(self.opacity)
        self.thinking=label('','accent');thinking_row=QHBoxLayout();self.thinking_icon=QLabel();self.thinking_icon.setPixmap(icon('bot').pixmap(18,18));thinking_row.addWidget(self.thinking_icon);thinking_row.addWidget(self.thinking,1);outer.addLayout(thinking_row);self.thinking.hide();self.thinking_icon.hide()
        self.pulse=QGraphicsOpacityEffect(self.thinking);self.thinking.setGraphicsEffect(self.pulse)
        self.animation=QVariantAnimation(self);self.animation.setStartValue(0.0);self.animation.setEndValue(2*math.pi);self.animation.setDuration(1800);self.animation.setLoopCount(-1)
        self.animation.valueChanged.connect(lambda value:self.pulse.setOpacity(.7+.3*math.sin(float(value))))
        self.key=None
        self.expanded_groups=set()
        self.sticky_day=label(tr("today"),"muted");self.sticky_day.setAlignment(Qt.AlignmentFlag.AlignCenter);outer.insertWidget(3,self.sticky_day)
        self.follow_bottom=True
        self.scroll.viewport().installEventFilter(self)
        bar=self.scroll.verticalScrollBar()
        bar.actionTriggered.connect(lambda action:QTimer.singleShot(0,self.user_scrolled))
        bar.sliderPressed.connect(lambda:setattr(self,'follow_bottom',False))
        bar.sliderReleased.connect(self.user_scrolled)
        bar.rangeChanged.connect(lambda low,high:bar.setValue(high) if self.follow_bottom else None)
        bar.valueChanged.connect(self.update_day_header)
    def user_scrolled(self):
        bar=self.scroll.verticalScrollBar();self.follow_bottom=bar.value()>=bar.maximum()-24
    def eventFilter(self,obj,event):
        if event.type()==QEvent.Type.Wheel: QTimer.singleShot(0,self.user_scrolled)
        return super().eventFilter(obj,event)
    def render(self,data,now,stopped,feed=(),image_dir=None,paused=None,count=0,refused=False):
        from .ui import label,layout_for,card
        rows=merge_timeline(data,list(feed));thinking=thinking_camera(data,now) if not stopped else ''
        self.thinking.setText(tr('ai_thinking',camera=thinking));self.thinking.setVisible(bool(thinking));self.thinking_icon.setVisible(bool(thinking))
        if thinking and self.animation.state()!=QVariantAnimation.State.Running: self.animation.start()
        if not thinking: self.animation.stop()
        note=status_note(data,now,stopped) or (tr('ai_delivery_banner') if refused else tr('paused_until',time=time.strftime('%H:%M',time.localtime(paused))) if paused else tr('ai_watching',count=count))
        settings=data.get("settings",{}) if isinstance(data,dict) else {}
        try: conf=float(settings["conf"])
        except (KeyError,TypeError,ValueError): conf=None
        self.sensitivity.setVisible(conf is not None and math.isfinite(conf))
        if conf is not None and math.isfinite(conf): self.sensitivity.setText(tr("sensitivity_in_force",percent=round(conf*100)))
        self.note.setText(note);self.note.setObjectName('error' if refused else 'muted');self.note.style().unpolish(self.note);self.note.style().polish(self.note)
        self.opacity.setOpacity(.45 if stopped else 1)
        key=tuple(rows)
        if key==self.key: return
        stick=self.follow_bottom
        value=self.scroll.verticalScrollBar().value();self.key=key
        content=QWidget();content.setObjectName("timelineBody");layout=layout_for(content,0);layout.setSpacing(16)
        if not rows: layout.addWidget(label(tr('ai_empty'),'muted'))
        day=None;previous=None
        for entry in group_quiet(rows):
            record=entry.records[-1] if isinstance(entry,QuietGroup) else entry
            date=time.strftime('%d %b',time.localtime(record.ts))
            if date!=day:
                if day is not None or time.localtime(record.ts)[:3]!=time.localtime(now)[:3]:
                    separator=label(tr('today') if time.localtime(record.ts)[:3]==time.localtime(now)[:3] else date,'muted');separator.setAlignment(Qt.AlignmentFlag.AlignCenter);layout.addWidget(separator)
                day=date;previous=None
            stamp=time.strftime('%H:%M',time.localtime(record.ts))
            if isinstance(entry,QuietGroup):
                group=QWidget();group.setProperty('feedDay',date);group.setObjectName('quietGroup');group_layout=layout_for(group,8);group_layout.setSpacing(8)
                heading=QHBoxLayout();glyph=QLabel();glyph.setPixmap(icon('bell-off' if record.muted else 'bot').pixmap(18,18));heading.addWidget(glyph)
                first=time.strftime('%H:%M',time.localtime(entry.records[0].ts));name=record.camera.replace('_',' ').title();count_=len(entry.records)
                if record.muted and record.text==tr('demo_pause_summary'):
                    text=tr('quiet_paused_group' if count_>1 else 'quiet_paused_once',camera=name,count=count_,time=first)
                else:
                    sentence=tr('quiet_paused_description' if record.muted else 'quiet_no_alert',text=record.text.rstrip('.'))
                    text=tr('quiet_group' if count_>1 else 'quiet_once',camera=name,count=count_,time=first,text=sentence)
                heading.addWidget(label(text,'muted'),1)
                group_layout.addLayout(heading)
                if count_>1:
                    key=(record.camera,entry.records[0].ts)
                    arrow=QPushButton();arrow.setObjectName('iconButton');arrow.setIcon(icon('chevron-down'));arrow.setToolTip(tr('details'));arrow.setCheckable(True);arrow.setChecked(key in self.expanded_groups);heading.addWidget(arrow)
                    expanded=QWidget();lines=layout_for(expanded,4);lines.setSpacing(8)
                    for quiet in entry.records:
                        text=tr('quiet_paused_once',camera=name,time=time.strftime('%H:%M',time.localtime(quiet.ts))) if quiet.muted else time.strftime('%H:%M',time.localtime(quiet.ts))+'  '+quiet.text
                        line=label(text,'muted');self.direction(line,text);lines.addWidget(line)
                    expanded.setVisible(key in self.expanded_groups);group_layout.addWidget(expanded)
                    arrow.toggled.connect(lambda checked,k=key,w=expanded:self.expand_group(k,w,checked))
                    arrow.toggled.connect(lambda checked,b=arrow:b.setIcon(icon('chevron-up' if checked else 'chevron-down')))
                layout.addWidget(group);previous=None;continue
            if record.kind=='button':
                event=QWidget();event.setProperty('feedDay',date);event.setObjectName('timelineBody');note_row=QHBoxLayout(event);note_row.setContentsMargins(8,4,8,4);note_row.addStretch();glyph=QLabel();glyph.setPixmap(icon('check').pixmap(16,16));note_row.addWidget(glyph)
                text=tr('chat_button',name=record.name,text=record.text);line=label(text,'muted');self.direction(line,text);note_row.addWidget(line);note_row.addStretch();layout.addWidget(event);previous=None;continue
            item=card();item.setProperty('feedDay',date);item.setObjectName('familyBubble' if record.who=='owner' else 'assistantBubble' if record.who=='assistant' else 'alertCard')
            if record.urgent: item.setProperty('urgent',True)
            row=layout_for(item,16);row.setSpacing(8)
            same_run=previous is not None and previous.who==record.who and previous.name==record.name and record.who in ('owner','assistant')
            meta=QHBoxLayout()
            if not same_run:
                glyph=QLabel();glyph.setPixmap(icon('user' if record.who=='owner' else 'bot' if record.who=='assistant' else 'camera').pixmap(18,18))
                if record.who=='assistant': glyph.setObjectName('botAvatar');glyph.setFixedSize(30,30);glyph.setAlignment(Qt.AlignmentFlag.AlignCenter)
                meta.addWidget(glyph)
                name=record.name if record.who=='owner' else tr('ai_assistant') if record.who=='assistant' else record.camera.replace('_',' ').title()
                meta.addWidget(label(name,'muted'),1)
            else: meta.addStretch()
            if record.label in ('suspicious','escalation'):
                chip=label(record.label.capitalize(),'warning' if record.label=='suspicious' else 'error');chip.setObjectName('decisionChip');chip.setStyleSheet('color: '+(__import__('home_guard_project.box.app.theme',fromlist=['WARNING']).WARNING if record.label=='suspicious' else __import__('home_guard_project.box.app.theme',fromlist=['ERROR']).ERROR)+'; padding: 4px 8px; border-radius: 6px; background: #202c36;');meta.addWidget(chip)
            meta.addWidget(label(stamp,'muted'));row.addLayout(meta)
            if record.who=='box' and record.image:
                path=image_path(image_dir,record.image) if image_dir else None
                row.addWidget(AlertPicture(path))
            message=label(record.text,'section' if record.who=='box' else None);self.direction(message,record.text);row.addWidget(message)
            if record.who=='box':
                mark=label(tr('chat_delivered' if record.delivered else 'chat_refused'),'ok' if record.delivered else 'error');delivery=QHBoxLayout();glyph=QLabel();glyph.setPixmap(icon('check' if record.delivered else 'warning').pixmap(18,18));delivery.addWidget(glyph);delivery.addWidget(mark,1);row.addLayout(delivery)
                if not record.delivered: row.addWidget(label(record.error or tr('ai_no_error'),'error'))
            bubble=QHBoxLayout()
            if record.who=='owner': bubble.addStretch(1)
            bubble.addWidget(item,4)
            if record.who=='assistant': bubble.addStretch(1)
            layout.addLayout(bubble);previous=record
        layout.addStretch()
        old=self.scroll.takeWidget()
        if old: old.deleteLater()
        self.scroll.setWidget(content)
        QTimer.singleShot(0,lambda:self.scroll.verticalScrollBar().setValue(self.scroll.verticalScrollBar().maximum() if stick else value))

    def direction(self,label,text):
        rtl=text_direction(text)=='rtl'
        label.setLayoutDirection(Qt.LayoutDirection.RightToLeft if rtl else Qt.LayoutDirection.LeftToRight)
        label.setAlignment((Qt.AlignmentFlag.AlignRight if rtl else Qt.AlignmentFlag.AlignLeft)|Qt.AlignmentFlag.AlignVCenter|Qt.AlignmentFlag.AlignAbsolute)

    def expand_group(self,key,widget,checked):
        if checked: self.expanded_groups.add(key)
        else: self.expanded_groups.discard(key)
        widget.setVisible(checked)

    def update_day_header(self,value):
        content=self.scroll.widget()
        if content is None: return
        records=sorted((widget.mapTo(content,widget.rect().topLeft()).y(),widget.property('feedDay')) for widget in content.findChildren(QWidget) if widget.property('feedDay'))
        if not records: return
        date=next((day for top,day in reversed(records) if top<=value+30),records[0][1])
        self.sticky_day.setText(tr('today') if date==time.strftime('%d %b') else date)
