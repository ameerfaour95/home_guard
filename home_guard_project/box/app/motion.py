"""Shared interaction timings and transient feedback."""
from PySide6.QtCore import QTimer, QEasingCurve, QPropertyAnimation, QPoint
from PySide6.QtWidgets import QLabel
HOVER_MS=120
TOGGLE_MS=180
PANE_MS=240
EASING=QEasingCurve.Type.OutCubic

def toast(window,text):
    previous=getattr(window,'_toast',None)
    if previous: previous.deleteLater()
    note=QLabel(text,window);note.setObjectName('toast');note.setStyleSheet('background:#243641;color:#edf4f6;padding:14px 24px;border-radius:10px;');note.adjustSize()
    end=QPoint((window.width()-note.width())//2,window.height()-note.height()-24)
    note.move(end+QPoint(0,12));note.show();note.raise_()
    animation=QPropertyAnimation(note,b'pos',note);animation.setDuration(PANE_MS);animation.setEasingCurve(EASING);animation.setStartValue(note.pos());animation.setEndValue(end);animation.start();note._animation=animation
    window._toast=note
    def dismiss():
        animation.setStartValue(note.pos());animation.setEndValue(note.pos()+QPoint(0,12));animation.finished.connect(note.hide);animation.start()
    QTimer.singleShot(3000,note,dismiss)


from PySide6.QtCore import Qt, QObject, QEvent, QVariantAnimation, QRectF, QSize
from PySide6.QtGui import QPainter, QColor, QPen, QIcon, QPixmap
from PySide6.QtWidgets import QWidget, QStackedWidget, QAbstractButton, QAbstractScrollArea, QAbstractItemView, QCheckBox, QComboBox, QSlider, QApplication
from .theme import ACTION

class Transition(QWidget):
    """Snapshot animation leaves all real layout geometry and focus untouched."""
    def __init__(self,parent,before,after,rect):
        super().__init__(parent)
        self.before,self.after=before,after;self.progress=0.
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.setGeometry(rect);self.show();self.raise_()
        self.animation=QVariantAnimation(self);self.animation.setDuration(PANE_MS);self.animation.setEasingCurve(EASING)
        self.animation.setStartValue(0.);self.animation.setEndValue(1.)
        self.animation.valueChanged.connect(self.advance);self.animation.finished.connect(self.deleteLater);self.animation.start()
    def advance(self,value): self.progress=float(value);self.update()
    def paintEvent(self,event):
        p=QPainter(self);p.fillRect(self.rect(),self.palette().window())
        p.setOpacity(1-self.progress);p.drawPixmap(round(-12*self.progress),0,self.before)
        p.setOpacity(self.progress);p.drawPixmap(round(12*(1-self.progress)),0,self.after)

class AnimatedStack(QStackedWidget):
    def setCurrentIndex(self,index):
        if index==self.currentIndex() or not self.isVisible(): return super().setCurrentIndex(index)
        old=getattr(self,'transition',None)
        if old:
            old.animation.stop();old.hide();old.deleteLater();self.transition=None
        before=self.grab()
        super().setCurrentIndex(index)
        self.layout().activate()
        after=self.grab()
        self.transition=Transition(self,before,after,self.rect())
        transition=self.transition
        transition.animation.finished.connect(lambda:setattr(self,'transition',None) if self.transition is transition else None)

def reveal(widget,opened):
    parent=widget.parentWidget()
    if not parent or not parent.isVisible(): widget.setVisible(opened);return
    if opened:
        widget.show();parent.layout().activate() if parent.layout() else None
        after=widget.grab();before=QPixmap(after.size());before.fill(Qt.GlobalColor.transparent)
    else:
        before=widget.grab();after=QPixmap(before.size());after.fill(Qt.GlobalColor.transparent)
    rect=widget.geometry()
    widget.setVisible(opened)
    widget._reveal=Transition(parent,before,after,rect)

class DecisionChip(QLabel):
    """Paint the fade directly: nested graphics effects erase chips on Windows."""
    def __init__(self,text,color):
        super().__init__(text);self.setObjectName('decisionChip');self.color=QColor(color);self.alpha=0.
        self.setContentsMargins(8,4,8,4)
        self.fade=QVariantAnimation(self);self.fade.setDuration(TOGGLE_MS);self.fade.setEasingCurve(EASING);self.fade.setStartValue(0.);self.fade.setEndValue(1.)
        self.fade.valueChanged.connect(self.advance)
    def showEvent(self,event):
        super().showEvent(event);self.fade.start()
    def advance(self,value): self.alpha=float(value);self.update()
    def paintEvent(self,event):
        painter=QPainter(self);painter.setRenderHint(QPainter.RenderHint.Antialiasing);painter.setOpacity(self.alpha)
        painter.setPen(Qt.PenStyle.NoPen);painter.setBrush(QColor('#202c36'));painter.drawRoundedRect(QRectF(self.rect()),6,6)
        painter.setPen(self.color);painter.drawText(self.contentsRect(),Qt.AlignmentFlag.AlignCenter,self.text())

class Switch(QCheckBox):
    def __init__(self,text='',parent=None):
        super().__init__(text,parent);self.position=0.;self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.animation=QVariantAnimation(self);self.animation.setDuration(TOGGLE_MS);self.animation.setEasingCurve(EASING)
        self.animation.valueChanged.connect(self.advance);self.toggled.connect(self.animate)
    def animate(self,checked):
        self.animation.stop();self.animation.setStartValue(self.position);self.animation.setEndValue(1. if checked else 0.);self.animation.start()
    def advance(self,value): self.position=float(value);self.update()
    def sizeHint(self): return QSize(self.fontMetrics().horizontalAdvance(self.text())+68,38)
    def hitButton(self,pos): return self.rect().contains(pos)
    def paintEvent(self,event):
        p=QPainter(self);p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setOpacity(1 if self.isEnabled() else .45);p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(ACTION if self.isChecked() else '#344650'));p.drawRoundedRect(QRectF(2,(self.height()-22)/2,40,22),11,11)
        p.setBrush(QColor('#edf4f6'));p.drawEllipse(QRectF(5+18*self.position,(self.height()-16)/2,16,16))
        p.setPen(self.palette().text().color());p.drawText(self.rect().adjusted(52,0,0,0),Qt.AlignmentFlag.AlignVCenter,self.text())
        if self.property('keyboardFocus'):
            p.setBrush(Qt.BrushStyle.NoBrush);p.setPen(QPen(QColor(ACTION),2));p.drawRoundedRect(QRectF(self.rect()).adjusted(1,1,-1,-1),7,7)

class InteractionOverlay(QWidget):
    def __init__(self,button):
        super().__init__(button);self.level=0.;self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.animation=QVariantAnimation(self);self.animation.setDuration(HOVER_MS);self.animation.setEasingCurve(EASING);self.animation.valueChanged.connect(self.advance)
        if isinstance(button,QAbstractButton): button.toggled.connect(self.toggle_feedback)
        self.setGeometry(button.rect());self.show()
    def advance(self,value): self.level=float(value);self.update()
    def toggle_feedback(self,checked):
        self.animation.setDuration(TOGGLE_MS);self.level=.8;self.target(0.)
    def target(self,value):
        self.animation.stop();self.animation.setStartValue(self.level);self.animation.setEndValue(value);self.animation.start()
    def paintEvent(self,event):
        p=QPainter(self);p.setRenderHint(QPainter.RenderHint.Antialiasing);p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(0,0,0,round(-self.level*70)) if self.level<0 else QColor(180,215,225,round(self.level*22)))
        p.drawRoundedRect(QRectF(self.rect()).adjusted(1,1,-1,-1),8,8)
        if self.parentWidget().property('keyboardFocus'):
            p.setBrush(Qt.BrushStyle.NoBrush);p.setPen(QPen(QColor(ACTION),2));p.drawRoundedRect(QRectF(self.rect()).adjusted(1,1,-1,-1),8,8)

def busy(button,working):
    if button.property('busyIndicator') == 'bar':
        reveal(button._busy_bar, working)
        button.setEnabled(not working)
        return
    if working and not getattr(button,'_busy_timer',None):
        button._saved_icon=button.icon();button._angle=0
        timer=QTimer(button);button._busy_timer=timer
        def frame():
            pix=QPixmap(20,20);pix.fill(Qt.GlobalColor.transparent)
            p=QPainter(pix);p.setRenderHint(QPainter.RenderHint.Antialiasing);p.setPen(QPen(QColor(ACTION),2));p.drawArc(3,3,14,14,button._angle*16,260*16);p.end()
            button.setIcon(QIcon(pix));button._angle=(button._angle+24)%360
        timer.timeout.connect(frame);timer.start(40);frame()
    elif not working and getattr(button,'_busy_timer',None):
        button._busy_timer.stop();button._busy_timer.deleteLater();button._busy_timer=None;button.setIcon(button._saved_icon)
    button.setEnabled(not working)

class MotionSystem(QObject):
    def eventFilter(self,obj,event):
        kind=event.type()
        if isinstance(obj,(QAbstractButton,QComboBox,QSlider)):
            if kind==QEvent.Type.Show:
                obj.setCursor(Qt.CursorShape.PointingHandCursor)
                if not isinstance(obj,Switch) and not obj.property('handlesMotion') and not hasattr(obj,'_motion_overlay'):
                    obj._motion_overlay=InteractionOverlay(obj)
            overlay=getattr(obj,'_motion_overlay',None)
            if overlay:
                if kind==QEvent.Type.Resize: overlay.setGeometry(obj.rect())
                elif kind==QEvent.Type.Enter: overlay.animation.setDuration(HOVER_MS);overlay.target(1.)
                elif kind==QEvent.Type.Leave: overlay.animation.setDuration(HOVER_MS);overlay.target(0.)
                elif kind==QEvent.Type.MouseButtonPress: obj.setProperty('keyboardFocus',False);overlay.animation.setDuration(HOVER_MS);overlay.target(-1.)
                elif kind==QEvent.Type.MouseButtonRelease: overlay.target(1. if obj.underMouse() else 0.)
            if kind==QEvent.Type.FocusIn:
                obj.setProperty('keyboardFocus',event.reason() in (Qt.FocusReason.TabFocusReason,Qt.FocusReason.BacktabFocusReason,Qt.FocusReason.ShortcutFocusReason));obj.update()
                if overlay: overlay.update()
            elif kind==QEvent.Type.FocusOut:
                obj.setProperty('keyboardFocus',False)
                if overlay: overlay.update()
        if kind==QEvent.Type.Show and isinstance(obj,QAbstractItemView): obj.setVerticalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel)
        if kind==QEvent.Type.Wheel and isinstance(obj,QWidget) and isinstance(obj.parentWidget(),QAbstractScrollArea):
            area=obj.parentWidget();bar=area.verticalScrollBar()
            if bar.maximum()==0: return False
            if not event.pixelDelta().isNull(): bar.setValue(bar.value()-event.pixelDelta().y());return True
            animation=getattr(area,'_wheel',None)
            target=animation.endValue() if animation and animation.state()==QPropertyAnimation.State.Running else bar.value()
            if animation: animation.stop();animation.deleteLater()
            animation=QPropertyAnimation(bar,b'value',area);area._wheel=animation;animation.setDuration(HOVER_MS);animation.setEasingCurve(EASING)
            animation.setStartValue(bar.value());animation.setEndValue(max(0,min(bar.maximum(),int(target-event.angleDelta().y()))));animation.start();return True
        return False

def install():
    app=QApplication.instance()
    if not hasattr(app,'_homeguard_motion'):
        app._homeguard_motion=MotionSystem(app);app.installEventFilter(app._homeguard_motion)
