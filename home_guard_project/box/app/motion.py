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
