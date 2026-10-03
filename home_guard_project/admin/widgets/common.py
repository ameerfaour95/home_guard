from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QPainter, QColor
from PySide6.QtWidgets import QLabel, QPushButton, QWidget, QVBoxLayout
from ..theme import PALETTES


def label(text, style='', wrap=False):
    widget = QLabel(text)
    widget.setTextFormat(Qt.TextFormat.PlainText)
    widget.setObjectName(style)
    widget.setWordWrap(wrap)
    return widget


def button(text, callback=None, style=''):
    widget = QPushButton(text)
    widget.setObjectName(style)
    widget.setCursor(Qt.CursorShape.PointingHandCursor)
    if callback:
        widget.clicked.connect(callback)
    return widget


class EmptyState(QWidget):
    def __init__(self, title, description, *, eyebrow='HOME GUARD', parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(48, 48, 48, 48)
        layout.setSpacing(16)
        layout.addStretch()
        for text, style in [(eyebrow, 'eyebrow'), (title, 'title'), (description, 'muted')]:
            item = label(text, style, True)
            item.setAlignment(Qt.AlignmentFlag.AlignCenter)
            layout.addWidget(item)
        self.action = button('Try again')
        self.action.hide()
        layout.addWidget(self.action, alignment=Qt.AlignmentFlag.AlignCenter)
        layout.addStretch()


class Skeleton(QWidget):
    def __init__(self, theme='dark'):
        super().__init__()
        self.tokens = PALETTES[theme]
        self.phase = 0
        self.setAccessibleName('Loading Home Guard data')
        self.timer = QTimer(self)
        self.timer.setInterval(120)
        self.timer.timeout.connect(self.tick)

    def showEvent(self, event):
        self.timer.start()
        super().showEvent(event)

    def hideEvent(self, event):
        self.timer.stop()
        super().hideEvent(event)

    def tick(self):
        self.phase = (self.phase + 1) % 12
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setPen(Qt.PenStyle.NoPen)
        color = QColor(self.tokens['raised'])
        color.setAlpha(150 + abs(6-self.phase)*16)
        painter.setBrush(color)
        for row in range(5):
            y = 24 + row * 80
            for x, width in [(16, 160), (200, 240), (480, 80), (600, 80)]:
                painter.drawRoundedRect(x, y, width, 14, 4, 4)
            painter.drawRoundedRect(16, y+24, 110, 8, 4, 4)
