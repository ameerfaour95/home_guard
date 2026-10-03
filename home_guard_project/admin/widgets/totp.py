from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QRegularExpressionValidator, QKeySequence
from PySide6.QtCore import QRegularExpression
from PySide6.QtWidgets import QWidget, QHBoxLayout, QLineEdit, QApplication


class Digit(QLineEdit):
    def __init__(self, owner, position):
        super().__init__()
        self.owner, self.position = owner, position
        self.setMaxLength(1)
        self.setFixedSize(48, 48)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setValidator(QRegularExpressionValidator(QRegularExpression('[0-9]'), self))
        self.setAccessibleName(f'Authentication code digit {position+1}')
        self.textEdited.connect(self.advance)

    def advance(self, text):
        if text and self.position < 5:
            self.owner.digits[self.position+1].setFocus()
            self.owner.digits[self.position+1].selectAll()

    def keyPressEvent(self, event):
        if event.matches(QKeySequence.StandardKey.Paste):
            text = QApplication.clipboard().text().strip()
            if len(text) == 6 and text.isascii() and text.isdigit():
                self.owner.set_code(text)
            return
        if event.key() == Qt.Key.Key_Backspace and not self.text() and self.position:
            previous = self.owner.digits[self.position-1]
            previous.clear()
            previous.setFocus()
            return
        super().keyPressEvent(event)


class TotpInput(QWidget):
    submitted = Signal()

    def __init__(self):
        super().__init__()
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        self.digits = [Digit(self, i) for i in range(6)]
        for digit in self.digits:
            layout.addWidget(digit)
            digit.returnPressed.connect(self.submitted)
        layout.addStretch()

    def code(self):
        return ''.join(d.text() for d in self.digits)

    def set_code(self, code):
        for digit, value in zip(self.digits, code):
            digit.setText(value)
        self.digits[-1].setFocus()

    def clear(self):
        for digit in self.digits:
            digit.clear()
