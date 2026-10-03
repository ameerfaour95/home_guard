"""Owner details and separate, opt-in permissions for the setup wizard."""
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QWidget, QHBoxLayout, QVBoxLayout, QCheckBox, QLabel
from .strings import tr
from .setup_pages import Page


class ConsentLabel(QLabel):
    def __init__(self, text, toggle):
        super().__init__(text)
        self.toggle = toggle
        self.setWordWrap(True)
        self.setTextFormat(Qt.TextFormat.PlainText)
        self.setBuddy(toggle)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.toggle.setFocus()
            self.toggle.toggle()
        super().mouseReleaseEvent(event)


def build_owner_page(window):
    from .ui import label
    from .preferences import InstallerPreference
    layout = window.page_layouts[Page.OWNER]
    layout.setContentsMargins(24, 24, 24, 24)
    layout.setSpacing(8)
    fields = QHBoxLayout(); fields.setSpacing(24)
    for key in ('owner_name', 'owner_phone', 'installer'):
        column = QVBoxLayout(); column.setSpacing(8)
        column.setAlignment(Qt.AlignmentFlag.AlignTop)
        start = layout.count()
        window.add_input(Page.OWNER, key, tr(key + '_label'))
        for _ in range(3): column.addWidget(layout.takeAt(start).widget())
        if key == 'installer': column.addWidget(label(tr('installer_remembered'), 'muted'))
        fields.addLayout(column, 1)
    layout.addLayout(fields)
    layout.addSpacing(8)
    layout.addWidget(label(tr('owner_permissions'), 'section'))
    layout.addWidget(label(tr('owner_permissions_hint'), 'muted'))
    window.consent_toggles = {}
    for key in ('live', 'recordings', 'training'):
        row = QWidget(); row.setStyleSheet('background: transparent;')
        line = QHBoxLayout(row); line.setContentsMargins(0, 0, 0, 0); line.setSpacing(12)
        toggle = QCheckBox(); toggle.setChecked(False)
        toggle.setAccessibleName(tr('consent_' + key))
        toggle.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        line.addWidget(toggle, 0, Qt.AlignmentFlag.AlignVCenter)
        line.addWidget(ConsentLabel(tr('consent_' + key), toggle), 1)
        layout.addWidget(row)
        window.consent_toggles[key] = toggle
    layout.addWidget(label(tr('consent_change_help'), 'muted'))
    window.installer_preference = InstallerPreference()
    if not window.args.demo:
        window.inputs['installer'].setText(window.installer_preference.load())
    order = [window.inputs[k] for k in ('owner_name', 'owner_phone', 'installer')] + list(window.consent_toggles.values())
    for first, second in zip(order, order[1:]): QWidget.setTabOrder(first, second)
