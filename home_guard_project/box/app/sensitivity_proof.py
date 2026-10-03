"""Offline 1366x768 demo proof: python -m home_guard_project.box.app.sensitivity_proof."""
import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
from pathlib import Path
from types import SimpleNamespace
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont, QFontDatabase, QPainter, QColor
from PySide6.QtWidgets import QApplication, QWidget, QHBoxLayout, QScrollArea
from PySide6.QtTest import QTest
from .ui import Window
from .box_controls import BoxControls, Settings
from .settings_ui import SettingsPage
from .theme import stylesheet


def main():
    app = QApplication.instance() or QApplication([])
    for name in ('segoeui.ttf', 'consola.ttf'):
        QFontDatabase.addApplicationFont(str(Path(os.environ.get('WINDIR', 'C:/Windows')) / 'Fonts' / name))
    app.setFont(QFont('Segoe UI', 11))
    destination = Path('docs/ui/screenshots'); destination.mkdir(parents=True, exist_ok=True)
    args = SimpleNamespace(demo=True, setup=False, theme='dark', panel='settings', fail=None, wifi=False, skip_cameras=False, alerts=False, details=False, state='inference', cameras=3, page=None, size='1366x768', screenshot=None, detections=False)
    window = Window(args); window.show(); QTest.qWait(300)
    window.settings_page.reset_sensitivity(); QTest.qWait(300)

    def capture(name, dialog=None):
        app.processEvents(); pix = window.grab()
        if dialog:
            painter = QPainter(pix); painter.fillRect(pix.rect(), QColor(0, 0, 0, 65))
            painter.drawPixmap(dialog.pos()-window.pos(), dialog.grab()); painter.end()
        assert (pix.width(), pix.height()) == (1366, 768)
        assert pix.save(str(destination / ('sensitivity-' + name + '.png')))

    capture('house')
    window.open_cameras(); QTest.qWait(400)
    page = window.cameras_page
    page.open_alerts('driveway'); QTest.qWait(400)
    dialog = page.alert_dialog; panel = dialog.sensitivity_panel
    panel.custom.click()
    panel.sliders.sliders['person'].setValue(10); panel.sliders.sliders['person'].sliderReleased.emit(); QTest.qWait(400)
    panel.sliders.sliders['vehicle'].setValue(16); panel.sliders.sliders['vehicle'].sliderReleased.emit(); QTest.qWait(1200)
    panel.sliders.sliders['person'].setFocus(Qt.FocusReason.TabFocusReason)
    dialog.move(window.pos().x()+(1366-dialog.width())//2, window.pos().y()+(768-dialog.height())//2)
    capture('camera-popover', dialog)
    dialog.accept(); window.close()

    host = QWidget(); host.setStyleSheet(stylesheet()); host.resize(1366, 768)
    row = QHBoxLayout(host); row.setContentsMargins(24, 16, 24, 16); row.addStretch()
    page = SettingsPage(BoxControls(demo=True, settings=Settings(mode='inference', conf_person=.5, conf_vehicle=.7, conf_animal=.6)), lambda: None); page.reload()
    scroll = QScrollArea(); scroll.setWidgetResizable(True); scroll.setFixedWidth(560); scroll.setWidget(page.widget)
    row.addWidget(scroll); row.addStretch(); host.show(); QTest.qWait(300)
    scroll.verticalScrollBar().setValue(page.sensitivities.parentWidget().mapTo(page.widget, page.sensitivities.parentWidget().rect().topLeft()).y()-24); QTest.qWait(300)
    page.sensitivities.sliders['person'].setFocus(Qt.FocusReason.TabFocusReason)
    assert host.grab().save(str(destination/'sensitivity-narrow.png'))
    page.timer.stop(); host.close()


if __name__ == '__main__': main()
