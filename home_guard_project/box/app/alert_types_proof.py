"""Reproducible, offline demo proof: python -m home_guard_project.box.app.alert_types_proof."""
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
    font_dir = Path(os.environ.get('WINDIR', 'C:/Windows')) / 'Fonts'
    for name in ('segoeui.ttf', 'consola.ttf'):
        if (font_dir/name).exists(): QFontDatabase.addApplicationFont(str(font_dir/name))
    app.setFont(QFont('Segoe UI', 11))
    destination = Path('docs/ui/screenshots'); destination.mkdir(parents=True, exist_ok=True)
    args = SimpleNamespace(demo=True,setup=False,theme='dark',panel='settings',fail=None,wifi=False,skip_cameras=False,alerts=False,details=False,state='inference',cameras=3,page=None,size='1366x768',screenshot=None,detections=False)
    window = Window(args); window.show(); QTest.qWait(350)

    def capture(name, dialog=None):
        app.processEvents()
        pix = window.grab()
        if dialog:
            # Offscreen platform windows are separate surfaces. Composite the
            # actual Qt dialog surface at its real position over the demo window.
            painter = QPainter(pix)
            painter.fillRect(pix.rect(), QColor(0, 0, 0, 65))
            painter.drawPixmap(dialog.pos()-window.pos(), dialog.grab()); painter.end()
        assert pix.size().width() == 1366 and pix.size().height() == 768
        assert pix.save(str(destination / ('alert-types-' + name + '.png')))

    capture('house')
    window.settings_page.alert_types.tiles['person'].click(); QTest.qWait(500)
    capture('last-one')
    window.open_cameras(); QTest.qWait(500)
    page = window.cameras_page
    page.open_alerts('driveway'); QTest.qWait(350)
    dialog = page.alert_dialog
    dialog.custom.click(); QTest.qWait(400)
    dialog.tiles.tiles['vehicle'].click(); QTest.qWait(1100)
    dialog.move(window.pos().x()+(1366-dialog.width())//2, window.pos().y()+(768-dialog.height())//2)
    capture('camera-popover', dialog)
    dialog.accept(); QTest.qWait(300)
    capture('camera-custom')
    window.content_stack.setCurrentIndex(0); window.settings_changed(); QTest.qWait(350)
    capture('status')
    window.close()

    # A real 560-pixel Settings viewport on the same 1366x768 proof canvas.
    # This tests the widget's responsive layout without shrinking or scaling it.
    host = QWidget(); host.setStyleSheet(stylesheet()); host.resize(1366, 768)
    row = QHBoxLayout(host); row.setContentsMargins(24, 16, 24, 16); row.addStretch()
    page = SettingsPage(BoxControls(demo=True, settings=Settings(mode='inference')), lambda: None); page.reload()
    scroll = QScrollArea(); scroll.setWidgetResizable(True); scroll.setFixedWidth(560); scroll.setWidget(page.widget)
    row.addWidget(scroll); row.addStretch(); host.show(); QTest.qWait(350)
    assert all(t.compact for t in page.alert_types.tiles.values())
    assert host.grab().save(str(destination/'alert-types-narrow.png'))
    page.timer.stop(); host.close()


if __name__ == '__main__': main()
