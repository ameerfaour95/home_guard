"""Regenerate the review gallery offline, with no credentials or camera input."""

from pathlib import Path
import subprocess
import sys
import os

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).with_name("screenshots")
CASES = []
for state in (
    "mixed",
    "live",
    "offline",
    "stopped",
    "hidden",
    "empty",
    "error",
    "loading",
    "inference",
    "quiet",
):
    CASES.append(("box-" + state, ["--demo", "--state", state]))
for count in (1, 4, 9):
    CASES.append(
        (
            "box-" + str(count) + "-cameras",
            [
                "--demo",
                "--cameras",
                str(count),
                "--state",
                "live" if count < 9 else "mixed",
            ],
        )
    )
CASES.append(("box-details", ["--demo", "--details"]))
for page in (
    "address",
    "network",
    "house",
    "cameras",
    "progress",
    "summary",
    "failure",
    "validation",
):
    CASES.append(("setup-" + page, ["--setup", "--demo", "--page", page]))
CASES.extend(
    [
        ("setup-wifi", ["--setup", "--demo", "--page", "network", "--wifi"]),
        ("setup-alerts", ["--setup", "--demo", "--page", "house", "--alerts"]),
        (
            "setup-cameras-later",
            ["--setup", "--demo", "--page", "cameras", "--skip-cameras"],
        ),
        (
            "setup-summary-cameras-later",
            ["--setup", "--demo", "--page", "summary", "--skip-cameras"],
        ),
    ]
)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    if '--zones-only' in sys.argv:
        capture_zones()
        return
    for size in ("1366x768", "1920x1080"):
        for name, flags in CASES:
            path = OUT / (name + "-" + size + ".png")
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "home_guard_project.box.app",
                    *flags,
                    "--size",
                    size,
                    "--screenshot",
                    str(path),
                ],
                cwd=ROOT,
                check=True,
            )
            print(path.name, flush=True)
    capture_zones()


def capture_zones():
    """The dialog at its real opening size, on a canvas of the target screen size."""
    os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
    sys.path.insert(0, str(ROOT))
    from types import SimpleNamespace
    from threading import Event
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QColor, QFont, QFontDatabase, QPainter, QPixmap
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QApplication, QWidget
    from home_guard_project.box.app.ui import Window
    from home_guard_project.box.app.zone_editor import ZoneEditorDialog
    from home_guard_project.box.app.camera_controls import CameraControls
    from home_guard_project.box.app.box_controls import BoxControls
    from home_guard_project.box.app.demo_media import picture
    from home_guard_project.box.app.theme import PALETTES, stylesheet
    from home_guard_project.box.app import motion

    app = QApplication.instance() or QApplication([])
    font_dir = Path(os.environ.get('WINDIR', 'C:/Windows')) / 'Fonts'
    for font in ('segoeui.ttf', 'seguisb.ttf', 'consola.ttf'):
        if (font_dir/font).exists(): QFontDatabase.addApplicationFont(str(font_dir/font))
    app.setFont(QFont('Segoe UI', 11))
    motion.install()
    points = [[.12, .40], [.45, .27], [.89, .43], [.94, .94], [.10, .94]]
    for size in ('1366x768', '1920x1080'):
        width, height = map(int, size.split('x'))
        host = QWidget(); host.setStyleSheet(stylesheet()); host.resize(width, height); host.show()
        for state in ('empty', 'drawing', 'closed', 'tiny-warning', 'saving', 'error'):
            controls = CameraControls(BoxControls(demo=True), ['front_door'])
            zone = [] if state == 'empty' else [[.44, .55], [.55, .55], [.50, .66]] if state == 'tiny-warning' else points
            if state == 'drawing': zone = points[:2]
            dialog = ZoneEditorDialog(controls, 'front_door', picture(0, '4:3'), zone, host)
            # Qt's synthetic offscreen screen is 800x800. Set the actual review
            # screen bounds explicitly; production uses availableGeometry().
            dialog.setMinimumSize(1100, 700)
            dialog.resize(min(1280, width-48), min(800, height-64))
            dialog.show()
            QTest.qWait(motion.PANE_MS + 40)
            if state == 'error':
                def fail(*args): raise RuntimeError('Synthetic save failure')
                controls.set_zone = fail
                dialog.save()
                QTest.qWait(motion.PANE_MS * 2)
            if state == 'saving':
                release = Event()
                controls.set_zone = lambda *args: (release.wait(5), zone)[1]
                dialog.save()
                QTest.qWait(motion.PANE_MS + 40)
            canvas = QPixmap(width, height); canvas.fill(QColor(PALETTES['dark']['bg']))
            painter = QPainter(canvas)
            painter.drawPixmap((width-dialog.width())//2, (height-dialog.height())//2, dialog.grab()); painter.end()
            path = OUT / f'zone-{state}-{size}.png'
            assert canvas.save(str(path)); print(path.name, flush=True)
            if state == 'saving':
                release.set(); dialog.future.result(timeout=3); dialog.poll()
            dialog.reject(); QTest.qWait(motion.TOGGLE_MS + 20); dialog.deleteLater(); app.processEvents()
        host.close()
        args = SimpleNamespace(demo=True, setup=False, theme='dark', panel='cameras', fail=None,
                               wifi=False, skip_cameras=False, alerts=False, details=False,
                               state='live', cameras=2, page=None, size=size, screenshot=None,
                               detections=False)
        window = Window(args); window.show(); QTest.qWait(motion.PANE_MS * 3)
        page = window.cameras_page
        name = window.camera_controls.records[0].name
        window.camera_controls.set_zone(name, points)
        page.zone_saved(name, points)
        QTest.qWait(motion.PANE_MS + 40)
        path = OUT / f'zone-camera-page-{size}.png'
        assert window.grab().save(str(path)); print(path.name, flush=True)
        window.close(); app.processEvents()


if __name__ == "__main__":
    main()
