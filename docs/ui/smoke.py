"""Offscreen UI integration checks, exclusively with synthetic providers."""

import os

os.environ["QT_QPA_PLATFORM"] = "offscreen"
from argparse import Namespace
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from unittest import mock
from PySide6.QtCore import QTimer
from PySide6.QtGui import QFontDatabase
from PySide6.QtWidgets import QApplication
from home_guard_project.box.app.ui import Window
from home_guard_project.box.app import launcher


def args(**overrides):
    values = dict(
        demo=True,
        setup=False,
        fail=False,
        wifi=False,
        skip_cameras=True,
        alerts=False,
        details=False,
        state="live",
        cameras=3,
        page=None,
        size="1366x768",
        screenshot=None,
    )
    values.update(overrides)
    return Namespace(**values)


def main():
    app = QApplication([])
    app.setQuitOnLastWindowClosed(False)
    QFontDatabase.addApplicationFont("C:/Windows/Fonts/segoeui.ttf")
    with (
        mock.patch("subprocess.run", return_value=Namespace(returncode=0)) as run,
        mock.patch("subprocess.Popen") as popen,
    ):
        launcher.main()
        run.assert_called_once()
        popen.assert_not_called()
    with (
        mock.patch("subprocess.run", return_value=Namespace(returncode=1)),
        mock.patch("subprocess.Popen") as popen,
    ):
        launcher.main()
        assert popen.call_args.args[0][-1] == "--legacy"
    for size in ("1366x768", "1920x1080"):
        for count in range(1, 10):
            window = Window(args(cameras=count, size=size))
            window.show()
            app.processEvents()
            assert len(window.tiles) == count
            assert all(tile.picture is not None for tile in window.tiles)
            assert window.size().width() == int(size.split("x")[0])
            assert window.size().height() == int(size.split("x")[1])
            window.close()
    for failure in (False, True):
        window = Window(args(setup=True, fail=failure))
        window.show()
        for _ in range(4):
            window.next_page()
        assert window.pages.currentIndex() == 4
        outcome = []

        def poll():
            if not failure and window.pages.currentIndex() == 5:
                assert len(window.sequence.results) == 5
                assert len(window.sequence.results[-1].checks) == 4
                outcome.append("success")
                window.close()
                app.quit()
            elif failure and window.sequence.failed and window.next.isVisible():
                assert len(window.sequence.results) == 3
                window.next.click()
                assert window.pages.currentIndex() == 1
                outcome.append("retry")
                window.close()
                app.quit()
            else:
                QTimer.singleShot(100, poll)

        timeout = QTimer()
        timeout.setSingleShot(True)
        timeout.timeout.connect(lambda: app.exit(2))
        timeout.start(20000)
        QTimer.singleShot(100, poll)
        assert app.exec() == 0
        timeout.stop()
        assert outcome
    print(
        "PASS: 1-9 cameras at both sizes; timed success/failure/retry; launcher fallback."
    )


if __name__ == "__main__":
    main()
