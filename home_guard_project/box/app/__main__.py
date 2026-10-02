"""Run with python -m home_guard_project.box.app [--demo | --setup]."""

import argparse
import os
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--panel", choices=("settings", "cameras"))
    parser.add_argument("--setup", action="store_true")
    parser.add_argument("--fail", action="store_true")
    parser.add_argument("--wifi", action="store_true")
    parser.add_argument("--skip-cameras", action="store_true")
    parser.add_argument("--alerts", action="store_true")
    parser.add_argument("--details", action="store_true")
    parser.add_argument(
        "--state",
        choices=(
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
        ),
        default="mixed",
    )
    parser.add_argument("--cameras", type=int, choices=range(1, 10), default=3)
    parser.add_argument(
        "--page",
        choices=(
            "address",
            "network",
            "house",
            "cameras",
            "progress",
            "summary",
            "failure",
            "validation",
        ),
    )
    parser.add_argument("--size", default="1366x768")
    parser.add_argument("--screenshot")
    args = parser.parse_args()
    if args.screenshot:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication
    from PySide6.QtCore import QTimer
    from .ui import Window

    app = QApplication(sys.argv[:1])
    from PySide6.QtGui import QFontDatabase, QFont

    # Explicit loading also fixes Windows offscreen font discovery.
    font_path = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts" / "segoeui.ttf"
    if font_path.exists():
        QFontDatabase.addApplicationFont(str(font_path))
    app.setFont(QFont("Segoe UI", 11))
    window = Window(args)
    window.show()
    if args.screenshot:

        def capture():
            path = Path(args.screenshot)
            path.parent.mkdir(parents=True, exist_ok=True)
            if not window.grab().save(str(path)):
                app.exit(2)
                return
            window.close()
            app.quit()

        QTimer.singleShot(700, capture)
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
