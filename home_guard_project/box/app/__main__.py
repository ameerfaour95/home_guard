"""Run with python -m home_guard_project.box.app [--demo | --setup]."""

import argparse
import os
import sys
from pathlib import Path


def _drop_own_console():
    """Close the terminal window Windows opened only for this program.

    Started from a shortcut, a console Python gets a terminal window of its
    own. If this process is the only one attached to its console, nobody ran
    it from a terminal, so the window is let go. Run from a terminal, the
    console is shared and stays.
    """
    if os.name != "nt":
        return
    try:
        import ctypes

        kernel32 = ctypes.windll.kernel32
        attached = (ctypes.c_uint32 * 2)()
        if kernel32.GetConsoleProcessList(attached, 2) == 1:
            kernel32.FreeConsole()
    except Exception:  # never worth failing the app over
        pass


def main():
    _drop_own_console()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--remote-box", metavar="USER@BOX", help="View a box through the existing SSH key (read only)")
    parser.add_argument("--aspect",choices=("16:9","4:3"),default="16:9")
    parser.add_argument("--detections", action="store_true")
    parser.add_argument("--theme", choices=("dark","light"), default="dark")
    parser.add_argument("--panel", choices=("settings", "cameras"))
    parser.add_argument("--setup", action="store_true")
    parser.add_argument("--fail", nargs="?", const="network", choices=("connect","update","site","register","network","cameras","alerts","readiness"))
    parser.add_argument("--wifi", action="store_true")
    parser.add_argument("--skip-cameras", action="store_true")
    parser.add_argument("--alerts", action="store_true")
    parser.add_argument("--details", action="store_true")
    parser.add_argument("--technical-log", action="store_true")
    parser.add_argument(
        "--state",
        choices=(
            "mixed",
            "live", "live-detections", "off-camera",
            "offline",
            "stopped",
            "hidden",
            "empty",
            "error",
            "loading",
            "inference", "paused", "ai-stopped", "ai-stale", "ai-empty", "ai-refused", "ai-delivered", "ai-urgent", "ai-paused", "ai-training", "ai-thinking",
            "ai-group", "ai-conversation", "quiet", "no-cameras", "box-unreachable",
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
            "owner-consent",
            "cameras",
            "progress",
            "summary",
            "failure",
            "validation",
            "review",
            "camera-check",
            "scene-map",
        ),
    )
    parser.add_argument("--scene", choices=("card", "loading", "regions", "grid", "drawing", "line", "summary", "saved", "error"),
                        help="Demo: open the camera map editor in this state (with --panel cameras or --page scene-map)")
    parser.add_argument("--lang", choices=("he", "en"), default="he", help="The camera map editor's language")
    parser.add_argument("--size", default="1366x768")
    parser.add_argument("--screenshot")
    args = parser.parse_args()
    if args.remote_box and (args.demo or args.setup): parser.error("--remote-box is a live dashboard option")
    if args.remote_box:
        from .remote_cameras import target_user
        try: target_user(args.remote_box)
        except ValueError: parser.error("Invalid remote box address")
    args.details = args.details or args.technical_log
    from .scene_strings import set_language
    set_language(args.lang)
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
    mono_path=font_path.with_name('consola.ttf')
    if mono_path.exists(): QFontDatabase.addApplicationFont(str(mono_path))
    app.setFont(QFont("Segoe UI", 11))
    window = Window(args)
    window.show()
    if args.screenshot:

        def capture():
            path = Path(args.screenshot)
            path.parent.mkdir(parents=True, exist_ok=True)
            from .scene_editor import grab_with_dialogs
            if not (grab_with_dialogs(window) if args.scene else window.grab()).save(str(path)):
                app.exit(2)
                return
            window.close()
            app.quit()

        QTimer.singleShot(700, capture)
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
