"""Console-free Windows shortcut entry point, with visible legacy fallback."""

import os
from pathlib import Path
import subprocess
import sys


def main():
    root = Path(__file__).resolve().parents[3]
    try:
        result = subprocess.run(
            [sys.executable, "-m", "home_guard_project.box.app"], cwd=root
        )
        if result.returncode == 0:
            return
    except OSError:
        pass
    bash = (
        Path(os.environ.get("ProgramFiles", "C:/Program Files"))
        / "Git"
        / "bin"
        / "bash.exe"
    )
    script = root / "home_guard_project" / "box" / "screen.sh"
    subprocess.Popen(
        [str(bash), "-l", str(script), "--legacy"],
        cwd=root,
        creationflags=getattr(subprocess, "CREATE_NEW_CONSOLE", 0),
    )


if __name__ == "__main__":
    main()
