"""Start the Admin Center with no console window: `<base python>\\pythonw.exe -I windowless.py [app arguments]`.

-I (isolated) keeps this file's folder off sys.path: admin\\collections.py would shadow the standard library.

uv makes the venv's Scripts\\pythonw.exe a console-subsystem launcher that runs python.exe, so starting the app with
it opened a terminal window behind the Admin Center (2026-10-09: WindowsTerminal titled ...\\.venv\\Scripts\\pythonw.exe).
launch_admin.ps1 runs the base interpreter's real pythonw.exe (pyvenv.cfg `home`) on this file instead; it puts the
venv's packages and this folder's code on the path and runs `python -m home_guard_project.admin` in-process.
"""
import os
import runpy
import site
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def main():
    venv = Path(os.environ.get('HG_ADMIN_VENV') or REPO / '.venv')
    site.addsitedir(str(venv / 'Lib' / 'site-packages'))  # also runs its .pth files (pywin32)
    sys.path.insert(0, str(REPO))
    sys.argv[0] = 'home_guard_project.admin'
    runpy.run_module('home_guard_project.admin', run_name='__main__', alter_sys=True)


if __name__ == '__main__':
    main()
