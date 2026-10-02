"""Start the Home Guard app without a console window.

A uv environment's own ``pythonw.exe`` hands over to the console interpreter,
which opens a terminal window next to the app. The shortcuts therefore run the
base install's ``pythonw.exe`` on this file, which adds the environment's
packages and the repository to the path and then runs the app as usual:

    <base>\\pythonw.exe start.pyw            the box window
    <base>\\pythonw.exe start.pyw --setup    the setup wizard
"""

import os
import runpy
import site
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
site.addsitedir(os.path.join(ROOT, ".venv", "Lib", "site-packages"))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
sys.argv[0] = "home_guard_project.box.app"
runpy.run_module("home_guard_project.box.app", run_name="__main__")
