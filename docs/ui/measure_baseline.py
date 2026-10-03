"""Run the current probe against unmodified 7f59234 modules loaded in memory.

This reads git objects only; no worktree or tracked source is changed.
"""
import os
from pathlib import Path
import runpy
import subprocess
import sys
import types

os.environ.pop("VIRTUAL_ENV",None)
os.environ.pop("SSLKEYLOGFILE",None)
root=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(root))
for suffix in ("preview","app.ai_activity_ui","app.ui"):
    name="home_guard_project.box."+suffix
    path="home_guard_project/box/"+suffix.replace(".","/")+".py"
    source=subprocess.check_output(["git","show","7f59234:"+path],cwd=root).decode("utf-8")
    module=types.ModuleType(name);module.__file__=str(root/path);module.__package__=name.rpartition('.')[0]
    sys.modules[name]=module
    exec(compile(source,module.__file__,"exec"),module.__dict__)
runpy.run_path(str(Path(__file__).with_name("measure_live.py")),run_name="__main__")
