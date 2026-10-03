"""Discover all box tests when the checkout has no test-package __init__.py files.

Adds namespace packages in memory only, avoiding the unrelated site-packages
package named tests. Does not modify files outside the Round L allowlist.
"""
import os
from pathlib import Path
import sys
import types
import unittest

os.environ.pop("VIRTUAL_ENV",None)
os.environ.pop("SSLKEYLOGFILE",None)
root=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(root))
for name,path in (("tests",root/"tests"),("tests.box",root/"tests"/"box")):
    package=types.ModuleType(name);package.__path__=[str(path)]
    sys.modules[name]=package
suite=unittest.defaultTestLoader.discover(str(root/"tests"/"box"))
if "--without-serve" in sys.argv:
    def keep(test):
        if isinstance(test,unittest.TestSuite): return unittest.TestSuite(keep(t) for t in test)
        return unittest.TestSuite() if test.__class__.__module__=="test_serve" else test
    suite=keep(suite)
result=unittest.TextTestRunner(verbosity=2).run(suite)
raise SystemExit(not result.wasSuccessful())
