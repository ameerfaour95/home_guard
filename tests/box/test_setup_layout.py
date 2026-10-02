import os
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
import unittest
from types import SimpleNamespace
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QPoint
from home_guard_project.box.app.ui import Window

class SetupLayoutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.app=QApplication.instance() or QApplication([])
    def test_details_never_shrink_or_overlap_step_rows(self):
        for size in ('1366x768','1920x1080','1000x650'):
            for opened in (False,True):
                with self.subTest(size=size,details=opened):
                    args=SimpleNamespace(demo=True,setup=True,theme='dark',panel=None,fail=None,wifi=False,skip_cameras=False,alerts=False,details=opened,state='mixed',cameras=3,page='progress',size=size,screenshot=None)
                    window=Window(args);window.show();self.app.processEvents()
                    workspace=window.setup_workspace;rects=[]
                    for row in workspace.rows:
                        rect=row.rect().translated(row.mapTo(workspace.step_list,QPoint(0,0)))
                        self.assertTrue(workspace.step_list.rect().contains(rect),(size,rect,workspace.step_list.rect()))
                        self.assertEqual(row.height(),50)
                        self.assertFalse(any(rect.intersects(other) for other in rects))
                        rects.append(rect)
                    if size!='1000x650':
                        bounds=workspace.step_list.rect().translated(workspace.step_list.mapTo(workspace.viewport(),QPoint(0,0)))
                        self.assertTrue(workspace.viewport().rect().contains(bounds),(bounds,workspace.viewport().rect()))
                    window.close();self.app.processEvents()
