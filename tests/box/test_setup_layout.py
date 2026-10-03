import os
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
import unittest
from types import SimpleNamespace
from pathlib import Path
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

    def test_recorded_summary_and_failure_keep_details_and_copy_usable(self):
        from home_guard_project.box.app.engine_backend import OutputParser
        args=SimpleNamespace(demo=True,setup=True,theme='dark',panel=None,fail=None,wifi=False,skip_cameras=False,alerts=False,details=False,state='mixed',cameras=3,page='progress',size='1366x768',screenshot=None)
        window=Window(args);window.show();window.setup_details.reset()
        recording=Path(__file__).with_name('fixtures').joinpath('setup_engine_success.txt').read_text()
        parser=OutputParser()
        for line in recording.splitlines(): window.engine_events.put(parser.parse(line))
        window.present_engine_events();window.set_page(6);window.summary_details_action.setChecked(True);self.app.processEvents()
        workspace=window.summary_workspace
        bounds=workspace.step_list.rect().translated(workspace.step_list.mapTo(workspace.viewport(),QPoint(0,0)))
        self.assertTrue(workspace.viewport().rect().contains(bounds),(bounds,workspace.viewport().rect()))
        details=window.summary_details
        details.select('network');self.assertFalse(details.selection.following)
        details.copy();self.assertEqual(details.copy_button.text(),'Copied')
        self.assertEqual(details.copy_timer.interval(),2000)
        details.copy_timer.timeout.emit();self.assertEqual(details.copy_button.text(),'Copy')
        self.assertTrue(QApplication.clipboard().text())
        details.technical.setChecked(True);details.copy()
        self.assertEqual(QApplication.clipboard().text(),details.model.technical_log('network'))
        self.assertNotIn('@@step connect',QApplication.clipboard().text())
        details.follow_progress();self.assertTrue(details.selection.following)
        window.close();self.app.processEvents()

    def test_details_twice_restore_exact_geometry(self):
        for size in ('1920x1080','1366x768','1000x650'):
            args=SimpleNamespace(demo=True,setup=True,theme='dark',panel=None,fail=None,wifi=False,skip_cameras=False,alerts=False,details=False,state='mixed',cameras=3,page='progress',size=size,screenshot=None)
            window=Window(args);window.show();self.app.processEvents()
            workspace=window.setup_workspace
            running=next(row for row in workspace.rows if row.indicator.isVisible())
            def geometry(widget):
                return widget.rect().translated(widget.mapTo(window,QPoint()))
            before=(geometry(running.indicator),geometry(workspace.rows[0]))
            for _ in range(2):
                workspace.toggle.setChecked(True);self.app.processEvents()
                self.assertEqual(before,(geometry(running.indicator),geometry(workspace.rows[0])))
                workspace.toggle.setChecked(False);self.app.processEvents()
                self.assertEqual(before,(geometry(running.indicator),geometry(workspace.rows[0])))
            window.close();self.app.processEvents()
