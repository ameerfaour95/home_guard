"""Subprocess proof: the designed states start without Qt warnings or hardware."""
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from home_guard_project.box.app.theme import stylesheet

class StartupTests(unittest.TestCase):
    def test_theme_never_sets_pixel_font_with_negative_point_size(self):
        for theme in ('dark','light'):
            self.assertNotRegex(stylesheet(theme),r'font-size:\s*[\d.]+px')
            self.assertIn('font-size: 12pt',stylesheet(theme))

    def test_screenshots_start_with_empty_stderr(self):
        env=os.environ.copy();env.pop('VIRTUAL_ENV',None);env['QT_QPA_PLATFORM']='offscreen'
        screens=(['--state','no-cameras'],['--state','box-unreachable'],['--setup','--page','failure','--fail','cameras'],['--setup','--page','failure','--fail','network'])
        with tempfile.TemporaryDirectory() as directory:
            for i,args in enumerate(screens):
                with self.subTest(screen=args):
                    path=Path(directory)/f'{i}.png'
                    result=subprocess.run([sys.executable,'-m','home_guard_project.box.app','--demo',*args,'--screenshot',str(path)],env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,timeout=30)
                    self.assertEqual(result.returncode,0,result.stderr)
                    self.assertEqual(result.stderr,'')
                    self.assertTrue(path.is_file())

    def test_styled_qt_fonts_keep_valid_point_sizes(self):
        env=os.environ.copy();env.pop('VIRTUAL_ENV',None);env['QT_QPA_PLATFORM']='offscreen'
        code='''from PySide6.QtWidgets import QApplication,QWidget,QLabel,QComboBox,QTextEdit,QVBoxLayout
from home_guard_project.box.app.theme import stylesheet
app=QApplication([])
root=QWidget();root.setStyleSheet(stylesheet());layout=QVBoxLayout(root)
for kind in (QLabel,QComboBox,QTextEdit): layout.addWidget(kind())
root.show();app.processEvents()
for widget in root.findChildren(QWidget):
    assert widget.font().pointSizeF()>0, (type(widget).__name__,widget.font().pointSizeF())
'''
        result=subprocess.run([sys.executable,'-c',code],env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,timeout=30)
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertEqual(result.stderr,'')
