"""Local recorded proof: python -m home_guard_project.box.app.proof."""
import os
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
from pathlib import Path
from types import SimpleNamespace
from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QFont, QFontDatabase
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from .ui import Window

def main():
    app=QApplication.instance() or QApplication([])
    font_dir=Path(os.environ.get('WINDIR','C:/Windows'))/'Fonts'
    for name in ('segoeui.ttf','consola.ttf'):
        if (font_dir/name).exists(): QFontDatabase.addApplicationFont(str(font_dir/name))
    app.setFont(QFont('Segoe UI',11))
    destination=Path(__file__).parent/'assets'/'proof';destination.mkdir(exist_ok=True)
    for size in ('1920x1080','1366x768'):
        for name,options in (
            ('wizard-closed',dict(setup=True,page='progress')),
            ('wizard-open',dict(setup=True,page='progress',details=True)),
            ('detections',dict(state='live-detections',detections=True)),
            ('off-stage',dict(state='off-camera')),
            ('off-cameras',dict(state='off-camera',panel='cameras')),
            ('settings',dict(panel='settings')),
            ('transition',dict()),
        ):
            values=dict(demo=True,setup=False,theme='dark',panel=None,fail=None,wifi=False,skip_cameras=False,alerts=False,details=False,state='live-detections',cameras=3,page=None,size=size,screenshot=None,detections=False)
            values.update(options);window=Window(SimpleNamespace(**values));window.show();QTest.qWait(350)
            if name=='transition':
                window.open_settings();window.content_stack.transition.animation.pause();window.content_stack.transition.animation.setCurrentTime(120)
            assert window.grab().save(str(destination/(size+'-'+name+'.png')))
            if name=='detections':
                button=window.detection_toggle
                QTest.mousePress(button,Qt.MouseButton.LeftButton)
                for step in (0,60,120):
                    button._motion_overlay.animation.pause();button._motion_overlay.animation.setCurrentTime(step)
                    window.grab().save(str(destination/(size+'-press-'+str(step)+'.png')))
                QTest.mouseRelease(button,Qt.MouseButton.LeftButton)
                window.open_settings();transition=window.content_stack.transition;transition.animation.pause()
                for step in (0,60,120,180,239):
                    transition.animation.setCurrentTime(step);window.grab().save(str(destination/(size+'-page-'+str(step)+'.png')))
            window.close();app.processEvents()
    return 0

if __name__=='__main__': raise SystemExit(main())
