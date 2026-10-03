import os
import time
import sys
from pathlib import Path
import pytest
from shiboken6 import isValid

os.environ['QT_QPA_PLATFORM'] = 'offscreen'
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QThreadPool
from home_guard_project.admin.theme import apply_theme


@pytest.fixture(scope='session')
def app():
    application = QApplication.instance() or QApplication([])
    apply_theme(application)
    yield application
    QThreadPool.globalInstance().waitForDone()


@pytest.fixture
def wait(app):
    def until(predicate, timeout=4):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            app.processEvents()
            if predicate():
                return
            time.sleep(.005)
        assert predicate(), 'Timed out waiting for Qt result'
    return until


@pytest.fixture
def widgets(app):
    created = []
    yield created
    QThreadPool.globalInstance().waitForDone()
    app.processEvents()
    for widget in created:
        if isValid(widget):
            widget.close()
            widget.deleteLater()
    app.processEvents()
