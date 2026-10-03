import json
import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication
from home_guard_project.box.app.ui import Window
from home_guard_project.box.app.engine_backend import answers_payload, ENGINE_STEPS, OutputParser


class OwnerConsentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.app = QApplication.instance() or QApplication([])

    def window(self, size='1366x768'):
        args = SimpleNamespace(demo=True, setup=True, theme='dark', panel=None, fail=None,
                               wifi=False, skip_cameras=False, alerts=False, details=False,
                               state='mixed', cameras=3, page='owner-consent', size=size, screenshot=None)
        window = Window(args); window.show(); self.app.processEvents()
        self.addCleanup(window.close)
        return window

    def test_defaults_validation_and_navigation(self):
        window = self.window()
        for toggle in window.consent_toggles.values(): self.assertFalse(toggle.isChecked())
        window.inputs['owner_name'].clear()
        window.next_page()
        self.assertTrue(window.field_guidance['owner_name'].isVisible())
        self.assertTrue(window.inputs['owner_name'].hasFocus())
        for value in (' ', 'x' * 121, 'bad\x01name'):
            window.inputs['owner_name'].setText(value)
            self.assertFalse(window.valid_page(3))
        for value in ('A', 'x' * 120, 'דנה כהן'):
            window.inputs['owner_name'].setText(value)
            self.assertTrue(window.valid_page(3))
        window.back.click(); self.assertEqual(window.pages.currentIndex(), 2)
        window.next_page(); self.assertEqual(window.pages.currentIndex(), 3)
        window.next_page(); self.assertEqual(window.pages.currentIndex(), 4)
        window.back.click(); self.assertEqual(window.pages.currentIndex(), 3)

    def test_keyboard_and_answers_payload(self):
        window = self.window()
        window.inputs['owner_name'].setText('דנה כהן')
        window.inputs['owner_phone'].setText('+972 50 123 4567')
        window.inputs['installer'].setText('Sam')
        toggle = window.consent_toggles['live']; toggle.setFocus()
        QTest.keyClick(toggle, Qt.Key.Key_Space)
        self.assertTrue(toggle.isChecked())
        QTest.keyClick(toggle, Qt.Key.Key_Tab)
        self.assertTrue(window.consent_toggles['recordings'].hasFocus())
        payload = answers_payload(window.collect_answers())
        for key, value in dict(owner_name='דנה כהן', owner_phone='+972 50 123 4567', installer='Sam',
                               consent_live=True, consent_recordings=False, consent_training=False).items():
            self.assertEqual(payload[key], value)
        self.assertNotIn('דנה כהן', repr(window.collect_answers()))
        self.assertNotIn('+972', repr(window.collect_answers()))

    def test_installer_preference_only_and_atomic_failure(self):
        from home_guard_project.box.app.preferences import InstallerPreference
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'installer.json'; preference = InstallerPreference(path)
            window = self.window(); window.installer_preference = preference
            window.args.demo = False
            window.inputs['owner_name'].setText('Private owner')
            window.inputs['installer'].setText('Sam'); window.next_page()
            self.assertEqual(InstallerPreference(path).load(), 'Sam')
            self.assertEqual(json.loads(path.read_text()), {'installer': 'Sam'})
            with patch('home_guard_project.box.app.preferences.os.replace', side_effect=OSError):
                with self.assertRaises(OSError): preference.save('Changed')
            self.assertEqual(preference.load(), 'Sam')
            self.assertEqual(list(path.parent.glob('*.tmp')), [])
            path.write_text('[]'); self.assertEqual(preference.load(), '')

    def test_register_step_display_and_parser(self):
        window = self.window()
        self.assertEqual(ENGINE_STEPS[3], 'register')
        self.assertEqual(window.setup_workspace.rows[3].title.text(), 'Adding the customer to Home Guard')
        event = OutputParser().parse('@@step register warn saved; retrying publication')
        window.setup_details.feed(event)
        self.assertEqual(window.setup_details.model.groups['register'].status, 'warn')

    def test_page_fits_both_sizes(self):
        for size in ('1366x768', '1920x1080'):
            window = self.window(size)
            scroll = window.owner_scroll
            self.assertEqual(scroll.verticalScrollBar().maximum(), 0)
            self.assertEqual(scroll.horizontalScrollBar().maximum(), 0)
            window.next_page(); self.app.processEvents()
            self.assertEqual(scroll.verticalScrollBar().maximum(), 0)
            positions = [window.inputs[key].mapTo(window, window.inputs[key].rect().topLeft()).y()
                         for key in ('owner_name', 'owner_phone', 'installer')]
            self.assertEqual(len(set(positions)), 1)
            for widget in (*window.consent_toggles.values(), window.inputs['owner_name']):
                self.assertTrue(widget.isVisible())
