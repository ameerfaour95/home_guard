"""The app in Hebrew: every string translated, the language chosen the right way, right to left on screen."""
import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import json
import string
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QLabel

from home_guard_project.box.app import strings
from home_guard_project.box.app.strings_he import TEXT_HE

HEBREW = range(0x05D0, 0x05EB)


def fields(text):
    return sorted(f for _, f, _, _ in string.Formatter().parse(text) if f)


class Hebrew(unittest.TestCase):
    def setUp(self):
        self.addCleanup(strings.set_language, 'en')        # the language is process-wide: always give it back


class CopyTest(Hebrew):
    def test_every_string_has_hebrew_with_the_same_placeholders(self):
        self.assertEqual(set(TEXT_HE), set(strings.TEXT_EN))
        for key, english in strings.TEXT_EN.items():
            hebrew = TEXT_HE[key]
            if isinstance(english, list):
                self.assertEqual(len(hebrew), len(english), key)
                continue
            self.assertEqual(fields(hebrew), fields(english), key)
            hebrew.format(**{f.split(':')[0].split('.')[0]: 1 for f in fields(english)}) if key not in (
                'review_text',) else None

    def test_hebrew_copy_is_hebrew(self):
        # Product names, addresses and technical words stay in Latin letters; everything else reads in Hebrew.
        latin_only = {'premium_home', 'brand', 'chat_button', 'chat_meta', 'ai_record_meta', 'detection_label',
                      'object_count', 'box_peer', 'step_number', 'step_done', 'event_line', 'separator', 'bullet',
                      'gb', 'demo_address', 'demo_site', 'demo_user', 'wifi', 'language_en', 'demo_chat_arabic',
                      'demo_chat_arabic_answer', 'detail_channel_ok', 'date_format'}
        for key, value in TEXT_HE.items():
            if isinstance(value, list) or key in latin_only:
                continue
            self.assertTrue(any(ord(c) in HEBREW for c in value), (key, value))

    def test_set_language_rewrites_text_in_place(self):
        text = strings.TEXT
        strings.set_language('he')
        self.assertIs(strings.TEXT, text)                      # modules that imported TEXT see Hebrew too
        self.assertEqual(strings.tr('next'), 'המשך')
        self.assertEqual(strings.TEXT['step_names'][5], 'המפה')
        self.assertTrue(strings.is_rtl())
        strings.set_language('en')
        self.assertEqual(strings.tr('next'), 'Continue')
        self.assertEqual(strings.set_language('fr'), 'en')

    def test_latin_inside_hebrew_is_isolated_and_the_line_reads_right_to_left(self):
        strings.set_language('he')
        line = strings.tr('summary_house', house='cedar_house')
        self.assertIn('⁨cedar_house⁩', line)
        self.assertEqual(strings.tr('camera_count', count=3), '‏3 מצלמות')   # starts with a number: RLM
        self.assertEqual(strings.tr('stopped_title'), '‏Home Guard כבוי')
        self.assertEqual(strings.tr('next'), 'המשך')                                # plain Hebrew: untouched
        strings.set_language('en')
        self.assertEqual(strings.tr('summary_house', house='cedar_house'), 'Home: cedar_house')

    def test_dates_and_times_inside_hebrew_keep_their_order(self):
        # "מ-08.10 20:20" showed as "00:20 08.10-מ": a digits-only run reorders unless it is isolated.
        from home_guard_project.box.app import scene_strings
        self.assertEqual(scene_strings.st('restore_from', 'he', when='08.10 20:20'), 'מ-⁨08.10 20:20⁩')
        self.assertEqual(scene_strings.st('restore_from', 'en', when='08.10 20:20'), 'from 08.10 20:20')
        strings.set_language('he')
        self.assertIn('⁨14:32⁩', strings.tr('event_line', time='14:32', text='נשמר קטע'))
        self.assertEqual(strings.isolate('הגינה'), 'הגינה')                       # plain Hebrew: untouched
        self.assertEqual(strings.isolate(3), 3)                                  # numbers for format specs: untouched
        strings.set_language('en')
        self.assertEqual(strings.isolate('14:32'), '14:32')

    def test_relative_times_and_live_badges_follow_the_language(self):
        from home_guard_project.box.app.liveness import frame_health, relative_time
        strings.set_language('he')
        self.assertEqual(frame_health(100, 103, 95), ('שידור חי', False))
        self.assertIn('לפני 30', relative_time(70, 100))
        strings.set_language('en')
        self.assertEqual(frame_health(100, 103, 95), ('LIVE', False))
        self.assertEqual(relative_time(70, 100), '30 s ago')


class ChoosingTheLanguageTest(Hebrew):
    def test_the_box_speaks_its_owners_language(self):
        from home_guard_project.box.app.preferences import app_language
        box = SimpleNamespace(lang=None, demo=False, setup=False, remote_box=None)
        with mock.patch('home_guard_project.box.boxconfig.load_box_settings', return_value={'owner_language': 'he'}):
            self.assertEqual(app_language(box), 'he')
        with mock.patch('home_guard_project.box.boxconfig.load_box_settings', side_effect=OSError('no box.yaml')):
            self.assertEqual(app_language(box), 'en')
        self.assertEqual(app_language(SimpleNamespace(lang='he', demo=True)), 'he')      # --lang wins
        self.assertEqual(app_language(SimpleNamespace(lang=None, demo=True)), 'en')

    def test_setup_on_a_laptop_remembers_the_installers_choice(self):
        from home_guard_project.box.app import preferences
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'language.json'
            pref = preferences.LanguagePreference(path)
            self.assertEqual(pref.load(), 'en')
            pref.save('he')
            self.assertEqual(json.loads(path.read_text(encoding='utf-8')), {'language': 'he'})
            with self.assertRaises(ValueError):
                pref.save('fr')
            with mock.patch.object(preferences, 'LanguagePreference', return_value=pref):
                setup = SimpleNamespace(lang=None, demo=False, setup=True, remote_box=None)
                self.assertEqual(preferences.app_language(setup), 'he')


class ScreenTest(Hebrew):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        super().setUp()
        self.addCleanup(self.app.setLayoutDirection, Qt.LayoutDirection.LeftToRight)

    def window(self, lang, **extra):
        from home_guard_project.box.app.ui import Window
        strings.set_language(lang)
        self.app.setLayoutDirection(Qt.LayoutDirection.RightToLeft if lang == 'he' else Qt.LayoutDirection.LeftToRight)
        values = dict(demo=True, remote_box=None, aspect='16:9', detections=False, theme='dark', panel=None, setup=True,
                      fail=None, wifi=False, skip_cameras=False, alerts=False, details=False, technical_log=False,
                      state='mixed', cameras=3, page=None, scene=None, lang=lang, size='1366x768', screenshot=None)
        values.update(extra)
        window = Window(SimpleNamespace(**values))
        from home_guard_project.box.app import camera_display
        self.addCleanup(camera_display.set_names, {}, ())
        self.addCleanup(lambda: (window.close(), window.deleteLater()))
        return window

    def test_setup_in_hebrew_runs_right_to_left_with_the_map_step(self):
        window = self.window('he')
        self.assertEqual(window.layoutDirection(), Qt.LayoutDirection.RightToLeft)
        steps = [l.text().lstrip('‏') for l in window.step_labels]
        self.assertEqual(steps[:2], ['1. חיבור', '2. רשת'])
        self.assertEqual(steps[5], '6. המפה')
        self.assertEqual(window.windowTitle(), strings.tr('setup_window_title'))
        self.assertTrue(window.language_buttons['he'].isChecked())
        self.assertFalse(window.language_widget.isHidden())

    def test_choosing_english_on_the_first_page_reopens_setup_in_english(self):
        window = self.window('he')
        window.language_buttons['en'].click()
        english = window.language_window
        self.addCleanup(lambda: (english.close(), english.deleteLater()))
        self.assertEqual(strings.LANG, 'en')
        self.assertEqual(self.app.layoutDirection(), Qt.LayoutDirection.LeftToRight)
        self.assertEqual(english.step_labels[0].text(), '1. Connect')
        self.assertFalse(window.isVisible())

    def test_the_box_screen_in_hebrew_shows_no_english_words_of_ours(self):
        window = self.window('he', setup=False, page=None)
        window.tick()
        english = set()
        for widget in window.findChildren(QLabel):
            text = widget.text()
            for word in strings.TEXT_EN.values():
                if isinstance(word, str) and len(word) > 6 and word == text:
                    english.add(text)
        self.assertEqual(english, set())


if __name__ == '__main__':
    unittest.main()
