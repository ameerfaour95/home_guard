"""Access notices in the app: the words for each kind (never the cloud's message), RTL times, the bell's count."""
import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import json
import unittest
from types import SimpleNamespace
from unittest import mock

from PySide6.QtWidgets import QApplication, QLabel

from home_guard_project.box.app import notices_ui as nu
from home_guard_project.box.app import strings

FSI, PDI = '⁨', '⁩'
MESSAGE = 'Home Guard support viewed recordings from x (99:99)'      # the cloud's sentence: never shown for a known kind


def notice(kind='recording', cameras=('Front door', 'Driveway'), frm='2026-10-09T13:15', to='2026-10-09T13:40',
           read=False, nid='1', staff='Noa Cohen', message=MESSAGE):
    return {'id': nid, 'kind': kind, 'cameras': list(cameras), 'staff_name': staff, 'from_local': frm,
            'to_local': to, 'message': message, 'read': read}


class WordsTest(unittest.TestCase):
    def setUp(self):
        self.addCleanup(strings.set_language, 'en')

    def test_time_is_a_moment_a_range_in_one_day_or_two_dates(self):
        self.assertEqual(nu.time_text('2026-10-09T13:15', '2026-10-09T13:15'), '09.10 13:15')
        self.assertEqual(nu.time_text('2026-10-09T13:15', '2026-10-09T13:40'), '09.10 13:15–13:40')
        self.assertEqual(nu.time_text('2026-10-09T23:50', '2026-10-10T00:20'), '09.10 23:50 – 10.10 00:20')
        self.assertEqual(nu.time_text('bad'), '')

    def test_english_lines_for_each_kind(self):
        strings.set_language('en')
        self.assertEqual(nu.notice_line(notice()),
                         'Home Guard support viewed recordings from Front door and Driveway (09.10 13:15–13:40)')
        self.assertEqual(nu.notice_line(notice(cameras=('Garden', 'Front door', 'Driveway'))),
                         'Home Guard support viewed recordings from Garden, Front door and Driveway '
                         '(09.10 13:15–13:40)')
        self.assertEqual(nu.notice_line(notice(cameras=())),
                         'Home Guard support viewed recordings (09.10 13:15–13:40)')
        self.assertEqual(nu.notice_line(notice('chat', cameras=(), to='2026-10-09T13:15')),
                         'Home Guard support viewed your chat with the assistant (09.10 13:15)')
        self.assertEqual(nu.secondary_line(notice()), 'By Noa Cohen')
        self.assertEqual(nu.secondary_line(notice(staff='')), '')

    def test_hebrew_lines_for_each_kind_with_times_isolated(self):
        strings.set_language('he')
        when = FSI + '09.10 13:15–13:40' + PDI
        self.assertEqual(nu.notice_line(notice(cameras=('דלת הכניסה', 'החניה'))),
                         f'צוות Home Guard צפה בהקלטות מדלת הכניסה והחניה ({when})')
        self.assertEqual(nu.notice_line(notice(cameras=())), f'צוות Home Guard צפה בהקלטות ({when})')
        self.assertEqual(nu.notice_line(notice('chat', cameras=())),
                         f'צוות Home Guard צפה בשיחה שלך עם העוזר ({when})')
        self.assertEqual(nu.secondary_line(notice()), f'על ידי {FSI}Noa Cohen{PDI}')

    def test_hebrew_joins_a_latin_name_with_a_hyphen_and_isolates_it(self):
        strings.set_language('he')
        line = nu.notice_line(notice(cameras=('Garage',)))
        self.assertIn(f'בהקלטות מ-{FSI}Garage{PDI}', line)
        self.assertIn(f'ו-{FSI}Garage{PDI}', nu.cameras_text(['הגינה', 'Garage'], 'he'))
        self.assertEqual(nu.cameras_text(['הגינה', 'מצלמה 6'], 'he'), f'הגינה ו{FSI}מצלמה 6{PDI}')

    def test_the_clouds_message_is_never_parsed_or_shown_for_a_known_kind(self):
        for lang in ('en', 'he'):
            strings.set_language(lang)
            for kind in nu.KINDS:
                self.assertNotIn('99:99', nu.notice_line(notice(kind)))

    def test_an_unknown_kind_shows_the_clouds_english_message(self):
        strings.set_language('he')
        self.assertEqual(nu.notice_line(notice('other', message='Support watched live video')),
                         'Support watched live video')
        self.assertIn('צפה במידע שלך', nu.notice_line(notice('other', message='')))

    def test_the_app_shows_the_boxs_camera_names_and_never_names_cameras_itself(self):
        refuse = AssertionError('the box names the cameras (notices list --lang)')
        with mock.patch('home_guard_project.box.camera_names.display_name', side_effect=refuse):
            line = nu.notice_line(notice(cameras=('Camera 6',)))
        self.assertIn('from Camera 6 (', line)

    def test_unread_count(self):
        self.assertEqual(nu.unread_count([notice(), notice(read=True), notice(nid='3')]), 2)
        self.assertEqual(nu.unread_count([]), 0)


class BackendTest(unittest.TestCase):
    def test_commands_run_the_boxs_notices_cli_here_or_over_ssh(self):
        runner = mock.Mock(return_value=SimpleNamespace(returncode=0, stdout='INFO x\n' + json.dumps(
            {'notices': [notice()], 'unread': 1})))
        backend = nu.NoticesBackend(runner=runner)
        backend.box.python = 'py'
        self.assertEqual(backend.list('en')[0]['id'], '1')
        self.assertEqual(runner.call_args.args[0],
                         ['py', '-m', 'home_guard_project.box', 'notices', 'list', '--lang', 'en', '--json'])
        remote = nu.NoticesBackend('admin@10.0.0.5', runner=mock.Mock())
        remote.box.runner.run.return_value = SimpleNamespace(returncode=0, stdout='{"marked": 2, "unread": 0}')
        self.assertEqual(remote.read_all(), {'marked': 2, 'unread': 0})
        line = remote.box.runner.run.call_args.args[0]
        self.assertEqual(line[0], 'ssh.exe')
        self.assertTrue(line[-1].endswith('-m home_guard_project.box notices read-all --json'), line[-1])

    def test_what_the_cloud_pushes_is_what_the_app_lists(self):
        # The real box command in a child process, its home folder a temporary one (paths.py's HOMEGUARD_HOME).
        import base64, subprocess, sys, tempfile
        home = tempfile.TemporaryDirectory(); self.addCleanup(home.cleanup)
        env = dict(os.environ, HOMEGUARD_HOME=home.name)
        data = {'schema_version': 1, 'id': 41, 'kind': 'recording', 'staff_name': 'Noa', 'cameras': ['Front door'],
                'from_utc': '2026-10-09T10:15:00Z', 'to_utc': '2026-10-09T10:40:00Z', 'message': 'x'}
        value = base64.b64encode(json.dumps(data).encode()).decode()
        added = subprocess.run([sys.executable, '-m', 'home_guard_project.box', 'notices', 'add', '--b64', value,
                                '--json'], env=env, capture_output=True, text=True, timeout=120)
        self.assertEqual((added.returncode, json.loads(added.stdout)), (0, {'result': 'added', 'id': 41}))
        with mock.patch.dict(os.environ, {'HOMEGUARD_HOME': home.name}):
            backend = nu.NoticesBackend(); backend.box.python = sys.executable
            rows = backend.list('en')
            self.assertEqual([(r['id'], r['cameras'], r['read']) for r in rows], [('41', ['Front door'], False)])
            backend.read_all()
            self.assertTrue(backend.list('en')[0]['read'])

    def test_a_refusal_raises_for_the_page_to_show(self):
        runner = mock.Mock(return_value=SimpleNamespace(returncode=1, stdout='{"error": "boom"}'))
        with self.assertRaises(Exception):
            nu.NoticesBackend(runner=runner, ).list()

    def test_backend_for_the_demo_this_box_or_the_remote_box(self):
        self.assertIsInstance(nu.notices_backend_for(SimpleNamespace(demo=True)), nu.DemoNoticesBackend)
        self.assertTrue(nu.notices_backend_for(SimpleNamespace(demo=True, notices='empty')).empty)
        self.assertIsNone(nu.notices_backend_for(SimpleNamespace(demo=False, remote_box=None)).box.target)
        self.assertEqual(nu.notices_backend_for(SimpleNamespace(demo=False, remote_box='a@b')).box.target, 'a@b')


def window_args(**extra):
    args = dict(demo=True, remote_box=None, aspect='16:9', detections=False, theme='dark', panel=None, setup=False,
                fail=None, wifi=False, skip_cameras=False, alerts=False, details=False, technical_log=False,
                state='mixed', cameras=2, page=None, scene=None, lang='en', size='1366x768', screenshot=None)
    args.update(extra)
    return SimpleNamespace(**args)


class ScreenTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def wait(self, page):
        if page.future is not None:
            from concurrent.futures import wait
            wait([page.future], timeout=5)
        page.poll()

    def window(self, **extra):
        from home_guard_project.box.app.ui import Window
        window = Window(window_args(**extra))
        self.addCleanup(lambda: (window.close(), window.deleteLater()))
        self.wait(window.notices_page)
        return window

    def test_the_bell_counts_the_unread_and_opening_the_panel_reads_them(self):
        window = self.window()
        page = window.notices_page
        self.assertEqual(window.notices_button.count, 2)
        self.assertEqual(window.notices_button.toolTip(), 'Access notices (2 new)')
        self.assertEqual(page.backend.calls[:1], ['list'])                  # read when the app opens
        window.show(); window.open_notices()
        self.assertEqual(window.content_stack.currentIndex(), 3)
        self.assertEqual(window.notices_button.count, 0)
        self.assertEqual(window.notices_button.toolTip(), 'Access notices')
        page.read_future.result(timeout=5)
        self.assertIn('read-all', page.backend.calls)
        news = [w for w in page.widget.findChildren(QLabel) if w.text() == 'New' and not w.isHidden()]
        self.assertEqual(len(news), 2)                                    # what was new stays marked while open
        page.refresh(); self.wait(page)
        self.assertEqual(window.notices_button.count, 0)                  # the box remembers they were read

    def test_rows_are_newest_first_and_the_empty_state_reads_plainly(self):
        window = self.window()
        lines = [w.text() for w in window.notices_page.widget.findChildren(QLabel) if w.objectName() == 'noticeLine']
        self.assertEqual(len(lines), 4)
        self.assertIn('Front door and Driveway', lines[0])
        empty = self.window(notices='empty')
        texts = [w.text() for w in empty.notices_page.widget.findChildren(QLabel)]
        self.assertIn(strings.tr('notices_empty'), texts)
        self.assertEqual(empty.notices_button.count, 0)

    def test_a_box_that_cannot_answer_shows_a_note_and_keeps_the_app_running(self):
        backend = mock.Mock(); backend.list.side_effect = RuntimeError('ssh failed')
        page = nu.NoticesPage(backend)
        self.addCleanup(lambda: (page.close(), page.widget.deleteLater()))
        page.start(); self.wait(page)
        self.assertTrue(page.failed)
        self.assertFalse(page.note.isHidden())
        self.assertEqual(page.note.text(), strings.tr('notices_unavailable'))

    def test_the_list_is_read_again_every_few_minutes(self):
        backend = mock.Mock(); backend.list.return_value = [notice()]
        counts = []
        page = nu.NoticesPage(backend, counts.append)
        self.addCleanup(lambda: (page.close(), page.widget.deleteLater()))
        page.start(); self.wait(page)
        self.assertEqual(counts, [1])
        self.assertEqual(page.refresh_timer.interval(), nu.POLL_SECONDS * 1000)
        self.assertTrue(page.refresh_timer.isActive())
        backend.list.return_value = [notice(), notice(nid='2')]
        page.refresh_timer.timeout.emit(); self.wait(page)
        self.assertEqual(counts, [1, 2])

    def test_a_list_read_before_the_box_marked_them_does_not_bring_the_count_back(self):
        backend = mock.Mock(); backend.list.return_value = [notice(), notice(nid='2')]
        counts = []
        page = nu.NoticesPage(backend, counts.append)
        self.addCleanup(lambda: (page.close(), page.widget.deleteLater()))
        page.start(); self.wait(page)
        page.open()
        page.refresh(); self.wait(page)                         # the box still says unread (not marked yet)
        self.assertEqual(counts, [2, 0, 0])
        backend.list.return_value = [notice(cameras=('Front door', 'Driveway', 'Garden')), notice(nid='2')]
        page.refresh(); self.wait(page)                         # the cloud rewrote notice 1: it is new again
        self.assertEqual(counts[-1], 1)

    def test_the_badge_paints_in_both_directions(self):
        from PySide6.QtCore import Qt
        for direction in (Qt.LayoutDirection.LeftToRight, Qt.LayoutDirection.RightToLeft):
            button = nu.BadgeButton(); button.setLayoutDirection(direction); button.resize(44, 44)
            button.set_count(12)
            self.assertFalse(button.grab().isNull())
            button.deleteLater()


if __name__ == '__main__':
    unittest.main()
