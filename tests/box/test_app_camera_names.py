"""Camera names on the app's screens come only from the box; an id is never shown (owner rule, 2026-10-08)."""
import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import unittest
from unittest import mock

from PySide6.QtWidgets import QAbstractButton, QApplication, QLabel, QLineEdit

from home_guard_project.box.app import camera_display
from home_guard_project.box.app.scene_backend import DemoSceneBackend

IDS = ['front_door', 'ameer_week_0_1_ch3']


def tr_(key, **values):
    from home_guard_project.box.app.strings import tr
    return tr(key, **values)


def texts(widget):
    out = [w.text() for w in widget.findChildren(QLabel)]
    out += [w.text() for w in widget.findChildren(QAbstractButton)]
    out += [w.text() for w in widget.findChildren(QLineEdit)]
    return [t for t in out if t]


class CameraNamesTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        camera_display.set_names({}, ())
        self.addCleanup(camera_display.set_names, {}, ())
        # The app never asks camera_names: the box answers names --json.
        refuse = AssertionError('the app must not name cameras itself')
        for f in ('display_name', 'replace_ids', 'family_names'):
            patcher = mock.patch(f'home_guard_project.box.camera_names.{f}', side_effect=refuse)
            patcher.start(); self.addCleanup(patcher.stop)

    def page(self):
        from home_guard_project.box.app.box_controls import BoxControls
        from home_guard_project.box.app.camera_controls import CameraControls
        from home_guard_project.box.app.camera_ui import CameraPage
        page = CameraPage(CameraControls(BoxControls(demo=True), IDS), lambda: None)
        page.widget.resize(1200, 650); page.widget.show()
        self.addCleanup(lambda: (page.close(), page.widget.deleteLater()))
        return page

    def test_the_card_shows_the_boxs_name_and_never_the_id(self):
        page = self.page()
        page.render(page.load_photos())
        fields = [field for _id, field, _switch in page.rows]
        self.assertEqual([f.text() for f in fields], ['Front door', ''])     # the box names one...
        self.assertEqual(fields[1].placeholderText(), 'Camera 2 of 2')     # ...the other reads by its place
        shown = texts(page.widget) + [f.placeholderText() for f in fields]
        self.assertFalse([t for t in shown if any(i in t for i in IDS)], shown)

    def test_saving_switches_cameras_and_never_renames_an_id(self):
        page = self.page()
        page.render(page.load_photos())
        page.rows[1][2].blockSignals(True); page.rows[1][2].setChecked(False)
        self.assertEqual(page.changes(), [('front_door', 'front_door', True),
                                          ('ameer_week_0_1_ch3', 'ameer_week_0_1_ch3', False)])

    def test_editing_the_name_sets_it_on_the_box_and_never_renames_the_id(self):
        page = self.page()
        page.render(page.load_photos())
        backend = mock.Mock()
        backend.set_name.return_value = {'he': 'הגינה', 'en': 'The garden'}
        page.rows[1][1].setText('The garden')
        self.assertEqual(page.name_edits(), {'ameer_week_0_1_ch3': 'The garden'})
        self.assertTrue(page.save.isEnabled())
        with mock.patch('home_guard_project.box.app.scene_backend.scene_backend_for', return_value=backend):
            page.save_clicked()
            page.future.result(timeout=5); page.poll()
        backend.set_name.assert_called_once_with('ameer_week_0_1_ch3', 'The garden')
        self.assertEqual([c.name for c in page.controls.records], IDS)      # the ids stay as they are
        self.assertEqual(page.rows[1][1].text(), 'The garden')
        self.assertEqual(camera_display.shown('ameer_week_0_1_ch3'), 'The garden')

    def test_a_new_name_reaches_the_tiles_the_feed_and_the_map_title(self):
        from types import SimpleNamespace
        from home_guard_project.box.app.ui import Window
        from home_guard_project.box.app.scene_editor import open_map_dialog
        window = Window(SimpleNamespace(demo=True, remote_box=None, aspect='16:9', detections=False, theme='dark',
                                        panel='cameras', setup=False, fail=None, wifi=False, skip_cameras=False,
                                        alerts=False, details=False, technical_log=False, state='mixed', cameras=2,
                                        page=None, scene=None, lang='en', size='1366x768', screenshot=None))
        self.addCleanup(lambda: (window.close(), window.deleteLater()))
        window.tick()
        page = window.cameras_page
        page.future.result(timeout=5); page.poll()
        self.assertEqual([t.caption.text() for t in window.tiles], ['Front door', 'Garden'])
        backend = mock.Mock(); backend.set_name.return_value = {'he': 'הגינה', 'en': 'The lawn'}
        page.rows[1][1].setText('The lawn')
        with mock.patch('home_guard_project.box.app.scene_backend.scene_backend_for', return_value=backend):
            page.save_clicked()
            page.future.result(timeout=5); page.poll()
        backend.set_name.assert_called_once_with('garden', 'The lawn')
        self.assertEqual([t.caption.text() for t in window.tiles], ['Front door', 'The lawn'])
        from home_guard_project.box.app.model import parse_activity
        self.assertEqual(parse_activity('12:00:00 INFO [garden] trigger saved: clip.mp4').text, 'Clip saved from The lawn')
        dialog = open_map_dialog(page, 'garden', demo_state='loading', display=page.camera_names.get('garden', ''))
        self.addCleanup(lambda: (dialog.editor.close_jobs(), dialog.deleteLater()))
        self.assertEqual(dialog.editor.title.text(), 'The lawn')

    def test_two_cameras_cannot_share_a_name_and_a_refusal_reads_plainly(self):
        page = self.page()
        page.render(page.load_photos())
        page.rows[1][1].setText('front DOOR')                             # the other camera's name
        self.assertFalse(page.save.isEnabled())
        self.assertEqual(page.note.text(), tr_('camera_names_invalid'))
        page.rows[1][1].setText('Garden')
        backend = mock.Mock(); backend.set_name.side_effect = RuntimeError('"Garden" already names x')
        with mock.patch('home_guard_project.box.app.scene_backend.scene_backend_for', return_value=backend):
            page.save_clicked()
            with self.assertRaises(Exception):
                page.future.result(timeout=5)
            page.poll()
        self.assertEqual(page.note.text(), tr_('camera_name_save_failed', name='Garden'))
        self.assertFalse([t for t in texts(page.widget) if any(i in t for i in IDS)])

    def test_a_box_that_cannot_say_leaves_the_place_in_the_list(self):
        from home_guard_project.box.app.camera_ui import box_names
        failing = mock.Mock(); failing.names.side_effect = RuntimeError('old box')
        with mock.patch('home_guard_project.box.app.scene_backend.scene_backend_for', return_value=failing):
            self.assertEqual(box_names(object()), {})
        camera_display.set_names({}, IDS)
        self.assertEqual([camera_display.shown(i) for i in IDS], ['Camera 1 of 2', 'Camera 2 of 2'])
        self.assertEqual(camera_display.shown('unknown_ch9'), 'Camera')

    def test_dashboard_tiles_and_the_feed_use_the_boxs_names(self):
        from home_guard_project.box.app import model, timeline
        from home_guard_project.box.app.ui import CameraTile
        camera_display.set_names({'front_door': 'Front door'}, IDS)
        tiles = [CameraTile(i) for i in IDS]
        self.addCleanup(lambda: [t.deleteLater() for t in tiles])
        self.assertEqual([t.caption.text() for t in tiles], ['Front door', 'Camera 2 of 2'])
        line = '14:32:10 INFO [ameer_week_0_1_ch3] trigger saved: local clip'
        self.assertNotIn('ameer', model.parse_activity(line).text)
        import time
        now = time.time()
        self.assertEqual(timeline.thinking_camera({'thinking': {'camera': 'front_door', 'ts': now}}, now), 'Front door')

    def test_the_demo_box_names_every_demo_camera(self):
        from home_guard_project.box.app.strings import TEXT
        ids = [n.lower().replace(' ', '_') for n in TEXT['demo_names']]
        names = DemoSceneBackend().names('he')
        self.assertTrue(all(names.get(i) for i in ids), names)


if __name__ == '__main__':
    unittest.main()
