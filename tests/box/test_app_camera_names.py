"""Camera names on the app's screens come only from the box; an id is never shown (owner rule, 2026-10-08)."""
import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import unittest
from unittest import mock

from PySide6.QtWidgets import QAbstractButton, QApplication, QLabel, QLineEdit

from home_guard_project.box.app import camera_display
from home_guard_project.box.app.scene_backend import DemoSceneBackend

IDS = ['front_door', 'ameer_week_0_1_ch3']


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
        names = [field.text() for _id, field, _switch in page.rows]
        self.assertEqual(names, ['Front door', 'Camera 2 of 2'])      # the box names one; the other by its place
        self.assertEqual([f for f in page.widget.findChildren(QLineEdit) if f not in (page.search_user, page.search_password)], [])
        shown = texts(page.widget)
        self.assertFalse([t for t in shown if any(i in t for i in IDS)], shown)

    def test_saving_switches_cameras_and_never_renames_an_id(self):
        page = self.page()
        page.render(page.load_photos())
        page.rows[1][2].blockSignals(True); page.rows[1][2].setChecked(False)
        self.assertEqual(page.changes(), [('front_door', 'front_door', True),
                                          ('ameer_week_0_1_ch3', 'ameer_week_0_1_ch3', False)])

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
