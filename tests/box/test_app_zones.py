import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import unittest
from unittest.mock import Mock
from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QPixmap
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from home_guard_project.box.app import motion
from home_guard_project.box.app.strings import tr
from home_guard_project.box.app.zone_editor import ZoneEditorDialog, ZoneStage


class ZoneEditorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.pix = QPixmap(800, 600); self.pix.fill(Qt.GlobalColor.gray)
        self.controls = Mock()
        self.controls.set_zone.side_effect = lambda name, points: points
        self.controls.clear_zone.return_value = []
        self.dialog = ZoneEditorDialog(self.controls, 'yard', self.pix)
        self.dialog.show()
        QTest.qWait(motion.PANE_MS + 30)

    def tearDown(self):
        self.dialog.reject()
        QTest.qWait(motion.TOGGLE_MS + 30)
        self.dialog.deleteLater()
        self.app.processEvents()

    def test_picture_center_round_trip_and_resize_with_side_fill(self):
        stage = ZoneStage(self.pix); stage.resize(1000, 600); stage.show()
        self.assertGreater(stage.picture_rect().left(), 7)
        center = stage.picture_rect().center()
        self.assertEqual(stage.to_fraction(center), (.5, .5))
        self.assertEqual(stage.to_stage((.5, .5)), center)
        QTest.mouseClick(stage, Qt.MouseButton.LeftButton, pos=center.toPoint())
        self.assertEqual(stage.points, [[.5, .5]])
        stage.resize(1100, 680)
        self.assertEqual(stage.to_stage(stage.points[0]), stage.picture_rect().center())
        # Side fill is never a drawable part of the camera picture.
        QTest.mouseClick(stage, Qt.MouseButton.LeftButton, pos=QPointF(10, 200).toPoint())
        self.assertEqual(len(stage.points), 1)
        stage.close()

    def test_undo_clear_backspace_control_z_and_right_click(self):
        stage = self.dialog.stage
        stage.set_points([(.1, .1), (.8, .1), (.8, .8), (.1, .8)])
        self.dialog.undo_button.click(); self.assertEqual(len(stage.points), 3)
        QTest.keyClick(stage, Qt.Key.Key_Backspace); self.assertEqual(len(stage.points), 2)
        QTest.keyClick(stage, Qt.Key.Key_Z, Qt.KeyboardModifier.ControlModifier); self.assertEqual(len(stage.points), 1)
        QTest.mouseClick(stage, Qt.MouseButton.RightButton); self.assertEqual(stage.points, [])
        stage.set_points([(.1, .1), (.8, .1), (.8, .8)])
        self.dialog.clear_button.click(); self.assertEqual(stage.points, [])

    def test_save_rounds_and_disables_all_controls_until_reply(self):
        self.dialog.stage.set_points([(.123456, .234567), (.9, .2), (.5, .9)])
        self.dialog.save()
        for widget in (self.dialog.stage, self.dialog.save_button, self.dialog.cancel_button, self.dialog.clear_button, self.dialog.undo_button):
            self.assertFalse(widget.isEnabled())
        self.assertTrue(self.dialog.progress.isVisible())
        self.dialog.future.result(timeout=3); self.dialog.poll()
        self.controls.set_zone.assert_called_once_with('yard', [[.1235, .2346], [.9, .2], [.5, .9]])
        QTest.qWait(motion.TOGGLE_MS + 30)
        self.assertFalse(self.dialog.isVisible())

    def test_empty_save_clears_and_incomplete_save_is_disabled(self):
        self.assertTrue(self.dialog.save_button.isEnabled())
        for count in (1, 2):
            self.dialog.stage.set_points([(.1, .2)] * count)
            self.assertFalse(self.dialog.save_button.isEnabled())
            QTest.keyClick(self.dialog, Qt.Key.Key_Return)
            self.controls.clear_zone.assert_not_called()
        self.dialog.stage.clear(); self.dialog.save_button.click()
        self.dialog.future.result(timeout=3); self.dialog.poll()
        self.controls.clear_zone.assert_called_once_with('yard')

    def test_programmatic_incomplete_save_clears(self):
        self.dialog.stage.set_points([(.1, .2), (.3, .4)])
        self.dialog.save()
        self.dialog.future.result(timeout=3); self.dialog.poll()
        self.controls.clear_zone.assert_called_once_with('yard')

    def test_small_area_warning(self):
        self.dialog.stage.set_points([(.1, .1), (.2, .1), (.2, .2)])
        self.assertEqual(self.dialog.warning.text(), tr('camera_zone_small'))
        self.assertTrue(self.dialog.warning.isVisible())
        self.assertTrue(self.dialog.save_button.isEnabled())
        self.dialog.stage.set_points([(.1, .1), (.9, .1), (.9, .9)])
        self.assertEqual(self.dialog.warning.text(), '')

    def test_failed_save_keeps_dialog_open_enabled_and_preserves_points(self):
        self.controls.set_zone.side_effect = RuntimeError('offline')
        points = [[.1, .1], [.9, .1], [.9, .9]]
        self.dialog.stage.set_points(points); self.dialog.save()
        try: self.dialog.future.result(timeout=3)
        except RuntimeError: pass
        self.dialog.poll()
        self.assertEqual(self.dialog.error.text(), tr('camera_zone_save_failed'))
        self.assertTrue(self.dialog.isVisible())
        self.assertTrue(self.dialog.save_button.isEnabled())
        self.assertTrue(self.dialog.stage.isEnabled())
        self.assertEqual(self.dialog.stage.points, points)

    def test_fourth_click_snaps_without_adding_corner_and_drag_clamps(self):
        stage = self.dialog.stage
        for point in ((.15, .2), (.85, .2), (.5, .85)):
            QTest.mouseClick(stage, Qt.MouseButton.LeftButton, pos=stage.to_stage(point).toPoint())
        first = stage.to_stage(stage.points[0])
        QTest.mouseClick(stage, Qt.MouseButton.LeftButton, pos=(first + QPointF(8, 0)).toPoint())
        self.assertEqual(len(stage.points), 3)
        self.assertTrue(stage.closed)
        QTest.mousePress(stage, Qt.MouseButton.LeftButton, pos=first.toPoint())
        QTest.mouseMove(stage, stage.picture_rect().bottomRight().toPoint())
        QTest.mouseRelease(stage, Qt.MouseButton.LeftButton)
        self.assertEqual(stage.points[0], [1., 1.])

    def test_cancel_never_writes_and_rtl_mirrors_column(self):
        self.dialog.setLayoutDirection(Qt.LayoutDirection.RightToLeft)
        self.app.processEvents()
        self.assertGreater(self.dialog.stage.x(), self.dialog.eyebrow.parentWidget().x())
        QTest.keyClick(self.dialog, Qt.Key.Key_Escape)
        QTest.qWait(motion.TOGGLE_MS + 30)
        self.assertFalse(self.dialog.isVisible())
        self.controls.set_zone.assert_not_called(); self.controls.clear_zone.assert_not_called()
