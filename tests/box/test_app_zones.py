import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import unittest
from unittest.mock import Mock
import time

from PySide6.QtCore import QAbstractAnimation, QPointF, QSize, Qt
from PySide6.QtGui import QColor, QPixmap
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from home_guard_project.box.app.strings import tr
from home_guard_project.box.app.zone_editor import ZoneEditorDialog, ZoneStage


def wait_until(condition, timeout_ms=3000):
    """Run the event loop until condition() holds. Animations end on an event-loop
    tick, so on a busy machine they finish late; a fixed qWait races them."""
    deadline = time.monotonic() + timeout_ms / 1000
    while not condition() and time.monotonic() < deadline:
        QTest.qWait(10)
    return condition()


def settled(owner, name='animation'):
    animation = getattr(owner, name, None)
    return animation is None or animation.state() == QAbstractAnimation.State.Stopped


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
        self.assertTrue(wait_until(lambda: settled(self.dialog, 'transition_group')))

    def tearDown(self):
        self.dialog.reject()
        wait_until(lambda: not self.dialog.isVisible())
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
        self.assertEqual(self.dialog.progress.height(), 3)
        self.assertTrue(self.dialog.save_button.rect().contains(self.dialog.progress.geometry()))
        self.dialog.future.result(timeout=3); self.dialog.poll()
        self.controls.set_zone.assert_called_once_with('yard', [[.1235, .2346], [.9, .2], [.5, .9]])
        self.assertTrue(wait_until(lambda: not self.dialog.isVisible()))

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

    def test_coverage_and_footer_alignment_in_each_state(self):
        for screen in (QSize(1366, 768), QSize(1920, 1080)):
            self.dialog.resize(self.dialog.opening_size(screen))
            for points in ([], [[.1, .1]], [[0, 0], [1, 0], [1, .46], [0, .46]]):
                self.dialog.stage.set_points(points)
                self.assertTrue(wait_until(lambda: settled(self.dialog.coverage)))
                stage = self.dialog.stage.geometry()
                top = self.dialog.eyebrow.mapTo(self.dialog, self.dialog.eyebrow.rect().topLeft())
                bottom = self.dialog.save_button.mapTo(self.dialog, self.dialog.save_button.rect().bottomLeft())
                self.assertEqual(top.y(), stage.top())
                self.assertEqual(bottom.y(), stage.bottom())
                self.assertEqual(self.dialog.eyebrow.parentWidget().width(), 360)
                self.assertEqual(stage.width(), self.dialog.width()-360-32*3)
                self.assertEqual(stage.top(), 32)
                self.assertEqual(stage.bottom(), self.dialog.height()-33)
                self.assertLessEqual(self.dialog.width(), screen.width())
                self.assertLessEqual(self.dialog.height(), screen.height())
                picture = self.dialog.stage.picture_rect()
                self.assertTrue(self.dialog.stage.core_rect().contains(picture))
                self.assertAlmostEqual(picture.width()/picture.height(), 4/3)
                self.assertEqual(self.dialog.undo_button.height(), 32)
                self.assertEqual(self.dialog.clear_button.height(), 32)
                self.assertEqual(self.dialog.save_button.y() - self.dialog.undo_button.geometry().bottom() - 1, 12)
                expected = 1. if not points else .46 if len(points) >= 3 else 0.
                self.assertAlmostEqual(self.dialog.coverage.fraction, expected)
        self.assertEqual(self.dialog.status.text(), 'Watching 46 % of the picture')
        self.assertEqual(self.dialog.corners.text(), '4 corners')

    def test_error_chip_takes_warning_slot_without_moving_footer(self):
        self.controls.set_zone.side_effect = RuntimeError('offline')
        self.dialog.stage.set_points([[.1, .1], [.2, .1], [.2, .2]])
        self.app.processEvents()
        slot = self.dialog.warning.geometry()
        footer = self.dialog.save_button.geometry()
        self.dialog.save()
        try: self.dialog.future.result(timeout=3)
        except RuntimeError: pass
        self.dialog.poll(); self.app.processEvents()
        self.assertIs(self.dialog.messages.currentWidget(), self.dialog.error)
        self.assertEqual(self.dialog.error.geometry(), slot)
        self.assertFalse(self.dialog.warning.isVisible())
        self.assertEqual(self.dialog.save_button.geometry(), footer)

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
        # Cross the picture boundary rather than rounding a fractional edge inward.
        QTest.mouseMove(stage, (stage.picture_rect().bottomRight() + QPointF(20, 20)).toPoint())
        QTest.mouseRelease(stage, Qt.MouseButton.LeftButton)
        self.assertEqual(stage.points[0], [1., 1.])

    def test_cancel_never_writes_and_rtl_mirrors_column(self):
        self.dialog.setLayoutDirection(Qt.LayoutDirection.RightToLeft)
        self.app.processEvents()
        self.assertGreater(self.dialog.stage.x(), self.dialog.eyebrow.parentWidget().x())
        QTest.keyClick(self.dialog, Qt.Key.Key_Escape)
        self.assertTrue(wait_until(lambda: not self.dialog.isVisible()))
        self.controls.set_zone.assert_not_called(); self.controls.clear_zone.assert_not_called()


class ZoneTileTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        from home_guard_project.box.app.box_controls import BoxControls
        from home_guard_project.box.app.camera_controls import CameraControls
        from home_guard_project.box.app.camera_ui import CameraPage
        from home_guard_project.box.app.theme import stylesheet
        self.controls = CameraControls(BoxControls(demo=True), ['yard', 'street', 'door'])
        self.page = CameraPage(self.controls, lambda: None)
        self.page.widget.setStyleSheet(stylesheet())
        self.page.widget.resize(1200, 650); self.page.widget.show()

    def tearDown(self):
        if self.page.zone_dialog:
            self.page.zone_dialog.reject(); wait_until(lambda: self.page.zone_dialog is None)
        self.page.close(); self.page.widget.close(); self.page.widget.deleteLater()
        self.app.processEvents()

    def test_cover_crop_dims_only_outside_saved_zone(self):
        from home_guard_project.box.app.zone_picture import ZonePicture
        from home_guard_project.box.app.theme import PALETTES
        photo = ZonePicture(None)
        photo.pix = QPixmap(800, 600); photo.pix.fill(QColor(180, 160, 140))
        photo.height_limit = 200
        photo.resize(900, 200); photo.show(); self.app.processEvents()
        try:
            whole = photo.grab().toImage()
            photo.set_zone([[0, 0], [.5, 0], [.5, 1], [0, 1]])
            self.assertTrue(wait_until(lambda: settled(photo)))
            zoned = photo.grab().toImage()
            self.assertEqual(zoned.pixelColor(225, 100), whole.pixelColor(225, 100))
            before, after = whole.pixelColor(675, 100), zoned.pixelColor(675, 100)
            bg = QColor(PALETTES['dark']['bg'])
            for channel in ('red', 'green', 'blue'):
                self.assertLess(getattr(after, channel)(), getattr(before, channel)())
                expected = .4 * getattr(before, channel)() + .6 * getattr(bg, channel)()
                self.assertAlmostEqual(getattr(after, channel)(), expected, delta=1)
            # A horizontal edge at y=.4 is cropped/scaled with the image:
            # 600 * 1.125 high, centred in a 200px tile -> y=32.5.
            photo.set_zone([[0, .4], [1, .4], [1, 1], [0, 1]])
            self.assertTrue(wait_until(lambda: settled(photo)))
            cropped = photo.grab().toImage()
            self.assertLess(cropped.pixelColor(225, 20).red(), whole.pixelColor(225, 20).red())
            self.assertEqual(cropped.pixelColor(225, 50), whole.pixelColor(225, 50))
            edge = cropped.pixelColor(225, 32)
            self.assertGreater(edge.green(), edge.red())
            photo.set_zone([]); self.assertTrue(wait_until(lambda: settled(photo)))
            self.assertEqual(photo.grab().toImage(), whole)
        finally:
            photo.close()

    def test_load_once_after_snapshots_and_success_updates_tile(self):
        calls = []
        snapshots, zones = self.controls.snapshots, self.controls.zones
        self.controls.snapshots = lambda: (calls.append('snapshots'), snapshots())[1]
        self.controls.zones = lambda: (calls.append('zones'), zones())[1]
        records = self.page.load_photos(); self.page.render(records)
        self.assertEqual(calls, ['snapshots', 'zones'])
        button, status, photo = self.page.zone_widgets['yard']
        self.app.processEvents()
        self.assertEqual(button.parentWidget().height(), 36)
        self.assertLessEqual(button.height(), 36)
        self.assertEqual(status.text(), tr('camera_zone_whole'))
        raw_key = photo.pix.cacheKey()
        button.click()
        dialog = self.page.zone_dialog
        self.assertEqual(dialog.stage.pix.cacheKey(), raw_key)
        points = [[.1, .2], [.9, .2], [.5, .9]]
        dialog.stage.set_points(points); dialog.save()
        dialog.future.result(timeout=3); dialog.poll()
        self.assertEqual(status.text(), tr('camera_zone_drawn'))
        self.assertEqual(photo.points, points)
        self.assertEqual(photo.pix.cacheKey(), raw_key)
        self.assertEqual(calls, ['snapshots', 'zones'])
        # The close animation ends on an event-loop tick; wait for it, not a fixed time.
        self.assertTrue(wait_until(lambda: self.page.zone_dialog is None))
        # Reopening restores the saved polygon, clearing restores the whole image.
        button.click(); dialog = self.page.zone_dialog
        self.assertEqual(dialog.stage.points, points)
        dialog.stage.clear(); dialog.save(); dialog.future.result(timeout=3); dialog.poll()
        self.assertEqual(status.text(), tr('camera_zone_whole'))
        self.assertEqual(photo.points, [])

    def test_zone_load_failure_preserves_photos_and_refresh_can_recover(self):
        zones = self.controls.zones
        self.controls.zones = Mock(side_effect=RuntimeError('offline'))
        self.page.render(self.page.load_photos())
        button, status, photo = self.page.zone_widgets['yard']
        self.assertFalse(button.isEnabled())
        self.assertFalse(photo.pix.isNull())
        self.controls.zones = zones
        self.page.render(self.page.load_photos())
        self.assertTrue(self.page.zone_widgets['yard'][0].isEnabled())
