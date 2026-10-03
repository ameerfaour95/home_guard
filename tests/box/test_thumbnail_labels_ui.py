"""Exercise painted text rectangles at the rail floor and real window sizes."""
import itertools
import os
import time
import unittest
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PySide6.QtWidgets import QApplication
from home_guard_project.box.app.ui import CameraTile, Window, demo_picture
from home_guard_project.box.app.detector_view import Detection


class ThumbnailLabelsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def check_labels(self, tile):
        tile.show_detections = True
        tile.detections = (Detection("person", .91, (.1, .2, .8, .95)),
                           Detection("car", .92, (.4, .8, .6, .98)))
        tile.caption.setText("A very long driveway camera name")
        tile.detector_note.setText("Sees: 1 person, 2 vehicles")
        for age in (0, 12000):
            tile.update_picture(demo_picture(0), time.time()-age)
            tile.grab()  # actual paint path, including chips if ever reintroduced
            rects = tile.painted_label_rects
            self.assertEqual(set(rects), {"name", "caption", "badge"})
            for name, rect in rects.items():
                self.assertTrue(tile.rect().contains(rect.toAlignedRect()), name)
            for (a, ra), (b, rb) in itertools.combinations(rects.items(), 2):
                self.assertFalse(ra.intersects(rb), (tile.size(), a, b))
            scrim=tile.thumbnail_scrim()
            if age==0: self.assertTrue(tile.painted_detection_rects)
            for rect in tile.painted_detection_rects:
                self.assertFalse(rect.intersects(scrim), (tile.size(), rect, scrim))
            # Exercise the actual pixels too: enabling outlines must not alter
            # even a single scrim pixel (including antialiased stroke edges).
            with_boxes=tile.grab().toImage()
            tile.show_detections=False
            without_boxes=tile.grab().toImage()
            scale=with_boxes.devicePixelRatio()
            pixels=scrim.adjusted(0,1,0,-1)
            from PySide6.QtCore import QRect
            region=QRect(round(pixels.x()*scale),round(pixels.y()*scale),
                         round(pixels.width()*scale),round(pixels.height()*scale))
            self.assertEqual(with_boxes.copy(region),without_boxes.copy(region))
            tile.show_detections=True

    def test_narrowest_thumbnail(self):
        tile = CameraTile("garden")
        tile.setMinimumSize(80, 144)
        tile.resize(80, 144)
        self.check_labels(tile)
        tile.close()

    def test_thumbnail_layout_in_both_window_sizes(self):
        for size in ("1366x768", "1920x1080"):
            args = SimpleNamespace(demo=True, setup=False, state="inference", size=size,
                                   cameras=6, details=False, detections=True, theme="dark", aspect="16:9")
            window = Window(args)
            try:
                window.show()
                window.resize(*map(int, size.split("x")))
                self.app.processEvents()
                self.assertEqual((window.width(), window.height()), tuple(map(int, size.split("x"))))
                for tile in window.tiles:
                    if not tile.hero:
                        self.check_labels(tile)
            finally:
                window.close()
