import os,tempfile,unittest
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QPixmap,QColor
from home_guard_project.box.app.preferences import ViewerPreference,ViewerSettings
from home_guard_project.box.app.ui import CameraTile,Window
from home_guard_project.box.app.detector_view import Detection

class ViewerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.app=QApplication.instance() or QApplication([])
    def test_preference_roundtrip_and_safe_defaults(self):
        with tempfile.TemporaryDirectory() as directory:
            preference=ViewerPreference(Path(directory)/'viewer.json')
            self.assertEqual(preference.load(),ViewerSettings())
            value=ViewerSettings(True,'name');preference.save(value)
            self.assertEqual(preference.load(),value)
            preference.path.write_text('{broken')
            self.assertEqual(preference.load(),ViewerSettings())
    def test_painting_off_matches_no_detections_and_on_draws_boxes(self):
        tile=CameraTile('front');tile.resize(640,420);tile.hero=True
        picture=QPixmap(640,420);picture.fill(QColor('#202020'));tile.update_picture(picture)
        tile.show();self.app.processEvents()
        baseline=tile.grab().toImage()
        tile.detections=(Detection('person',.71,(.1,.1,.5,.7)),);tile.box_opacity=.5
        tile.repaint();self.assertEqual(tile.grab().toImage(),baseline)
        tile.show_detections=True;tile.repaint();self.assertNotEqual(tile.grab().toImage(),baseline)
        tile.detection_labels='none';tile.repaint();no_label=tile.grab().toImage()
        tile.detection_labels='confidence';tile.repaint();self.assertNotEqual(tile.grab().toImage(),no_label)
        tile.close()
    def test_stage_and_settings_controls_stay_in_sync(self):
        with tempfile.TemporaryDirectory() as directory:
            preference=ViewerPreference(Path(directory)/'viewer.json')
            args=SimpleNamespace(demo=True,setup=False,theme='dark',panel=None,fail=None,wifi=False,skip_cameras=False,alerts=False,details=False,state='inference',cameras=3,page=None,size='1366x768',screenshot=None,detections=False)
            with patch('home_guard_project.box.app.preferences.ViewerPreference',return_value=preference): window=Window(args)
            self.assertFalse(window.detection_toggle.isChecked())
            window.detection_toggle.setChecked(True)
            self.assertTrue(window.settings_page.detections.isChecked())
            window.settings_page.labels.setCurrentIndex(1)
            self.assertEqual(window.viewer_settings.detection_labels,'name')
            window.settings_page.detections.setChecked(False)
            self.assertFalse(window.detection_toggle.isChecked())
            self.assertTrue(all(not tile.show_detections for tile in window.tiles))
            self.assertFalse(preference.load().detections)
            window.close()
