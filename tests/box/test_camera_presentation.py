import os
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
import unittest
from unittest.mock import patch
from home_guard_project.box.app.camera_presentation import image_rect,active_names
class CameraPresentationTests(unittest.TestCase):
    def test_hero_and_symmetric_cover(self):
        self.assertEqual(image_rect((0,0,1200,600),(1280,960)),(200,0,800,600))
        self.assertEqual(image_rect((0,0,300,150),(1280,960),False),(0,-37.5,300,225))
    def test_disabled_leaves_stage_and_active_names_survive_stale_preview(self):
        with patch('home_guard_project.box.find_cameras._read_cameras_raw',return_value={'cameras':{'front':'secret','new':'secret'},'disabled':{'garden':'secret'}}):
            self.assertEqual(active_names(['front','garden'],[]),['front','new'])
            self.assertEqual(active_names([],[]),['front','new'])
    def test_camera_list_refresh_keeps_picture(self):
        from home_guard_project.box.app.camera_controls import CameraControls,Camera
        from types import SimpleNamespace
        controls=CameraControls(SimpleNamespace(demo=False));controls.records=[Camera('front',True,'preview.jpg',True)]
        with patch('home_guard_project.box.find_cameras._read_cameras_raw',return_value={'cameras':{},'disabled':{'front':'secret'}}):
            self.assertEqual(controls.load(),[Camera('front',False,'preview.jpg',True)])
