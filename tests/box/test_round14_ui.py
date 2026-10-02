import os
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
import unittest
from types import SimpleNamespace
from unittest.mock import patch
from PySide6.QtWidgets import QApplication
from home_guard_project.box.app.ui import Window
from home_guard_project.box.app.ai_demo import recorded_status
from home_guard_project.box.app.detector_view import camera_view, box_rect, fade_opacity
from home_guard_project.box.app.timeline import merge_timeline

def args(**values):
    result=dict(demo=True,setup=False,theme='dark',panel=None,fail=None,wifi=False,skip_cameras=False,alerts=False,details=False,state='live-detections',cameras=3,page=None,size='1366x768',screenshot=None,detections=True)
    result.update(values);return SimpleNamespace(**result)

class RecordedDetectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.app=QApplication.instance() or QApplication([])
    def test_recorded_contract_age_and_labels(self):
        data=recorded_status(['front'],1000)
        self.assertEqual(len(camera_view(data,'front',1000)[0]),2)
        data['cameras']['front']['checked_ts']=1004
        self.assertFalse(camera_view(data,'front',1004)[0])
        self.assertEqual(fade_opacity(1000,1002.5),.5)
        self.assertEqual(box_rect((.1,.2,.5,.8),(100,50,400,200)),(140,90,160,120.00000000000001))
        self.assertEqual({r.label for r in merge_timeline(data,[])},{'', 'suspicious','escalation'})
    def test_toggle_updates_hero_and_thumbnails_without_new_frame(self):
        w=Window(args());w.show();self.app.processEvents()
        w.current_state.mode='data_collection'
        w.set_viewer_settings(detections=False)
        self.assertTrue(all(not t.show_detections for t in w.tiles))
        w.set_viewer_settings(detections=True)
        self.assertTrue(all(t.show_detections and len(t.detections)==2 for t in w.tiles))
        self.assertTrue(w.expanded_tile.hero)
        self.assertEqual(w.detection_toggle.toolTip(),'Show detections (D)')
        w.close()
    def test_stopped_hint_and_wall_clock(self):
        w=Window(args(state='ai-stopped',detections=False));w.show()
        w.set_viewer_settings(detections=True)
        self.assertFalse(w.detection_hint.isHidden())
        self.assertTrue(all(not t.detections for t in w.tiles))
        w.close()
