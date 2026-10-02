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


class OffCameraTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.app=QApplication.instance() or QApplication([])
    def test_disabled_camera_stays_and_switch_applies_without_login(self):
        w=Window(args(state='off-camera'));w.show();self.app.processEvents()
        off=next(t for t in w.tiles if t.off)
        self.assertIsNot(w.expanded_tile,off)
        self.assertFalse(off.turn_on.isHidden())
        page=w.cameras_page;page.render(w.camera_controls.records)
        from PySide6.QtWidgets import QWidget
        slots=page.widget.findChildren(QWidget,'cameraActionSlot')
        self.assertEqual(len(slots),3)
        self.assertTrue(all(slot.layout().count()==0 for slot in slots))
        off.turn_on.click()
        self.assertTrue(all(c.enabled for c in w.camera_controls.records))
        page.future.result();page.poll()
        self.assertEqual(len(w.tiles),3)
        self.assertTrue(all(not t.off for t in w.tiles))
        w.close()
    def test_failed_toggle_rolls_back(self):
        from concurrent.futures import Future
        w=Window(args(state='off-camera'));page=w.cameras_page;page.render(w.camera_controls.records)
        failed=Future();failed.set_exception(ValueError('recorded failure'))
        with patch.object(page.pool,'submit',return_value=failed):
            page.set_camera_enabled(w.camera_controls.records[-1].name,True)
        page.poll()
        self.assertFalse(w.camera_controls.records[-1].enabled)
        self.assertTrue(w.tiles[-1].off)
        w.close()
