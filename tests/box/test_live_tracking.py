"""Synthetic detector cadence and caption collision regressions; no devices."""
import os
import time
import unittest
from dataclasses import replace
from unittest.mock import patch

os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
from PySide6.QtCore import QRectF
from PySide6.QtGui import QImage
from PySide6.QtWidgets import QApplication
from home_guard_project.box.app.live_tracking import OverlayTracker
from home_guard_project.box.app.detector_view import Detection, entry_opacity
from home_guard_project.box.app import motion
from home_guard_project.box.app.ui import CameraTile, demo_picture


class TrackingBoundsTests(unittest.TestCase):
    def setUp(self):
        self.tracker=OverlayTracker()
        self.image=QImage(320,180,QImage.Format.Format_Grayscale8)
        self.image.fill(0)

    def sample(self,stamp,box=(.2,.2,.3,.6)):
        return {'updated':stamp,'cameras':{'front':{'ts':stamp,'checked_ts':stamp,
                'objects':[{'label':'person','conf':.91,'box':box}]}}}

    def update(self,data,now):
        return self.tracker.update('front',self.image,data,now)

    def test_distance_and_one_sample_interval_caps_then_freezes(self):
        self.update(self.sample(100),100)
        data=self.sample(100.5)
        self.update(data,100.5)
        self.update(data,100.5+motion.HOVER_MS/1000+.001)
        def drift(previous,current,objects):
            return tuple(replace(o,box=tuple(v+.04 for v in o.box)) for o in objects)
        with patch.object(self.tracker,'flow',side_effect=drift) as flow:
            for now in (100.7,100.8,100.9,101):
                box=self.update(data,now)[0].box
                self.assertLessEqual(((box[0]-.2)**2+(box[1]-.2)**2)**.5,.05+1e-9)
            calls=flow.call_count
            for now in (101.01,101.26,102):
                self.assertEqual(self.update(data,now)[0].box,box)
            self.assertEqual(flow.call_count,calls)

    def test_real_sample_eases_without_snap_and_arrives_within_hover(self):
        start=self.update(self.sample(100),100)[0].box
        target=(.24,.22,.36,.64)
        data=self.sample(100.5,target)
        self.assertEqual(self.update(data,100.5)[0].box,start)
        halfway=self.update(data,100.5+motion.HOVER_MS/2000)[0].box
        for a,b,c in zip(start,halfway,target): self.assertLess(a,b);self.assertLess(b,c)
        arrived=self.update(data,100.5+motion.HOVER_MS/1000)[0].box
        for a,b in zip(arrived,target): self.assertAlmostEqual(a,b)

    def test_freeze_preserves_detector_entry_fade_and_slow_box_visibility(self):
        data=self.sample(100)
        first=self.update(data,100)
        self.assertEqual(self.update(data,104),first)
        self.assertEqual(entry_opacity(data['cameras']['front'],104),1)
        data['cameras']['front']['checked_ts']=102
        self.assertAlmostEqual(entry_opacity(data['cameras']['front'],102.5),.5)
        self.assertEqual(self.update(data,104),())


class HeroCaptionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.app=QApplication.instance() or QApplication([])

    def test_caption_moves_to_other_corner_and_never_paints_over_box(self):
        tile=CameraTile('front_door');tile.hero=True;tile.resize(1000,600)
        tile.show_detections=True
        tile.update_picture(demo_picture(0),time.time())
        tile.detections=(Detection('person',.91,(.1,.6,.2,.97)),)
        try:
            tile.grab()
            self.assertEqual(tile.caption_corner,1.)
            self.assertEqual(tile.caption_animation.duration(),motion.PANE_MS)
            for elapsed in (0,60,120,240):
                tile.caption_animation.setCurrentTime(elapsed)
                tile.grab()
                for key in ('name','caption'):
                    rect=tile.painted_label_rects.get(key)
                    if rect:
                        self.assertFalse(any(rect.intersects(box) for box in tile.painted_detection_rects))
            self.assertIn('caption',tile.painted_label_rects)
            self.assertGreater(tile.painted_label_rects['caption'].left(),tile.width()/2)
            # Stay in the clear corner, rather than bouncing back on each paint.
            tile.grab();self.assertEqual(tile.caption_corner,1.)
        finally: tile.close()

    def test_both_bottom_corners_occupied_conceals_caption(self):
        tile=CameraTile('front')
        try:
            _,visible=tile.hero_caption_rect(QRectF(0,0,1000,600),[QRectF(0,500,1000,100)])
            self.assertFalse(visible)
        finally: tile.close()

    def test_demo_boxes_enclose_painted_figures_for_every_camera(self):
        from home_guard_project.box.app.demo_media import detections
        from home_guard_project.box.app.ai_demo import demo_status
        names=[f'camera{i}' for i in range(6)]
        data=demo_status(names,100,'live-detections')
        for i,name in enumerate(names):
            self.assertEqual(data['cameras'][name]['objects'],detections(i))
            person,car=detections(i)
            x=160+i*65
            for px,py in ((x-41,285),(x+51,350),(x-28,526),(x+36,526),(x,238)):
                a,b,c,d=person['box']
                self.assertTrue(a<=px/1280<=c and b<=py/720<=d)
            a,b,c,d=car['box']
            self.assertLessEqual(a,530/1280);self.assertGreaterEqual(c,770/1280)
            self.assertLessEqual(b,304/720);self.assertGreaterEqual(d,441/720)
