import tempfile,time,unittest
from pathlib import Path
import cv2
import numpy as np
from home_guard_project.box.preview import PreviewReader,PreviewWriter
from home_guard_project.box.app.detector_view import picture_rect,box_rect,fade_opacity

class PremiumPreviewTests(unittest.TestCase):
    def test_hero_rate_and_thumbnail_rate(self):
        with tempfile.TemporaryDirectory() as directory:
            now=[time.time()]
            reader=PreviewReader(directory);reader.touch('front_door')
            writer=PreviewWriter(directory,enabled=True,clock=lambda:now[0])
            frame=np.zeros((720,1280,3),dtype=np.uint8)
            self.assertTrue(writer.publish('front_door',frame))
            self.assertTrue(writer.publish('garden',frame))
            now[0]+=.2
            self.assertTrue(writer.wanted('front_door'));self.assertFalse(writer.wanted('garden'))
            now[0]+=.31
            self.assertTrue(writer.wanted('garden'))
    def test_letterbox_no_upscale_and_fade(self):
        rect=picture_rect((0,0,1000,800),(1280,720))
        self.assertEqual(rect,(0,118.75,1000,562.5))
        box=box_rect((.1,.2,.5,.6),rect)
        self.assertAlmostEqual(box[0],100);self.assertAlmostEqual(box[1],231.25)
        self.assertEqual(picture_rect((0,0,2000,1000),(1280,720))[2:],(1280,720))
        self.assertEqual(fade_opacity(100,102),1);self.assertEqual(fade_opacity(100,102.5),.5);self.assertEqual(fade_opacity(100,103),0)
    def test_publish_cost_and_dimensions(self):
        with tempfile.TemporaryDirectory() as directory:
            PreviewReader(directory).touch('hero')
            writer=PreviewWriter(directory,enabled=True)
            frame=np.random.default_rng(7).integers(0,256,(720,1280,3),dtype=np.uint8)
            costs=[]
            for i in range(12):
                start=time.perf_counter();self.assertTrue(writer.publish('camera_'+str(i),frame));costs.append((time.perf_counter()-start)*1000)
            print('1280x720 JPEG 88 publish: mean %.2f ms, max %.2f ms'%(sum(costs)/len(costs),max(costs)))
            self.assertLess(sum(costs)/len(costs),40)
