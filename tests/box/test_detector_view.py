import json
import tempfile
import unittest
from pathlib import Path
from home_guard_project.box.ai_status import read_status
from home_guard_project.box.app.detector_view import camera_view,box_rect

class DetectorViewTests(unittest.TestCase):
    def setUp(self):
        self.data=json.loads((Path(__file__).parent/'fixtures'/'ai_status_sample.json').read_text())
        self.now=self.data['updated']
    def test_fresh_last_detection_and_recent_empty_check(self):
        objects,text=camera_view(self.data,'bian_ch2',self.now)
        self.assertEqual(len(objects),2);self.assertIn('1 person',text);self.assertIn('1 car',text)
        self.assertEqual(objects[1].caption(),'Person \u00b7 71%')
        self.assertNotEqual(objects[0].color,objects[1].color)
        self.assertEqual(camera_view(self.data,'bian_ch3',self.now),((),'Nothing right now'))
        self.assertEqual(camera_view(self.data,'bian_ch2',self.now+4),((),'Nothing right now'))
        self.assertIn('not looking',camera_view(self.data,'bian_ch2',self.now+11)[1])
    def test_stopped_stale_and_bad_values_never_draw_boxes(self):
        self.assertEqual(camera_view(self.data,'bian_ch2',self.now,True),((),'Stopped'))
        self.data['updated']=self.now-16
        self.assertFalse(camera_view(self.data,'bian_ch2',self.now)[0])
        self.data['updated']=self.now;self.data['cameras']['bian_ch2']['objects']=[None,{'label':'person','conf':'bad','box':[] }]
        self.assertFalse(camera_view(self.data,'bian_ch2',self.now)[0])
    def test_box_coordinates_follow_picture_offset_and_size(self):
        box=(.1,.3,.18,.75)
        for picture in ((20,40,1000,500),(-200,0,1000,500)):
            x,y,w,h=box_rect(box,picture)
            self.assertAlmostEqual(x,picture[0]+100);self.assertAlmostEqual(y,picture[1]+150)
            self.assertAlmostEqual(w,80);self.assertAlmostEqual(h,225)
    def test_empty_and_half_written_file_are_safe(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'status.json'
            for text in ('','{','[]','{"cameras": {}}'):
                path.write_text(text)
                self.assertFalse(camera_view(read_status(str(path)),'camera',self.now)[0])
