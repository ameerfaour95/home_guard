import unittest
from pathlib import Path
from home_guard_project.box.app.timeline import merge_timeline,thinking_camera,image_path
class TimelineTests(unittest.TestCase):
    def test_merge_order_deduplication_and_limit(self):
        data={'decisions':[{'ts':100,'camera':'door','summary':'Person at door','sent':True},{'ts':120,'camera':'driveway','summary':'Parked cars','false_positive':True}]}
        feed=[{'ts':101,'who':'box','kind':'alert','camera':'door','text':'Person at door','delivered':True},{'ts':110,'who':'owner','kind':'message','name':'Ameer','text':'Thanks'}]
        rows=merge_timeline(data,feed);self.assertEqual(len(rows),3);self.assertEqual([r.ts for r in rows],[101,110,120])
        self.assertEqual(rows[-1].kind,'quiet');self.assertEqual(len(merge_timeline({},feed*150)),200)
    def test_safe_image_path_and_thinking_freshness(self):
        self.assertIsNone(image_path(Path('images'),'../secret.jpg'));self.assertEqual(image_path(Path('images'),'door_123.jpg'),Path('images/door_123.jpg'))
        data={'thinking':{'camera':'front_door','ts':100}}
        self.assertEqual(thinking_camera(data,150),'Front Door');self.assertEqual(thinking_camera(data,161),'')
