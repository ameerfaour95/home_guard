"""Offline liveness, bounded delivery, scroll and demand regressions."""
import json
import os
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch, Mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PySide6.QtCore import QCoreApplication, QEvent
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QApplication
from home_guard_project.box.preview import PreviewWriter, PreviewReader
from home_guard_project.box.app.liveness import frame_health, relative_time


class ClockTests(unittest.TestCase):
    def test_reconnect_counts_from_last_distinct_frame(self):
        self.assertEqual(frame_health(100,103,95),("LIVE",False))
        self.assertEqual(frame_health(100,104,95),("Reconnecting… 4 s",True))
        self.assertEqual(frame_health(100,112.8,95),("Reconnecting… 12 s",True))
        self.assertEqual(frame_health(None,101,100),("Connecting…",False))
        self.assertEqual(frame_health(None,113,100),("Reconnecting… 13 s",True))
        self.assertEqual(frame_health(113,113,100),("LIVE",False))

    def test_relative_time_boundaries(self):
        for age,expected in ((0,"now"),(1,"now"),(2,"2 s ago"),(59,"59 s ago"),
                             (60,"1 min ago"),(240,"4 min ago"),(3600,"1 h ago"),(86400,"1 d ago")):
            self.assertEqual(relative_time(100000-age,100000),expected)
        self.assertEqual(relative_time(float('nan'),100),"Unknown")
        self.assertEqual(relative_time(None,100),"Unknown")
        self.assertEqual(relative_time(110,100),"now")


class DemandTests(unittest.TestCase):
    def test_large_thumbnail_hidden_and_expired_demand(self):
        with tempfile.TemporaryDirectory() as directory:
            now=time.time()+.01
            reader=PreviewReader(directory)
            writer=PreviewWriter(directory,enabled=True,clock=lambda:now)
            reader.touch("front",visible=True,cameras=("front","yard"))
            writer.last={"front":now,"yard":now}
            now+=.07
            self.assertTrue(writer.wanted("front"))
            self.assertFalse(writer.wanted("yard"))
            self.assertFalse(writer.wanted("unseen"))
            now+=.14
            self.assertTrue(writer.wanted("yard"))
            reader.touch("yard",visible=True,cameras=("front","yard"))
            now+=.3
            writer.wanted("yard")
            writer.last={"front":now,"yard":now};now+=.07
            self.assertTrue(writer.wanted("yard"))
            self.assertFalse(writer.wanted("front"))
            reader.touch("yard",visible=False,cameras=())
            now+=.3
            with patch("cv2.imencode") as encode:
                self.assertFalse(writer.publish("yard",None))
                encode.assert_not_called()
            reader.touch("front",visible=True,cameras=("front",))
            now+=4
            self.assertFalse(writer.wanted("front"))

    def test_async_capture_is_masked_and_does_not_wait_for_encode(self):
        import threading
        import numpy as np
        from home_guard_project.box.inference_preview import adapt_stream
        with tempfile.TemporaryDirectory() as directory:
            reader=PreviewReader(directory);reader.touch("front",visible=True,cameras=("front",))
            writer=PreviewWriter(directory,enabled=True)
            entered=threading.Event();release=threading.Event()
            class Stream:
                def __init__(self,name,url): self.name=name;self._running=True;self._frame=None
                def _ingest(self,frame,now): self._frame=frame.copy();self._frame[:]=7
                def read(self): return self._frame.copy()
            stream=adapt_stream(Stream,writer)("front","synthetic")
            def slow_publish(name,frame,source=None):
                self.assertTrue((frame==7).all());entered.set();release.wait(1)
            try:
                with patch.object(writer,"publish",side_effect=slow_publish):
                    stream._ingest(np.zeros((40,40,3),np.uint8),time.time())
                    self.assertTrue(entered.wait(1))
                    start=time.perf_counter();stream.read()
                    stream._ingest(np.zeros((40,40,3),np.uint8),time.time())
                    self.assertLess(time.perf_counter()-start,.05)
                    self.assertEqual(len(writer._pending),1)
                    release.set()
            finally: release.set();writer.close()


class WidgetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.app=QApplication.instance() or QApplication([])
    def settle(self):
        for _ in range(5): self.app.processEvents()
    def test_incremental_append_keeps_scroll_and_widget_identity(self):
        from home_guard_project.box.app.ai_activity_ui import AiActivity
        from home_guard_project.box.app.theme import stylesheet
        panel=AiActivity();panel.setStyleSheet(stylesheet());panel.resize(430,550);panel.show()
        now=time.time()
        feed=[{"ts":now-100+i,"who":"owner","name":"Owner","text":f"Message {i} with enough text to occupy a row."} for i in range(30)]
        data={"updated":now,"decisions":[]}
        panel.render(data,now,False,feed);self.settle()
        content=panel.scroll.widget();bar=panel.scroll.verticalScrollBar()
        bar.setValue(bar.maximum()//2);panel.user_scrolled()
        value=bar.value();widgets=dict(panel.row_widgets)
        feed.append({"ts":now,"who":"assistant","text":"The front path is clear."})
        panel.render(data,now,False,feed);self.settle()
        self.assertIs(panel.scroll.widget(),content)
        self.assertEqual(bar.value(),value)
        self.assertTrue(all(panel.row_widgets[k] is w for k,w in widgets.items()))
        bar.setValue(bar.maximum());panel.user_scrolled()
        feed.append({"ts":now+1,"who":"owner","name":"Owner","text":"Thank you."})
        panel.render(data,now+1,False,feed);self.settle()
        self.assertEqual(bar.value(),bar.maximum())
        panel.close();panel.deleteLater();self.settle()

    def test_same_picture_does_not_reset_heartbeat(self):
        from home_guard_project.box.app.ui import CameraTile
        tile=CameraTile("front");pix=QPixmap(20,20);pix.fill()
        tile.update_picture(pix,100)
        tile.update_picture(tile.picture)
        self.assertEqual(tile.frame_stamp,100)
        tile.close();tile.deleteLater();self.settle()

    def test_changed_files_decode_off_gui_and_lease_revokes(self):
        from home_guard_project.box.app.live_transport import LiveTransport
        import threading
        from home_guard_project.box.preview import camera_key
        with tempfile.TemporaryDirectory() as directory:
            preview=Path(directory)/"preview";preview.mkdir()
            transport=LiveTransport(preview)
            packets=[];transport.frames.connect(packets.append)
            transport.demand(["front"],"front",True)
            image=QImage(20,20,QImage.Format.Format_RGB32);image.fill(0xff00ff00)
            image.save(str(preview/(camera_key("front")+".jpg")))
            end=time.monotonic()+2
            try:
                while not packets and time.monotonic()<end:
                    self.app.processEvents();time.sleep(.005)
                self.assertTrue(packets)
                self.assertIsNot(transport.worker.thread(),self.app.thread())
            finally: transport.close()
            self.assertFalse(json.loads((preview/"viewer.alive").read_text())["visible"])

    def test_overlay_tracks_video_translation_then_expires(self):
        import numpy as np
        import cv2
        from home_guard_project.box.app.live_tracking import OverlayTracker
        rng=np.random.default_rng(7)
        first=np.zeros((180,320),np.uint8)
        first[50:100,80:130]=rng.integers(20,240,(50,50),dtype=np.uint8)
        second=cv2.warpAffine(first,np.float32([[1,0,6],[0,1,2]]),(320,180))
        def image(array): return QImage(array.data,320,180,320,QImage.Format.Format_Grayscale8).copy()
        data={"updated":100,"cameras":{"front":{"ts":100,"objects":[{"label":"person","conf":.9,"box":[80/320,50/180,130/320,100/180]}]}}}
        tracker=OverlayTracker()
        tracker.update("front",image(first),data,100)
        objects=tracker.update("front",image(second),data,100.06)
        self.assertEqual(len(objects),1)
        self.assertAlmostEqual(objects[0].box[0],86/320,delta=.003)
        self.assertAlmostEqual(objects[0].box[1],52/180,delta=.003)
        self.assertEqual(tracker.update("front",image(second),data,104),())


class RemoteTests(unittest.TestCase):
    def test_tunnel_is_loopback_only_and_rejects_shell_text(self):
        from home_guard_project.box.app.remote_live import tunnel_command
        command=tunnel_command("owner@box")
        self.assertIn("127.0.0.1:8765:127.0.0.1:8765",command)
        self.assertIn("BatchMode=yes",command)
        with self.assertRaises(ValueError): tunnel_command("owner@box & calc")

    def test_remote_same_version_cannot_refresh_frozen_frame(self):
        from concurrent.futures import Future
        from home_guard_project.box.app.remote_live import _RemoteReader
        reader=_RemoteReader("owner@box");reader.process=Mock();reader.process.poll.return_value=None
        reader.pool=Mock();reader.pool.submit.return_value=Future()
        version=("123",b"same-image")
        reader.versions["frame:front"]=version
        future=Future();future.set_result((version,(QImage(),time.time())))
        reader.jobs["frame:front"]=future
        packets=[];reader.ready.connect(packets.append)
        reader.scan()
        self.assertEqual(packets,[])


if __name__ == "__main__": unittest.main()
