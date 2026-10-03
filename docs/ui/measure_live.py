"""Offline six-camera replay through the real publisher, transport and Qt window.

Run from the checkout: .venv/Scripts/python.exe docs/ui/measure_live.py --seconds 12
No camera, model, Telegram or network access. Results are workstation measurements.
"""
import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import statistics
import sys
import tempfile
import threading
import time
from types import SimpleNamespace
from unittest.mock import patch

os.environ.pop("VIRTUAL_ENV", None)
os.environ.pop("SSLKEYLOGFILE", None)
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import cv2
import numpy as np
from PySide6.QtCore import QTimer, QEvent, QCoreApplication
from PySide6.QtGui import QFont
from PySide6.QtWidgets import QApplication
from home_guard_project.box.preview import PreviewWriter, PreviewReader, camera_key
from home_guard_project.box.app import ui


def summary(values):
    values = sorted(values)
    return {"p50_ms": round(statistics.median(values)*1000, 2),
            "p95_ms": round(values[min(len(values)-1, int(len(values)*.95))]*1000, 2),
            "max_ms": round(max(values)*1000, 2)} if values else {}


def synthetic_remote_server(directory,connection):
    """Separate box process: don't make its Python threads contend with the GUI."""
    from home_guard_project.box.serve import BoxData
    from home_guard_project.box.app.live_server import LiveHandler,LiveServer
    root=Path(directory)
    data=BoxData(str(root),str(root/"cameras.yaml"),status=lambda:{"stopped":False,"site":"Cedar House"},
        settings=lambda:{"mode":"inference","show_cameras":True,"site":"Cedar House"})
    server=LiveServer(("127.0.0.1",0),type("DemoLiveHandler",(LiveHandler,),{"data":data}))
    connection.send(server.server_port);connection.close()
    try: server.serve_forever()
    finally: server.server_close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=float, default=12)
    parser.add_argument("--output")
    parser.add_argument("--capture", action="store_true")
    parser.add_argument("--screenshots-only", action="store_true", help="Capture the three proof states without recording a GIF")
    parser.add_argument("--size", default="1366x768")
    parser.add_argument("--screenshot-suffix", default="")
    parser.add_argument("--remote-simulated", action="store_true", help="Use a synthetic HTTP server on loopback; no box/SSH")
    args = parser.parse_args()
    app = QApplication([])
    app.setFont(QFont("Segoe UI", 11))
    frames, latencies, stalls, status_latencies, input_latencies, callbacks, chat_latencies = defaultdict(list), [], [], [], [], [], []
    start=float('inf')
    stamps, painted = {}, {}
    status = {"written": 0, "seen": 0, "chat_written":0, "chat_seen":0}
    original_read = PreviewReader.read
    original_update = ui.CameraTile.update_picture
    original_paint = ui.CameraTile.paintEvent
    original_event = ui.CameraTile.event
    input_type=QEvent.Type(QEvent.registerEventType())

    def event(tile,event):
        if event.type()==input_type:
            tile._probe_input_at=event.sent_at
            tile.update()
            return True
        return original_event(tile,event)

    def read(reader, name):
        data = original_read(reader, name)
        if data:
            stamps[name] = (reader.directory/(camera_key(name)+".jpg")).stat().st_mtime
        return data

    def update(tile, pix, *rest, **kwargs):
        original_update(tile, pix, *rest, **kwargs)
        tile._measure_stamp = getattr(tile, "frame_stamp", stamps.get(tile.name, 0))

    def paint(tile, event):
        began=time.perf_counter()
        original_paint(tile, event)
        if time.time()>=start+1: callbacks.append(time.perf_counter()-began)
        sent_at=getattr(tile,"_probe_input_at",None)
        if sent_at is not None:
            if time.time()>=start+1: input_latencies.append(time.perf_counter()-sent_at)
            tile._probe_input_at=None
        stamp = getattr(tile, "frame_stamp", getattr(tile, "_measure_stamp", 0))
        now = time.time()
        if stamp and stamp >= start and painted.get(tile.name) != stamp and now >= start+1:
            painted[tile.name] = stamp
            frames[tile.name].append(now)
            latencies.append(max(0, now-stamp))

    from home_guard_project.box.app.ai_activity_ui import AiActivity
    original_ai_paint = AiActivity.paintEvent

    def ai_paint(panel, event):
        original_ai_paint(panel, event)
        if window.ai_data.get("measure_id") == status["written"] and status["seen"] != status["written"]:
            status["seen"] = status["written"]
            if time.time() >= start+1:
                status_latencies.append(time.time()-status["written"])
        if window.chat_data and window.chat_data[-1].get("ts")==status["chat_written"] and status["chat_seen"]!=status["chat_written"]:
            status["chat_seen"]=status["chat_written"]
            if time.time()>=start+1: chat_latencies.append(time.time()-status["chat_written"])

    def timed(method):
        def run(*args,**kwargs):
            began=time.perf_counter()
            try: return method(*args,**kwargs)
            finally:
                if time.time()>=start+1: callbacks.append(time.perf_counter()-began)
        return run

    with tempfile.TemporaryDirectory(prefix="homeguard-live-measure-") as directory:
        root = Path(directory)
        (root/"preview").mkdir()
        with patch.object(ui.bc, "LOG_DIR", str(root)), patch.object(ui.bc, "get_option", return_value=True), \
             patch.object(PreviewReader, "read", read), patch.object(ui.CameraTile, "update_picture", update), \
             patch.object(ui.CameraTile, "paintEvent", paint), patch.object(ui.CameraTile,"event",event), \
             patch.object(AiActivity, "paintEvent", ai_paint), patch.object(AiActivity,"render",timed(AiActivity.render)), \
             patch.object(ui.Window,"tick",timed(ui.Window.tick)), patch.object(ui.Window,"update_detector",timed(ui.Window.update_detector)):
            options = SimpleNamespace(demo=True, setup=False, state="inference", size=args.size,
                                      cameras=6, details=False, detections=True, theme="dark", aspect="16:9")
            window = ui.Window(options)
            window.show()
            # Windows constrains the initial show to this monitor's work area.
            # Resize after show so proof captures exercise the requested layout.
            window.resize(*map(int, args.size.split("x")))
            window.args.demo = False  # controls remain simulated; transport uses temporary files
            window.box_unreachable = False
            window.fetch = lambda: (window.current_state, True, [])
            window.alert_pause.status = lambda names: (None, False)
            window.ai_data = {}
            window.chat_data = []
            names = [tile.name for tile in window.tiles]
            server=None
            remote_patches=[]
            if args.remote_simulated:
                import multiprocessing
                from home_guard_project.box.app import remote_live
                cameras=root/"cameras.yaml"
                cameras.write_text("cameras:\n"+"".join(f"  {name}: synthetic\n" for name in names),encoding="utf-8")
                receiver,sender=multiprocessing.Pipe(duplex=False)
                server=multiprocessing.Process(target=synthetic_remote_server,args=(str(root),sender),daemon=True)
                server.start();sender.close()
                if not receiver.poll(10):
                    server.terminate();server.join(2)
                    raise RuntimeError("Synthetic HTTP server failed to start")
                port=receiver.recv();receiver.close()
                def synthetic_tunnel(reader):
                    reader.retry_at=time.monotonic()+5
                    reader.process=SimpleNamespace(poll=lambda:None,stdin=None,terminate=lambda:None,wait=lambda timeout:None)
                remote_patches=[patch.object(remote_live,"BASE_URL",f"http://127.0.0.1:{port}/"),
                                patch.object(remote_live._RemoteReader,"connect_tunnel",synthetic_tunnel)]
                for item in remote_patches: item.start()
                window.remote_target="demo@localhost"
            writer = PreviewWriter(root/"preview", enabled=True)
            writer.set_cameras(names)
            stop = threading.Event()
            costs = []
            from home_guard_project.box.app.demo_media import picture, detections
            bases = []
            for i in range(6):
                image = picture(i).toImage().convertToFormat(__import__('PySide6.QtGui', fromlist=['QImage']).QImage.Format.Format_RGB888)
                rgb = np.frombuffer(image.bits(), np.uint8).reshape(image.height(), image.bytesPerLine())[:, :image.width()*3].reshape(image.height(), image.width(), 3)
                bases.append(cv2.cvtColor(rgb.copy(), cv2.COLOR_RGB2BGR))
            start = time.time()

            def produce():
                next_status = 0
                next_input = 0
                index = 0
                decisions=[]
                while not stop.is_set():
                    now = time.time()
                    if now>=next_input:
                        event=QEvent(input_type);event.sent_at=time.perf_counter()
                        QCoreApplication.postEvent(window.tiles[0],event)
                        next_input=now+.2
                    for i, name in enumerate(names):
                        if (args.capture or args.screenshots_only) and i == 5 and now-start > 4: continue
                        if not writer.wanted(name): continue
                        frame = bases[i].copy()
                        before = time.thread_time()
                        writer.publish(name, frame)
                        costs.append(time.thread_time()-before)
                    if now >= next_status:
                        index += 1
                        observations={name:{"checked_ts":now,"ts":now,"objects":detections(i)} for i,name in enumerate(names)}
                        data = {"updated": now, "measure_id": now, "cameras": observations, "decisions": decisions}
                        if index % 2:
                            data["thinking"] = {"camera": names[0], "ts": now}
                        else:
                            decisions.append({"ts": now, "camera": names[0], "summary": "Front path checked. A person is arriving.", "sent": True, "command": "[send_message]"})
                        temp = root/"ai_status.tmp"
                        temp.write_text(json.dumps(data), encoding="utf-8")
                        status["written"] = now
                        for attempt in range(20):
                            try:
                                os.replace(temp, root/"ai_status.json")
                                break
                            except PermissionError:
                                if attempt==19: raise
                                stop.wait(.005)
                        chat_now=time.time()
                        status['chat_written']=chat_now
                        message={'ts':chat_now,'who':'owner','name':'Owner','text':('Please check the front path.','Keep watching the house.','Thank you.')[index%3]}
                        with (root/'telegram_chat.jsonl').open('a',encoding='utf-8') as feed:
                            feed.write(json.dumps(message)+'\n')
                        next_status = now+1.7
                    stop.wait(.004)

            worker = threading.Thread(target=produce, daemon=True)
            worker.start()
            last = time.perf_counter()

            def probe():
                nonlocal last
                now = time.perf_counter()
                if time.time() > start+1: stalls.append(max(0, now-last-.01))
                last = now

            timer = QTimer();timer.timeout.connect(probe);timer.start(10)
            captures = []
            capture_times = []
            capture_timer = QTimer()
            if args.capture or args.screenshots_only:
                from PIL import Image
                def capture():
                    from PySide6.QtGui import QImage
                    # QWidget.render(QImage) omits Qt's scroll-area opacity effect
                    # on Windows; grab preserves the actual composed timeline.
                    image=window.grab().toImage().convertToFormat(QImage.Format.Format_RGBA8888)
                    captures.append(Image.frombytes("RGBA", (image.width(),image.height()), bytes(image.bits())).convert("RGB"))
                    capture_times.append(time.time()-start)
                if args.screenshots_only:
                    for milliseconds in (2600, 4200, 9500): QTimer.singleShot(milliseconds, capture)
                else:
                    capture_timer.timeout.connect(capture);capture_timer.start(100)
                shots = Path(__file__).parent/"screenshots";shots.mkdir(exist_ok=True)
            QTimer.singleShot(round(args.seconds*1000), app.quit)
            app.exec()
            elapsed = time.time()-start
            if not worker.is_alive(): raise RuntimeError("Synthetic source stopped early; discard this run")
            stop.set();worker.join(2)
            capture_timer.stop()
            window.close()
            if server: server.terminate();server.join(2)
            for item in reversed(remote_patches): item.stop()
            result = {"duration_s": round(elapsed,2), "source": "six synthetic 1280x720 camera frames; local Windows workstation",
                      "window_logical_size": [window.width(), window.height()],
                      "tile_logical_sizes": {tile.name: [tile.width(), tile.height()] for tile in window.tiles},
                      "transport":"simulated remote over loopback HTTP (separate server process; no SSH)" if args.remote_simulated else "local files",
                      "fps": {name: round(len(frames[name])/max(1,elapsed-1),2) for name in names},
                      "file_to_paint": summary(latencies), "ai_write_to_paint": summary(status_latencies),
                      "event_loop_lateness_10ms_probe": summary(stalls),
                      "ui_callback_block":summary(callbacks),
                      "chat_write_to_paint":summary(chat_latencies),
                      "posted_input_to_paint":summary(input_latencies),
                      "publisher_one_core_percent": round(sum(costs)/elapsed*100,2),
                      "publisher_frames": len(costs)}
            print(json.dumps(result, indent=2))
            if args.output: Path(args.output).write_text(json.dumps(result,indent=2)+"\n", encoding="utf-8")
            if captures:
                for second,name in ((2.6,"live_overview"),(4.2,"live_looking"),(9.5,"live_reconnecting")):
                    index=min(range(len(capture_times)),key=lambda i:abs(capture_times[i]-second))
                    captures[index].save(shots/(name+args.screenshot_suffix+".png"))
                if not args.screenshots_only:
                    captures[0].save(Path(__file__).parent/"live_demo.gif", save_all=True, append_images=captures[1:], duration=100, loop=0)


if __name__ == "__main__": main()
