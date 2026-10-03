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
from PySide6.QtCore import QTimer
from PySide6.QtGui import QFont
from PySide6.QtWidgets import QApplication
from home_guard_project.box.preview import PreviewWriter, PreviewReader, camera_key
from home_guard_project.box.app import ui


def summary(values):
    values = sorted(values)
    return {"p50_ms": round(statistics.median(values)*1000, 2),
            "p95_ms": round(values[min(len(values)-1, int(len(values)*.95))]*1000, 2),
            "max_ms": round(max(values)*1000, 2)} if values else {}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=float, default=12)
    parser.add_argument("--output")
    parser.add_argument("--capture", action="store_true")
    args = parser.parse_args()
    app = QApplication([])
    app.setFont(QFont("Segoe UI", 11))
    frames, latencies, stalls, status_latencies = defaultdict(list), [], [], []
    stamps, painted = {}, {}
    status = {"written": 0, "seen": 0}
    original_read = PreviewReader.read
    original_update = ui.CameraTile.update_picture
    original_paint = ui.CameraTile.paintEvent

    def read(reader, name):
        data = original_read(reader, name)
        if data:
            stamps[name] = (reader.directory/(camera_key(name)+".jpg")).stat().st_mtime
        return data

    def update(tile, pix, *rest, **kwargs):
        original_update(tile, pix, *rest, **kwargs)
        tile._measure_stamp = getattr(tile, "frame_stamp", stamps.get(tile.name, 0))

    def paint(tile, event):
        original_paint(tile, event)
        stamp = getattr(tile, "frame_stamp", getattr(tile, "_measure_stamp", 0))
        now = time.time()
        if stamp and painted.get(tile.name) != stamp and now >= start+1:
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

    with tempfile.TemporaryDirectory(prefix="homeguard-live-measure-") as directory:
        root = Path(directory)
        (root/"preview").mkdir()
        with patch.object(ui.bc, "LOG_DIR", str(root)), patch.object(ui.bc, "get_option", return_value=True), \
             patch.object(PreviewReader, "read", read), patch.object(ui.CameraTile, "update_picture", update), \
             patch.object(ui.CameraTile, "paintEvent", paint), patch.object(AiActivity, "paintEvent", ai_paint):
            options = SimpleNamespace(demo=True, setup=False, state="inference", size="1366x768",
                                      cameras=6, details=False, detections=True, theme="dark", aspect="16:9")
            window = ui.Window(options)
            window.show()
            window.args.demo = False  # controls remain simulated; transport uses temporary files
            window.box_unreachable = False
            window.fetch = lambda: (window.current_state, True, [])
            window.alert_pause.status = lambda names: (None, False)
            window.ai_data = {}
            window.chat_data = []
            names = [tile.name for tile in window.tiles]
            writer = PreviewWriter(root/"preview", enabled=True)
            writer.set_cameras(names)
            stop = threading.Event()
            costs = []
            from home_guard_project.box.app.demo_media import picture
            bases = []
            for i in range(6):
                image = picture(i).toImage().convertToFormat(__import__('PySide6.QtGui', fromlist=['QImage']).QImage.Format.Format_RGB888)
                rgb = np.frombuffer(image.bits(), np.uint8).reshape(image.height(), image.bytesPerLine())[:, :image.width()*3].reshape(image.height(), image.width(), 3)
                bases.append(cv2.cvtColor(rgb.copy(), cv2.COLOR_RGB2BGR))
            start = time.time()

            def produce():
                next_status = 0
                index = 0
                while not stop.is_set():
                    now = time.time()
                    for i, name in enumerate(names):
                        if args.capture and i == 5 and now-start > 4: continue
                        if not writer.wanted(name): continue
                        frame = bases[i].copy()
                        x = 80+int((now-start)*75) % 1000
                        cv2.rectangle(frame, (x, 530), (x+70, 610), (150, 195, 205), -1)
                        before = time.thread_time()
                        writer.publish(name, frame)
                        costs.append(time.thread_time()-before)
                    if now >= next_status:
                        index += 1
                        data = {"updated": now, "measure_id": now, "cameras": {}, "decisions": []}
                        if index % 2:
                            data["thinking"] = {"camera": names[0], "ts": now}
                        else:
                            data["decisions"] = [{"ts": now, "camera": names[0], "summary": "Front path checked. A person is arriving.", "sent": True, "command": "[send_message]"}]
                        temp = root/"ai_status.tmp"
                        temp.write_text(json.dumps(data), encoding="utf-8")
                        status["written"] = now
                        os.replace(temp, root/"ai_status.json")
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
            capture_timer = QTimer()
            if args.capture:
                from PIL import Image
                def capture():
                    pix = window.grab()
                    image = pix.toImage().convertToFormat(__import__('PySide6.QtGui', fromlist=['QImage']).QImage.Format.Format_RGBA8888)
                    captures.append(Image.frombytes("RGBA", (image.width(),image.height()), bytes(image.bits())).convert("RGB"))
                capture_timer.timeout.connect(capture);capture_timer.start(100)
                shots = Path(__file__).parent/"screenshots";shots.mkdir(exist_ok=True)
                for ms, name in ((2600,"live_overview"),(4200,"live_looking"),(9500,"live_reconnecting")):
                    QTimer.singleShot(ms, lambda name=name: window.grab().save(str(shots/(name+".png"))))
            QTimer.singleShot(round(args.seconds*1000), app.quit)
            app.exec()
            elapsed = time.time()-start
            stop.set();worker.join(2)
            capture_timer.stop()
            window.close()
            result = {"duration_s": round(elapsed,2), "source": "six synthetic 1280x720 camera frames; local Windows workstation",
                      "fps": {name: round(len(frames[name])/max(1,elapsed-1),2) for name in names},
                      "file_to_paint": summary(latencies), "ai_write_to_paint": summary(status_latencies),
                      "event_loop_lateness_10ms_probe": summary(stalls),
                      "publisher_one_core_percent": round(sum(costs)/elapsed*100,2),
                      "publisher_frames": len(costs)}
            print(json.dumps(result, indent=2))
            if args.output: Path(args.output).write_text(json.dumps(result,indent=2)+"\n", encoding="utf-8")
            if captures:
                captures[0].save(Path(__file__).parent/"live_demo.gif", save_all=True, append_images=captures[1:], duration=100, loop=0)


if __name__ == "__main__": main()
