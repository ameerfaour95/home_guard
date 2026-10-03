"""Local latest-frame transport. All filesystem access and decoding is off Qt GUI.

Directory watches survive atomic replacement. A bounded recovery scan covers lost
Windows watcher events; only changed files are decoded. One delivery is in flight.
"""
from pathlib import Path
import time
from PySide6.QtCore import QObject, QThread, QTimer, QFileSystemWatcher, Signal, Slot, Qt
from PySide6.QtGui import QImage
from ..preview import PreviewReader, camera_key


class _Reader(QObject):
    ready = Signal(object)
    def __init__(self, directory):
        super().__init__()
        self.directory = Path(directory)
        self.reader = PreviewReader(self.directory)
        self.names = ()
        self.hero = None
        self.visible = False
        self.stamps = {}
        self.inflight = False
        self.last_touch = 0

    @Slot()
    def start(self):
        self.directory.mkdir(parents=True, exist_ok=True)
        self.watcher = QFileSystemWatcher([str(self.directory), str(self.directory.parent)], self)
        self.watcher.directoryChanged.connect(self.scan)
        self.watcher.fileChanged.connect(self.scan)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.scan)
        self.timer.start(250)

    @Slot(object)
    def demand(self, request):
        self.names, self.hero, self.visible = request
        self.last_touch = 0
        self.scan()

    @Slot()
    def acknowledge(self):
        self.inflight = False
        self.scan()

    @Slot()
    def scan(self):
        now = time.monotonic()
        if now-self.last_touch >= 1:
            try:
                self.reader.touch(self.hero, visible=self.visible, cameras=self.names)
                self.last_touch = now
            except OSError:
                pass
        if self.inflight:
            return
        status = {}
        for filename, key in (("ai_status.json", "status"), ("telegram_chat.jsonl", "chat")):
            path = self.directory.parent/filename
            try:
                stamp = (path.stat().st_mtime_ns, path.stat().st_size)
                if str(path) not in self.watcher.files(): self.watcher.addPath(str(path))
                if self.stamps.get(filename) == stamp: continue
                if key == "status":
                    from ..ai_status import read_status
                    data = read_status(str(path))
                    if not data: continue  # retry a partial/malformed write
                else:
                    from ..chat_feed import read_feed
                    data = read_feed(str(path), limit=200)
                status[key] = data
                self.stamps[filename] = stamp
            except (OSError, ValueError, UnicodeError):
                continue
        frames = {}
        for name in self.names if self.visible else ():
            path = self.directory/(camera_key(name)+".jpg")
            try:
                stamp = path.stat().st_mtime_ns
                if self.stamps.get(name) == stamp: continue
                image = QImage.fromData(path.read_bytes())
                if image.isNull(): continue
                # Don't associate an old decode with a file replaced during the read.
                if path.stat().st_mtime_ns != stamp: continue
                self.stamps[name] = stamp
                frames[name] = (image, stamp/1e9)
            except OSError:
                continue
        if frames or status:
            self.inflight = True
            self.ready.emit(dict(status, frames=frames))

    @Slot()
    def stop(self):
        self.timer.stop()
        try: self.reader.touch(self.hero, visible=False, cameras=())
        except OSError: pass


class LiveTransport(QObject):
    frames = Signal(object)
    status = Signal(object)
    request = Signal(object)
    ack = Signal()
    stopping = Signal()

    def __init__(self, directory, parent=None):
        super().__init__(parent)
        self.thread = QThread(self)
        self.worker = _Reader(directory)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.start)
        self.request.connect(self.worker.demand)
        self.ack.connect(self.worker.acknowledge)
        self.stopping.connect(self.worker.stop, Qt.ConnectionType.BlockingQueuedConnection)
        self.worker.ready.connect(self.deliver)
        self.thread.finished.connect(self.worker.deleteLater)
        self.thread.start()

    @Slot(object)
    def deliver(self, packet):
        if "status" in packet or "chat" in packet: self.status.emit(packet)
        self.frames.emit(packet["frames"])
        self.ack.emit()

    def demand(self, names, hero, visible):
        self.request.emit((tuple(names), hero, bool(visible)))

    def close(self):
        if self.thread.isRunning():
            self.stopping.emit()
            self.thread.quit()
            self.thread.wait()
