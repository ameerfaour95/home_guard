"""Bounded asynchronous laptop transport over the existing SSH trust/key setup."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import subprocess
import re
import time
import urllib.request
from urllib.parse import quote
from PySide6.QtCore import QObject, QThread, QTimer, Signal, Slot, Qt
from PySide6.QtGui import QImage
from .remote_cameras import target_user


def tunnel_command(target):
    target_user(target)
    return ["ssh.exe", "-T", "-i", str(Path.home()/".ssh"/"homeguard_box"),
            "-o", "BatchMode=yes", "-o", "ExitOnForwardFailure=yes", "-o", "ConnectTimeout=5",
            "-L", "127.0.0.1:8765:127.0.0.1:8765", target,
            r"cd /d C:\home_guard && .venv\Scripts\python.exe -m home_guard_project.box.app.live_server"]


class _RemoteReader(QObject):
    ready=Signal(object)
    def __init__(self,target):
        super().__init__()
        self.target=target;self.names=();self.hero=None;self.visible=False
        self.jobs={};self.due={};self.versions={};self.pending={};self.inflight=False
        self.process=None
        self.image_names=[]

    @Slot()
    def start(self):
        self.pool=ThreadPoolExecutor(max_workers=8,thread_name_prefix="remote-live")
        self.timer=QTimer(self);self.timer.timeout.connect(self.scan);self.timer.start(20)
        self.connect_tunnel()

    def connect_tunnel(self):
        self.retry_at=time.monotonic()+5
        try:
            self.process=subprocess.Popen(tunnel_command(self.target),stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,
                creationflags=getattr(subprocess,"CREATE_NO_WINDOW",0))
        except OSError: self.process=None

    @Slot(object)
    def demand(self,request):
        self.names,self.hero,self.visible=request
        self.due["viewer"]=0

    def fetch(self,key,request=None):
        path = ("preview/"+quote(key[6:],safe="")+".jpg") if key.startswith("frame:") else ("chat_images/"+quote(key[6:],safe="")) if key.startswith("image:") else "chat?limit=200" if key=="chat" else key
        url="http://127.0.0.1:8765/"+path
        req=urllib.request.Request(url,data=json.dumps(request).encode() if request is not None else None,
                                   headers={"Content-Type":"application/json"})
        # Bypass corporate proxies for the SSH loopback only.
        with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(req,timeout=2) as response:
            data=response.read(8*1024*1024+1)
            if len(data)>8*1024*1024: raise ValueError("Oversized reply")
            version=(response.headers.get("X-Modified"),hashlib.sha256(data).digest())
        if key.startswith(("frame:","image:")):
            image=QImage.fromData(data)
            if image.isNull(): raise ValueError("Invalid frame")
            return version,(image,time.time()) if key.startswith("frame:") else image
        return version,json.loads(data)

    @Slot()
    def scan(self):
        now=time.monotonic()
        if (self.process is None or self.process.poll() is not None):
            if now>=self.retry_at: self.connect_tunnel()
            return
        for key,future in list(self.jobs.items()):
            if not future.done(): continue
            del self.jobs[key]
            try:
                version,data=future.result()
                if self.versions.get(key)!=version:
                    self.versions[key]=version
                    if key!="viewer": self.pending[key]=data
                    if key=="chat":
                        self.image_names=list(dict.fromkeys(row.get("image","") for row in data
                            if isinstance(row,dict) and re.fullmatch(r"[A-Za-z0-9_-]+\.jpe?g",str(row.get("image","")),re.I)))[-32:]
            except (OSError,ValueError):
                self.due[key]=now+.5
        rates={"status":.35,"chat":.5,"viewer":1}
        if self.visible:
            rates.update({"frame:"+n:(.09 if n==self.hero else .18) for n in self.names})
        for key,interval in rates.items():
            if key in self.jobs or now<self.due.get(key,0): continue
            request={"hero":self.hero,"cameras":self.names,"visible":self.visible} if key=="viewer" else None
            self.jobs[key]=self.pool.submit(self.fetch,key,request);self.due[key]=now+interval
        # At most two image downloads; timeline text/status never wait for pictures.
        image_jobs=sum(key.startswith("image:") for key in self.jobs)
        for name in self.image_names if self.visible else ():
            key="image:"+name
            if image_jobs>=2: break
            if key in self.versions or key in self.jobs or now<self.due.get(key,0): continue
            self.jobs[key]=self.pool.submit(self.fetch,key);image_jobs+=1
        if self.pending and not self.inflight:
            packet={"frames":{}}
            for key,data in self.pending.items():
                if key.startswith("frame:"): packet["frames"][key[6:]]=data
                elif key=="status": packet.update(status=data.get("ai",{}),overview=data)
                elif key=="chat": packet["chat"]=data
                elif key.startswith("image:"): packet.setdefault("images",{})[key[6:]]=data
            self.pending={};self.inflight=True;self.ready.emit(packet)

    @Slot()
    def acknowledge(self): self.inflight=False

    @Slot()
    def stop(self):
        self.timer.stop()
        # Best effort revocation before closing the tunnel; the lease also expires.
        self.pool.submit(self.fetch,"viewer",{"visible":False,"cameras":[],"hero":None})
        self.pool.shutdown(wait=False,cancel_futures=True)
        if self.process is not None:
            if self.process.stdin: self.process.stdin.close()
            if self.process.poll() is None: self.process.terminate()
            self.process.wait(timeout=5)


class RemoteLiveTransport(QObject):
    frames=Signal(object)
    status=Signal(object)
    request=Signal(object)
    ack=Signal()
    stopping=Signal()
    def __init__(self,target,parent=None):
        super().__init__(parent)
        target_user(target)
        self.thread=QThread(self);self.worker=_RemoteReader(target)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.start)
        self.request.connect(self.worker.demand);self.ack.connect(self.worker.acknowledge)
        self.worker.ready.connect(self.deliver)
        self.stopping.connect(self.worker.stop,Qt.ConnectionType.BlockingQueuedConnection)
        self.thread.finished.connect(self.worker.deleteLater);self.thread.start()
    @Slot(object)
    def deliver(self,packet):
        self.status.emit(packet);self.frames.emit(packet["frames"]);self.ack.emit()
    def demand(self,names,hero,visible): self.request.emit((tuple(names),hero,bool(visible)))
    def close(self):
        if self.thread.isRunning():
            self.stopping.emit();self.thread.quit();self.thread.wait()
