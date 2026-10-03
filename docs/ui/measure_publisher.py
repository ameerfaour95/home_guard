"""Compare complete publisher CPU (including its worker) on synthetic captures.

No detector/model/camera is started. This does not measure YOLO throughput.
"""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

os.environ.pop('VIRTUAL_ENV',None)
os.environ.pop('SSLKEYLOGFILE',None)
root=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(root))
import cv2
import numpy as np
from PySide6.QtGui import QImage
from PySide6.QtWidgets import QApplication
from home_guard_project.box.preview import PreviewWriter,PreviewReader
from home_guard_project.box.app.demo_media import picture

app=QApplication([])
image=picture().toImage().convertToFormat(QImage.Format.Format_RGB888)
base=cv2.cvtColor(np.frombuffer(image.bits(),np.uint8).reshape(720,image.bytesPerLine())[:,:1280*3].reshape(720,1280,3).copy(),cv2.COLOR_RGB2BGR)
original={}
exec(subprocess.check_output(['git','show','7f59234:home_guard_project/box/preview.py'],cwd=root).decode(),original)


def run(modern,visible,duration):
    with tempfile.TemporaryDirectory(prefix='homeguard-publish-cpu-') as directory:
        writer=(PreviewWriter if modern else original['PreviewWriter'])(directory,enabled=True)
        reader=PreviewReader(directory)
        names=[f'camera_{i}' for i in range(6)]
        writer.set_cameras(names)
        counts=dict.fromkeys(names,0)
        publish=writer.publish
        def counted(name,frame,source=None):
            success=publish(name,frame,source)
            if success: counts[name]+=1
            return success
        writer.publish=counted
        worker_cpu=[]
        if modern:
            drain=writer._drain
            def measured_drain():
                started=time.thread_time()
                try: drain()
                finally: worker_cpu.append(time.thread_time()-started)
            writer._drain=measured_drain
        start=time.monotonic();next_touch=0;next_capture=start;caller_cpu=0
        while time.monotonic()-start<duration:
            now=time.monotonic()
            if now>=next_touch:
                if modern: reader.touch(names[0],visible=visible,cameras=names)
                elif visible: reader.touch(names[0])
                next_touch=now+1
            for name in names:
                frame=base.copy()  # capture simulation, excluded from publisher CPU
                before=time.thread_time()
                if modern: writer.offer(name,frame)
                else: writer.publish(name,frame,source=frame)
                caller_cpu+=time.thread_time()-before
            next_capture+=1/30
            time.sleep(max(0,next_capture-time.monotonic()))
        elapsed=time.monotonic()-start
        if modern: writer.close()
        return {'duration_s':round(elapsed,2),'one_core_percent':round((caller_cpu+sum(worker_cpu))/elapsed*100,2),
                'published_frames':counts,'fps':{name:round(count/elapsed,2) for name,count in counts.items()}}


result={'hardware':'AMD Ryzen 9 9955HX3D; not the N150 box',
        'before_open':run(False,True,8),'after_open':run(True,True,8),
        'before_hidden':run(False,False,2),'after_hidden':run(True,False,2)}
result['extra_one_core_percentage_points']=round(result['after_open']['one_core_percent']-result['before_open']['one_core_percent'],2)
text=json.dumps(result,indent=2)
print(text)
Path(__file__).with_name('live_publisher_cpu.json').write_text(text+'\n',encoding='utf-8')
