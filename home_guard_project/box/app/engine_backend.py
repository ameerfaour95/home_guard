"""Streaming adapter for the installer's PowerShell engine contract."""
from dataclasses import dataclass, field
from pathlib import Path
import json
import os
import queue
import re
import shutil
import subprocess
import tempfile
import threading
from .model import redact
from .strings import tr

ENGINE_STEPS = ('connect','update','site','network','cameras','alerts','readiness')
OWNERS = {'connect':0,'update':0,'site':2,'network':1,'cameras':3,'alerts':2,'readiness':0}

def is_progress_warning(event):
    return event.kind=='step' and event.step=='network' and event.status=='warn' and 'switching' in event.text.lower()

@dataclass(frozen=True)
class Event:
    kind: str
    step: str = ''
    status: str = ''
    text: str = ''
    name: str = ''
    password: str = field(default='',repr=False)
    facts: dict = field(default_factory=dict, repr=False)

class OutputParser:
    def __init__(self, secrets=()):
        self.secrets=list(filter(None,secrets))
    def safe(self,text):
        for secret in sorted(self.secrets,key=len,reverse=True):
            text=text.replace(secret,tr('hidden'))
        text=re.sub(r'\b[^\s@]+@[^\s@]+\b',tr('hidden_address'),text)
        text=re.sub(r'(?<![\w.])(?:\d{1,3}\.){3}\d{1,3}(?::\d+)?',tr('hidden_address'),text)
        return redact(text)
    def parse(self,line):
        line=line.strip()
        from .setup_failure import network_facts
        facts=network_facts(line)
        match=re.fullmatch(r'@@step (\w+) (start|ok|warn|fail|skip)(?: (.*))?',line)
        if match and match[1] in ENGINE_STEPS:
            return Event('step',match[1],match[2],self.safe(match[3] or ''), facts=facts)
        match=re.fullmatch(r'@@camera ([a-z0-9_]+) (\d+)x(\d+)',line)
        if match: return Event('camera',text=match[2]+'x'+match[3],name=match[1])
        match=re.fullmatch(r'@@check (PASS|WARN|FAIL) (.*)',line)
        if match: return Event('check',status=match[1],text=self.safe(match[2]))
        match=re.fullmatch(r'@@rescue (\S+) (\S+)',line)
        if match:
            self.secrets.append(match[2])
            return Event('rescue',name=match[1],password=match[2])
        match=re.fullmatch(r'@@done (ok|fail)',line)
        if match: return Event('done',status=match[1])
        return Event('detail',text=self.safe(line),facts=facts)

def answers_payload(answers):
    return dict(target=answers.address,network=answers.network,wifi_ssid=answers.ssid,wifi_password=answers.wifi_password,site=answers.house,show_cameras=answers.show_cameras,find_cameras=answers.find_cameras,camera_user=answers.camera_user,camera_password=answers.camera_password,alerts=answers.alerts,alert_start_hour=answers.start_hour,alert_end_hour=answers.end_hour,alert_cooldown_sec=answers.cooldown_sec)

class ProcessRunner:
    def spawn(self,args):
        env=os.environ.copy();env.pop('VIRTUAL_ENV',None)
        return subprocess.Popen(args,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf-8',errors='replace',env=env,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    def stop(self,process):
        if process.poll() is not None: return
        if os.name=='nt':
            subprocess.run(['taskkill','/PID',str(process.pid),'/T','/F'],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,creationflags=subprocess.CREATE_NO_WINDOW)
        else: process.terminate()
        try: process.wait(timeout=5)
        except subprocess.TimeoutExpired: process.kill();process.wait()

class EngineBackend:
    def __init__(self,runner=None,script=None,executable=None):
        self.runner=runner or ProcessRunner()
        self.script=Path(script) if script else Path(__file__).parents[1]/'setup_customer.ps1'
        self.executable=executable
        self.cancelled=threading.Event()
    def cancel(self): self.cancelled.set()
    def run(self,answers,emit):
        parser=OutputParser((answers.address,answers.address.rsplit("@",1)[-1],answers.wifi_password,answers.camera_password))
        process=None
        try:
            executable=self.executable or shutil.which('powershell')
            if not executable:
                emit(Event('step','connect','fail',tr('powershell_missing')));return False
            if not self.script.is_file():
                emit(Event('step','connect','fail',tr('engine_missing')));return False
            with tempfile.TemporaryDirectory(prefix='homeguard-setup-') as directory:
                path=Path(directory)/'answers.json'
                path.write_text(json.dumps(answers_payload(answers),ensure_ascii=False),encoding='utf-8')
                if self.cancelled.is_set(): return False
                try:
                    process=self.runner.spawn([executable,'-NoProfile','-ExecutionPolicy','Bypass','-File',str(self.script),'-AnswersFile',str(path)])
                    lines=queue.Queue()
                    def read_output():
                        try:
                            for line in process.stdout: lines.put(line)
                        except (OSError,ValueError): pass
                        finally: lines.put(None)
                    thread=threading.Thread(target=read_output,daemon=True);thread.start()
                    failed=False;done=False;completed=[];running='connect'
                    while True:
                        if self.cancelled.is_set(): self.runner.stop(process);return False
                        try: line=lines.get(timeout=.1)
                        except queue.Empty: continue
                        if line is None: break
                        event=parser.parse(line)
                        if event.kind=='step':
                            running=event.step
                            if event.status=='start':
                                if len(completed)>=len(ENGINE_STEPS) or event.step!=ENGINE_STEPS[len(completed)]:
                                    emit(Event('step',running,'fail',tr('engine_protocol_error')));failed=True;break
                            elif event.status in ('ok','warn','skip'):
                                if not is_progress_warning(event) and event.step not in completed: completed.append(event.step)
                            elif event.status=='fail': failed=True
                        if event.kind=='check' and event.status=='FAIL': failed=True
                        if event.kind=='done':
                            done=event.status=='ok'
                            if not done and not failed: emit(Event('step',running,'fail',tr('engine_incomplete')))
                            failed=failed or not done
                        emit(event)
                        if event.kind=="check" and event.status=="FAIL":
                            emit(Event("step","readiness","fail",event.text))
                        if failed: break
                    if failed: self.runner.stop(process)
                    while True:
                        if self.cancelled.is_set(): self.runner.stop(process);return False
                        try: code=process.wait(timeout=.1);break
                        except subprocess.TimeoutExpired: continue
                    if self.cancelled.is_set(): return False
                    if failed: return False
                    if code!=0 or not done or completed!=list(ENGINE_STEPS):
                        emit(Event('step',running,'fail',tr('engine_incomplete')));return False
                    return True
                finally:
                    if process is not None:
                        self.runner.stop(process)
                        if process.stdout: process.stdout.close()
                        process=None
        except Exception:
            emit(Event('step','connect','fail',tr('engine_launch_error')))
            return False
        finally:
            if process is not None:
                self.runner.stop(process)
                if process.stdout: process.stdout.close()
            answers.wifi_password='';answers.camera_password=''

class DemoEngine:
    def __init__(self,failure=False):
        self.failure=failure;self.cancelled=threading.Event()
    def cancel(self): self.cancelled.set()
    def run(self,answers,emit,instant=False):
        for step in ENGINE_STEPS:
            emit(Event('step',step,'start'))
            if not instant and self.cancelled.wait(.6): return False
            failure_step = 'network' if self.failure is True else self.failure
            if step=='network':
                emit(Event('detail',text='The box joined the home network.',facts={'network':answers.ssid or 'ameer2','address':'192.168.68.120'}))
            if step==failure_step and not (step=='cameras' and not answers.find_cameras):
                if step=='cameras': emit(Event('detail',text='WARNING No device answers on the camera port. Is the box on the cameras network?'))
                emit(Event('step',step,'fail',tr('failure_cameras_title' if step=='cameras' else 'failure_connect_title' if step=='connect' else 'failure_update_title' if step=='update' else 'network_fail')));return False
            if step=='cameras':
                if answers.find_cameras:
                    for name in ('front_door','garden','driveway'): emit(Event('camera',name=name,text='1920x1080'))
                else:
                    emit(Event('step',step,'skip',tr('skipped')));continue
            if step=='readiness':
                for key,status in (('check_collector','PASS'),('check_network','PASS'),('check_folder','PASS'),('check_power','WARN')): emit(Event('check',status=status,text=tr(key)))
            emit(Event('step',step,'ok',tr('setup_step_complete')))
        emit(Event('done',status='ok'));return True
