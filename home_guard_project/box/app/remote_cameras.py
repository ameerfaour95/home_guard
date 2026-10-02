"""Remote camera management; only the injected runner launches processes."""
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import json
import re
import subprocess
import tempfile
import threading
import time
from .camera_controls import Camera,changes_payload
from .engine_backend import ProcessRunner

def target_user(target):
    match=re.fullmatch(r'([A-Za-z0-9_][A-Za-z0-9_.-]*)@([A-Za-z0-9][A-Za-z0-9_.:-]*|\[[A-Fa-f0-9:]+\])',target)
    if not match: raise ValueError('Invalid box address')
    return match[1]

class CommandRunner:
    def __init__(self):
        self.cancelled=threading.Event();self.processes=ProcessRunner()
    def cancel(self): self.cancelled.set()
    def run(self,args):
        if self.cancelled.is_set(): raise RuntimeError('Cancelled')
        process=self.processes.spawn(args)
        deadline=time.monotonic()+480
        try:
            while True:
                if self.cancelled.is_set(): raise RuntimeError('Cancelled')
                try:
                    output,_=process.communicate(timeout=.1)
                    return SimpleNamespace(returncode=process.returncode,stdout=output)
                except subprocess.TimeoutExpired:
                    if time.monotonic()>deadline: raise RuntimeError('Command timed out')
        finally:
            self.processes.stop(process)
            if process.stdout: process.stdout.close()

class RemoteCameras:
    remote=True
    def __init__(self,target,names=(),runner=None,key=None):
        self.target=target;self.user=target_user(target)
        self.runner=runner or CommandRunner()
        self.key=Path(key) if key else Path.home()/'.ssh'/'homeguard_box'
        self.box=SimpleNamespace(demo=False,is_stopped=lambda:False)
        self.records=[Camera(name) for name in dict.fromkeys(names)]
        self.directory=None;self.cancelled=False
    def load(self): return list(self.records)
    def local_directory(self):
        if self.directory is None: self.directory=tempfile.TemporaryDirectory(prefix='homeguard-cameras-')
        return Path(self.directory.name)
    def command(self,args,allow_failed=False):
        if self.cancelled: raise RuntimeError('Cancelled')
        self.diagnostic_output=""
        result=self.runner.run(args)
        self.diagnostic_output=result.stdout
        if result.returncode and not allow_failed: raise RuntimeError('Camera command failed')
        return result
    def ssh(self,operation):
        command=r'cd /d C:\home_guard && .venv\Scripts\python.exe -m home_guard_project.box.find_cameras --json '+operation
        if '"' in command or "'" in command: raise ValueError('Quotes are not allowed in the remote command')
        return ['ssh.exe','-i',str(self.key),'-o','LogLevel=ERROR',self.target,command]
    def scp(self,source,destination):
        return ['scp.exe','-i',str(self.key),'-o','LogLevel=ERROR',str(source),str(destination)]
    def parse(self,result,key):
        # The box prints its result as indented JSON over several lines, and the
        # output can carry other lines before it (decoder messages, setup notes).
        lines=result.stdout.splitlines()
        starts=[i for i,line in enumerate(lines) if line.strip()=='{']
        ends=[i for i,line in enumerate(lines) if line.strip()=='}']
        candidates=['\n'.join(lines[s:e+1]) for s in reversed(starts) for e in reversed(ends) if e>s]
        candidates+=list(reversed(lines))          # a result printed on one line
        for text in candidates:
            try: data=json.loads(text)
            except ValueError: continue
            if not isinstance(data,dict): continue
            if 'error' in data: raise RuntimeError('Camera command failed')
            if isinstance(data.get(key),list): return data
        raise RuntimeError('Invalid camera result')
    def snapshots(self):
        try:
            local=self.local_directory()
            remote=rf'C:\Users\{self.user}\hg_snapshots'
            result=self.command(self.ssh('snapshots --out '+remote),allow_failed=True)
            data=self.parse(result,'snapshots')
            if result.returncode not in (0,1): raise RuntimeError('Camera command failed')
            shots=data['snapshots']
            if any(row.get('ok') for row in shots):
                self.command(self.scp(self.target+f':C:/Users/{self.user}/hg_snapshots/*.jpg',local))
            current={camera.name:camera for camera in self.records}
            for row in shots:
                name=row['name']
                if not re.fullmatch(r'[a-z0-9_]+',name): raise ValueError('Invalid camera name')
                filename=str(row.get('file') or '').replace('\\','/').rsplit('/',1)[-1]
                path=local/filename
                ok=bool(row.get('ok')) and bool(re.fullmatch(r'[a-z0-9_]+\.jpg',filename)) and path.is_file()
                camera=current.get(name,Camera(name))
                current[name]=replace(camera,file=str(path) if ok else '',ok=ok)
            self.records=list(current.values())
            return list(self.records)
        finally:
            if self.cancelled: self.cleanup()
    def save(self,changes):
        payload=changes_payload(changes)
        if {row['name'] for row in payload['cameras']}!={camera.name for camera in self.records}: raise ValueError('Camera list changed')
        path=None
        try:
            path=self.local_directory()/'changes.json'
            path.write_text(json.dumps(payload),encoding='utf-8')
            self.command(self.scp(path,self.target+f':C:/Users/{self.user}/hg_camera_changes.json'))
            result=self.command(self.ssh(rf'apply --changes C:\Users\{self.user}\hg_camera_changes.json'),allow_failed=True)
            data=self.parse(result,'active')
            if result.returncode or not isinstance(data.get('disabled'),list): raise RuntimeError('Camera command failed')
            old={camera.name:camera for camera in self.records}
            expected_active=[row['new_name'] for row in payload['cameras'] if row['enabled']]
            expected_disabled=[row['new_name'] for row in payload['cameras'] if not row['enabled']]
            if set(data['active'])!=set(expected_active) or set(data['disabled'])!=set(expected_disabled): raise RuntimeError('Camera result changed')
            self.records=[replace(old[row['name']],name=row['new_name'],enabled=row['enabled']) for row in payload['cameras']]
            return list(self.records)
        finally:
            if path: path.unlink(missing_ok=True)
            if self.cancelled: self.cleanup()
    def cancel(self):
        self.cancelled=True
        if hasattr(self.runner,'cancel'): self.runner.cancel()
    def cleanup(self):
        if self.directory is not None: self.directory.cleanup();self.directory=None
