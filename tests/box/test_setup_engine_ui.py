import io,json,secrets,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from home_guard_project.box.app.backend import Answers
from home_guard_project.box.app.engine_backend import EngineBackend,OutputParser,ENGINE_STEPS,OWNERS

class FakeProcess:
    def __init__(self,lines,code=0): self.stdout=io.StringIO(lines);self.code=code;self.stopped=False
    def wait(self,timeout=None): return self.code
    def poll(self): return self.code if self.stopped else None
class FakeRunner:
    def __init__(self,lines,code=0): self.process=FakeProcess(lines,code);self.path=None;self.payload=None;self.stops=0
    def spawn(self,args):
        self.args=args;self.path=Path(args[-1]);self.payload=json.loads(self.path.read_text(encoding='utf-8'));return self.process
    def stop(self,process): self.stops+=1;process.stopped=True

def recording():
    return Path(__file__).with_name('fixtures').joinpath('setup_engine_success.txt').read_text()

class EngineTests(unittest.TestCase):
    def test_success_streams_and_deletes_answers(self):
        runner=FakeRunner(recording());events=[]
        with tempfile.TemporaryDirectory() as directory:
            script=Path(directory)/'engine.ps1';script.touch()
            backend=EngineBackend(runner,script,'powershell')
            answers=Answers(address='installer@box.example',house='cedar_house',cooldown_sec=90,wifi_password=secrets.token_hex(12),camera_password=secrets.token_hex(12))
            self.assertTrue(backend.run(answers,events.append))
        self.assertEqual(runner.payload['alert_cooldown_sec'],90)
        self.assertEqual(set(runner.payload),{'target','network','wifi_ssid','wifi_password','site','show_cameras','find_cameras','camera_user','camera_password','alerts','alert_start_hour','alert_end_hour','alert_cooldown_sec'})
        self.assertEqual(runner.args[1:5],['-NoProfile','-ExecutionPolicy','Bypass','-File'])
        self.assertFalse(runner.path.exists())
        self.assertEqual(answers.wifi_password,'')
        self.assertEqual([e.step for e in events if e.kind=='step' and e.status=='start'],list(ENGINE_STEPS))
        self.assertTrue(any(e.kind=='camera' for e in events))
    def test_failure_stops_without_later_events(self):
        runner=FakeRunner(Path(__file__).with_name('fixtures').joinpath('setup_engine_failure.txt').read_text())
        with tempfile.TemporaryDirectory() as directory:
            script=Path(directory)/'engine.ps1';script.touch();events=[]
            self.assertFalse(EngineBackend(runner,script,'powershell').run(Answers(),events.append))
        self.assertEqual(events[-1].status,'fail');self.assertFalse(runner.path.exists())
        self.assertFalse(any(e.step=='update' for e in events))
        self.assertGreater(runner.stops,0)
    def test_failed_readiness_check_is_visible_and_owned(self):
        runner=FakeRunner('@@check FAIL Check the box connection\n@@done ok\n');events=[]
        with tempfile.TemporaryDirectory() as directory:
            script=Path(directory)/'engine.ps1';script.touch()
            self.assertFalse(EngineBackend(runner,script,'powershell').run(Answers(),events.append))
        self.assertEqual(events[-1].kind,'step')
        self.assertEqual(events[-1].step,'readiness')
        self.assertEqual(events[-1].status,'fail')
        self.assertFalse(runner.path.exists())

    def test_cancel_cleans_file_and_process(self):
        runner=FakeRunner(recording())
        with tempfile.TemporaryDirectory() as directory:
            script=Path(directory)/'engine.ps1';script.touch()
            backend=EngineBackend(runner,script,'powershell')
            original=runner.spawn
            def spawn(args):
                process=original(args);backend.cancel();return process
            runner.spawn=spawn
            self.assertFalse(backend.run(Answers(),lambda event:None))
        self.assertFalse(runner.path.exists());self.assertGreater(runner.stops,0)
    def test_spawn_failure_cleans_file(self):
        runner=FakeRunner('')
        def fail(args): runner.path=Path(args[-1]);raise OSError()
        runner.spawn=fail
        with tempfile.TemporaryDirectory() as directory:
            script=Path(directory)/'engine.ps1';script.touch();events=[]
            self.assertFalse(EngineBackend(runner,script,'powershell').run(Answers(),events.append))
        self.assertFalse(runner.path.exists());self.assertEqual(events[-1].status,'fail')
    def test_missing_engine_or_powershell_and_incomplete_run(self):
        with tempfile.TemporaryDirectory() as directory:
            script=Path(directory)/'engine.ps1';events=[]
            self.assertFalse(EngineBackend(FakeRunner(''),script,'powershell').run(Answers(),events.append))
            self.assertIn('missing',events[-1].text)
            script.touch()
            with patch('home_guard_project.box.app.engine_backend.shutil.which',return_value=None):
                self.assertFalse(EngineBackend(FakeRunner(''),script).run(Answers(),events.append))
            self.assertIn('PowerShell',events[-1].text)
            for lines,code in [('@@done ok\n',0),(recording(),1),('@@step update start\n',0)]:
                self.assertFalse(EngineBackend(FakeRunner(lines,code),script,'powershell').run(Answers(),events.append))
    def test_process_runner_clears_inherited_environment_and_hides_console(self):
        import os
        from home_guard_project.box.app.engine_backend import ProcessRunner
        with patch.dict(os.environ,{'VIRTUAL_ENV':'wrong-checkout'}),patch('home_guard_project.box.app.engine_backend.subprocess.Popen') as spawn:
            ProcessRunner().spawn(['powershell','-File','unused'])
        self.assertNotIn('VIRTUAL_ENV',spawn.call_args.kwargs['env'])
        import subprocess
        self.assertEqual(spawn.call_args.kwargs['creationflags'],getattr(subprocess,'CREATE_NO_WINDOW',0))

    def test_parser_redacts_secrets_addresses_and_hides_rescue_repr(self):
        secret=secrets.token_hex(12);parser=OutputParser((secret,))
        event=parser.parse('ordinary installer@box.example password='+secret)
        self.assertNotIn(secret,event.text);self.assertNotIn('installer@box.example',event.text)
        rescue=parser.parse('@@rescue hotspot '+secret)
        self.assertEqual(rescue.password,secret);self.assertNotIn(secret,repr(rescue))
        self.assertNotIn(secret,parser.parse('ordinary '+secret).text)
        self.assertEqual(parser.parse('@@step cameras skip Later').status,'skip')
        self.assertEqual(OWNERS['alerts'],2)
