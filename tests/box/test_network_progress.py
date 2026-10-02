import io
import tempfile
import unittest
from pathlib import Path
from home_guard_project.box.app.backend import Answers
from home_guard_project.box.app.engine_backend import EngineBackend,OutputParser,is_progress_warning
from home_guard_project.box.app.setup_details import DetailsModel

WARN="@@step network warn The box is switching to ameer2; the connection will drop for up to a minute"

class Process:
    def __init__(self,text): self.stdout=io.StringIO(text)
    def wait(self,timeout=None): return 0
class Runner:
    def __init__(self,text): self.process=Process(text)
    def spawn(self,args): return self.process
    def stop(self,process): pass

class ProgressTests(unittest.TestCase):
    def test_switch_warning_is_running_until_ok(self):
        parser=OutputParser();model=DetailsModel()
        model.feed(parser.parse('@@step network start'))
        event=parser.parse(WARN);self.assertTrue(is_progress_warning(event))
        model.feed(event)
        self.assertEqual(model.groups['network'].status,'start')
        self.assertIn(('warning',event.text),model.groups['network'].messages)
        model.feed(parser.parse('@@step network ok Home network is ready'))
        self.assertEqual(model.groups['network'].status,'ok')
        self.assertEqual(model.groups['network'].result,'Home network is ready')

    def test_full_recording_completes_with_warning_then_ok(self):
        recording=Path(__file__).with_name('fixtures').joinpath('setup_engine_success.txt').read_text()
        recording=recording.replace('@@step network start','@@step network start\n'+WARN)
        with tempfile.TemporaryDirectory() as directory:
            script=Path(directory)/'engine.ps1';script.touch()
            self.assertTrue(EngineBackend(Runner(recording),script,'powershell').run(Answers(),lambda e:None))

    def test_switch_warning_cannot_complete_network_without_ok(self):
        recording=Path(__file__).with_name('fixtures').joinpath('setup_engine_success.txt').read_text()
        recording='\n'.join(WARN if line.startswith('@@step network ok') else line for line in recording.splitlines())
        with tempfile.TemporaryDirectory() as directory:
            script=Path(directory)/'engine.ps1';script.touch()
            self.assertFalse(EngineBackend(Runner(recording),script,'powershell').run(Answers(),lambda e:None))

    def test_terminal_warning_stays_a_warning(self):
        event=OutputParser().parse('@@step update warn Software was already current')
        self.assertFalse(is_progress_warning(event))
        model=DetailsModel();model.feed(event)
        self.assertEqual(model.groups['update'].status,'warn')
