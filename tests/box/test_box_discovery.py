import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from home_guard_project.box.app.box_discovery import parse_peers, discover, compose_target
from home_guard_project.box.app.preferences import AddressPreference

class BoxDiscoveryTests(unittest.TestCase):
    def test_windows_only_online_first_deduplicated(self):
        peers={'a':{'OS':'windows','HostName':'Z box','Online':True,'TailscaleIPs':['100.100.100.10']},'b':{'OS':'windows','HostName':'A box','Online':False,'TailscaleIPs':['100.100.100.11']},'c':{'OS':'linux','Online':True,'TailscaleIPs':['100.100.100.12']}}
        result=parse_peers(json.dumps({'Peer':peers}))
        self.assertEqual([p.name for p in result],['Z box','A box'])
        self.assertEqual(compose_target(' installer ','100.100.100.10 '),'installer@100.100.100.10')
        with self.assertRaises(ValueError): compose_target('bad user','box.example')
    def test_local_runner_missing_and_empty(self):
        calls=[]
        def runner(command,**kwargs):
            calls.append(command)
            self.assertNotIn('VIRTUAL_ENV',kwargs['env'])
            return SimpleNamespace(returncode=0,stdout='{"Peer": {}}')
        self.assertEqual(discover(runner),[])
        self.assertEqual(calls,[['tailscale','status','--json']])
        def missing(*a,**kw): raise FileNotFoundError()
        self.assertEqual(discover(missing),[])
        self.assertEqual(parse_peers('bad JSON'),[])
    def test_user_changes_only_after_success(self):
        with tempfile.TemporaryDirectory() as directory:
            pref=AddressPreference(Path(directory)/'last_box.json')
            pref.save('installer@box.example')
            self.assertEqual(pref.load_user(),'')
            pref.save_success('installer@box.example')
            pref.save('new_user@another.example')
            self.assertEqual(pref.load(),'new_user@another.example')
            self.assertEqual(pref.load_user(),'installer')
            self.assertEqual(set(json.loads(pref.path.read_text())),{'target','successful_user'})
