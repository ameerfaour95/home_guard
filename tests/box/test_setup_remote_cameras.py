import json,secrets,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from home_guard_project.box.app.remote_cameras import RemoteCameras,target_user
from home_guard_project.box.app.camera_controls import Camera

class FakeRunner:
    def __init__(self): self.calls=[];self.fail=False;self.payload=None;self.cancelled=False
    def run(self,args):
        self.calls.append(args)
        if args[0]=='ssh.exe' and 'snapshots --out' in args[-1]:
            return SimpleNamespace(returncode=0,stdout=json.dumps({'snapshots':[{'name':'front_door','file':'C:/Users/installer/hg_snapshots/front_door.jpg','ok':True},{'name':'garden','file':'','ok':False}]}))
        if args[0]=='scp.exe' and '*.jpg' in args[-2]:
            (Path(args[-1])/'front_door.jpg').write_bytes(b'synthetic-image')
            return SimpleNamespace(returncode=0,stdout='')
        if args[0]=='scp.exe':
            self.local_change=Path(args[-2]);self.payload=json.loads(self.local_change.read_text())
            return SimpleNamespace(returncode=0,stdout='')
        if self.fail: return SimpleNamespace(returncode=1,stdout='{"error":"Private diagnostic"}')
        rows=self.payload['cameras']
        return SimpleNamespace(returncode=0,stdout=json.dumps({'active':[r['new_name'] for r in rows if r['enabled']],'disabled':[r['new_name'] for r in rows if not r['enabled']]}))
    def cancel(self): self.cancelled=True

class RemoteCameraTests(unittest.TestCase):
    def setUp(self):
        self.runner=FakeRunner()
        self.controls=RemoteCameras('installer@box.example',runner=self.runner,key='unused-test-key')
        self.addCleanup(self.controls.cleanup)
    def test_snapshots_fetch_over_same_connection_without_remote_quotes(self):
        cameras=self.controls.snapshots()
        self.assertTrue(cameras[0].ok);self.assertFalse(cameras[1].ok)
        self.assertTrue(Path(cameras[0].file).is_file())
        ssh,scp=self.runner.calls
        self.assertEqual(ssh[:5],['ssh.exe','-i','unused-test-key','-o','LogLevel=ERROR'])
        self.assertEqual(ssh[5],'installer@box.example')
        self.assertNotIn('"',ssh[-1]);self.assertNotIn("'",ssh[-1])
        self.assertIn('C:\\Users\\installer\\hg_snapshots',ssh[-1])
        self.assertEqual(scp[-2],'installer@box.example:C:/Users/installer/hg_snapshots/*.jpg')
    def test_save_rename_disable_reenable_and_cleanup_json(self):
        self.controls.snapshots()
        cameras=self.controls.save([('front_door','entrance',False),('garden','garden',True)])
        self.assertEqual(cameras[0].name,'entrance');self.assertFalse(cameras[0].enabled)
        self.assertFalse(self.runner.local_change.exists())
        self.assertIn('apply --changes C:\\Users\\installer\\hg_camera_changes.json',self.runner.calls[-1][-1])
        self.controls.save([('entrance','entrance',True),('garden','garden',True)])
        self.assertTrue(self.controls.records[0].enabled)
    def test_apply_failure_preserves_records_and_cleans_changes(self):
        self.controls.snapshots();before=list(self.controls.records);self.runner.fail=True
        with self.assertRaises(RuntimeError): self.controls.save([('front_door','entrance',False),('garden','garden',True)])
        self.assertEqual(self.controls.records,before);self.assertFalse(self.runner.local_change.exists())
    def test_failed_photos_do_not_scp(self):
        self.runner.run=lambda args: SimpleNamespace(returncode=1,stdout='{"snapshots":[{"name":"door","file":"","ok":false}]}')
        self.assertFalse(self.controls.snapshots()[0].ok)
    def test_rejects_unsafe_targets_and_names_before_process_calls(self):
        for target in ('-option@box','user name@box','user@box&command','user@box"','../user@box'):
            with self.assertRaises(ValueError): target_user(target)
        self.controls.records=[Camera('door')]
        with self.assertRaises(ValueError): self.controls.save([('door','unsafe name',True)])
        self.assertEqual(self.runner.calls,[])
    def test_cancel_prevents_commands_and_cleans_temp_images(self):
        self.controls.snapshots();path=Path(self.controls.directory.name)
        self.controls.cancel();self.assertTrue(self.runner.cancelled)
        with self.assertRaises(RuntimeError): self.controls.snapshots()
        self.assertFalse(path.exists())
