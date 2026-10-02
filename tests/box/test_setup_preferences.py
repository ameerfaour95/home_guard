import json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from home_guard_project.box.app.preferences import AddressPreference

class PreferenceTests(unittest.TestCase):
    def test_address_roundtrip_saves_only_target_atomically(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'profile'/'last_box.json';prefs=AddressPreference(path)
            self.assertEqual(prefs.load(),'')
            prefs.save('installer@box.example')
            self.assertEqual(AddressPreference(path).load(),'installer@box.example')
            self.assertEqual(json.loads(path.read_text()),{'target':'installer@box.example'})
            self.assertEqual(list(path.parent.glob('*.tmp')),[])
    def test_missing_damaged_and_unsafe_addresses_are_empty(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'last_box.json';prefs=AddressPreference(path)
            for value in ('broken','[]','null','{}','{"target":"user@box&command"}','{"target":false}'):
                path.write_text(value);self.assertEqual(prefs.load(),'')
            with self.assertRaises(ValueError): prefs.save('user@box&command')
    def test_replace_failure_keeps_old_address_and_removes_temp(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'last_box.json';prefs=AddressPreference(path)
            prefs.save('installer@first.example')
            with patch('home_guard_project.box.app.preferences.os.replace',side_effect=OSError):
                with self.assertRaises(OSError): prefs.save('installer@second.example')
            self.assertEqual(prefs.load(),'installer@first.example')
            self.assertEqual(list(path.parent.glob('*.tmp')),[])
