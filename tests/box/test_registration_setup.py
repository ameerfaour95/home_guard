"""Exercise the actual setup blocks in Windows PowerShell without SSH or AWS."""
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

SCRIPT = Path(__file__).parents[2] / 'home_guard_project/box/setup_customer.ps1'


@unittest.skipUnless(shutil.which('powershell'), 'Windows PowerShell required')
class RegistrationSetupTests(unittest.TestCase):
    def run_script(self, source):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'test.ps1'
            path.write_text(source, encoding='utf-8-sig')
            return subprocess.run(['powershell', '-NoProfile', '-File', str(path)],
                                  capture_output=True, text=True, timeout=30)

    def test_old_answers_and_non_boolean_consents_default_off(self):
        script = SCRIPT.read_text(encoding='utf-8')
        block = script.split('# ---- answers:')[1].split('# ---- which box')[0]
        block = block[block.index('$Mode ='):]
        for extras in ({}, {'consent_live': 'true', 'consent_recordings': 1, 'consent_training': 'yes'}):
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / 'answers.json'
                path.write_text(json.dumps(dict(target='installer@box.example', network='ethernet', site='test', **extras)))
                result = self.run_script("$ErrorActionPreference='Stop'\n$NonInteractive=$true\n$AnswersFile='" + str(path) + "'\n" + block +
                                         '\n@($ConsentLive,$ConsentRecordings,$ConsentTraining) | ConvertTo-Json -Compress')
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(json.loads(result.stdout), [False, False, False])
                self.assertFalse(path.exists())

    def test_registration_output_and_cleanup_are_private(self):
        block = SCRIPT.read_text(encoding='utf-8').split("Step-Start 'register'")[1].split('# ---- network configuration')[0]
        for cleanup_fails in (False, True):
            source = r"""
$ErrorActionPreference='Stop'
$OwnerName='PRIVATE OWNER'; $OwnerPhone='PRIVATE PHONE'; $Installer='Installer'
$Target='installer@box.example'; $RemoteHome='C:\Users\installer'
$InstallDir='C:\home_guard'; $Python='python'; $DryRun=$false
$ConsentLive=$false; $ConsentRecordings=$false; $ConsentTraining=$false
function Info($m) {}; function Note($m) { Write-Output $m }; function Ok($m) {}
function Bad($m) { Write-Output $m }; function Step-Ok($a,$b) {}; function Step-Warn($a,$b) {}
function Copy-ToBox($local,$remote) {
    $payload=Get-Content $local -Raw | ConvertFrom-Json
    if ($payload.owner_name -ne 'PRIVATE OWNER') { throw 'Payload missing' }
}
function Invoke-Box($command) {
    if ($command -match 'PRIVATE') { throw 'Private command line' }
    $global:LASTEXITCODE=0
    if ($command -match 'register --from-json') { return 'registered test for PRIVATE OWNER; PRIVATE PHONE' }
    if (CLEANUP_FAILS) { throw 'Cleanup failed' }
}
""".replace('CLEANUP_FAILS', '$true' if cleanup_fails else '$false')
            result = self.run_script(source + '\ntry {\n' + block +
                                     '\n} finally { if (Test-Path $regLocal) { Remove-Item $regLocal; throw "Local file leaked" } }')
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertNotIn('PRIVATE', result.stdout + result.stderr)

    def test_native_stderr_does_not_terminate_under_stop(self):
        script = SCRIPT.read_text(encoding='utf-8')
        helper = 'function Invoke-Native' + script.split('function Invoke-Native')[1].split('function Invoke-Box')[0]
        result = self.run_script("$ErrorActionPreference='Stop'\n" + helper +
                                 "\nInvoke-Native 'cmd.exe' @('/c','echo ordinary-progress 1>&2 & exit /b 7')\nWrite-Output \"exit=$LASTEXITCODE\"\nWrite-Output \"preference=$ErrorActionPreference\"")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('exit=7', result.stdout)
        self.assertIn('preference=Stop', result.stdout)

    def test_compiled_engine_dry_run_accepts_old_and_new_answers(self):
        probe = self.run_script("if (-not (Get-Module -ListAvailable ps2exe)) { exit 2 }")
        if probe.returncode == 2: self.skipTest('ps2exe not installed')
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root/'setup.ps1'; exe = root/'setup.exe'
            script = SCRIPT.read_text(encoding='utf-8').replace(
                "$HgDir = Join-Path $HOME '.homeguard'", "$HgDir = '" + str(root/'prefs') + "'")
            source.write_text(script, encoding='utf-8-sig')
            compiled = self.run_script("$ErrorActionPreference='Stop'\nImport-Module ps2exe\nInvoke-ps2exe -inputFile '" +
                                       str(source) + "' -outputFile '" + str(exe) + "' | Out-Null")
            self.assertEqual(compiled.returncode, 0, compiled.stderr)
            for fields in ({}, dict(owner_name='PRIVATE OWNER', owner_phone='PRIVATE PHONE',
                                   installer='Sam', consent_live=True, consent_recordings=False, consent_training=False)):
                answers = root/'answers.json'
                answers.write_text(json.dumps(dict(target='installer@box.example', network='ethernet',
                                                    site='test_house', find_cameras=False, **fields)), encoding='utf-8')
                try:
                    result = subprocess.run([str(exe), '-AnswersFile', str(answers), '-DryRun', '-SkipUpdate'],
                                            capture_output=True, text=True, timeout=45,
                                            creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
                except OSError as exc:
                    if getattr(exc, 'winerror', None) == 4551:
                        self.skipTest('Windows Application Control blocked the generated test executable')
                    raise
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn('@@step register ok', result.stdout)
                self.assertIn('@@done ok', result.stdout)
                self.assertNotIn('PRIVATE', result.stdout + result.stderr)
                self.assertFalse(answers.exists())
