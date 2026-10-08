"""The laptop's SSH commands: the code folder is fixed, the box says where its files are."""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

from home_guard_project.box import paths
from home_guard_project.box.app import box_layout
from home_guard_project.box.app.box_controls import RemoteSettingsBackend, Settings
from home_guard_project.box.app.remote_cameras import RemoteCameras
from home_guard_project.box.app.remote_live import tunnel_command

HOME = r"C:\ProgramData\HomeGuard"
MIGRATED = paths.home_layout(HOME, r"C:\home_guard").as_dict()


class FakeSsh:
    def __init__(self, paths_answer=None, paths_code=0):
        self.calls = []
        self.paths_answer, self.paths_code = paths_answer, paths_code

    def run(self, args, **kwargs):
        self.calls.append(args)
        command = args[-1]
        if command == box_layout.PATHS_COMMAND:
            return SimpleNamespace(returncode=self.paths_code, stdout=self.paths_answer or "")
        if command.startswith("type "):
            return SimpleNamespace(returncode=0, stdout=json.dumps({"updated": time.time(), "settings": {}}))
        return SimpleNamespace(returncode=0, stdout="{}")


class CommandTest(unittest.TestCase):
    def test_box_command_runs_the_boxs_python_in_its_code_folder(self):
        self.assertEqual(box_layout.box_command("-m home_guard_project.box status"),
                         r"cd /d C:\home_guard && .venv\Scripts\python.exe -m home_guard_project.box status")
        self.assertEqual(box_layout.PATHS_COMMAND,
                         r"cd /d C:\home_guard && .venv\Scripts\python.exe -m home_guard_project.box paths --json")

    def test_a_migrated_box_answers_its_places(self):
        noise = "[h264 @ 0x1] decoder noise\n" + json.dumps(MIGRATED, indent=2)
        self.assertEqual(box_layout.parse_paths(noise, 0)["logs_dir"], HOME + r"\logs")

    def test_an_old_box_is_the_old_layout(self):
        for output, code in (("usage: ... invalid choice: 'paths'", 2), ("", 0), ("{not json", 0), ('{"x": 1}', 0),
                             (json.dumps(MIGRATED), 1)):
            got = box_layout.parse_paths(output, code)
            self.assertEqual(got["mode"], paths.LEGACY)
            self.assertEqual(got["logs_dir"], r"C:\home_guard\logs")
            self.assertEqual(got["secrets_env"], r"C:\home_guard\api_key.env")
            self.assertEqual(got["network_json"], r"C:\home_guard\home_guard_project\box\network.json")

    def test_type_command_quotes_the_path(self):
        self.assertEqual(box_layout.type_command(HOME + r"\logs", "ai_status.json"),
                         r'type "C:\ProgramData\HomeGuard\logs\ai_status.json"')
        with self.assertRaises(ValueError):
            box_layout.type_command('C:\\x" & del', "a")

    def test_settings_command(self):
        ssh = FakeSsh()
        backend = RemoteSettingsBackend("user@box", Settings(), ssh, lambda: {}, key="k")
        backend.save_settings(Settings(alert_start_hour=22))
        self.assertEqual(ssh.calls[-1][-1], box_layout.box_command("-m home_guard_project.box set-option alert_start_hour=22"))

    def test_camera_command(self):
        cameras = RemoteCameras("user@box", runner=FakeSsh())
        self.assertEqual(cameras.ssh("list")[-1], box_layout.box_command("-m home_guard_project.box.find_cameras --json list"))

    def test_live_tunnel_command(self):
        self.assertEqual(tunnel_command("user@box")[-1], box_layout.box_command("-m home_guard_project.box.app.live_server"))


class ReportedStatusTest(unittest.TestCase):
    def test_reads_ai_status_where_a_migrated_box_keeps_its_logs_asking_once(self):
        ssh = FakeSsh(json.dumps(MIGRATED))
        cameras = RemoteCameras("user@box", runner=ssh)
        cameras.alert_reported_status()
        cameras.alert_reported_status()
        self.assertEqual(sum(1 for c in ssh.calls if c[-1] == box_layout.PATHS_COMMAND), 1)
        self.assertEqual([c[-1] for c in ssh.calls if c[-1].startswith("type ")],
                         [r'type "C:\ProgramData\HomeGuard\logs\ai_status.json"'] * 2)

    def test_an_old_box_is_read_in_its_code_folder(self):
        ssh = FakeSsh("usage: box [-h] {upload,...}\nbox: error: invalid choice: 'paths'", paths_code=2)
        RemoteCameras("user@box", runner=ssh).alert_reported_status()
        self.assertEqual(ssh.calls[-1][-1], r'type "C:\home_guard\logs\ai_status.json"')


SETUP = Path(__file__).parents[2] / "home_guard_project/box/setup_customer.ps1"


@unittest.skipUnless(shutil.which("powershell"), "Windows PowerShell required")
class WizardBoxPathsTest(unittest.TestCase):
    """setup_customer.ps1 asks the box where api_key.env and network.json are, and falls back on an old box."""

    def run_block(self, answer: str, code: int) -> list:
        script = SETUP.read_text(encoding="utf-8")
        block = script[script.index("$script:BoxPaths = $null"):script.index("# The box drops its network link")]
        source = "\n".join([
            "$ErrorActionPreference='Stop'",
            r"$InstallDir='C:\home_guard'; $BoxBox=" + '"$InstallDir' + r'\home_guard_project\box"',
            r"$Python='.venv\Scripts\python.exe'; $DryRun=$false; $global:asked=0",
            "function Invoke-Box($command) {",
            f"    $global:asked++; $global:LASTEXITCODE={code}",
            "    if ($command -notmatch 'paths --json') { throw 'unexpected' }",
            "    return @'", answer, "'@",
            "}",
            block,
            "$a = Get-BoxPaths; $b = Get-BoxPaths",
            "Write-Output $a.secrets_env; Write-Output $a.network_json; Write-Output $global:asked",
        ])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "test.ps1"
            path.write_text(source, encoding="utf-8-sig")
            out = subprocess.run(["powershell", "-NoProfile", "-File", str(path)], capture_output=True, text=True, timeout=120)
        self.assertEqual(out.returncode, 0, out.stderr)
        return out.stdout.strip().splitlines()

    def test_a_migrated_box(self):
        self.assertEqual(self.run_block(json.dumps(MIGRATED, indent=2), 0),
                         [HOME + r"\secrets\api_key.env", HOME + r"\config\network.json", "1"])

    def test_an_old_box_asks_once_and_uses_the_code_folder(self):
        self.assertEqual(self.run_block("box: error: invalid choice: 'paths'", 2),
                         [r"C:\home_guard\api_key.env", r"C:\home_guard\home_guard_project\box\network.json", "1"])


if __name__ == "__main__":
    unittest.main()
