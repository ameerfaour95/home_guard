"""migrate-layout on a fake box: dry run, run, rollback and finalize, with the box's tasks faked."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import unittest
from types import SimpleNamespace
from typing import Dict, List, Optional
from unittest import mock

from home_guard_project.box import layout_migration as lm
from home_guard_project.box import paths

BOX_YAML = """# Per-box settings. Created by setup_box.ps1 - not committed.
site: "house2"
mode: inference
min_age_minutes: 10
vlm_provider: openrouter
vlm_model: qwen/qwen3.5-4b
vlm_fallback_provider: dashscope-intl
vlm_fallback_model: qwen3.5-4b
"""
API_KEYS = "OPENAI_API_KEY=sk-test\nTELEGRAM_BOT_TOKEN=123:abc\nOPENROUTER_API_KEY=sk-or-test\n"

# The fake box's files, relative to the code folder, and where each lands in HOME.
LEGACY_FILES: Dict[str, Optional[str]] = {
    "home_guard_project/box/box.yaml": "config/box.yaml",
    "home_guard_project/box/registration.json": "config/registration.json",
    "home_guard_project/box/network.json": "config/network.json",
    "home_guard_project/data_collection/cameras.yaml": "config/cameras.yaml",
    "home_guard_project/data_collection/camera_alerts.yaml": "config/camera_alerts.yaml",
    "home_guard_project/data_collection/camera_aliases.yaml": "config/camera_aliases.yaml",
    "home_guard_project/data_collection/zones.yaml": "config/zones.yaml",
    "home_guard_project/data_collection/scene_maps.yaml": "config/scene_maps.yaml",
    "api_key.env": "secrets/api_key.env",
    "dataset_multi/clips/gate/2026-10-06/a.mp4": "data/live/clips/gate/2026-10-06/a.mp4",
    "dataset_multi/meta/gate/2026-10-06/a.meta.json": "data/live/meta/gate/2026-10-06/a.meta.json",
    "dataset_outbox/house2/clips/gate/2026-10-05/b.mp4": "data/outbox/house2/clips/gate/2026-10-05/b.mp4",
    "production_multi/clips/gate/2026-10-06/c_alert.mp4": "data/production/clips/gate/2026-10-06/c_alert.mp4",
    "production_multi/feedback/gate/2026-10-06/c_alert.json": "data/production/feedback/gate/2026-10-06/c_alert.json",
    "production_multi/.registry/house_state.jsonl": "data/state/house_state.jsonl",
    "production_multi/.registry/cases.jsonl": "data/state/cases.jsonl",
    "production_multi/.registry/quiet_since.json": "data/state/quiet_since.json",
    "production_multi/.conversations/-5.json": "data/state/.conversations/-5.json",
    "production_multi/.receipts/r1.json": "data/state/.receipts/r1.json",
    "production_multi/.alert_embeddings.json": "data/state/.alert_embeddings.json",
    "production_archive/house2/clips/gate/2026-10-01/d_alert.mp4": "data/archive/house2/clips/gate/2026-10-01/d_alert.mp4",
    "scene_interview/gate/numbered.jpg": "data/scene_interview/gate/numbered.jpg",
    "logs/runner.log": "logs/runner.log",
    "logs/collector-2026-10-06.log": "logs/collector-2026-10-06.log",
    "logs/collector.alive": "logs/collector.alive",
    "logs/ai_status.json": "logs/ai_status.json",
    "logs/alert_mute.json": "logs/alert_mute.json",
    "logs/telegram_chat.jsonl": "logs/telegram_chat.jsonl",
    "logs/preview/gate.jpg": "logs/preview/gate.jpg",
    "logs/registration.published": "config/registration.published",
    "logs/runner.winpid": None,                # processes of the old layout: not carried over
    "logs/collector.winpid": None,
    "yolo11s.pt": "models/yolo11s.pt",
    "yolo11s_openvino_model/metadata.yaml": "models/yolo11s_openvino_model/metadata.yaml",
    "FastSAM-s.pt": "models/FastSAM-s.pt",
    "api_key.env.bak-20261001": None,          # junk: only --finalize deletes it
}
CODE_FILES = ("home_guard_project/box/__init__.py", "home_guard_project/box/stop_collector.sh",
              "home_guard_project/data_collection/config.yaml", "pyproject.toml")


def write(path: str, text: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)


def read(path: str) -> str:
    with open(path, encoding="utf-8") as f:
        return f.read()


class FakeBox:
    """Records every command; the tasks, the processes and icacls all just work (or fail when asked)."""

    def __init__(self, code: str) -> None:
        self.code = code
        self.calls: List[List[str]] = []
        self.envs: List[Optional[dict]] = []
        self.alive: set = set()
        self.stuck: set = set()     # pids stop_collector.sh cannot end
        self.icacls_fails = False
        self.running = 0          # upload / heartbeat polls that still say Running

    def run(self, cmd, env=None):
        cmd = [str(c) for c in cmd]
        self.calls.append(cmd)
        self.envs.append(env)
        if cmd[0] == "git":
            return subprocess.run(cmd, capture_output=True, text=True, check=False)
        if cmd[0] == lm.GIT_BASH:            # stop_collector.sh: ends the processes, removes their pid files
            layout = paths.resolve({paths.ENV_HOME: env[paths.ENV_HOME]}, code_dir=self.code)
            for name in ("runner.winpid", "collector.winpid"):
                pid_file = os.path.join(layout.logs_dir(), name)
                if os.path.isfile(pid_file):
                    with open(pid_file, encoding="ascii") as f:
                        pid = f.read().strip()
                    if pid not in self.stuck:
                        self.alive.discard(pid)
                        os.remove(pid_file)
            return SimpleNamespace(returncode=0, stdout="")
        if cmd[0] == "tasklist":
            pid = cmd[2].split()[-1]
            return SimpleNamespace(returncode=0, stdout=f"python.exe {pid} Services" if pid in self.alive else "INFO: No tasks")
        if cmd[:2] == ["schtasks", "/Query"] and "/XML" in cmd:
            return SimpleNamespace(returncode=0, stdout="<Task><Principals><Principal><UserId>S-1-5-21-1-2-3-1001"
                                                        "</UserId></Principal></Principals></Task>")
        if cmd[:2] == ["schtasks", "/Query"]:
            if self.running:
                self.running -= 1
                return SimpleNamespace(returncode=0, stdout='"\\HomeGuard-Upload","N/A","Running"')
            return SimpleNamespace(returncode=0, stdout='"\\HomeGuard-Upload","N/A","Ready"')
        if cmd[0] == "icacls":
            return SimpleNamespace(returncode=5 if self.icacls_fails else 0, stdout="Access is denied." if self.icacls_fails else "")
        return SimpleNamespace(returncode=0, stdout="")

    def index(self, *prefix: str) -> int:
        return next(i for i, c in enumerate(self.calls) if c[:len(prefix)] == list(prefix))

    def has(self, *prefix: str) -> bool:
        return any(c[:len(prefix)] == list(prefix) for c in self.calls)


class MigrationTest(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.code = os.path.join(tmp.name, "home_guard")
        self.pd = os.path.join(tmp.name, "ProgramData")
        self.home = os.path.join(self.pd, "HomeGuard")
        for rel in LEGACY_FILES:
            write(self.src(rel), f"content of {rel}\n")
        write(self.src("home_guard_project/box/box.yaml"), BOX_YAML)
        write(self.src("api_key.env"), API_KEYS)
        write(self.src("logs/runner.winpid"), "4242")
        write(self.src("logs/collector.winpid"), "4343")
        for rel in CODE_FILES:
            write(self.src(rel), f"code {rel}\n")
        os.makedirs(self.src("production_outbox"))      # junk: an empty folder
        self.fake = FakeBox(self.code)
        self.said: List[str] = []
        self.box = lm.Box(self.code, run=self.fake.run, sleep=lambda s: None, wait_sec=30, say=self.said.append)

    def src(self, rel: str) -> str:
        return os.path.join(self.code, *rel.split("/"))

    def dst(self, rel: str) -> str:
        return os.path.join(self.home, *rel.split("/"))

    def migrate(self, **kw) -> int:
        return lm.migrate(self.code, self.home, self.box, **kw)

    def layout(self) -> paths.Layout:
        return paths.resolve({"ProgramData": self.pd}, code_dir=self.code)

    # -- dry run -------------------------------------------------------------------------------------
    def test_dry_run_prints_the_plan_and_changes_nothing(self) -> None:
        self.assertEqual(lm.main(["--dry-run", "--home", self.home, "--code-dir", self.code], box=self.box), 0)
        self.assertFalse(os.path.exists(self.home))
        self.assertEqual(self.fake.calls, [])
        text = "\n".join(self.said)
        for name in ("box.yaml", "cameras.yaml", "api_key.env", "live_dir", "production_dir", "state/house_state.jsonl",
                     "assistant/.conversations", "logs_dir", "models/yolo11s.pt", "models/yolo11s_openvino_model"):
            self.assertIn(name, text)
        self.assertIn("Dry run", text)

    # -- run -----------------------------------------------------------------------------------------
    def test_run_copies_everything_to_its_new_place(self) -> None:
        self.assertEqual(self.migrate(), 0)
        for rel, new in LEGACY_FILES.items():
            if not rel.endswith(".winpid"):    # stop_collector.sh removes those
                self.assertTrue(os.path.isfile(self.src(rel)), f"the original {rel} was kept")
            if new:
                self.assertEqual(read(self.dst(new)), read(self.src(rel)), rel)
        self.assertFalse(os.path.exists(self.dst("logs/runner.winpid")))
        self.assertFalse(os.path.exists(self.dst("logs/collector.winpid")))
        for rel in CODE_FILES:
            self.assertFalse(os.path.exists(self.dst(os.path.basename(rel))))

    def test_box_yaml_and_api_keys_are_carried_byte_for_byte(self) -> None:
        self.migrate()
        with open(self.dst("config/box.yaml"), "rb") as f:
            self.assertEqual(f.read(), BOX_YAML.encode())
        self.assertEqual(sum(1 for line in read(self.dst("config/box.yaml")).splitlines() if line.startswith("vlm_")), 4)
        with open(self.dst("secrets/api_key.env"), "rb") as f:
            self.assertEqual(f.read(), API_KEYS.encode())

    def test_after_the_run_the_box_finds_its_files_in_the_new_place(self) -> None:
        self.migrate()
        layout = self.layout()
        self.assertEqual((layout.mode, layout.home), (paths.HOME, os.path.abspath(self.home)))
        for path in (layout.box_yaml(), layout.cameras_yaml(), layout.zones_yaml(), layout.scene_maps_yaml(),
                     layout.camera_alerts_yaml(), layout.camera_aliases_yaml(), layout.registration_json(),
                     layout.registration_published(), layout.network_json(), layout.secrets_env(),
                     layout.alive_file(), layout.yolo_model(), os.path.join(layout.state_dir(), "house_state.jsonl"),
                     os.path.join(layout.assistant_dir(), ".alert_embeddings.json")):
            self.assertTrue(os.path.isfile(path), path)
        self.assertTrue(os.path.isdir(os.path.join(layout.assistant_dir(), ".conversations")))
        self.assertTrue(os.path.isdir(os.path.join(layout.models_dir(), "yolo11s_openvino_model")))

    def test_marker_and_manifest(self) -> None:
        self.migrate()
        marker = json.loads(read(self.dst("layout.json")))
        self.assertEqual(marker["layout_version"], paths.LAYOUT_VERSION)
        manifest = json.loads(read(self.dst(lm.MANIFEST_NAME)))
        listed = {os.path.normcase(lm._join(i["source"], rel)) for i in manifest["items"] for rel in i["files"]}
        expected = {os.path.normcase(self.src(rel)) for rel, new in LEGACY_FILES.items() if new}
        self.assertEqual(listed, expected)

    def test_the_box_is_stopped_first_locked_down_and_started_last(self) -> None:
        self.migrate()
        f = self.fake
        disable = [f.index("schtasks", "/Change", "/TN", t, "/DISABLE") for t in lm.PAUSED_TASKS]
        end = f.index("schtasks", "/End", "/TN", lm.COLLECTOR_TASK)
        stop = f.index(lm.GIT_BASH)
        acl = f.index("icacls")
        enable = [f.index("schtasks", "/Change", "/TN", t, "/ENABLE") for t in lm.PAUSED_TASKS]
        start = f.index("schtasks", "/Run", "/TN", lm.COLLECTOR_TASK)
        self.assertLess(max(disable), end)
        self.assertLess(end, stop)
        self.assertLess(stop, acl)
        self.assertLess(acl, min(enable))
        self.assertLess(max(enable), start)
        # stop_collector.sh is told which layout it stops (the old one), and is given as a Git Bash path.
        self.assertEqual(f.envs[stop][paths.ENV_HOME], paths.LEGACY)
        self.assertTrue(f.calls[stop][-1].endswith("/home_guard_project/box/stop_collector.sh"))
        self.assertFalse(f.calls[stop][-1][1:2] == ":")
        # The pids of the old layout were checked after the stop.
        self.assertTrue(f.has("tasklist", "/FI", "PID eq 4242"))
        self.assertTrue(f.has("tasklist", "/FI", "PID eq 4343"))

    def test_acl_is_administrators_system_and_the_task_user_only(self) -> None:
        self.migrate()
        acl = [c for c in self.fake.calls if c[0] == "icacls"]
        self.assertEqual([c[1] for c in acl], [os.path.abspath(self.home), os.path.join(os.path.abspath(self.home), "secrets")])
        for cmd in acl:
            self.assertEqual(cmd[2:], ["/inheritance:r", "/grant:r", "*S-1-5-32-544:(OI)(CI)F", "*S-1-5-18:(OI)(CI)F",
                                       "*S-1-5-21-1-2-3-1001:(OI)(CI)M"])

    def test_waits_for_a_running_upload(self) -> None:
        self.fake.running = 2
        self.assertEqual(self.migrate(), 0)
        self.assertTrue(any("waiting" in line for line in self.said))

    def test_a_second_run_does_nothing(self) -> None:
        self.migrate()
        self.fake.calls.clear()
        self.assertEqual(self.migrate(), 0)
        self.assertEqual(self.fake.calls, [])
        self.assertIn("Already migrated", self.said[-1])

    def test_a_program_that_will_not_stop_aborts_before_any_copy(self) -> None:
        self.fake.alive = self.fake.stuck = {"4343"}
        with self.assertRaises(lm.MigrationError):
            self.migrate()
        self.assertFalse(os.path.exists(self.dst("layout.json")))
        self.assertFalse(os.path.exists(self.dst("config/box.yaml")))
        self.assertTrue(self.fake.has("schtasks", "/Run", "/TN", lm.COLLECTOR_TASK))     # started again as it was
        self.assertTrue(self.fake.has("schtasks", "/Change", "/TN", "HomeGuard-Upload", "/ENABLE"))
        self.assertEqual(self.layout().mode, paths.LEGACY)

    def test_failed_permissions_leave_the_old_layout(self) -> None:
        self.fake.icacls_fails = True
        self.assertEqual(lm.main(["--home", self.home, "--code-dir", self.code], box=self.box), 1)
        self.assertFalse(os.path.exists(self.dst("layout.json")))
        self.assertTrue(self.fake.has("schtasks", "/Run", "/TN", lm.COLLECTOR_TASK))
        self.assertTrue(any("administrator" in line for line in self.said))

    def test_a_copy_that_does_not_match_switches_nothing(self) -> None:
        real = lm.copy_item

        def damaged(item, files=None):
            out = real(item, files)
            if item.name == "cameras.yaml":
                write(item.dest, "something else\n")     # same length? no: a different file
            return out

        with mock.patch.object(lm, "copy_item", damaged):
            with self.assertRaises(lm.MigrationError) as caught:
                self.migrate()
        self.assertIn("cameras.yaml", str(caught.exception))
        self.assertFalse(os.path.exists(self.dst("layout.json")))
        self.assertEqual(self.layout().mode, paths.LEGACY)
        self.assertTrue(self.fake.has("schtasks", "/Run", "/TN", lm.COLLECTOR_TASK))

    def test_a_same_size_change_in_config_is_caught(self) -> None:
        real = lm.copy_item

        def flipped(item, files=None):
            out = real(item, files)
            if item.name == "box.yaml":
                write(item.dest, BOX_YAML.replace("house2", "house3"))
            return out

        with mock.patch.object(lm, "copy_item", flipped):
            with self.assertRaises(lm.MigrationError):
                self.migrate()
        self.assertFalse(os.path.exists(self.dst("layout.json")))

    def test_clips_saved_while_the_box_was_stopping_are_copied_too(self) -> None:
        late = "dataset_multi/clips/gate/2026-10-06/late.mp4"
        real_stop = lm.Box.stop

        def stop_and_save(box, layout):
            write(self.src(late), "saved during the stop\n")
            real_stop(box, layout)

        with mock.patch.object(lm.Box, "stop", stop_and_save):
            self.migrate()
        self.assertTrue(os.path.isfile(self.dst("data/live/clips/gate/2026-10-06/late.mp4")))

    def test_not_enough_disk_stops_before_anything(self) -> None:
        with mock.patch.object(lm.shutil, "disk_usage", return_value=SimpleNamespace(free=10)):
            with self.assertRaises(lm.MigrationError):
                self.migrate()
        self.assertEqual(self.fake.calls, [])

    # -- rollback ------------------------------------------------------------------------------------
    def test_rollback_copies_back_what_changed_and_switches_back(self) -> None:
        self.migrate()
        layout = self.layout()
        write(layout.box_yaml(), BOX_YAML + "alert_start_hour: 22\n")                 # a setting changed since
        write(os.path.join(layout.live_dir(), "clips", "gate", "2026-10-07", "new.mp4"), "new clip\n")
        os.remove(os.path.join(layout.outbox_dir(), "house2", "clips", "gate", "2026-10-05", "b.mp4"))  # uploaded
        write(os.path.join(layout.assistant_dir(), ".desc", "c_alert.json"), "{}")    # new assistant state
        write(os.path.join(layout.state_dir(), "sees.json"), "{}")                    # new registry state
        write(os.path.join(layout.logs_dir(), "runner.winpid"), "999")                # a process of the new layout
        self.fake.alive = {"999"}
        self.fake.calls.clear()
        self.fake.envs.clear()

        self.assertEqual(lm.main(["--rollback", "--home", self.home, "--code-dir", self.code], box=self.box), 0)

        self.assertEqual(self.layout().mode, paths.LEGACY)
        self.assertTrue(read(self.src("home_guard_project/box/box.yaml")).endswith("alert_start_hour: 22\n"))
        self.assertEqual(read(self.src("dataset_multi/clips/gate/2026-10-07/new.mp4")), "new clip\n")
        self.assertFalse(os.path.exists(self.src("dataset_outbox/house2/clips/gate/2026-10-05/b.mp4")))
        self.assertTrue(os.path.isfile(self.src("production_multi/.desc/c_alert.json")))
        self.assertTrue(os.path.isfile(self.src("production_multi/.registry/sees.json")))
        self.assertTrue(os.path.isfile(self.src("logs/registration.published")))
        self.assertFalse(os.path.exists(self.src("logs/runner.winpid")))             # not carried back
        self.assertTrue(os.path.isdir(self.home))                       # left as it is
        # The stop was of the new layout, then the box started again.
        stop = self.fake.index(lm.GIT_BASH)
        self.assertEqual(os.path.normcase(self.fake.envs[stop][paths.ENV_HOME]), os.path.normcase(os.path.abspath(self.home)))
        self.assertTrue(self.fake.has("tasklist", "/FI", "PID eq 999"))
        self.assertTrue(self.fake.has("schtasks", "/Run", "/TN", lm.COLLECTOR_TASK))

    def test_rollback_without_a_migration_does_nothing(self) -> None:
        self.assertEqual(lm.rollback(self.code, self.home, self.box), 0)
        self.assertEqual(self.fake.calls, [])

    def test_migrate_rollback_migrate_again(self) -> None:
        self.migrate()
        lm.rollback(self.code, self.home, self.box)
        write(self.src("home_guard_project/data_collection/cameras.yaml"), "cameras: {gate: rtsp://x}\n")
        self.assertEqual(self.migrate(), 0)
        self.assertEqual(read(self.dst("config/cameras.yaml")), "cameras: {gate: rtsp://x}\n")
        self.assertEqual(self.layout().mode, paths.HOME)

    # -- finalize ------------------------------------------------------------------------------------
    def git(self, *args: str) -> None:
        subprocess.run(["git", "-C", self.code, *args], check=True, capture_output=True,
                       env=dict(os.environ, GIT_AUTHOR_NAME="t", GIT_AUTHOR_EMAIL="t@t", GIT_COMMITTER_NAME="t",
                                GIT_COMMITTER_EMAIL="t@t"))

    def detached_checkout(self) -> None:
        """The code folder as on the live box: a git checkout on a detached HEAD."""
        write(self.src("scene_interview/README.txt"), "tracked, inside a data folder\n")
        self.git("init", "-q")
        self.git("add", *CODE_FILES, "scene_interview/README.txt")
        self.git("commit", "-q", "-m", "code")
        self.git("checkout", "-q", "--detach")

    def test_finalize_before_migrating_refuses(self) -> None:
        self.detached_checkout()
        self.assertEqual(lm.main(["--finalize", "--yes", "--home", self.home, "--code-dir", self.code], box=self.box), 1)
        self.assertTrue(os.path.isfile(self.src("api_key.env")))

    def test_finalize_lists_and_deletes_nothing_without_yes(self) -> None:
        self.detached_checkout()
        self.migrate()
        self.said.clear()
        self.assertEqual(lm.main(["--finalize", "--home", self.home, "--code-dir", self.code], box=self.box), 0)
        text = "\n".join(self.said)
        self.assertIn(self.src("api_key.env"), text)
        self.assertIn(self.src("api_key.env.bak-20261001"), text)
        self.assertIn("production_outbox", text)
        self.assertIn("Nothing deleted", text)
        for rel in LEGACY_FILES:
            if not rel.endswith(".winpid"):
                self.assertTrue(os.path.isfile(self.src(rel)), rel)

    def test_finalize_yes_deletes_the_old_copies_and_never_code_or_git(self) -> None:
        self.detached_checkout()
        self.migrate()
        self.assertEqual(lm.main(["--finalize", "--yes", "--home", self.home, "--code-dir", self.code], box=self.box), 0)
        for rel, new in LEGACY_FILES.items():
            if new or rel.startswith("api_key.env.bak"):
                self.assertFalse(os.path.exists(self.src(rel)), rel)
            if new:
                self.assertTrue(os.path.isfile(self.dst(new)), new)       # the new copies stay
        for rel in CODE_FILES + ("scene_interview/README.txt",):
            self.assertTrue(os.path.isfile(self.src(rel)), rel)
        self.assertTrue(os.path.isdir(os.path.join(self.code, ".git")))
        for gone in ("dataset_multi", "dataset_outbox", "production_archive", "yolo11s_openvino_model", "production_outbox",
                     os.path.join("production_multi", ".registry")):
            self.assertFalse(os.path.exists(os.path.join(self.code, gone)), gone)
        self.assertTrue(os.path.isdir(os.path.join(self.code, "home_guard_project", "box")))
        out = subprocess.run(["git", "-C", self.code, "status", "--porcelain"], capture_output=True, text=True)
        self.assertEqual(out.stdout.strip(), "")          # nothing git knows about was touched

    def test_finalize_keeps_a_file_that_changed_since_the_copy(self) -> None:
        self.detached_checkout()
        self.migrate()
        write(self.src("dataset_multi/clips/gate/2026-10-06/a.mp4"), "changed after the copy, longer than before\n")
        lm.main(["--finalize", "--yes", "--home", self.home, "--code-dir", self.code], box=self.box)
        self.assertTrue(os.path.isfile(self.src("dataset_multi/clips/gate/2026-10-06/a.mp4")))

    def test_finalize_without_git_deletes_nothing(self) -> None:
        self.migrate()            # the code folder is not a git checkout here
        self.assertEqual(lm.main(["--finalize", "--yes", "--home", self.home, "--code-dir", self.code], box=self.box), 1)
        self.assertTrue(os.path.isfile(self.src("api_key.env")))

    def test_rollback_after_finalize_restores_the_old_places(self) -> None:
        self.detached_checkout()
        self.migrate()
        lm.main(["--finalize", "--yes", "--home", self.home, "--code-dir", self.code], box=self.box)
        lm.rollback(self.code, self.home, self.box)
        for rel, new in LEGACY_FILES.items():
            if new:
                self.assertTrue(os.path.isfile(self.src(rel)), rel)
        self.assertEqual(read(self.src("api_key.env")), API_KEYS)
        self.assertEqual(self.layout().mode, paths.LEGACY)


@unittest.skipUnless(shutil.which("git"), "needs git")
class FinalizeGuardTest(unittest.TestCase):
    def test_a_manifest_naming_code_or_git_is_never_deleted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            code, home = os.path.join(tmp, "code"), os.path.join(tmp, "home")
            write(os.path.join(code, "run.py"), "code\n")
            write(os.path.join(code, "notes.txt"), "untracked\n")
            subprocess.run(["git", "-C", code, "init", "-q"], check=True)
            subprocess.run(["git", "-C", code, "add", "run.py"], check=True)
            write(os.path.join(home, paths.MARKER_NAME), "{}")
            files = {"run.py": 5, "notes.txt": 10, ".git/HEAD": os.path.getsize(os.path.join(code, ".git", "HEAD"))}
            write(os.path.join(home, lm.MANIFEST_NAME), json.dumps({"items": [
                {"name": "x", "source": code, "dest": home, "is_dir": True, "files": files}]}))
            self.assertEqual(lm.finalize_list(code, home), [os.path.join(code, "notes.txt")])


if __name__ == "__main__":
    unittest.main()
