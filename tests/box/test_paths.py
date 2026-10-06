"""paths.py: which layout a machine uses, and where every place is in each layout."""

from __future__ import annotations

import contextlib
import io
import json
import os
import shutil
import subprocess
import tempfile
import unittest

from home_guard_project.box import boxconfig, paths

CODE = os.path.abspath(os.path.join(os.sep, "code"))
HOME = os.path.abspath(os.path.join(os.sep, "pd", "HomeGuard"))


def j(*parts: str) -> str:
    return os.path.join(*parts)


LEGACY_EXPECTED = {
    "config_dir": j(CODE, "home_guard_project", "data_collection"),
    "box_yaml": j(CODE, "home_guard_project", "box", "box.yaml"),
    "cameras_yaml": j(CODE, "home_guard_project", "data_collection", "cameras.yaml"),
    "camera_alerts_yaml": j(CODE, "home_guard_project", "data_collection", "camera_alerts.yaml"),
    "camera_aliases_yaml": j(CODE, "home_guard_project", "data_collection", "camera_aliases.yaml"),
    "zones_yaml": j(CODE, "home_guard_project", "data_collection", "zones.yaml"),
    "scene_maps_yaml": j(CODE, "home_guard_project", "data_collection", "scene_maps.yaml"),
    "registration_json": j(CODE, "home_guard_project", "box", "registration.json"),
    "registration_published": j(CODE, "logs", "registration.published"),
    "network_json": j(CODE, "home_guard_project", "box", "network.json"),
    "secrets_dir": CODE,
    "secrets_env": j(CODE, "api_key.env"),
    "data_dir": CODE,
    "live_dir": j(CODE, "dataset_multi"),
    "outbox_dir": j(CODE, "dataset_outbox"),
    "production_dir": j(CODE, "production_multi"),
    "archive_dir": j(CODE, "production_archive"),
    "state_dir": j(CODE, "production_multi", ".registry"),
    "assistant_dir": j(CODE, "production_multi"),
    "scene_interview_dir": j(CODE, "scene_interview"),
    "logs_dir": j(CODE, "logs"),
    "alive_file": j(CODE, "logs", "collector.alive"),
    "models_dir": CODE,
    "yolo_model": j(CODE, "yolo11s.pt"),
}

HOME_EXPECTED = {
    "config_dir": j(HOME, "config"),
    "box_yaml": j(HOME, "config", "box.yaml"),
    "cameras_yaml": j(HOME, "config", "cameras.yaml"),
    "camera_alerts_yaml": j(HOME, "config", "camera_alerts.yaml"),
    "camera_aliases_yaml": j(HOME, "config", "camera_aliases.yaml"),
    "zones_yaml": j(HOME, "config", "zones.yaml"),
    "scene_maps_yaml": j(HOME, "config", "scene_maps.yaml"),
    "registration_json": j(HOME, "config", "registration.json"),
    "registration_published": j(HOME, "config", "registration.published"),
    "network_json": j(HOME, "config", "network.json"),
    "secrets_dir": j(HOME, "secrets"),
    "secrets_env": j(HOME, "secrets", "api_key.env"),
    "data_dir": j(HOME, "data"),
    "live_dir": j(HOME, "data", "live"),
    "outbox_dir": j(HOME, "data", "outbox"),
    "production_dir": j(HOME, "data", "production"),
    "archive_dir": j(HOME, "data", "archive"),
    "state_dir": j(HOME, "data", "state"),
    "assistant_dir": j(HOME, "data", "state"),
    "scene_interview_dir": j(HOME, "data", "scene_interview"),
    "logs_dir": j(HOME, "logs"),
    "alive_file": j(HOME, "logs", "collector.alive"),
    "models_dir": j(HOME, "models"),
    "yolo_model": j(HOME, "models", "yolo11s.pt"),
}


class ResolveTest(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.pd = tmp.name
        self.home = os.path.join(self.pd, "HomeGuard")

    def mark(self) -> None:
        os.makedirs(self.home, exist_ok=True)
        with open(os.path.join(self.home, paths.MARKER_NAME), "w", encoding="utf-8") as f:
            f.write("{}")

    def test_no_marker_is_legacy(self) -> None:
        layout = paths.resolve({"ProgramData": self.pd}, code_dir=CODE)
        self.assertEqual((layout.mode, layout.home, layout.code_dir), (paths.LEGACY, None, CODE))

    def test_an_empty_programdata_folder_is_still_legacy(self) -> None:
        os.makedirs(self.home)   # created but not migrated (a rollback leaves it like this)
        self.assertEqual(paths.resolve({"ProgramData": self.pd}).mode, paths.LEGACY)

    def test_marker_in_programdata_is_home(self) -> None:
        self.mark()
        layout = paths.resolve({"ProgramData": self.pd}, code_dir=CODE)
        self.assertEqual((layout.mode, layout.home), (paths.HOME, os.path.abspath(self.home)))
        self.assertIn(paths.MARKER_NAME, layout.reason)

    def test_programdata_name_is_not_case_sensitive(self) -> None:
        self.mark()
        self.assertEqual(paths.resolve({"PROGRAMDATA": self.pd}).mode, paths.HOME)

    def test_env_folder_wins_over_everything(self) -> None:
        other = os.path.join(self.pd, "elsewhere")
        layout = paths.resolve({paths.ENV_HOME: other, "ProgramData": self.pd})
        self.assertEqual((layout.mode, layout.home), (paths.HOME, os.path.abspath(other)))

    def test_env_legacy_wins_over_the_marker(self) -> None:
        self.mark()
        for value in ("legacy", "LEGACY", " legacy "):
            self.assertEqual(paths.resolve({paths.ENV_HOME: value, "ProgramData": self.pd}).mode, paths.LEGACY)

    def test_an_empty_env_value_is_ignored(self) -> None:
        self.mark()
        self.assertEqual(paths.resolve({paths.ENV_HOME: "", "ProgramData": self.pd}).mode, paths.HOME)

    def test_no_programdata_in_a_given_environment_is_legacy(self) -> None:
        self.assertEqual(paths.resolve({}).mode, paths.LEGACY)
        self.assertIsNone(paths.default_home({}))

    def test_current_is_resolved_once_and_reset_resolves_again(self) -> None:
        self.addCleanup(paths.reset)
        fake = paths.home_layout(HOME, CODE)
        paths.reset(fake)
        self.assertIs(paths.current(), fake)
        self.assertEqual(paths.mode(), paths.HOME)
        self.assertEqual(paths.home(), HOME)
        paths.reset()
        self.assertIsNot(paths.current(), fake)

    def test_this_test_run_is_legacy_and_boxconfig_points_at_the_old_places(self) -> None:
        # A developer laptop: nothing migrated, so every alias is where it always was.
        self.assertEqual(paths.mode(), paths.LEGACY)
        self.assertEqual(boxconfig.PROJECT_ROOT, paths.CODE_DIR)
        self.assertEqual(boxconfig.LIVE_DIR, os.path.join(paths.CODE_DIR, "dataset_multi"))
        self.assertEqual(boxconfig.OUTBOX_DIR, os.path.join(paths.CODE_DIR, "dataset_outbox"))
        self.assertEqual(boxconfig.LOG_DIR, os.path.join(paths.CODE_DIR, "logs"))
        self.assertEqual(boxconfig.BOX_YAML, os.path.join(paths.CODE_DIR, "home_guard_project", "box", "box.yaml"))
        self.assertEqual(boxconfig.PRODUCTION_LIVE_DIR, os.path.join(paths.CODE_DIR, "production_multi"))
        self.assertEqual(boxconfig.PRODUCTION_ARCHIVE_DIR, os.path.join(paths.CODE_DIR, "production_archive"))


class AccessorTest(unittest.TestCase):
    def test_every_accessor_in_the_legacy_layout(self) -> None:
        layout = paths.legacy(CODE)
        self.assertEqual({name: getattr(layout, name)() for name in paths.PATH_NAMES}, LEGACY_EXPECTED)
        self.assertIsNone(layout.marker())

    def test_every_accessor_in_the_home_layout(self) -> None:
        layout = paths.home_layout(HOME, CODE)
        self.assertEqual({name: getattr(layout, name)() for name in paths.PATH_NAMES}, HOME_EXPECTED)
        self.assertEqual(layout.marker(), j(HOME, paths.MARKER_NAME))

    def test_module_functions_follow_the_current_layout(self) -> None:
        self.addCleanup(paths.reset)
        for layout, expected in ((paths.legacy(CODE), LEGACY_EXPECTED), (paths.home_layout(HOME, CODE), HOME_EXPECTED)):
            paths.reset(layout)
            self.assertEqual({name: getattr(paths, name)() for name in paths.PATH_NAMES}, expected)

    def test_as_dict_lists_every_place_and_the_mode(self) -> None:
        info = paths.home_layout(HOME, CODE).as_dict()
        self.assertEqual({k: info[k] for k in paths.PATH_NAMES}, HOME_EXPECTED)
        self.assertEqual((info["mode"], info["home"], info["code_dir"]), (paths.HOME, HOME, CODE))
        self.assertEqual(info["layout_version"], paths.LAYOUT_VERSION)

    def test_a_bare_model_name_is_in_the_models_folder(self) -> None:
        for layout in (paths.legacy(CODE), paths.home_layout(HOME, CODE)):
            self.assertEqual(layout.resolve_model("yolo11s.pt"), os.path.join(layout.models_dir(), "yolo11s.pt"))
            self.assertEqual(layout.resolve_model("FastSAM-s.pt"), os.path.join(layout.models_dir(), "FastSAM-s.pt"))
            for kept in (os.path.join("weights", "best.pt"), os.path.join(CODE, "x.pt"), "", "https://h/x.pt"):
                self.assertEqual(layout.resolve_model(kept), kept)

    def test_the_production_folders_state_has_its_own_place(self) -> None:
        home = paths.home_layout(HOME, CODE)
        self.assertEqual(home.state_paths_for(home.production_dir()), (j(HOME, "data", "state"), j(HOME, "data", "state")))
        old = paths.legacy(CODE)
        self.assertEqual(old.state_paths_for(old.production_dir()),
                         (j(CODE, "production_multi", ".registry"), j(CODE, "production_multi")))

    def test_any_other_alert_folder_keeps_its_state_inside(self) -> None:
        other = os.path.abspath("somewhere")
        for layout in (paths.legacy(CODE), paths.home_layout(HOME, CODE)):
            self.assertEqual(layout.state_paths_for(other), (os.path.join(other, ".registry"), other))


class CliTest(unittest.TestCase):
    def run_cli(self, *argv: str) -> str:
        self.addCleanup(paths.reset)
        paths.reset(paths.home_layout(HOME, CODE))
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            self.assertEqual(paths.main(list(argv)), 0)
        return out.getvalue()

    def test_json_is_every_place(self) -> None:
        info = json.loads(self.run_cli("--json"))
        self.assertEqual(info["mode"], paths.HOME)
        self.assertEqual({k: info[k] for k in paths.PATH_NAMES}, HOME_EXPECTED)

    def test_get_prints_one_place(self) -> None:
        self.assertEqual(self.run_cli("--get", "cameras_yaml").strip(), HOME_EXPECTED["cameras_yaml"])
        self.assertEqual(self.run_cli("--get", "mode").strip(), paths.HOME)

    def test_plain_lists_the_places(self) -> None:
        text = self.run_cli()
        self.assertIn("layout: home", text)
        self.assertIn(HOME_EXPECTED["logs_dir"], text)


BASH = shutil.which("bash")
POWERSHELL = shutil.which("powershell.exe") or shutil.which("powershell")
BOX_DIR = os.path.dirname(os.path.abspath(paths.__file__))


def _clean_env(**extra: str) -> dict:
    env = {k: v for k, v in os.environ.items() if k.upper() not in (paths.ENV_HOME, "PROGRAMDATA")}
    env.update(extra)
    return env


@unittest.skipUnless(BASH, "needs bash")
class ShellAgreesTest(unittest.TestCase):
    """_common.sh and box_paths.ps1 resolve the layout the same way paths.py does."""

    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.pd = tmp.name
        self.home = os.path.join(self.pd, "HomeGuard")
        os.makedirs(self.home)

    def mark(self) -> None:
        with open(os.path.join(self.home, paths.MARKER_NAME), "w", encoding="utf-8") as f:
            f.write("{}")

    def bash(self, env: dict) -> dict:
        script = (f'source "{BOX_DIR.replace(os.sep, "/")}/_common.sh"; '
                  'native() { if command -v cygpath >/dev/null; then cygpath -w "$1"; else printf "%s" "$1"; fi; }; '
                  'printf "%s|%s|%s|%s" "$HG_LAYOUT" "$HOMEGUARD_HOME" "$(native "$LOG_DIR")" "$(native "$CAMERAS_YAML")"')
        out = subprocess.run([BASH, "-c", script], env=env, capture_output=True, text=True, timeout=60)
        self.assertEqual(out.returncode, 0, out.stderr)
        layout, exported, logs, cameras = out.stdout.strip().split("|")
        return {"layout": layout, "exported": exported, "logs": logs, "cameras": cameras}

    def assertSamePath(self, got: str, want: str) -> None:
        self.assertEqual(os.path.normcase(os.path.abspath(got)), os.path.normcase(os.path.abspath(want)))

    def test_bash_with_the_marker_is_home_and_exports_it(self) -> None:
        self.mark()
        got = self.bash(_clean_env(ProgramData=self.pd))
        expected = paths.resolve({"ProgramData": self.pd})
        self.assertEqual(got["layout"], expected.mode)
        self.assertSamePath(got["exported"], expected.home)
        self.assertSamePath(got["logs"], expected.logs_dir())
        self.assertSamePath(got["cameras"], expected.cameras_yaml())
        # What it exports makes Python agree, even without ProgramData.
        self.assertEqual(paths.resolve({paths.ENV_HOME: got["exported"]}).home, os.path.abspath(expected.home))

    def test_bash_without_the_marker_is_legacy(self) -> None:
        got = self.bash(_clean_env(ProgramData=self.pd))
        expected = paths.resolve({"ProgramData": self.pd})
        self.assertEqual((got["layout"], got["exported"]), (paths.LEGACY, paths.LEGACY))
        self.assertSamePath(got["logs"], expected.logs_dir())
        self.assertSamePath(got["cameras"], expected.cameras_yaml())

    def test_bash_follows_homeguard_home(self) -> None:
        self.mark()
        self.assertEqual(self.bash(_clean_env(ProgramData=self.pd, HOMEGUARD_HOME="legacy"))["layout"], paths.LEGACY)
        other = os.path.join(self.pd, "other")
        got = self.bash(_clean_env(ProgramData=self.pd, HOMEGUARD_HOME=other))
        self.assertEqual(got["layout"], paths.HOME)
        self.assertSamePath(got["logs"], os.path.join(other, "logs"))

    @unittest.skipUnless(POWERSHELL, "needs PowerShell")
    def test_powershell_agrees(self) -> None:
        script = (f". '{os.path.join(BOX_DIR, 'box_paths.ps1')}'; $p = Get-HomeGuardPaths; "
                  "Write-Output \"$($p.Mode)|$($p.LogsDir)|$($p.CamerasYaml)|$($p.NetworkJson)|$($p.SecretsEnv)|$($p.BoxYaml)\"")
        for marked in (False, True):
            if marked:
                self.mark()
            out = subprocess.run([POWERSHELL, "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", script],
                                 env=_clean_env(ProgramData=self.pd), capture_output=True, text=True, timeout=60)
            self.assertEqual(out.returncode, 0, out.stderr)
            mode, logs, cameras, network, secrets, box_yaml = out.stdout.strip().split("|")
            expected = paths.resolve({"ProgramData": self.pd})
            self.assertEqual(mode, expected.mode)
            for got, want in ((logs, expected.logs_dir()), (cameras, expected.cameras_yaml()),
                              (network, expected.network_json()), (secrets, expected.secrets_env()),
                              (box_yaml, expected.box_yaml())):
                self.assertEqual(os.path.normcase(os.path.abspath(got)), os.path.normcase(want))


if __name__ == "__main__":
    unittest.main()
