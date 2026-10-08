"""Where the box keeps its config, secrets, data, logs and models. The one place that knows.

Two layouts:

``home`` (a box set up or migrated since 2026-10): everything that belongs to this box, in one folder,
``HOME = %ProgramData%\\HomeGuard``::

    config\\   box.yaml, cameras.yaml, camera_alerts.yaml, camera_aliases.yaml, zones.yaml, scene_maps.yaml,
              registration.json, registration.published, network.json
    secrets\\  api_key.env (Administrators, SYSTEM and the box's task user only)
    data\\     live\\ (the collector's clips), outbox\\ (waiting for S3), production\\ (alert clips),
              archive\\ (answered alerts, 14 days), state\\ (the assistant's and the house's memory),
              scene_interview\\
    logs\\     runner, collector, upload and heartbeat logs, pid and flag files, ai_status.json,
              alert_mute.json, telegram_*.json(l), chat_images\\, preview\\, app_snapshots\\
    models\\   yolo11s.pt and its OpenVINO copy, FastSAM-s.pt

``legacy``: the places inside the code folder used until then (box.yaml in ``box\\``, cameras.yaml in
``data_collection\\``, ``dataset_multi\\``, ``logs\\`` and api_key.env at the repo root, ...). A box that has
not migrated, and every developer laptop, run like this unchanged.

Which one: ``$HOMEGUARD_HOME`` (a folder, or ``legacy``), else ``%ProgramData%\\HomeGuard`` when its marker
``layout.json`` is there (written by ``migrate-layout`` and by setup_box.ps1 on a new box; ``--rollback``
removes it), else legacy. _common.sh and box_paths.ps1 use the same rule, and _common.sh exports
HOMEGUARD_HOME, so the runners and the Python programs they start agree.

The code folder is not here: it stays where it is (C:\\home_guard on a box).

Pure stdlib and cheap to import: data_collection imports it when it runs as a script too.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

ENV_HOME = "HOMEGUARD_HOME"
HOME = "home"
LEGACY = "legacy"
MARKER_NAME = "layout.json"
LAYOUT_VERSION = 1
DEFAULT_YOLO = "yolo11s.pt"

_BOX_DIR = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.path.abspath(os.path.join(_BOX_DIR, "..", ".."))   # the repo root: C:\home_guard on a box

# The assistant keeps hidden state in the legacy production folder (``production_multi/.registry``,
# ``.conversations`` ...). In the home layout it all goes to data\state.
REGISTRY_NAME = ".registry"


def _env_value(environ: Mapping[str, str], name: str) -> Optional[str]:
    """A variable from *environ*, by any spelling (Windows names are not case-sensitive; a plain dict is)."""
    if name in environ:
        return environ[name]
    wanted = name.upper()
    for key, value in environ.items():
        if key.upper() == wanted:
            return value
    return None


def default_home(environ: Optional[Mapping[str, str]] = None) -> Optional[str]:
    """``%ProgramData%\\HomeGuard``; None where there is no ProgramData (not Windows)."""
    env = os.environ if environ is None else environ
    base = _env_value(env, "ProgramData")
    if not base and environ is None and os.name == "nt":
        base = r"C:\ProgramData"
    return os.path.join(base, "HomeGuard") if base else None


@dataclass(frozen=True)
class Layout:
    """The places of one layout. *home* is None in legacy mode."""

    mode: str
    home: Optional[str] = None
    code_dir: str = CODE_DIR
    reason: str = ""

    @property
    def is_home(self) -> bool:
        return self.mode == HOME

    def _home(self, *parts: str) -> str:
        assert self.home is not None
        return os.path.join(self.home, *parts)

    def _code(self, *parts: str) -> str:
        return os.path.join(self.code_dir, *parts)

    # -- config -------------------------------------------------------------------------------------
    def config_dir(self) -> str:
        """The per-camera config (cameras.yaml and the files next to it). Legacy: data_collection\\."""
        return self._home("config") if self.is_home else self._code("home_guard_project", "data_collection")

    def _box_config(self, name: str) -> str:
        return self._home("config", name) if self.is_home else self._code("home_guard_project", "box", name)

    def box_yaml(self) -> str:
        return self._box_config("box.yaml")

    def registration_json(self) -> str:
        return self._box_config("registration.json")

    def registration_published(self) -> str:
        return self._home("config", "registration.published") if self.is_home \
            else os.path.join(self.logs_dir(), "registration.published")

    def network_json(self) -> str:
        return self._box_config("network.json")

    def cameras_yaml(self) -> str:
        return os.path.join(self.config_dir(), "cameras.yaml")

    def camera_alerts_yaml(self) -> str:
        return os.path.join(self.config_dir(), "camera_alerts.yaml")

    def camera_aliases_yaml(self) -> str:
        return os.path.join(self.config_dir(), "camera_aliases.yaml")

    def zones_yaml(self) -> str:
        return os.path.join(self.config_dir(), "zones.yaml")

    def scene_maps_yaml(self) -> str:
        return os.path.join(self.config_dir(), "scene_maps.yaml")

    # -- secrets ------------------------------------------------------------------------------------
    def secrets_dir(self) -> str:
        return self._home("secrets") if self.is_home else self.code_dir

    def secrets_env(self) -> str:
        """api_key.env: the model, Telegram and Twilio keys (read with load_dotenv)."""
        return os.path.join(self.secrets_dir(), "api_key.env")

    # -- data ---------------------------------------------------------------------------------------
    def data_dir(self) -> str:
        return self._home("data") if self.is_home else self.code_dir

    def live_dir(self) -> str:
        """The collector's clips (data collection mode)."""
        return self._home("data", "live") if self.is_home else self._code("dataset_multi")

    def outbox_dir(self) -> str:
        """Finished collector clips waiting for S3, one folder per site."""
        return self._home("data", "outbox") if self.is_home else self._code("dataset_outbox")

    def production_dir(self) -> str:
        """Inference mode's alert clips."""
        return self._home("data", "production") if self.is_home else self._code("production_multi")

    def archive_dir(self) -> str:
        """Answered alerts, one folder per site, kept PRODUCTION_RETENTION_DAYS."""
        return self._home("data", "archive") if self.is_home else self._code("production_archive")

    def state_dir(self) -> str:
        """house_state.jsonl, cases.jsonl, quiet_since.json, vision_budget.json, sees.json."""
        return self._home("data", "state") if self.is_home else os.path.join(self.production_dir(), REGISTRY_NAME)

    def assistant_dir(self) -> str:
        """Where the assistant keeps .conversations, .receipts, .desc, .live and .alert_embeddings.json."""
        return self._home("data", "state") if self.is_home else self.production_dir()

    def scene_interview_dir(self) -> str:
        return self._home("data", "scene_interview") if self.is_home else self._code("scene_interview")

    # -- logs and models ----------------------------------------------------------------------------
    def logs_dir(self) -> str:
        return self._home("logs") if self.is_home else self._code("logs")

    def alive_file(self) -> str:
        """Touched by run_collector.sh every 30 s; read by the heartbeat."""
        return os.path.join(self.logs_dir(), "collector.alive")

    def models_dir(self) -> str:
        return self._home("models") if self.is_home else self.code_dir

    def yolo_model(self, name: str = DEFAULT_YOLO) -> str:
        return os.path.join(self.models_dir(), name)

    def resolve_model(self, name: str) -> str:
        """A model given by bare file name ("yolo11s.pt") lives in models_dir; a path stays as it is.

        Ultralytics downloads a known model to the path it is given, so a missing file still works.
        """
        if not name or os.path.dirname(name) or "://" in name:
            return name
        return self.yolo_model(name)

    def marker(self) -> Optional[str]:
        return self._home(MARKER_NAME) if self.is_home else None

    def state_paths_for(self, live_dir: str) -> Tuple[str, str]:
        """``(state_dir, assistant_dir)`` for the alert folder *live_dir*.

        The production folder's state has its own place (data\\state); any other folder (a test's, an
        eval's) keeps it inside, the old way: ``<live_dir>/.registry`` and ``<live_dir>``.
        """
        if os.path.normcase(os.path.abspath(live_dir)) == os.path.normcase(os.path.abspath(self.production_dir())):
            return self.state_dir(), self.assistant_dir()
        return os.path.join(live_dir, REGISTRY_NAME), live_dir

    def as_dict(self) -> Dict[str, Any]:
        """Every place, for ``python -m home_guard_project.box paths --json`` (the laptop's tools ask this)."""
        return {
            "layout_version": LAYOUT_VERSION, "mode": self.mode, "home": self.home, "code_dir": self.code_dir,
            "reason": self.reason,
            **{name: getattr(self, name)() for name in PATH_NAMES},
        }


# The places as_dict() and the CLI list (each a Layout method without arguments).
PATH_NAMES: Tuple[str, ...] = (
    "config_dir", "box_yaml", "cameras_yaml", "camera_alerts_yaml", "camera_aliases_yaml", "zones_yaml",
    "scene_maps_yaml", "registration_json", "registration_published", "network_json",
    "secrets_dir", "secrets_env",
    "data_dir", "live_dir", "outbox_dir", "production_dir", "archive_dir", "state_dir", "assistant_dir",
    "scene_interview_dir",
    "logs_dir", "alive_file", "models_dir", "yolo_model",
)


def legacy(code_dir: str = CODE_DIR, reason: str = "") -> Layout:
    return Layout(LEGACY, None, code_dir, reason)


def home_layout(home: str, code_dir: str = CODE_DIR, reason: str = "") -> Layout:
    return Layout(HOME, os.path.abspath(home), code_dir, reason)


def resolve(environ: Optional[Mapping[str, str]] = None, code_dir: str = CODE_DIR) -> Layout:
    """The layout this machine uses: $HOMEGUARD_HOME, else ProgramData's marker, else legacy."""
    env = os.environ if environ is None else environ
    value = (_env_value(env, ENV_HOME) or "").strip()
    if value.lower() == LEGACY:
        return legacy(code_dir, f"{ENV_HOME}=legacy")
    if value:
        return home_layout(value, code_dir, f"{ENV_HOME} is set")
    home = default_home(environ)
    if home and os.path.isfile(os.path.join(home, MARKER_NAME)):
        return home_layout(home, code_dir, f"{MARKER_NAME} found in {home}")
    return legacy(code_dir, "not migrated (no layout.json in ProgramData)")


_current: Optional[Layout] = None


def current() -> Layout:
    """The layout of this process, resolved once (``reset()`` resolves it again)."""
    global _current
    if _current is None:
        _current = resolve()
    return _current


def reset(layout: Optional[Layout] = None) -> None:
    """Forget the resolved layout, or use *layout* from now on (tests)."""
    global _current
    _current = layout


def mode() -> str:
    return current().mode


def home() -> Optional[str]:
    return current().home


def config_dir() -> str:
    return current().config_dir()


def box_yaml() -> str:
    return current().box_yaml()


def registration_json() -> str:
    return current().registration_json()


def registration_published() -> str:
    return current().registration_published()


def network_json() -> str:
    return current().network_json()


def cameras_yaml() -> str:
    return current().cameras_yaml()


def camera_alerts_yaml() -> str:
    return current().camera_alerts_yaml()


def camera_aliases_yaml() -> str:
    return current().camera_aliases_yaml()


def zones_yaml() -> str:
    return current().zones_yaml()


def scene_maps_yaml() -> str:
    return current().scene_maps_yaml()


def secrets_dir() -> str:
    return current().secrets_dir()


def secrets_env() -> str:
    return current().secrets_env()


def data_dir() -> str:
    return current().data_dir()


def live_dir() -> str:
    return current().live_dir()


def outbox_dir() -> str:
    return current().outbox_dir()


def production_dir() -> str:
    return current().production_dir()


def archive_dir() -> str:
    return current().archive_dir()


def state_dir() -> str:
    return current().state_dir()


def assistant_dir() -> str:
    return current().assistant_dir()


def scene_interview_dir() -> str:
    return current().scene_interview_dir()


def logs_dir() -> str:
    return current().logs_dir()


def alive_file() -> str:
    return current().alive_file()


def models_dir() -> str:
    return current().models_dir()


def yolo_model(name: str = DEFAULT_YOLO) -> str:
    return current().yolo_model(name)


def resolve_model(name: str) -> str:
    return current().resolve_model(name)


def state_paths_for(live_dir: str) -> Tuple[str, str]:
    return current().state_paths_for(live_dir)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """``python -m home_guard_project.box paths [--json | --get NAME]``: where this box keeps things."""
    import argparse  # noqa: PLC0415

    p = argparse.ArgumentParser(prog="box paths", description="Where this box keeps its config, data and logs.")
    p.add_argument("--json", action="store_true", help="every place as JSON (read by the laptop's tools)")
    p.add_argument("--get", choices=PATH_NAMES + ("mode", "home", "code_dir"), help="print one place")
    args = p.parse_args(list(sys.argv[1:] if argv is None else argv))
    info = current().as_dict()
    if args.get:
        print(info[args.get] or "")
    elif args.json:
        print(json.dumps(info, indent=2))
    else:
        print(f"layout: {info['mode']} ({info['reason']})")
        for name in PATH_NAMES:
            print(f"  {name:<24} {info[name]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
