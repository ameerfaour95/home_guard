"""The scene map editor's transport: the box's scene_interview commands, on this box or over ssh, and a demo.

Every command is ``python -m home_guard_project.box.scene_interview <command> --camera NAME --json ...``:

- ``propose --embed`` numbers the camera's picture (FastSAM on the box) and answers with the clean picture, the
  numbered one, every region's polygon and the camera's map now, all in one line of JSON;
- ``confirm --map-b64 <map>`` saves the map built in the app as the camera's whole truth, and restarts the running
  mode once when its frame mask changed (``--no-restart`` leaves that to the caller);
- ``restore --check`` says whether the camera has a previous map (the one its last save replaced), ``restore``
  brings it back;
- ``names`` gives every camera's name as the owner reads it.

The box is the only source of camera names (the family sets them over Telegram, they live on the box): every
answer carries ``display_name`` / ``display_name_en`` and the app never shows a camera id.

Nothing is ever written on the laptop: from the laptop these run over ssh on the box. The map travels zlib-compressed
as base64, because the remote command line carries no spaces or quotes in a value. Only the injected runner
launches processes.
"""
import base64
import json
import os
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from .box_layout import box_command
from .scene_model import Region, encode_map, shrink_map

MODULE = "home_guard_project.box.scene_interview"
REMOTE_LIMIT = 8000            # cmd.exe runs the remote command: its line stops at 8191 characters
MAP_LIMIT = 7500               # the --map-b64 value, leaving room for the rest of the line
SHRINK_STEPS = (.002, .004, .008, .015)       # outline tolerances tried, in picture fractions, before refusing
_CAMERA = re.compile(r"[a-z0-9_]+")


class SceneError(RuntimeError):
    """A plain message for the owner (``kind`` picks the words); ``detail`` is what the box said, if anything."""
    def __init__(self, kind, detail=""):
        super().__init__(kind)
        self.kind = kind
        self.detail = str(detail or "")


def names_from(data):
    """``{"he": ..., "en": ...}``: the box's names for the camera in this answer ("" where it gave none)."""
    return {"he": str(data.get("display_name") or ""), "en": str(data.get("display_name_en") or "")}


@dataclass(frozen=True)
class Proposal:
    camera: str
    method: str                                  # sam | grid
    picture: bytes                               # the clean JPEG: the app draws its own overlay
    regions: tuple = ()
    current: dict = field(default_factory=dict)  # the camera's map now (SceneMap.to_dict): the whole truth
    names: dict = field(default_factory=dict)    # the box's names for it: {"he", "en"}

    @property
    def grid(self):
        return self.method == "grid"


@dataclass(frozen=True)
class Saved:
    camera: str
    map: dict
    restart_needed: bool = False
    restored: bool = False         # the previous map came back (restore), not a new one saved
    names: dict = field(default_factory=dict)


@dataclass(frozen=True)
class Previous:
    exists: bool
    saved_at: float = None         # when the map it holds was replaced (epoch seconds)


def camera_name(name):
    if not _CAMERA.fullmatch(str(name or "")):
        raise ValueError("Invalid camera name")
    return name


def map_argument(scene):
    """The map for ``--map-b64``, its outlines simplified step by step until it fits the command line;
    SceneError ``too_detailed`` when even the plainest outlines do not."""
    encoded = encode_map(scene)
    for tolerance in SHRINK_STEPS:
        if len(encoded) <= MAP_LIMIT:
            return encoded
        encoded = encode_map(shrink_map(scene, tolerance))
    if len(encoded) > MAP_LIMIT:
        raise SceneError("too_detailed")
    return encoded


def operation(command, camera=None, grid=False, scene=None, restart=True, check=False, lang="he"):
    """The scene_interview arguments for one step (no interpreter, no ssh)."""
    if command == "names":
        return ["names", "--json", "--lang", "en" if lang == "en" else "he"]
    args = [command, "--camera", camera_name(camera), "--json"]
    if command == "propose":
        return args + ["--embed"] + (["--grid"] if grid else [])
    if command == "confirm":
        if not isinstance(scene, dict):
            raise ValueError("A map is needed to save")
        return args + ["--map-b64", map_argument(scene)] + ([] if restart else ["--no-restart"])
    if command == "restore":
        return args + (["--check"] if check else [] if restart else ["--no-restart"])
    raise ValueError("Unknown scene map step")


def result_json(output):
    """The last JSON object the command printed. Log lines (FastSAM loading, decoder notes) can come before it."""
    decoder = json.JSONDecoder()
    found = None
    for match in re.finditer(r"(?m)^\{", output or ""):
        try:
            data, _ = decoder.raw_decode(output[match.start():])
        except ValueError:
            continue
        if isinstance(data, dict):
            found = data
    if found is None:
        raise SceneError("no_answer", (output or "")[-400:])
    if "error" in found:
        raise SceneError("box_refused", found["error"])
    return found


def _bytes(value):
    try:
        return base64.b64decode(str(value or ""), validate=True)
    except ValueError as exc:
        raise SceneError("bad_picture") from exc


def proposal_from(camera, data):
    if data.get("camera") != camera or not isinstance(data.get("regions"), list):
        raise SceneError("bad_answer")
    try:
        regions = tuple(Region(int(r["number"]), tuple((float(x), float(y)) for x, y in r["points"]),
                               float(r.get("area") or 0.)) for r in data["regions"])
    except (KeyError, TypeError, ValueError) as exc:
        raise SceneError("bad_answer") from exc
    picture = _bytes(data.get("picture_b64"))
    if not picture:
        raise SceneError("bad_picture")
    current = data.get("current_map") if isinstance(data.get("current_map"), dict) else {}
    return Proposal(camera, "grid" if data.get("method") == "grid" else "sam", picture, regions, current,
                    names_from(data))


def saved_from(camera, data, restored=False):
    if data.get("camera") != camera or not isinstance(data.get("map"), dict):
        raise SceneError("bad_answer")
    return Saved(camera, data["map"], data.get("restart_needed") is True, restored, names_from(data))


def previous_from(camera, data):
    if data.get("camera") != camera or not isinstance(data.get("has_previous"), bool):
        raise SceneError("bad_answer")
    saved_at = data.get("saved_at")
    return Previous(data["has_previous"], float(saved_at) if isinstance(saved_at, (int, float)) else None)


class SceneBackend:
    """On this box (no target) or on the box at ``user@address`` over the app's ssh key."""
    def __init__(self, target=None, key=None, runner=None, python=None):
        self.target = target
        self.key = Path(key) if key else Path.home() / ".ssh" / "homeguard_box"
        self.runner = runner
        self.own_runner = runner is None      # a cancelled runner of ours is dropped; the next step gets a new one
        self.python = python

    def command_line(self, args):
        if self.target:
            remote = box_command(f"-m {MODULE} " + " ".join(args))
            if '"' in remote or "'" in remote:
                raise ValueError("Quotes are not allowed in the remote command")
            if len(remote) > REMOTE_LIMIT:
                raise SceneError("too_detailed")
            return ["ssh.exe", "-i", str(self.key), "-o", "LogLevel=ERROR", self.target, remote]
        from .camera_controls import box_python
        return [self.python or box_python(), "-m", MODULE, *args]

    def run(self, args):
        line = self.command_line(args)
        try:
            if self.target:
                from .remote_cameras import CommandRunner
                self.runner = self.runner or CommandRunner()
                result = self.runner.run(line)
            else:
                env = os.environ.copy()
                env.pop("VIRTUAL_ENV", None)
                result = (self.runner or subprocess.run)(
                    line, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                    errors="replace", timeout=480, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        except (OSError, RuntimeError, subprocess.SubprocessError) as exc:
            raise SceneError("unreachable", str(exc)) from exc
        data = result_json(result.stdout)
        if result.returncode:
            raise SceneError("box_refused", data.get("error") or "")
        return data

    def propose(self, camera, grid=False):
        return proposal_from(camera, self.run(operation("propose", camera, grid=grid)))

    def confirm(self, camera, scene, restart=True):
        return saved_from(camera, self.run(operation("confirm", camera, scene=scene, restart=restart)))

    def previous(self, camera):
        return previous_from(camera, self.run(operation("restore", camera, check=True)))

    def names(self, lang="he"):
        """``{camera id: name}`` for every camera of the box; ids are for commands only, never shown."""
        data = self.run(operation("names", lang=lang))
        names = data.get("names")
        if not isinstance(names, dict):
            raise SceneError("bad_answer")
        return {str(k): str(v or "") for k, v in names.items()}

    def restore(self, camera, restart=True):
        return saved_from(camera, self.run(operation("restore", camera, restart=restart)), restored=True)

    def cancel(self):
        """Stop the command in flight (leaving a camera while the box works). The setup step goes on to the next
        camera with the same backend, so a runner of ours is replaced, never left cancelled."""
        runner = self.runner
        if self.own_runner and self.target:
            self.runner = None
        if runner is not None and hasattr(runner, "cancel"):
            runner.cancel()


def scene_backend_for(controls):
    """The backend beside a camera page's controls: the demo, this box, or the remote box."""
    if getattr(getattr(controls, "box", None), "demo", False):
        return DemoSceneBackend()
    if getattr(controls, "remote", False):
        return SceneBackend(controls.target, getattr(controls, "key", None))
    return SceneBackend()


# ----------------------------------------------------------------------------
# Demo: the generated house picture, its places outlined by hand, confirmed by the real engine rules.
# ----------------------------------------------------------------------------
DEMO_REGIONS = (
    ((0., .445), (1., .445), (1., 1.), (0., 1.)),                                                  # the lawn
    ((.195, .299), (.484, .153), (.805, .299), (.758, .299), (.758, .667), (.234, .667), (.234, .299)),  # house
    ((.39, .625), (.5625, .625), (.9375, 1.), (.078, 1.)),                                          # the driveway
    ((.414, .42), (.602, .42), (.602, .614), (.414, .614)),                                         # the car
    ((.055, .264), (.133, .264), (.133, .722), (.055, .722)),                                       # left hedge
    ((.867, .264), (.945, .264), (.945, .722), (.867, .722)),                                       # right hedge
)


DEMO_NAMES = {"front_door": {"he": "דלת הכניסה", "en": "Front door"}, "garden": {"he": "הגינה", "en": "Garden"},
              "driveway": {"he": "החניה", "en": "Driveway"}}


class DemoSceneBackend:
    """No box: the demo picture and regions; ``confirm`` validates and shapes the map as the box would.
    ``fail`` makes that step fail once (``propose`` | ``confirm``). *names*: the box's names per camera
    (``{"he", "en"}``); a camera without one gets none, as from a box that cannot say."""
    def __init__(self, fail=None, method="sam", current=None, names=None):
        self.names_by_camera = DEMO_NAMES if names is None else dict(names)
        self.fail = fail
        self.method = method
        self.current = dict(current or {})          # the first camera's map now; each camera keeps its own after
        self.maps = {}
        self.backups = {}                            # camera -> (the map a save replaced, when)
        self.calls = []

    def _check(self, step, camera):
        self.calls.append(step)
        if self.fail == step:
            self.fail = None
            raise SceneError("box_refused", f"could not get a picture from {camera} right now (is it online?)")

    def picture(self, camera):
        from PySide6.QtCore import QBuffer, QIODevice
        from .demo_media import picture
        buffer = QBuffer(); buffer.open(QIODevice.OpenModeFlag.WriteOnly)
        picture(sum(map(ord, camera)) % 3).save(buffer, "JPG", 90)
        return bytes(buffer.data())

    def regions(self):
        from ..scene_map import polygon_area
        if self.method == "grid":
            cells = [((c / 4, r / 3), ((c + 1) / 4, r / 3), ((c + 1) / 4, (r + 1) / 3), (c / 4, (r + 1) / 3))
                     for r in range(3) for c in range(4)]
            return tuple(Region(i + 1, cell, 1 / 12) for i, cell in enumerate(cells))
        return tuple(Region(i + 1, points, polygon_area(points)) for i, points in enumerate(DEMO_REGIONS))

    def propose(self, camera, grid=False):
        self._check("propose", camera)
        if grid:
            self.method = "grid"
        return Proposal(camera, self.method, self.picture(camera), self.regions(),
                        dict(self.maps.get(camera, self.current)), self.name_of(camera))

    def name_of(self, camera):
        return dict(self.names_by_camera.get(camera) or {"he": "", "en": ""})

    def names(self, lang="he"):
        self._check("names", "")
        return {camera: names.get(lang, "") for camera, names in self.names_by_camera.items()}

    def confirm(self, camera, scene, restart=True):
        self._check("confirm", camera)
        from .. import scene_map as sm
        from ..scene_interview import confirmed_preview
        from .scene_model import restart_expected
        current = self.maps.get(camera, self.current)
        draft = sm.SceneMap.from_dict(camera, scene, current.get("watched") or None)
        saved = confirmed_preview(draft).to_dict()
        self.backups[camera] = (current, __import__("time").time())
        self.maps[camera] = saved
        return Saved(camera, saved, restart_expected(current, scene), names=self.name_of(camera))

    def previous(self, camera):
        self._check("previous", camera)
        backup = self.backups.get(camera)
        return Previous(backup is not None, backup[1] if backup else None)

    def restore(self, camera, restart=True):
        self._check("restore", camera)
        from .scene_model import restart_expected
        if camera not in self.backups:
            raise SceneError("box_refused", "there is no previous map of this camera")
        now = self.maps.get(camera, self.current)
        back, _when = self.backups[camera]
        self.backups[camera] = (now, __import__("time").time())
        self.maps[camera] = back
        return Saved(camera, dict(back), restart_expected(now, back) or bool(back.get("watched")), restored=True,
                     names=self.name_of(camera))

    def cancel(self):
        pass
