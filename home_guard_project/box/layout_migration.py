"""Move a box's config, secrets, data, logs and models out of the code folder, and back (paths.py).

    python -m home_guard_project.box migrate-layout --dry-run      # print the plan; change nothing
    python -m home_guard_project.box migrate-layout                # stop, copy, check, lock down, switch, start
    python -m home_guard_project.box migrate-layout --rollback     # stop, copy changes back, switch back, start
    python -m home_guard_project.box migrate-layout --finalize     # list the old copies; with --yes delete them

It copies and never moves: until ``--finalize`` the old places keep a full copy. The switch is the
marker ``%ProgramData%\\HomeGuard\\layout.json``, written last, after every copy was checked; a failure
before it leaves the box on the old layout and starts it again as it was.

A run:
  1. pauses the upload and heartbeat tasks (waits for a running one to finish), ends the collector task
     and stop_collector.sh, and checks the runner and collector processes are gone;
  2. creates the folders and locks them down (icacls: Administrators, SYSTEM, the collector task's account
     and the auto sign-in account that runs the window);
  3. copies (a file already there with the same size and time is not copied again, so a second run
     resumes), then checks every file's size, and config and secrets byte for byte;
  4. writes migration_manifest.json (every file copied, and the commit the code was at) and then layout.json;
  5. turns the upload and heartbeat tasks back on and starts the collector task.

``--rollback`` stops the box the same way, copies back whatever changed in the new place since (and
removes from the old place the copied files the box has since deleted, such as uploaded clips),
removes layout.json and starts the box on the old layout.

``--finalize`` deletes, from the code folder, the files the manifest copied that are still as they were
copied, then the folders left empty, and known junk (api_key.env.bak-*, an empty production_outbox). It
never deletes a file git tracks or anything under .git, and does nothing without ``--yes``.

The Home Guard window (start.pyw) reads the places when it starts: restart it, or the box, afterwards.
"""

from __future__ import annotations

import argparse
import datetime as dt
import fnmatch
import getpass
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, FrozenSet, List, Optional, Sequence

from . import paths
from .paths import Layout

MANIFEST_NAME = "migration_manifest.json"
COLLECTOR_TASK = "HomeGuard-Collector"
PAUSED_TASKS = ("HomeGuard-Upload", "HomeGuard-Heartbeat")
GIT_BASH = r"C:\Program Files\Git\bin\bash.exe"

# Per-box config files, each moved to its own accessor's place (byte-for-byte checked).
CONFIG_FILES = ("box_yaml", "cameras_yaml", "camera_alerts_yaml", "camera_aliases_yaml", "zones_yaml",
                "scene_maps_yaml", "registration_json", "registration_published", "network_json")
# Whole folders, each to its accessor's place in the other layout.
DATA_DIRS = ("live_dir", "outbox_dir", "archive_dir", "scene_interview_dir")
# Not carried over from logs\: they describe processes of the old layout (registration.published is config).
LOG_SKIP = frozenset({"runner.winpid", "collector.winpid", "collector.restart", "registration.published"})
MODEL_PATTERNS = ("*.pt", "*_openvino_model")
JUNK_PATTERNS = ("api_key.env.bak-*",)
EMPTY_JUNK_DIRS = ("production_outbox",)

WINLOGON_KEY = r"HKLM\SOFTWARE\Microsoft\Windows NT\CurrentVersion\Winlogon"
ADMINISTRATORS = "*S-1-5-32-544"
SYSTEM = "*S-1-5-18"

Runner = Callable[..., Any]   # run(cmd, env=None) -> an object with returncode and stdout


class MigrationError(Exception):
    """The migration cannot go on; the box is left as it was."""


@dataclass(frozen=True)
class Item:
    """One thing to copy: a file, or a folder (minus *skip*, names at its top level)."""

    name: str
    source: str
    dest: str
    is_dir: bool
    exact: bool = False                 # config and secrets: compared byte for byte
    skip: FrozenSet[str] = frozenset()

    def files(self, root: Optional[str] = None) -> Dict[str, int]:
        """``{relative path: size}`` under *root* (default: the source). A file item has the key ""."""
        root = self.source if root is None else root
        if not self.is_dir:
            return {"": os.path.getsize(root)} if os.path.isfile(root) else {}
        out: Dict[str, int] = {}
        for folder, dirs, names in os.walk(root):
            if os.path.normcase(folder) == os.path.normcase(root):
                dirs[:] = [d for d in dirs if d not in self.skip]
                names = [n for n in names if n not in self.skip]
            for n in names:
                full = os.path.join(folder, n)
                try:
                    out[os.path.relpath(full, root).replace(os.sep, "/")] = os.path.getsize(full)
                except OSError:
                    pass
        return out


def _join(root: str, rel: str) -> str:
    return os.path.join(root, *rel.split("/")) if rel else root


def _entries(folder: str) -> List[str]:
    return sorted(os.listdir(folder)) if os.path.isdir(folder) else []


def plan(src: Layout, dst: Layout) -> List[Item]:
    """What to copy from layout *src* to layout *dst* (either direction); only what exists in *src*."""
    items: List[Item] = []

    def add(name: str, source: str, dest: str, **kw: Any) -> None:
        if os.path.isdir(source):
            items.append(Item(name, source, dest, True, **kw))
        elif os.path.isfile(source):
            items.append(Item(name, source, dest, False, **kw))

    for accessor in CONFIG_FILES:
        add(os.path.basename(getattr(src, accessor)()), getattr(src, accessor)(), getattr(dst, accessor)(), exact=True)
    add("api_key.env", src.secrets_env(), dst.secrets_env(), exact=True)
    for accessor in DATA_DIRS:
        add(accessor, getattr(src, accessor)(), getattr(dst, accessor)())
    # Alert clips; the hidden state inside the old production folder is copied on its own below.
    hidden = frozenset(e for e in _entries(src.production_dir()) if e.startswith("."))
    add("production_dir", src.production_dir(), dst.production_dir(), skip=hidden)
    shared = os.path.normcase(src.state_dir()) == os.path.normcase(src.assistant_dir())
    for entry in _entries(src.state_dir()):
        if not (shared and entry.startswith(".")):
            add(f"state/{entry}", os.path.join(src.state_dir(), entry), os.path.join(dst.state_dir(), entry))
    for entry in _entries(src.assistant_dir()):
        if entry.startswith(".") and entry != paths.REGISTRY_NAME:
            add(f"assistant/{entry}", os.path.join(src.assistant_dir(), entry),
                os.path.join(dst.assistant_dir(), entry))
    add("logs_dir", src.logs_dir(), dst.logs_dir(), skip=LOG_SKIP)
    for entry in _entries(src.models_dir()):
        if any(fnmatch.fnmatch(entry, p) for p in MODEL_PATTERNS):
            add(f"models/{entry}", os.path.join(src.models_dir(), entry), os.path.join(dst.models_dir(), entry))
    return items


def _same(src: str, dst: str) -> bool:
    """Already copied: the same size and modification time (copy2 keeps the time)."""
    try:
        a, b = os.stat(src), os.stat(dst)
    except OSError:
        return False
    return a.st_size == b.st_size and int(a.st_mtime) == int(b.st_mtime)


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def copy_item(item: Item, files: Optional[Dict[str, int]] = None) -> Dict[str, int]:
    """Copy *item*'s files to its destination; returns ``{relative path: size}`` of what it covers."""
    files = item.files() if files is None else files
    for rel in files:
        src, dst = _join(item.source, rel), _join(item.dest, rel)
        if _same(src, dst):
            continue
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(src, dst)
    if item.is_dir:
        os.makedirs(item.dest, exist_ok=True)   # an empty folder is still carried over
    return files


def verify_item(item: Item, files: Dict[str, int]) -> List[str]:
    """What is wrong with the copy of *item*: missing files, other sizes, other bytes."""
    problems = []
    for rel, size in files.items():
        dst = _join(item.dest, rel)
        if not os.path.isfile(dst):
            problems.append(f"{item.name}: {rel or os.path.basename(dst)} missing in {item.dest}")
        elif os.path.getsize(dst) != size:
            problems.append(f"{item.name}: {rel or os.path.basename(dst)} has {os.path.getsize(dst)} bytes, not {size}")
        elif item.exact and _sha256(dst) != _sha256(_join(item.source, rel)):
            problems.append(f"{item.name}: {rel or os.path.basename(dst)} differs from the original")
    return problems


# -- the box's tasks and processes -------------------------------------------------------------------
def default_run(cmd: Sequence[str], env: Optional[Dict[str, str]] = None) -> subprocess.CompletedProcess:
    return subprocess.run(list(cmd), capture_output=True, text=True, env=env, check=False, stdin=subprocess.DEVNULL)


def _posix(path: str) -> str:
    """C:\\home_guard\\x -> /c/home_guard/x, for Git Bash."""
    if len(path) > 1 and path[1] == ":":
        return "/" + path[0].lower() + path[2:].replace("\\", "/")
    return path.replace("\\", "/")


def task_user(run: Runner) -> str:
    """The account the collector task runs as (icacls form); the current user when there is no task."""
    out = run(["schtasks", "/Query", "/TN", COLLECTOR_TASK, "/XML"])
    match = re.search(r"<UserId>([^<]+)</UserId>", getattr(out, "stdout", "") or "") \
        if getattr(out, "returncode", 1) == 0 else None
    user = match.group(1).strip() if match else getpass.getuser()
    return "*" + user if user.upper().startswith("S-1-") else user


def window_user(run: Runner) -> Optional[str]:
    """The account Windows signs in by itself (auto sign-in): it runs the Home Guard window. None if off."""
    out = run(["reg", "query", WINLOGON_KEY, "/v", "DefaultUserName"])
    match = re.search(r"DefaultUserName\s+REG_SZ\s+(\S.*)", getattr(out, "stdout", "") or "") \
        if getattr(out, "returncode", 1) == 0 else None
    return match.group(1).strip() if match else None


def box_users(run: Runner) -> List[str]:
    """Who works with the box's files: the collector task's account, and the window's when it is another."""
    users = [task_user(run)]
    window = window_user(run)
    if window and window.lower() not in {u.lower().rsplit("\\", 1)[-1] for u in users}:
        users.append(window)
    return users


def acl_commands(home: str, users: Sequence[str]) -> List[List[str]]:
    """icacls lines: HOME and, again on its own, secrets\\ to Administrators, SYSTEM and *users* only."""
    grants = ["/grant:r", f"{ADMINISTRATORS}:(OI)(CI)F", f"{SYSTEM}:(OI)(CI)F", *(f"{u}:(OI)(CI)M" for u in users)]
    return [["icacls", home, "/inheritance:r", *grants],
            ["icacls", os.path.join(home, "secrets"), "/inheritance:r", *grants]]


def _pids(logs_dir: str) -> List[str]:
    out = []
    for name in ("runner.winpid", "collector.winpid"):
        try:
            with open(os.path.join(logs_dir, name), encoding="ascii", errors="replace") as f:
                pid = f.read().strip()
        except OSError:
            continue
        if pid.isdigit():
            out.append(pid)
    return out


def _alive(run: Runner, pid: str) -> bool:
    out = run(["tasklist", "/FI", f"PID eq {pid}", "/NH"])
    return bool(re.search(rf"\b{pid}\b", getattr(out, "stdout", "") or ""))


def _task_running(run: Runner, task: str) -> bool:
    out = run(["schtasks", "/Query", "/TN", task, "/FO", "CSV", "/NH"])
    return getattr(out, "returncode", 1) == 0 and "Running" in (getattr(out, "stdout", "") or "")


@dataclass
class Box:
    """The box's tasks and processes, as one migration step sees them (injected in tests)."""

    code_dir: str
    run: Runner = default_run
    sleep: Callable[[float], None] = time.sleep
    wait_sec: float = 1800.0
    say: Callable[[str], None] = print
    paused: List[str] = field(default_factory=list)

    def stop(self, layout: Layout) -> None:
        """Pause the upload and heartbeat, stop the collector of *layout*, and check it is gone."""
        for task in PAUSED_TASKS:
            if self.run(["schtasks", "/Change", "/TN", task, "/DISABLE"]).returncode == 0:
                self.paused.append(task)
        waited = 0.0
        while any(_task_running(self.run, t) for t in PAUSED_TASKS):
            if waited >= self.wait_sec:
                raise MigrationError("an upload or heartbeat is still running; try again later")
            self.say("waiting for a running upload or heartbeat to finish...")
            self.sleep(10)
            waited += 10
        pids = _pids(layout.logs_dir())
        self.run(["schtasks", "/End", "/TN", COLLECTOR_TASK])
        env = dict(os.environ, **{paths.ENV_HOME: layout.home or paths.LEGACY})
        script = os.path.join(self.code_dir, "home_guard_project", "box", "stop_collector.sh")
        self.run([GIT_BASH, "-lc", _posix(script)], env=env)
        for _ in range(10):
            left = [pid for pid in pids if _alive(self.run, pid)]
            if not left:
                return
            self.sleep(1)
        raise MigrationError(f"the box's program is still running (PID {', '.join(left)}); "
                             "stop it (schtasks /End, stop_collector.sh) and try again")

    def start(self) -> None:
        """Turn the paused tasks back on and start the collector task."""
        for task in self.paused:
            self.run(["schtasks", "/Change", "/TN", task, "/ENABLE"])
        self.paused.clear()
        if self.run(["schtasks", "/Run", "/TN", COLLECTOR_TASK]).returncode != 0:
            self.say(f"warning: could not start {COLLECTOR_TASK}; run: schtasks /Run /TN {COLLECTOR_TASK}")


# -- the steps ---------------------------------------------------------------------------------------
def _layouts(code_dir: str, home: str) -> tuple:
    return paths.legacy(code_dir), paths.home_layout(home, code_dir)


def _size(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} GB"


def describe(items: Sequence[Item], listed: Sequence[Dict[str, int]]) -> List[str]:
    lines = []
    for item, files in zip(items, listed):
        lines.append(f"  {item.name:<32} {len(files):>7} file(s) {_size(sum(files.values())):>10}   "
                     f"{item.source}  ->  {item.dest}")
    return lines


def migrate(code_dir: str, home: str, box: Box, dry_run: bool = False, set_acl: bool = True) -> int:
    old, new = _layouts(code_dir, home)
    marker = os.path.join(home, paths.MARKER_NAME)
    if os.path.isfile(marker):
        box.say(f"Already migrated: {marker} exists. Nothing to do (--rollback goes back).")
        return 0
    if os.environ.get(paths.ENV_HOME):
        box.say(f"warning: {paths.ENV_HOME}={os.environ[paths.ENV_HOME]} is set here; the box follows it, not the marker")
    items = plan(old, new)
    listed = [item.files() for item in items]
    total = sum(sum(files.values()) for files in listed)
    commit = code_commit(code_dir, box.run)
    box.say(f"Code: {code_dir} at {commit or 'an unknown commit (not a git checkout?)'}")
    box.say(f"Copy {len(items)} item(s), {_size(total)}, from {code_dir} into {home}:")
    for line in describe(items, listed):
        box.say(line)
    if dry_run:
        box.say("Dry run: nothing was changed.")
        return 0
    os.makedirs(home, exist_ok=True)
    free = shutil.disk_usage(home).free
    if free < total * 1.1 + 1e9:
        raise MigrationError(f"not enough free disk: {_size(free)} free, {_size(total)} to copy (plus 1 GB)")

    switched = False
    try:
        box.say("Stopping the box...")
        box.stop(old)
        # The folders are locked down before anything is copied in (the copies inherit it).
        users = box_users(box.run) if set_acl else []
        commands = acl_commands(home, users) if set_acl else []
        if commands:
            _icacls(box, commands[0])
        for folder in ("config", "secrets", "data", "logs", "models"):
            os.makedirs(os.path.join(home, folder), exist_ok=True)
        if commands:
            _icacls(box, commands[1])
            box.say(f"Locked down {home} (Administrators, SYSTEM, {', '.join(users)}).")
        # Planned again now the box is stopped: it may have saved or uploaded clips since the first look.
        items = plan(old, new)
        copied = [(item, copy_item(item)) for item in items]
        problems = [p for item, files in copied for p in verify_item(item, files)]
        if problems:
            raise MigrationError("the copy does not match:\n  " + "\n  ".join(problems[:20]))
        box.say(f"Copied and checked {sum(len(f) for _, f in copied)} file(s).")
        manifest = {
            "layout_version": paths.LAYOUT_VERSION, "created": dt.datetime.now().isoformat(timespec="seconds"),
            "code_dir": code_dir, "code_commit": commit, "home": home,
            "items": [{"name": i.name, "source": i.source, "dest": i.dest, "is_dir": i.is_dir,
                       "skip": sorted(i.skip), "files": files} for i, files in copied],
        }
        _write_json(os.path.join(home, MANIFEST_NAME), manifest)
        _write_json(marker, {"layout_version": paths.LAYOUT_VERSION, "migrated_at": manifest["created"],
                             "from": code_dir, "manifest": MANIFEST_NAME})
        switched = True
        box.say(f"Switched: the box now keeps its files in {home}.")
    finally:
        box.say("Starting the box" + ("." if switched else " again on the old layout."))
        box.start()
    box.say("Restart the Home Guard window (or the box) so it reads the new places too. "
            "The old copies stay until: migrate-layout --finalize")
    return 0


def _icacls(box: Box, cmd: List[str]) -> None:
    out = box.run(cmd)
    if out.returncode != 0:
        raise MigrationError(f"could not set the folder permissions on {cmd[1]}: "
                             f"{(getattr(out, 'stdout', '') or '').strip()} - run as administrator")


def rollback(code_dir: str, home: str, box: Box) -> int:
    old, new = _layouts(code_dir, home)
    marker = os.path.join(home, paths.MARKER_NAME)
    if not os.path.isfile(marker):
        box.say(f"Not migrated ({marker} is missing): nothing to roll back.")
        return 0
    switched = False
    try:
        box.say("Stopping the box...")
        box.stop(new)
        back = 0
        for item in plan(new, old):
            back += sum(1 for rel in item.files() if not _same(_join(item.source, rel), _join(item.dest, rel)))
            copy_item(item)
        removed = 0
        for entry in _load_manifest(home).get("items", []):
            for rel, size in (entry.get("files") or {}).items():
                src, dst = _join(entry["source"], rel), _join(entry["dest"], rel)
                if not os.path.exists(dst) and os.path.isfile(src) and os.path.getsize(src) == size:
                    os.remove(src)       # the box deleted it since (an uploaded clip, an expired answer)
                    removed += 1
        box.say(f"Copied back {back} changed file(s); removed {removed} the box had deleted since.")
        os.remove(marker)
        switched = True
        box.say(f"Switched back: the box keeps its files in {code_dir} again ({home} is left as it is).")
    finally:
        box.say("Starting the box.")
        box.start()
    box.say("Restart the Home Guard window (or the box) so it reads the old places too.")
    return 0


def code_commit(code_dir: str, run: Runner) -> Optional[str]:
    """The commit the code folder is at (a detached HEAD is fine); None when git cannot say."""
    try:
        out = run(["git", "-C", code_dir, "rev-parse", "HEAD"])
    except OSError:
        return None
    sha = (getattr(out, "stdout", "") or "").strip()
    return sha if getattr(out, "returncode", 1) == 0 and re.fullmatch(r"[0-9a-f]{40,64}", sha) else None


def _tracked(code_dir: str, run: Runner) -> Optional[FrozenSet[str]]:
    """The files git tracks in *code_dir* (normalised full paths); None when git cannot say."""
    try:
        out = run(["git", "-C", code_dir, "ls-files", "-z"])
    except OSError:
        return None
    if getattr(out, "returncode", 1) != 0:
        return None
    return frozenset(os.path.normcase(os.path.abspath(os.path.join(code_dir, p)))
                     for p in (out.stdout or "").split("\0") if p)


def finalize_list(code_dir: str, home: str, run: Runner = default_run) -> List[str]:
    """The old copies --finalize would delete: files (then their emptied folders) and junk."""
    if not os.path.isfile(os.path.join(home, paths.MARKER_NAME)):
        raise MigrationError("not migrated (no layout.json): nothing to finalize")
    tracked = _tracked(code_dir, run)
    if tracked is None:
        raise MigrationError("git cannot list the tracked files here; finalize deletes nothing without it")
    root = os.path.normcase(os.path.abspath(code_dir))
    git_dir = os.path.join(root, ".git")
    out = []

    def deletable(path: str) -> bool:
        full = os.path.normcase(os.path.abspath(path))
        return (full.startswith(root + os.sep) and not (full == git_dir or full.startswith(git_dir + os.sep))
                and full not in tracked and os.path.isfile(path))

    for entry in _load_manifest(home).get("items", []):
        for rel, size in (entry.get("files") or {}).items():
            src = _join(entry["source"], rel)
            if deletable(src) and os.path.getsize(src) == size:
                out.append(src)
    for name in _entries(code_dir):
        if any(fnmatch.fnmatch(name, p) for p in JUNK_PATTERNS) and deletable(os.path.join(code_dir, name)):
            out.append(os.path.join(code_dir, name))
    return out


def finalize(code_dir: str, home: str, yes: bool, box: Box) -> int:
    files = finalize_list(code_dir, home, box.run)
    empty_junk = [os.path.join(code_dir, d) for d in EMPTY_JUNK_DIRS
                  if os.path.isdir(os.path.join(code_dir, d)) and not any(os.scandir(os.path.join(code_dir, d)))]
    box.say(f"{len(files)} old file(s) to delete from {code_dir}:")
    for path in files:
        box.say(f"  {path}")
    for path in empty_junk:
        box.say(f"  {path}\\  (empty folder)")
    if not yes:
        box.say("Nothing deleted. Run again with --yes to delete these.")
        return 0
    for path in files:
        os.remove(path)
    folders = {entry["source"] for entry in _load_manifest(home).get("items", []) if entry.get("is_dir")}
    folders |= {os.path.dirname(path) for path in files}
    for folder in sorted(folders, key=len, reverse=True):
        _remove_empty_dirs(folder, code_dir)
    for path in empty_junk:
        os.rmdir(path)
    box.say(f"Deleted {len(files)} file(s).")
    return 0


def _remove_empty_dirs(top: str, code_dir: str) -> None:
    """Remove the empty folders under *top*, then *top* and its parents while empty, up to (not) *code_dir*."""
    root = os.path.normcase(os.path.abspath(code_dir))
    if not os.path.isdir(top) or not os.path.normcase(os.path.abspath(top)).startswith(root + os.sep):
        return
    for folder, _, _ in sorted(os.walk(top), key=lambda t: len(t[0]), reverse=True):
        try:
            os.rmdir(folder)      # only an empty folder goes
        except OSError:
            pass
    parent = os.path.dirname(os.path.abspath(top))
    while os.path.normcase(parent).startswith(root + os.sep):
        try:
            os.rmdir(parent)
        except OSError:
            break
        parent = os.path.dirname(parent)


def _write_json(path: str, data: Dict[str, Any]) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=1)
    os.replace(tmp, path)


def _load_manifest(home: str) -> Dict[str, Any]:
    try:
        with open(os.path.join(home, MANIFEST_NAME), encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def main(argv: Optional[Sequence[str]] = None, box: Optional[Box] = None) -> int:
    p = argparse.ArgumentParser(prog="box migrate-layout", description=__doc__.split("\n", 1)[0])
    what = p.add_mutually_exclusive_group()
    what.add_argument("--dry-run", action="store_true", help="print the plan; change nothing")
    what.add_argument("--rollback", action="store_true", help="go back to the old places")
    what.add_argument("--finalize", action="store_true", help="delete the old copies (with --yes)")
    p.add_argument("--yes", action="store_true", help="with --finalize: really delete")
    p.add_argument("--home", default=paths.default_home(), help="the new place (default %%ProgramData%%\\HomeGuard)")
    p.add_argument("--code-dir", default=paths.CODE_DIR, help=argparse.SUPPRESS)
    p.add_argument("--no-acl", action="store_true", help=argparse.SUPPRESS)
    args = p.parse_args(list(sys.argv[1:] if argv is None else argv))
    if not args.home:
        p.error("no ProgramData folder here; give --home")
    home, code_dir = os.path.abspath(args.home), os.path.abspath(args.code_dir)
    box = box or Box(code_dir)
    try:
        if args.rollback:
            return rollback(code_dir, home, box)
        if args.finalize:
            return finalize(code_dir, home, args.yes, box)
        return migrate(code_dir, home, box, dry_run=args.dry_run, set_acl=not args.no_acl)
    except MigrationError as exc:
        box.say(f"error: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
