"""Access notices: the owner is told, in his Home Guard app only, when staff looked at his recordings or his chat.

Owner decision (2026-10-03): every staff look at the owner's videos or chat is audit-logged, and the owner is told
inside the app (never over Telegram). The cloud (the Admin Center) PUSHES each notice to the box over its Tailscale
ssh path; the box has no S3 read, and nothing goes back to the cloud. The cloud runs::

    cd /d C:\\home_guard && .venv\\Scripts\\python.exe -m home_guard_project.box notices add --b64 <BASE64> --json

``<BASE64>`` is standard base64 of the UTF-8 JSON body (audit._notice_body in the cloud)::

    {"schema_version": 1, "id", "kind", "staff_name", "cameras", "from_utc", "to_utc", "message"}

("version": 1 is accepted for schema_version; unknown extra fields are ignored). ``kind`` is data the app branches
on (NOTICE_KINDS); an unknown kind is stored and shows the cloud's English ``message``, which is otherwise never
shown or parsed. ``cameras`` are already the owner's names. Answers, one line of ASCII JSON:

- exit 0, ``{"result": "added" | "updated" | "unchanged", "id": <id as sent, e.g. 42>}``: the notice is stored
  (the id is a number in the cloud's body; the box keeps it as text: the file name and the read marks);
- exit 1, ``{"error": "..."}``: a permanent rejection (bad base64 or JSON, no version, id or kind, or a --b64 longer
  than MAX_B64). The cloud never resends the same bytes, so only genuinely bad input is refused;
- exit 2, ``{"error": "..."}``: the notice is fine but could not be saved (the box's disk): send it again later,
  like an ssh failure (255) or a timeout.

Storage, under ``paths.state_dir()``: ``notices/<YYYYmmddTHHMMSS of from_utc>_<id>.json``, one per notice (the
newest body for an id replaces the older one, written atomically), and ``notices/read.json``: id -> a digest of the
body the owner read. A re-push of the same body keeps it read; a changed body (more cameras, a later end) is unread
again. The cloud never touches read.json.

The app's commands (one line of ASCII JSON each; ``{"error"}`` and exit 1 on failure)::

    python -m home_guard_project.box notices list --json [--lang he|en]   # newest first, with read flags and unread
    python -m home_guard_project.box notices read --id X --json
    python -m home_guard_project.box notices read-all --json
"""

from __future__ import annotations

import base64
import binascii
import datetime as dt
import hashlib
import json
import os
import re
import tempfile
import time
from typing import Any, Dict, List, Optional, Sequence

# Mirrored from HomeGuardAdmin home_guard_project/fleet_contract/notices.py (commit 4bada96, "pins the whole
# contract"), where the same values are kept (NOTICE_KINDS, and NOTICE_CLI with these arguments and MAX_B64). A new
# kind is added there first; until this box
# knows it, it is stored and shown as OTHER with the cloud's English message.
NOTICE_KINDS = ("recording", "chat")
NOTICE_CLI = ("notices", "add", "--b64", "<BASE64>", "--json")   # after: python -m home_guard_project.box
MAX_B64 = 7000              # the --b64 value; cmd.exe runs the ssh command and its line stops at 8191 characters
SCHEMA_VERSION = 1
OTHER = "other"
DIR_NAME = "notices"
READ_NAME = "read.json"
_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9-]{0,63}")      # the cloud's ids are numbers; no "_" (the file name's split)
_CAMERA_ID = re.compile(r"[a-z0-9_]+")
_UTC = "%Y-%m-%dT%H:%M:%SZ"


class NoticeError(ValueError):
    """A notice the box refuses, in plain words (the cloud logs them)."""


def notices_dir() -> str:
    from . import paths  # noqa: PLC0415

    return os.path.join(paths.state_dir(), DIR_NAME)


# ----------------------------------------------------------------------------
# Files
# ----------------------------------------------------------------------------
def _write_json(path: str, data: Any) -> None:
    """Write *data* atomically: a temporary file next to *path*, then os.replace."""
    folder = os.path.dirname(os.path.abspath(path))
    os.makedirs(folder, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".notice-", suffix=".tmp", dir=folder)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=1, sort_keys=True)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise


def _read_json(path: str) -> Any:
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def _utc(value: Any) -> Optional[str]:
    """*value* when it is a "YYYY-mm-ddTHH:MM:SSZ" time, else None."""
    try:
        dt.datetime.strptime(str(value), _UTC)
    except ValueError:
        return None
    return str(value)


def digest(body: Dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(body, sort_keys=True, ensure_ascii=True).encode("ascii")).hexdigest()[:16]


# ----------------------------------------------------------------------------
# One notice
# ----------------------------------------------------------------------------
def validate(raw: Any) -> Dict[str, Any]:
    """The notice in *raw* (the parsed body) as the box stores it; NoticeError when it is not a notice.

    Required: schema_version (or version), id, kind. Another version or an unknown kind is stored as it came and
    shown as OTHER; missing optional fields read as empty."""
    if not isinstance(raw, dict):
        raise NoticeError("the notice is not a JSON object")
    version = raw.get("schema_version", raw.get("version"))
    if isinstance(version, bool) or not isinstance(version, int):
        raise NoticeError("the notice has no schema_version")
    notice_id = raw.get("id")
    if isinstance(notice_id, bool) or not isinstance(notice_id, (int, str)) or not _ID.fullmatch(str(notice_id)):
        raise NoticeError("the notice has no usable id (letters, digits and dashes, up to 64)")
    kind = raw.get("kind")
    if not isinstance(kind, str) or not kind.strip():
        raise NoticeError("the notice has no kind")
    from_utc = _utc(raw.get("from_utc"))
    to_utc = _utc(raw.get("to_utc")) or from_utc
    cameras = raw.get("cameras") if isinstance(raw.get("cameras"), list) else []
    staff, message = raw.get("staff_name"), raw.get("message")
    return {
        "schema_version": version,
        "id": str(notice_id),
        "kind": kind.strip(),
        "staff_name": staff.strip() if isinstance(staff, str) else "",
        "cameras": [str(c).strip() for c in cameras if isinstance(c, (str, int)) and str(c).strip()],
        "from_utc": from_utc,
        "to_utc": max(from_utc, to_utc) if from_utc and to_utc else to_utc,
        "message": message.strip() if isinstance(message, str) else "",
    }


def decode(b64: str) -> Dict[str, Any]:
    """The notice in the --b64 value; NoticeError for anything the box must refuse."""
    b64 = str(b64 or "").strip()
    if len(b64) > MAX_B64:
        raise NoticeError(f"the notice is too long ({len(b64)} base64 characters, at most {MAX_B64})")
    try:
        text = base64.b64decode(b64, validate=True).decode("utf-8")
    except (binascii.Error, ValueError) as exc:
        raise NoticeError("the notice is not base64 of UTF-8 text") from exc
    try:
        raw = json.loads(text)
    except ValueError as exc:
        raise NoticeError("the notice is not JSON") from exc
    return validate(raw)


def _files(folder: str) -> Dict[str, str]:
    """id -> file name, for every notice file in *folder* (the newest name wins if an id has two)."""
    found: Dict[str, str] = {}
    try:
        names = sorted(os.listdir(folder))
    except OSError:
        return found
    for name in names:
        if name.endswith(".json") and name != READ_NAME and "_" in name and not name.startswith("."):
            found[name[:-5].split("_", 1)[1]] = name
    return found


def add(notice: Dict[str, Any], folder: Optional[str] = None, now: Optional[float] = None) -> str:
    """Store *notice* (validate's result). Returns "added", "updated" (it replaced an older body of the same id)
    or "unchanged" (the same body again: nothing is written, the read state stays)."""
    folder = folder or notices_dir()
    stamp = notice["from_utc"] or time.strftime(_UTC, time.gmtime(time.time() if now is None else now))
    name = f"{dt.datetime.strptime(stamp, _UTC):%Y%m%dT%H%M%S}_{notice['id']}.json"
    old_name = _files(folder).get(notice["id"])
    if old_name is not None:
        old = _read_json(os.path.join(folder, old_name))
        if isinstance(old, dict) and digest(old) == digest(notice) and old_name == name:
            return "unchanged"
    _write_json(os.path.join(folder, name), notice)
    if old_name is not None and old_name != name:
        try:
            os.remove(os.path.join(folder, old_name))
        except OSError:
            pass
    return "added" if old_name is None else "updated"


# ----------------------------------------------------------------------------
# What the app reads
# ----------------------------------------------------------------------------
def load(folder: Optional[str] = None) -> List[Dict[str, Any]]:
    """Every stored notice (a damaged file is left out)."""
    folder = folder or notices_dir()
    out = []
    for name in _files(folder).values():
        data = _read_json(os.path.join(folder, name))
        try:
            out.append(validate(data))
        except NoticeError:
            continue
    return out


def read_marks(folder: Optional[str] = None) -> Dict[str, str]:
    data = _read_json(os.path.join(folder or notices_dir(), READ_NAME))
    return {str(k): str(v) for k, v in data.items()} if isinstance(data, dict) else {}


def is_read(notice: Dict[str, Any], marks: Dict[str, str]) -> bool:
    return marks.get(notice["id"]) == digest(notice)


def _local(utc: str) -> str:
    """The box's clock time for *utc*, "YYYY-mm-ddTHH:MM" (the box runs on the house's time zone)."""
    return dt.datetime.strptime(utc, _UTC).replace(tzinfo=dt.timezone.utc).astimezone().strftime("%Y-%m-%dT%H:%M")


def shown_camera(name: str, lang: str = "he", aliases: Optional[Dict[str, List[str]]] = None) -> str:
    """The cloud sends the owner's names; one that still looks like a camera id goes through display_name."""
    if not _CAMERA_ID.fullmatch(name):
        return name
    from .camera_names import display_name  # noqa: PLC0415

    return display_name(name, lang, aliases)


def listing(folder: Optional[str] = None, lang: str = "he",
            aliases: Optional[Dict[str, List[str]]] = None) -> Dict[str, Any]:
    """``{"notices": [...], "unread": n}``, newest first (by from_utc), ready for the app to word."""
    marks = read_marks(folder)
    rows = []
    for n in load(folder):
        known = n["schema_version"] == SCHEMA_VERSION and n["kind"] in NOTICE_KINDS
        frm = n["from_utc"] or ""
        rows.append({
            "id": n["id"],
            "kind": n["kind"] if known else OTHER,
            "cameras": [shown_camera(c, lang, aliases) for c in n["cameras"]],
            "staff_name": n["staff_name"],
            "from_utc": frm,
            "to_utc": n["to_utc"] or frm,
            "from_local": _local(frm) if frm else "",
            "to_local": _local(n["to_utc"] or frm) if frm else "",
            "message": "" if known else n["message"],
            "read": is_read(n, marks),
        })
    rows.sort(key=lambda r: (r["from_utc"], r["id"]), reverse=True)
    return {"notices": rows, "unread": sum(1 for r in rows if not r["read"])}


def mark_read(notice_id: Optional[str] = None, folder: Optional[str] = None) -> int:
    """Mark one notice (or, with no id, every notice) read as it is now; returns how many changed.
    KeyError for an unknown id."""
    folder = folder or notices_dir()
    stored = {n["id"]: n for n in load(folder)}
    if notice_id is not None and str(notice_id) not in stored:
        raise KeyError(str(notice_id))
    marks = read_marks(folder)
    changed = 0
    for nid in ([str(notice_id)] if notice_id is not None else list(stored)):
        if not is_read(stored[nid], marks):
            marks[nid] = digest(stored[nid])
            changed += 1
    marks = {k: v for k, v in marks.items() if k in stored}      # a notice that is gone keeps no mark
    if changed:
        _write_json(os.path.join(folder, READ_NAME), marks)
    return changed


# ----------------------------------------------------------------------------
# Command line
# ----------------------------------------------------------------------------
def _print(data: Dict[str, Any]) -> None:
    print(json.dumps(data, ensure_ascii=True, separators=(",", ":")))


class _Parser:
    @staticmethod
    def build():
        import argparse  # noqa: PLC0415

        class Refusing(argparse.ArgumentParser):
            def error(self, message):              # one line of JSON, never argparse's usage text
                raise NoticeError(message)

        parser = Refusing(prog="box notices", description="Staff access notices for the owner's app.")
        parser.add_argument("command", choices=("add", "list", "read", "read-all"))
        parser.add_argument("--b64", help="add: the notice, base64 of its UTF-8 JSON")
        parser.add_argument("--id")
        parser.add_argument("--lang", choices=("he", "en"), default="he")
        parser.add_argument("--json", action="store_true")
        return parser


def main(argv: Optional[Sequence[str]] = None, folder: Optional[str] = None) -> int:
    try:
        args = _Parser.build().parse_args(argv)
        folder = folder or notices_dir()
        if args.command == "add":
            if args.b64 is None:
                raise NoticeError("add needs --b64")
            notice = decode(args.b64)
            sent_id = json.loads(base64.b64decode(args.b64.strip())).get("id")   # the reply gives the id as sent
            try:
                result = add(notice, folder)
            except OSError as exc:            # the box's disk, not the notice: worth sending again
                _print({"error": f"the notice could not be saved on the box ({type(exc).__name__}: {exc})"})
                return 2
            _print({"result": result, "id": sent_id})
        elif args.command == "list":
            _print(listing(folder, args.lang))
        elif args.command == "read":
            if not args.id:
                raise NoticeError("read needs --id")
            changed = mark_read(args.id, folder)
            _print({"id": args.id, "marked": changed, "unread": listing(folder)["unread"]})
        else:
            _print({"marked": mark_read(None, folder), "unread": listing(folder)["unread"]})
    except NoticeError as exc:
        _print({"error": str(exc)})
        return 1
    except KeyError as exc:
        _print({"error": f"no notice {exc.args[0]}"})
        return 1
    except Exception as exc:  # noqa: BLE001 - the caller reads one line of JSON
        _print({"error": f"{type(exc).__name__}: {exc}"})
        return 1
    return 0
