"""Owner notices: what staff looked at, written by the cloud to fleet/<device_id>/notices/<time>_<id>.json (the
cloud's record) and pushed to the box over Tailscale SSH (NOTICE_CLI below; the box never reads S3).

Each notice is ``{"schema_version": 1, "id", "kind", "staff_name", "cameras", "from_utc", "to_utc", "message"}``.
``kind`` is data a consumer (the box, the owner's app) branches on; ``message`` is only the English sentence for
display. A consumer never parses the message to learn what was viewed.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any, Optional

# recording: staff opened a recording (clip, picture) of these cameras; chat: staff opened the owner's Telegram
# conversation with the assistant (the Admin Center's Chat tab, or a picture sent in it).
NOTICE_KINDS = ("recording", "chat")


def check_kind(kind: str) -> str:
    """*kind* when it is a notice kind, else ValueError (a new kind is added here first, then used)."""
    if kind not in NOTICE_KINDS:
        raise ValueError(f"unknown notice kind {kind!r}; known: {', '.join(NOTICE_KINDS)}")
    return kind


def notice_message(kind: str, cameras: list[str], when: str) -> str:
    """The display sentence for a notice of *kind* (cameras already in the owner's names, *when* "HH:MM" or a range)."""
    check_kind(kind)
    if kind == "chat":
        return f"Home Guard support viewed your chat with the assistant ({when})"
    names = "" if not cameras else cameras[0] if len(cameras) == 1 else ", ".join(cameras[:-1]) + " and " + cameras[-1]
    return f"Home Guard support viewed recordings{' from ' + names if names else ''} ({when})"


# ---------------------------------------------------------------- push delivery to the box (agreed with the box side)
# The box has no S3 read access: the cloud runs this over Tailscale SSH. Body: the notice object above
# (schema_version, id and kind are required; the box replaces by id, so a re-push of the same id is safe).
NOTICE_CLI = r"cd /d C:\home_guard && .venv\Scripts\python.exe -m home_guard_project.box notices add --b64 {b64} --json"
NOTICE_SSH_OPTIONS = ("-o", "BatchMode=yes", "-o", "ConnectTimeout=15")
MAX_B64 = 7000  # the box refuses longer --b64 values (cmd line limit); the cloud refuses them before sending


class NoticeTooLong(ValueError):
    """The encoded notice is over MAX_B64: the box would reject it, so it is never sent."""


def notice_b64(body: dict[str, Any]) -> str:
    """BASE64(JSON utf-8) of a notice body."""
    return base64.b64encode(json.dumps(body, ensure_ascii=False, separators=(",", ":")).encode("utf-8")).decode("ascii")


def notice_command(body: dict[str, Any]) -> str:
    """The command the box runs (in cmd.exe); NoticeTooLong over MAX_B64."""
    b64 = notice_b64(body)
    if len(b64) > MAX_B64:
        raise NoticeTooLong(f"notice is {len(b64)} base64 chars, over {MAX_B64}")
    return NOTICE_CLI.format(b64=b64)


def notice_ssh_argv(ssh_user: str, host: str, body: dict[str, Any], key: Optional[Path] = None) -> list[str]:
    """``ssh -i ~/.ssh/homeguard_box -o BatchMode=yes -o ConnectTimeout=15 <user>@<host> "<NOTICE_CLI>"``."""
    key = key if key is not None else Path.home() / ".ssh" / "homeguard_box"
    return ["ssh", "-i", str(key), *NOTICE_SSH_OPTIONS, f"{ssh_user}@{host}", notice_command(body)]


def notice_reply(returncode: Optional[int], stdout: str) -> tuple[str, dict[str, Any]]:
    """("delivered" | "rejected" | "retry", the box's JSON reply or {}). Exit 0 with a JSON reply without "error" is
    delivered; exit 1 with {"error": ...} is rejected for good (that body is never re-sent); anything else (ssh exit
    255, a timeout, a box without the command yet, output that is not JSON) is retried."""
    reply: dict[str, Any] = {}
    for line in (stdout or "").splitlines():
        line = line.strip()
        if line.startswith("{"):
            try:
                parsed = json.loads(line)
            except ValueError:
                continue
            if isinstance(parsed, dict):
                reply = parsed
                break
    if returncode == 0 and reply and "error" not in reply:
        return "delivered", reply
    if returncode == 1 and "error" in reply:
        return "rejected", reply
    return "retry", reply
