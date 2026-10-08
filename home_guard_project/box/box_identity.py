"""The box's own id: made once, kept for the life of the box, the same across site renames.

Why (2026-10-09): the Admin Center saw the owner's one box as six houses (ameer_test, ameer_tes2, ameer_week_0_1, ...)
because every identity it had changed with the site name; only the host name stayed. A random id written once next
to registration.json, sent in the heartbeat and the registration, lets the cloud group a box's history by box.
"""

from __future__ import annotations

import os
import re
import uuid
from typing import Optional

from . import paths

FILE_NAME = "box_id.txt"
_ID = re.compile(r"^[0-9a-f]{32}$")


def default_path() -> str:
    return os.path.join(os.path.dirname(paths.registration_json()), FILE_NAME)


def box_id(path: Optional[str] = None, create: bool = True) -> Optional[str]:
    """This box's id (32 hex characters). Made and saved on first use when *create*; None when missing and not
    created, or when the disk refuses (the caller then sends no id rather than a new one each time)."""
    path = path or default_path()
    try:
        with open(path, encoding="ascii") as f:
            value = f.read().strip()
        if _ID.fullmatch(value):
            return value
    except (OSError, UnicodeDecodeError):
        pass
    if not create:
        return None
    value = uuid.uuid4().hex
    try:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        tmp = f"{path}.tmp"
        with open(tmp, "w", encoding="ascii") as f:
            f.write(value + "\n")
        os.replace(tmp, path)
    except OSError:
        return None
    return value
