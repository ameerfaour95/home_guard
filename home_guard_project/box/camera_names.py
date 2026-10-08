"""What the owner reads for a camera: the name the family gave it, never the internal id.

Owner rule (2026-10-06, again 2026-10-08): "don't show the camera ID if we decided to call it differently".
Every owner-facing text (alerts, receipts, the assistant's answers) goes through ``display_name``.

- The family's newest name for the camera wins (``brain/aliases.py`` keeps them, oldest first).
- A site rename (``ameer_tes2`` -> ``ameer_week_0_1``) changes every camera id but not the channel, so a name
  saved under the old id is still found by the channel suffix (``_ch6``) when exactly one old id matches.
- Without a name: "מצלמה 6" / "Camera 6" from the channel number, else the id with underscores as spaces.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

_CHANNEL = re.compile(r"(?:^|_)ch(\d+)$", re.IGNORECASE)


def channel_of(camera: str) -> Optional[str]:
    """``"6"`` for ``ameer_week_0_1_ch6``; None when the id has no ``_chN`` ending."""
    m = _CHANNEL.search(str(camera or ""))
    return m.group(1) if m else None


def _load(aliases: Optional[Dict[str, List[str]]]) -> Dict[str, List[str]]:
    if aliases is not None:
        return aliases
    try:
        from .brain.aliases import load_aliases  # noqa: PLC0415 - keep this module import-light

        return load_aliases()
    except Exception:  # noqa: BLE001 - a damaged or missing file only means "no names"
        return {}


def family_names(camera: str, aliases: Optional[Dict[str, List[str]]] = None) -> List[str]:
    """The family's names for *camera*, oldest first; names saved under a pre-rename id are included."""
    data = _load(aliases)
    names = list(data.get(str(camera), []))
    if names:
        return names
    ch = channel_of(camera)
    if ch is None:
        return []
    same_channel = [k for k in data if k != camera and channel_of(k) == ch and data.get(k)]
    if len(same_channel) == 1:          # two old sites with a ch6 each: ambiguous, show none rather than guess
        return list(data[same_channel[0]])
    return []


def display_name(camera: str, lang: str = "he", aliases: Optional[Dict[str, List[str]]] = None) -> str:
    """The name the owner reads for *camera* in *lang* ("he" or "en")."""
    names = family_names(camera, aliases)
    if names:
        return names[-1].strip()
    ch = channel_of(camera)
    if ch is not None:
        return f"מצלמה {ch}" if str(lang).startswith("he") else f"Camera {ch}"
    return str(camera or "").replace("_", " ").strip()


def replace_ids(text: str, cameras: List[str], lang: str = "he",
                aliases: Optional[Dict[str, List[str]]] = None) -> str:
    """*text* with every known camera id swapped for its display name (a last guard before a message goes out)."""
    out = str(text or "")
    data = _load(aliases)
    for cam in sorted({c for c in cameras if c}, key=len, reverse=True):
        if cam in out:
            out = out.replace(cam, display_name(cam, lang, data))
    return out


# ----------------------------------------------------------------------------
# Command line: the app sets a camera's name (the family's names, brain/aliases)
# ----------------------------------------------------------------------------
def set_name(camera: str, name: str, cameras: List[str], path: Optional[str] = None) -> Dict[str, str]:
    """Make *name* the camera's name: its newest alias (brain/aliases.add_alias; an alias it already has is
    taken off first with remove_alias, so choosing it again makes it the newest). ValueError in plain words."""
    from .brain import aliases  # noqa: PLC0415

    path = path or aliases.ALIASES_PATH
    if any(aliases.normalize(a) == aliases.normalize(name) for a in aliases.load_aliases(path).get(str(camera), [])):
        aliases.remove_alias(camera, name, path)
    aliases.add_alias(camera, name, cameras, path)
    names = aliases.load_aliases(path)
    return {"camera": camera, "display_name": display_name(camera, "he", names),
            "display_name_en": display_name(camera, "en", names)}


def main(argv: Optional[List[str]] = None) -> int:
    """``set --camera X --name-b64 <base64 of UTF-8> --json``: one line of ASCII JSON, ``{"error"}`` on failure."""
    import argparse  # noqa: PLC0415
    import base64  # noqa: PLC0415
    import binascii  # noqa: PLC0415
    import json  # noqa: PLC0415

    parser = argparse.ArgumentParser(prog="camera_names", description="The family's names for the cameras.")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("set", help="give a camera its name (the newest alias)")
    p.add_argument("--camera", required=True)
    p.add_argument("--name-b64", required=True, help="the name, as base64 of UTF-8 text")
    p.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    try:
        from .find_cameras import CAMERAS_PATH, _read_cameras_raw  # noqa: PLC0415

        try:
            name = base64.b64decode(args.name_b64, validate=True).decode("utf-8").strip()
        except (binascii.Error, ValueError):
            raise ValueError("the name could not be read") from None
        raw = _read_cameras_raw(CAMERAS_PATH)
        cameras = list(dict(raw.get("cameras") or {})) + list(dict(raw.get("disabled") or {}))
        result: Dict[str, Any] = set_name(args.camera, name, cameras)
    except ValueError as exc:
        result = {"error": str(exc)}
    print(json.dumps(result))
    return 1 if "error" in result else 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
