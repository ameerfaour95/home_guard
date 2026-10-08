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
from typing import Dict, List, Optional

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
