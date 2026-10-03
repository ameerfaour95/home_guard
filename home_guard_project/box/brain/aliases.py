"""The owner's own names for the cameras ("the entrance", "כניסה", "المدخل").

Kept in their own small file next to cameras.yaml (that file is rewritten from
its two sections whenever a camera changes, so it cannot carry extra keys).
Matching is case-, space- and underscore-insensitive. An alias may never name
two cameras, and a rename carries a camera's aliases with it.
"""

from __future__ import annotations

import os
from typing import Dict, List, Sequence

import yaml

ALIASES_PATH = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", "data_collection",
                                             "camera_aliases.yaml"))
MAX_ALIAS_CHARS = 40


def normalize(text: str) -> str:
    return " ".join(str(text or "").casefold().replace("_", " ").split())


def load_aliases(path: str = ALIASES_PATH) -> Dict[str, List[str]]:
    """``{camera: [alias, ...]}``; empty when the file is missing or damaged."""
    try:
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    except (OSError, ValueError, yaml.YAMLError):
        return {}
    if not isinstance(data, dict):
        return {}
    return {str(k): [str(a) for a in v if str(a).strip()] for k, v in data.items() if isinstance(v, list)}


def _save(data: Dict[str, List[str]], path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write("# The owner's names for each camera (set in the app or by chat).\n")
        yaml.safe_dump({k: v for k, v in sorted(data.items()) if v}, f, allow_unicode=True)
    os.replace(tmp, path)


def add_alias(camera: str, alias: str, cameras: Sequence[str], path: str = ALIASES_PATH) -> List[str]:
    """Add *alias* to *camera*; returns its aliases. Raises ValueError on a bad or colliding alias."""
    wanted = normalize(alias)
    if not wanted or len(wanted) > MAX_ALIAS_CHARS:
        raise ValueError("an alias must be 1 to 40 characters")
    if camera not in cameras:
        raise ValueError(f"unknown camera: {camera!r}")
    data = load_aliases(path)
    for other in cameras:
        if other == camera:
            continue
        if normalize(other) == wanted or wanted in (normalize(a) for a in data.get(other, [])):
            raise ValueError(f'"{alias.strip()}" already names {other}')
    mine = data.setdefault(camera, [])
    if wanted not in (normalize(a) for a in mine):
        mine.append(alias.strip())
        _save(data, path)
    return list(mine)


def remap_aliases(renames: Dict[str, str], path: str = ALIASES_PATH) -> None:
    """Apply camera renames ``{old: new}`` in one pass (swaps and chains included)."""
    data = load_aliases(path)
    if not data or not renames:
        return
    out: Dict[str, List[str]] = {}
    for name, aliases in data.items():          # renamed entries first: they win a name clash
        if name in renames:
            out[renames[name]] = aliases
    for name, aliases in data.items():
        if name not in renames and name not in out:
            out[name] = aliases
    _save(out, path)
