"""The box's api_key.env, merged rather than replaced.

Setup used to write a fresh api_key.env holding only OPENAI_API_KEY and TELEGRAM_BOT_TOKEN, which silently
removed every other key (OPENROUTER_API_KEY on 2026-10-06 and again 2026-10-10: no vision model, no translator,
no describer). Setup now copies its keys to ``api_key.env.incoming`` and runs ``python -m home_guard_project.box
merge-secrets <file>``: a key it sends is set, every other key the box had stays, an empty value never wipes one.
The file is written to a temp file synced to disk, then replaced. Only key names are ever printed.
"""

from __future__ import annotations

import os
from typing import Dict, List


def _pairs(text: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key.strip() and value.strip():
            out[key.strip()] = value.strip()
    return out


def merge_text(existing: str, incoming: str) -> str:
    """*existing* with every non-empty key of *incoming* set (in place, or added at the end); lines kept."""
    new = _pairs(incoming)
    done = set()
    lines: List[str] = []
    for line in existing.splitlines():
        key = line.split("=", 1)[0].strip() if "=" in line and not line.lstrip().startswith("#") else ""
        if key in new:
            lines.append(f"{key}={new[key]}")
            done.add(key)
        else:
            lines.append(line.rstrip("\r"))
    lines += [f"{k}={v}" for k, v in new.items() if k not in done]
    return "\n".join(lines) + "\n" if lines else ""


def merge_file(incoming_path: str, target_path: str) -> Dict[str, List[str]]:
    """Merge *incoming_path* into *target_path*, then delete *incoming_path*. ``{"set": [...], "kept": [...]}``."""
    with open(incoming_path, encoding="utf-8-sig") as f:
        incoming = f.read()
    existing = ""
    if os.path.isfile(target_path):
        with open(target_path, encoding="utf-8-sig") as f:
            existing = f.read()
    merged = merge_text(existing, incoming)
    os.makedirs(os.path.dirname(os.path.abspath(target_path)), exist_ok=True)
    tmp = target_path + ".tmp"
    try:
        with open(tmp, "w", encoding="utf-8", newline="\n") as f:
            f.write(merged)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, target_path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    os.remove(incoming_path)
    sent = sorted(_pairs(incoming))
    return {"set": sent, "kept": sorted(k for k in _pairs(existing) if k not in sent)}


def main(argv: List[str]) -> int:
    """``merge-secrets FILE``: merge FILE into this box's api_key.env and delete FILE."""
    import json  # noqa: PLC0415

    from . import paths  # noqa: PLC0415

    if len(argv) != 1:
        print("usage: merge-secrets FILE")
        return 2
    try:
        print(json.dumps(merge_file(argv[0], paths.secrets_env())))
    except OSError as exc:
        print(json.dumps({"error": f"{type(exc).__name__}: {exc.strerror or exc}"}))
        return 1
    return 0
