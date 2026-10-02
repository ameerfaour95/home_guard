"""
Build dist/home_guard_box.zip — the code a collector box needs, and nothing else.

Usage:
    python -m home_guard_project.box.make_bundle

The bundle is built from an explicit include list rather than from git: the
repo ignores the config loaders the box needs, and some tracked scripts hold
the founder's own camera credentials, which must not travel to another house.
"""

from __future__ import annotations

import logging
import os
import re
import zipfile
from typing import List

from .boxconfig import PROJECT_ROOT

log = logging.getLogger("box.bundle")

BUNDLE_PATH = os.path.join(PROJECT_ROOT, "dist", "home_guard_box.zip")

ROOT_FILES = ("pyproject.toml", "uv.lock", ".python-version", "yolo11s.pt")

# Whole directories (walked recursively).
INCLUDE_DIRS = (
    "home_guard_project/box",
    "home_guard_project/s3_upload",
    "home_guard_project/labeling/utils",
)

INCLUDE_FILES = (
    "home_guard_project/labeling/__init__.py",
    "home_guard_project/data_collection/__init__.py",
    "home_guard_project/data_collection/config.py",
    "home_guard_project/data_collection/config.yaml",
    "home_guard_project/data_collection/data_collection.py",
    "home_guard_project/data_collection/discover.py",
    "home_guard_project/data_collection/roi_editor.py",
    "home_guard_project/data_collection/start.sh",
)

# Per-machine or secret files that must never be bundled.
EXCLUDE_NAMES = frozenset({"cameras.yaml", "zones.yaml", "box.yaml", "api_key.env"})
EXCLUDE_SUFFIXES = (".pyc", ".pt", ".log", ".zip")

# An RTSP URL carrying user and password. A placeholder password such as "pass" still matches,
# so docs with example URLs are kept out too.
CREDENTIAL_URL = re.compile(r"rtsp://[^\s:/@\"']+:[^\s/@\"']+@")

_LF_SUFFIXES = (".sh",)


def _excluded(rel: str) -> bool:
    name = os.path.basename(rel)
    return (
        name in EXCLUDE_NAMES
        or "__pycache__" in rel.split("/")
        or name.endswith(EXCLUDE_SUFFIXES)
    )


def _has_credentials(path: str) -> bool:
    with open(path, encoding="utf-8", errors="ignore") as f:
        return CREDENTIAL_URL.search(f.read()) is not None


def bundle_files(root: str = PROJECT_ROOT) -> List[str]:
    """Relative POSIX paths of every file that goes into the bundle, sorted."""
    candidates = [f for f in ROOT_FILES + INCLUDE_FILES if os.path.isfile(os.path.join(root, f))]
    for rel_dir in INCLUDE_DIRS:
        for dirpath, _, filenames in os.walk(os.path.join(root, rel_dir)):
            for name in filenames:
                rel = os.path.relpath(os.path.join(dirpath, name), root).replace("\\", "/")
                candidates.append(rel)

    files: List[str] = []
    for rel in candidates:
        if rel in ROOT_FILES:
            files.append(rel)
        elif _excluded(rel):
            continue
        elif _has_credentials(os.path.join(root, rel)):
            log.warning("Skipping %s: contains a camera URL with credentials", rel)
        else:
            files.append(rel)
    return sorted(set(files))


def build_bundle(root: str = PROJECT_ROOT, out_path: str = BUNDLE_PATH) -> List[str]:
    files = bundle_files(root)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for rel in files:
            path = os.path.join(root, rel)
            if rel.endswith(_LF_SUFFIXES):
                # bash on the box rejects CRLF line endings.
                with open(path, "rb") as f:
                    zf.writestr(rel, f.read().replace(b"\r\n", b"\n"))
            else:
                zf.write(path, rel)
    return files


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)-8s %(message)s")
    files = build_bundle()
    size_mb = os.path.getsize(BUNDLE_PATH) / 1e6
    log.info("Wrote %s (%d files, %.1f MB)", BUNDLE_PATH, len(files), size_mb)


if __name__ == "__main__":
    main()
