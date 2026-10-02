"""Shared fixtures for box tests: build a clip's files in a dataset directory."""
from __future__ import annotations

import os
from typing import List


def clip_files(camera: str, date: str, stem: str, with_crop: bool = True) -> List[str]:
    """Relative paths of every file the collector writes for one clip (meta last)."""
    rel = f"{camera}/{date}"
    files = [f"clips/{rel}/{stem}.mp4"]
    if with_crop:
        files.append(f"vlm_crops/{rel}/{stem}.mp4")
    files += [
        f"responses/{rel}/{stem}.model_raw.txt",
        f"yolo/images/{rel}/{stem}_f0000.jpg",
        f"yolo/images/{rel}/{stem}_f0002.jpg",
        f"yolo/labels/{rel}/{stem}_f0000.txt",
        f"yolo/labels/{rel}/{stem}_f0002.txt",
        f"meta/{rel}/{stem}.meta.json",
    ]
    return files


def make_clip(
    dataset_dir: str,
    camera: str,
    stem: str,
    meta_mtime: float,
    date: str = "2026-10-02",
    with_crop: bool = True,
) -> List[str]:
    """Create the clip's files under *dataset_dir*; return their relative paths."""
    files = clip_files(camera, date, stem, with_crop)
    for rel in files:
        path = os.path.join(dataset_dir, rel)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(rel)
    os.utime(os.path.join(dataset_dir, files[-1]), (meta_mtime, meta_mtime))
    return files
