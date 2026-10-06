"""YOLO boxes of the unified dataset's clips, as editable tracks.

The dataset's auto-tagger (week 0: YOLO11x + RT-DETR, tracked) wrote one YOLO label file per sampled frame:
``<dataset>/yolo/labels/<folder>/<clip_id>_f<frame>.txt`` with lines ``class cx cy w h`` (normalised), classes in
``yolo/classes.txt`` (the contiguous 0-8 training ids). The frame number is the clip's own frame; its time is
frame / fps (the clip meta's ``fps_estimated``). The per-frame boxes are linked into tracks with the shared
``fleet_contract.tracks.tracks_from_weak_labels`` and marked ``source="yolo"``: boxes nobody has checked yet.
"""
from __future__ import annotations

import glob
import os
import re
from typing import Dict, List, Optional, Tuple

from ...fleet_contract import tracks as ft

_FRAME = re.compile(r"_f(\d+)\.txt$")
DEFAULT_CLASSES = ("person", "bicycle", "car", "motorcycle", "bus", "truck", "bird", "cat", "dog")


def classes(dataset: str) -> Tuple[str, ...]:
    path = os.path.join(dataset, "yolo", "classes.txt")
    try:
        names: Dict[int, str] = {}
        with open(path, encoding="utf-8") as f:
            for line in f:
                parts = line.split()
                if len(parts) >= 2 and parts[0].isdigit():
                    names[int(parts[0])] = parts[1]
        return tuple(names[i] for i in sorted(names)) or DEFAULT_CLASSES
    except OSError:
        return DEFAULT_CLASSES


def label_files(dataset: str, clip_id: str) -> List[Tuple[int, str]]:
    """``[(frame, path)]`` of the clip's YOLO label files, in frame order."""
    pattern = os.path.join(glob.escape(os.path.join(dataset, "yolo", "labels")), "*", glob.escape(clip_id) + "_f*.txt")
    out = []
    for path in glob.glob(pattern):
        m = _FRAME.search(os.path.basename(path))
        if m and os.path.basename(path)[:-len(m.group(0))] == clip_id:
            out.append((int(m.group(1)), path))
    return sorted(out)


def read_boxes(path: str, names: Tuple[str, ...]) -> List[Tuple[str, List[float]]]:
    out = []
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                parts = line.split()
                if len(parts) < 5 or not parts[0].isdigit() or int(parts[0]) >= len(names):
                    continue
                cx, cy, w, h = (float(v) for v in parts[1:5])
                box = [max(0.0, cx - w / 2), max(0.0, cy - h / 2), min(1.0, cx + w / 2), min(1.0, cy + h / 2)]
                if box[2] > box[0] and box[3] > box[1]:
                    out.append((names[int(parts[0])], box))
    except (OSError, ValueError):
        return []
    return out


def dataset_tracks(dataset: str, clip_id: str, fps: Optional[float]) -> List[ft.Track]:
    """The clip's YOLO boxes linked into tracks ``t-1``, ``t-2``, ... in order of first appearance; [] if none."""
    files = label_files(dataset, clip_id)
    if not files:
        return []
    names, fps = classes(dataset), fps or 7.0
    frames = [(frame, round(frame / fps, 4), read_boxes(path, names)) for frame, path in files]
    tracks = ft.tracks_from_weak_labels(frames)
    for n, tr in enumerate(tracks, start=1):
        tr.track_id, tr.source = f"t-{n}", "yolo"
    return tracks
