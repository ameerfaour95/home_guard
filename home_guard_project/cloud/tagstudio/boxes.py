"""YOLO boxes of the unified dataset's clips, as editable tracks.

The dataset's auto-tagger (week 0: YOLO11x + RT-DETR, tracked) wrote one YOLO label file per sampled frame:
``<dataset>/yolo/labels/<folder>/<clip_id>_f<frame>.txt`` with lines ``class cx cy w h`` (normalised). The frame
number is the clip's own frame; its time is frame / fps (the clip meta's ``fps_estimated``). The per-frame boxes are
linked into tracks with the shared ``fleet_contract.tracks.tracks_from_weak_labels`` and marked ``source="yolo"``:
boxes nobody has checked yet.

Two class-id systems exist (``fleet_contract.classes``): training labels use the contiguous 0-8 ids (named by
``yolo/classes.txt``), the box's weak labels keep the sparse COCO ids (0 person, 2 car, 5 bus, 7 truck, 14 bird, ...).
Read one as the other and a truck becomes a cat. Each label folder's system is, in order: declared by an
``ids.txt`` in the folder (``coco`` or ``contiguous``); seen in the clip's own files or a sample of the folder (an id
above 8 only exists in COCO, 4/6/8 only in the contiguous system); else contiguous when the dataset has a
``yolo/classes.txt`` (a training dataset) and COCO when it has none (a box's weak-label folder).
"""
from __future__ import annotations

import functools
import glob
import os
import re
from typing import Dict, Iterable, List, Optional, Tuple

from ...fleet_contract import tracks as ft
from ...fleet_contract.classes import COCO_IDS, CONTIGUOUS_IDS, class_name

_FRAME = re.compile(r"_f(\d+)\.txt$")
DEFAULT_CLASSES = ("person", "bicycle", "car", "motorcycle", "bus", "truck", "bird", "cat", "dog")
ID_FILE = "ids.txt"
SAMPLE_FILES = 400
_ONLY_COCO = lambda i: i > 8  # noqa: E731
_ONLY_CONTIGUOUS = frozenset({4, 6, 8})


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


def _ids(paths: Iterable[str]) -> set:
    found = set()
    for path in paths:
        try:
            with open(path, encoding="utf-8") as f:
                for line in f:
                    head = line.split(maxsplit=1)
                    if head and head[0].isdigit():
                        found.add(int(head[0]))
        except OSError:
            continue
    return found


def _seen_system(ids: set) -> Optional[str]:
    """The system a set of ids can only belong to, else None (0, 1, 2, 3, 5, 7 read the same way in both)."""
    if any(_ONLY_COCO(i) for i in ids):
        return COCO_IDS
    if ids & _ONLY_CONTIGUOUS:
        return CONTIGUOUS_IDS
    return None


@functools.lru_cache(maxsize=256)
def _folder_system(folder: str, mtime: float) -> Optional[str]:
    declared = os.path.join(folder, ID_FILE)
    try:
        with open(declared, encoding="utf-8") as f:
            word = f.read().strip().lower()
        if word in (COCO_IDS, CONTIGUOUS_IDS):
            return word
    except OSError:
        pass
    sample = sorted(glob.glob(os.path.join(glob.escape(folder), "*.txt")))[:SAMPLE_FILES]
    return _seen_system(_ids(p for p in sample if os.path.basename(p) != ID_FILE))


def id_system(dataset: str, folder: str, clip_files: Iterable[str] = ()) -> str:
    """COCO_IDS or CONTIGUOUS_IDS for the label files of `folder` (see the module docstring for the order)."""
    try:
        mtime = os.path.getmtime(folder)
    except OSError:
        mtime = 0.0
    declared = os.path.join(folder, ID_FILE)
    if os.path.exists(declared):
        system = _folder_system(folder, mtime)
        if system:
            return system
    system = _seen_system(_ids(clip_files)) or _folder_system(folder, mtime)
    if system:
        return system
    return CONTIGUOUS_IDS if os.path.exists(os.path.join(dataset, "yolo", "classes.txt")) else COCO_IDS


def read_boxes(path: str, names: Tuple[str, ...], system: str = CONTIGUOUS_IDS) -> List[Tuple[str, List[float]]]:
    """``[(class name, xyxy)]`` of one label file. Contiguous ids are named by the dataset's `names`; COCO ids by
    ``fleet_contract.classes``. Rows of an unknown class are dropped."""
    out = []
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                parts = line.split()
                if len(parts) < 5 or not parts[0].isdigit():
                    continue
                cls = int(parts[0])
                if system == COCO_IDS:
                    name = class_name(cls, COCO_IDS)
                else:
                    name = names[cls] if cls < len(names) else None
                if name is None:
                    continue
                cx, cy, w, h = (float(v) for v in parts[1:5])
                box = [max(0.0, cx - w / 2), max(0.0, cy - h / 2), min(1.0, cx + w / 2), min(1.0, cy + h / 2)]
                if box[2] > box[0] and box[3] > box[1]:
                    out.append((name, box))
    except (OSError, ValueError):
        return []
    return out


def dataset_tracks(dataset: str, clip_id: str, fps: Optional[float]) -> List[ft.Track]:
    """The clip's YOLO boxes linked into tracks ``t-1``, ``t-2``, ... in order of first appearance; [] if none."""
    files = label_files(dataset, clip_id)
    if not files:
        return []
    names, fps = classes(dataset), fps or 7.0
    systems: Dict[str, str] = {}
    for folder in {os.path.dirname(path) for _, path in files}:
        systems[folder] = id_system(dataset, folder, [p for _, p in files if os.path.dirname(p) == folder])
    frames = [(frame, round(frame / fps, 4), read_boxes(path, names, systems[os.path.dirname(path)]))
              for frame, path in files]
    tracks = ft.tracks_from_weak_labels(frames)
    for n, tr in enumerate(tracks, start=1):
        tr.track_id, tr.source = f"t-{n}", "yolo"
    return tracks


def preload_tracks(dataset: str, clip_id: str, fps: Optional[float],
                   paths: Iterable[str] = ()) -> Tuple[List[ft.Track], Optional[str]]:
    """(tracks, source) a never-saved dataset clip opens with: the box tracker's tracks.json next to its meta or video
    (`paths`) when there is one ("tracker"), else its YOLO label files linked by IoU ("yolo"); ([], None) if none."""
    from .tracks_file import local_tracks  # noqa: PLC0415

    tracks = local_tracks(paths, fps)
    if tracks:
        return tracks, "tracker"
    tracks = dataset_tracks(dataset, clip_id, fps)
    return tracks, ("yolo" if tracks else None)
