"""
Check every frame of a Label Studio export against cached detector boxes and
list the frames YOLO training should treat specially.

Two caches from ``analysis.detections``: ``--big`` (a large model, e.g.
yolo11x at 1280, as a second opinion) and ``--deployed`` (the model the box
runs, yolo11s at 640, person only is enough).

* **hard negative**: the deployed model sees a person (conf >= 0.4) where no
  human person box is, and the big model agrees nobody is there.  These are
  the false alarms the fine-tune must unlearn, so the dataset builder always
  keeps them and repeats them in train.
* **excluded** (disputed, left out of training and listed for a human):
  - ``missed``: the big model sees an object (conf >= 0.6) that no human box
    covers.  Training on that frame would teach the model to ignore it.  (At
    0.5 too many were shadows and plants the annotators rightly skipped.)
  - ``held_unsupported``: past a track's last keyframe Label Studio holds the
    box still; when the big model sees nothing under a held person / animal
    box, that one has most likely left.  Vehicles are not checked: a parked
    car cut off by the frame edge is held correctly and often undetected.

Writes ``<export>.review.json``::

    {"hard_negatives": [{"task": id, "clip": c, "frames": [native, ...]}],
     "excluded": [{"task": id, "clip": c, "frames": {native: reason}}],
     "summary": {...}}

Usage:
    py -m home_guard_project.analysis.review_frames EXPORT.json --big BIG.json [--deployed SMALL.json]
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

from . import detections as dt
from .utils.ls_convert import box_at
from .vehicle_classes import iou

log = logging.getLogger("analysis.review_frames")

PERSON = 0
VEHICLE_LABELS = {"car", "truck", "bus", "motorcycle", "bicycle"}
MISSED_CONF = 0.6
MISSED_MIN_AREA = 0.5          # percent^2 of the frame (1% x 0.5%)
COVER_IOU = 0.2                # a human box this close explains a detection
HELD_SUPPORT_IOU = 0.1
DEPLOYED_PERSON_CONF = 0.4
BIG_PERSON_CONF = 0.45
PERSON_MATCH_IOU = 0.3


def _tracks(ann: Dict[str, Any]) -> List[Tuple[str, List[Dict[str, Any]], int]]:
    """[(label, sequence, last keyframe frame)] for the annotation's boxes."""
    out = []
    for r in ann.get("result", []):
        if r.get("type") != "videorectangle":
            continue
        seq = r.get("value", {}).get("sequence", [])
        if not seq:
            continue
        label = (r["value"].get("labels") or [""])[0]
        out.append((label, seq, max(int(k.get("frame", 0)) for k in seq)))
    return out


def _human_boxes(tracks, f: int, ls_fps: float) -> List[Tuple[str, Tuple[float, ...], bool]]:
    """[(label, xywh, held)] visible at LS frame f; held = past the track's last keyframe."""
    out = []
    for label, seq, last in tracks:
        b = box_at(seq, f, ls_fps)
        if b and b["width"] > 0 and b["height"] > 0:
            out.append((label, (b["x"], b["y"], b["width"], b["height"]), f > last))
    return out


def review_task(
    task: Dict[str, Any], ann: Dict[str, Any], big: Dict[str, Any], deployed: Optional[Dict[str, Any]],
) -> Tuple[Dict[int, str], List[int], Counter]:
    """({native: exclusion reason}, [hard-negative native frames], counters)."""
    ls_fps = float(big.get("ls_fps") or task.get("data", {}).get("fps") or 7.0)
    native_fps = float(big.get("native_fps") or ls_fps)
    tracks = _tracks(ann)
    excluded: Dict[int, str] = {}
    stats: Counter = Counter()

    for n, dets in big["frames"].items():
        human = _human_boxes(tracks, dt.native_to_ls(n, native_fps, ls_fps), ls_fps)
        for cls, conf, *db in dets:
            if conf < MISSED_CONF or db[2] * db[3] < MISSED_MIN_AREA:
                continue
            if not any(iou(tuple(db), hb) >= COVER_IOU for _, hb, _ in human):
                excluded.setdefault(n, f"missed:{cls}")
                stats[f"missed_frames_cls{cls}"] += 1
                break
        for label, hb, held in human:
            if held and label not in VEHICLE_LABELS and not any(iou(hb, tuple(d[2:])) >= HELD_SUPPORT_IOU for d in dets):
                excluded.setdefault(n, f"held_unsupported:{label}")
                stats["held_unsupported_frames"] += 1
                break

    hard: List[int] = []
    if deployed is not None:
        gap = max(1, math.ceil(native_fps / ls_fps))
        for n, dets in deployed["frames"].items():
            persons = [d for d in dets if d[0] == PERSON and d[1] >= DEPLOYED_PERSON_CONF]
            if not persons:
                continue
            human = [hb for lab, hb, _ in _human_boxes(tracks, dt.native_to_ls(n, native_fps, ls_fps), ls_fps)
                     if lab == "person"]
            unexplained = [d for d in persons if not any(iou(tuple(d[2:]), hb) >= PERSON_MATCH_IOU for hb in human)]
            if not unexplained:
                continue
            m = dt.nearest(big["frames"], n, gap)
            if m is None:
                stats["deployed_person_no_second_opinion"] += 1
                continue
            big_persons = [d for d in big["frames"][m] if d[0] == PERSON and d[1] >= BIG_PERSON_CONF]
            if any(iou(tuple(u[2:]), tuple(b[2:])) >= PERSON_MATCH_IOU for u in unexplained for b in big_persons):
                excluded.setdefault(n, "missed:0")
                stats["deployed_person_confirmed_by_big"] += 1
            elif n not in excluded:
                hard.append(n)
    stats["excluded_frames"] = len(excluded)
    stats["hard_negative_frames"] = len(hard)
    return excluded, sorted(hard), stats


def review_export(
    tasks: List[Dict[str, Any]], big: Dict[str, Dict[str, Any]], deployed: Optional[Dict[str, Dict[str, Any]]],
) -> Dict[str, Any]:
    hard_out, excl_out = [], []
    summary: Counter = Counter()
    for task, ann in dt.kept_tasks(tasks):
        tid = str(task.get("id"))
        if tid not in big:
            summary["tasks_without_detections"] += 1
            continue
        excluded, hard, stats = review_task(task, ann, big[tid], (deployed or {}).get(tid) if deployed else None)
        summary.update(stats)
        summary["tasks"] += 1
        summary["frames_checked"] += len(big[tid]["frames"])
        if hard:
            hard_out.append({"task": task["id"], "clip": dt.clip_id(task), "frames": hard})
        if excluded:
            excl_out.append({"task": task["id"], "clip": dt.clip_id(task),
                             "frames": {str(n): r for n, r in sorted(excluded.items())}})
    summary["tasks_with_hard_negatives"] = len(hard_out)
    summary["tasks_with_excluded_frames"] = len(excl_out)
    return {"hard_negatives": hard_out, "excluded": excl_out, "summary": dict(sorted(summary.items()))}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("export_path")
    p.add_argument("--big", required=True, help="Detections cache of the large model.")
    p.add_argument("--deployed", help="Detections cache of the deployed model (for hard negatives).")
    a = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    with open(a.export_path, "r", encoding="utf-8") as f:
        tasks = json.load(f)
    out = review_export(tasks, dt.load(a.big), dt.load(a.deployed) if a.deployed else None)
    path = os.path.splitext(a.export_path)[0] + ".review.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    log.info("%s -> %s", out["summary"], path)


if __name__ == "__main__":
    main()
