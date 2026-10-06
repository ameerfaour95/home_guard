"""
Split the human "car" tracks of a Label Studio export into car / truck / bus /
motorcycle.

Annotators drew every vehicle as ``car``.  A large COCO detector (its boxes
cached by ``analysis.detections``, e.g. yolo11x at 1280) looks at every frame
where a vehicle track has a box; each vehicle detection that overlaps the human
box (IoU >= 0.45) votes for its class, weighted by confidence x IoU.  A ``car``
track takes the winning class when it won at least ``--min-share`` of the vote
over at least ``--min-frames`` frames.  Tracks an annotator already labelled
truck / bus / motorcycle are never changed, only listed for review when the
detector disagrees.

Writes ``<export>.vehicles.json`` (same format, labels fixed) and
``<export>.vehicles.csv`` (one row per vehicle track) next to the export.  Run
the normal analysis on the fixed export afterwards.

Usage:
    py -m home_guard_project.analysis.vehicle_classes EXPORT.json --detections CACHE.json
        [--min-share 0.6] [--min-frames 3]
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
from collections import Counter
from typing import Any, Dict, List, Tuple

from . import detections as dt
from .utils.ls_convert import box_at

log = logging.getLogger("analysis.vehicle_classes")

VEHICLES = ("car", "truck", "bus", "motorcycle")
COCO_VEHICLE_IDS = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
MATCH_IOU = 0.45


def iou(a: Tuple[float, ...], b: Tuple[float, ...]) -> float:
    """IoU of two (x, y, w, h) boxes."""
    iw = max(0.0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]))
    ih = max(0.0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
    inter = iw * ih
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


def vote_task(
    task: Dict[str, Any], ann: Dict[str, Any], cached: Dict[str, Any],
) -> Tuple[Dict[str, Counter], Counter]:
    """({result_id: Counter(class -> vote weight)}, {result_id: frames matched}) for the task's vehicle tracks."""
    ls_fps = float(cached.get("ls_fps") or task.get("data", {}).get("fps") or 7.0)
    native_fps = float(cached.get("native_fps") or ls_fps)
    tracks = [
        r for r in ann.get("result", [])
        if r.get("type") == "videorectangle"
        and (r.get("value", {}).get("labels") or [""])[0] in VEHICLES
    ]
    votes: Dict[str, Counter] = {r["id"]: Counter() for r in tracks}
    frames_hit: Counter = Counter()
    frames = cached["frames"]
    for n, dets in frames.items():
        f = dt.native_to_ls(n, native_fps, ls_fps)
        for r in tracks:
            b = box_at(r["value"].get("sequence", []), f, ls_fps)
            if not b or b["width"] <= 0 or b["height"] <= 0:
                continue
            hb = (b["x"], b["y"], b["width"], b["height"])
            hit = False
            for cls, conf, *db in dets:
                if cls not in COCO_VEHICLE_IDS:
                    continue
                o = iou(hb, tuple(db))
                if o >= MATCH_IOU:
                    votes[r["id"]][COCO_VEHICLE_IDS[cls]] += conf * o
                    hit = True
            frames_hit[r["id"]] += hit
    return votes, frames_hit


def fix_tasks(
    tasks: List[Dict[str, Any]], cache: Dict[str, Dict[str, Any]],
    min_share: float = 0.6, min_frames: int = 3,
) -> List[Dict[str, Any]]:
    """Relabel confident car tracks in place; return one row per vehicle track."""
    rows: List[Dict[str, Any]] = []
    for task, ann in dt.kept_tasks(tasks):
        cached = cache.get(str(task.get("id")))
        if cached is None:
            continue
        votes, frames_hit = vote_task(task, ann, cached)
        for r in ann.get("result", []):
            if r.get("id") not in votes:
                continue
            v = votes[r["id"]]
            frames = frames_hit[r["id"]]
            total = sum(v.values())
            human = r["value"]["labels"][0]
            top, top_w = (v.most_common(1)[0] if v else ("", 0.0))
            share = top_w / total if total else 0.0
            confident = frames >= min_frames and share >= min_share
            if human == "car" and confident and top != "car":
                r["value"]["labels"] = [top]
                decision = "changed"
            elif human == "car" and confident:
                decision = "kept"
            elif human == "car":
                decision = "review: detector unsure" if frames else "review: detector never matched"
            elif confident and top != human:
                decision = f"review: annotator said {human}, detector says {top}"
            else:
                decision = "kept"
            rows.append({
                "task_id": task.get("id"), "clip": dt.clip_id(task),
                "track_id": r.get("id"), "human_label": human, "final_label": r["value"]["labels"][0],
                "detector_class": top, "share": round(share, 3), "matched_frames": frames,
                "votes": json.dumps({k: round(x, 2) for k, x in v.most_common()}), "decision": decision,
            })
    return rows


def fix_export(
    export_path: str, detections_path: str, min_share: float = 0.6, min_frames: int = 3,
) -> Tuple[str, str]:
    with open(export_path, "r", encoding="utf-8") as f:
        tasks = json.load(f)
    rows = fix_tasks(tasks, dt.load(detections_path), min_share, min_frames)

    base = os.path.splitext(export_path)[0]
    out_json, out_csv = base + ".vehicles.json", base + ".vehicles.csv"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(tasks, f, ensure_ascii=False)
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        cols = ["task_id", "clip", "track_id", "human_label", "final_label",
                "detector_class", "share", "matched_frames", "votes", "decision"]
        wr = csv.DictWriter(f, fieldnames=cols)
        wr.writeheader()
        wr.writerows(rows)
    changed = Counter(r["final_label"] for r in rows if r["decision"] == "changed")
    review = sum(1 for r in rows if r["decision"].startswith("review"))
    log.info("%d vehicle tracks: changed %s, %d to review -> %s", len(rows), dict(changed), review, out_json)
    return out_json, out_csv


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("export_path")
    p.add_argument("--detections", required=True, help="Cache from analysis.detections (large model).")
    p.add_argument("--min-share", type=float, default=0.6)
    p.add_argument("--min-frames", type=int, default=3)
    a = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    fix_export(a.export_path, a.detections, a.min_share, a.min_frames)


if __name__ == "__main__":
    main()
