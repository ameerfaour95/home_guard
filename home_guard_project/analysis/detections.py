"""
Run a COCO detector over the clips of a Label Studio export and cache the boxes.

The cache is what ``vehicle_classes`` and ``review_frames`` read, so the slow
detector pass happens once per export and model.  Format (JSON)::

    {"weights": "yolo11x.pt", "imgsz": 1280,
     "tasks": {"<task id>": {"clip": "<clip id>", "native_fps": 7.0, "ls_fps": 7.0,
                             "n_frames": 61,
                             "frames": {"<native index>": [[coco_id, conf, x, y, w, h], ...]}}}}

Boxes are in Label Studio percent units (x, y = top-left).  ``--every N``
keeps every Nth native frame (1 = all).

Usage:
    py -m home_guard_project.analysis.detections EXPORT.json --dataset-dir DIR --out CACHE.json
        [--weights yolo11x.pt] [--imgsz 1280] [--conf 0.15] [--every 1]
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from typing import Any, Dict, Iterable, List, Optional, Tuple

log = logging.getLogger("analysis.detections")

COCO_IDS = (0, 1, 2, 3, 5, 7, 14, 15, 16)


def latest_annotation(task: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """The newest non-cancelled annotation (a re-tagged task keeps the old ones too)."""
    live = [a for a in task.get("annotations", []) if not a.get("was_cancelled")]
    if not live:
        return None
    return max(live, key=lambda a: a.get("updated_at") or a.get("created_at") or "")


def description(ann: Dict[str, Any]) -> str:
    for r in ann.get("result", []):
        if r.get("type") == "textarea" and r.get("from_name") == "vlm_description":
            return (r.get("value", {}).get("text") or [""])[0]
    return ""


def kept_tasks(tasks: Iterable[Dict[str, Any]]) -> Iterable[Tuple[Dict[str, Any], Dict[str, Any]]]:
    """(task, latest annotation) for every tagged task not marked ``[delete]``."""
    for task in tasks:
        ann = latest_annotation(task)
        if ann is not None and description(ann).strip().lower() != "[delete]":
            yield task, ann


def clip_id(task: Dict[str, Any]) -> str:
    return os.path.basename(task.get("data", {}).get("meta_path", "")).replace(".meta.json", "")


def clip_path(dataset_dir: str, meta_path: str) -> str:
    rel = meta_path.replace("\\", "/")
    if rel.startswith("meta/"):
        rel = "clips/" + rel[len("meta/"):]
    return os.path.join(dataset_dir, rel[: -len(".meta.json")] + ".mp4")


def ls_to_native(ls_frame: int, native_fps: float, ls_fps: float) -> int:
    """Native frame shown at Label Studio frame ``ls_frame`` (1-based, at ``ls_fps``)."""
    return int(round((ls_frame - 1) * native_fps / ls_fps))


def native_to_ls(native: int, native_fps: float, ls_fps: float) -> int:
    """Label Studio frame (1-based) for native frame ``native``, as labeling/utils/yolo.py maps it."""
    return int(round(native * ls_fps / native_fps)) + 1


def nearest(frames: Dict[int, Any], n: int, max_gap: int) -> Optional[int]:
    """The cached native index closest to ``n`` within ``max_gap`` frames, or None."""
    if n in frames:
        return n
    best = min(frames, key=lambda k: abs(k - n), default=None)
    return best if best is not None and abs(best - n) <= max_gap else None


def load(path: str) -> Dict[str, Dict[str, Any]]:
    """{task_id: {..., "frames": {native int: [[cls, conf, x, y, w, h], ...]}}}"""
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    tasks = {}
    for tid, t in raw["tasks"].items():
        t = dict(t)
        t["frames"] = {int(n): [tuple(d) for d in dets] for n, dets in t["frames"].items()}
        tasks[str(tid)] = t
    return tasks


def save(path: str, tasks: Dict[str, Dict[str, Any]], weights: str, imgsz: int) -> None:
    out = {"weights": weights, "imgsz": imgsz, "tasks": {
        str(tid): {**{k: v for k, v in t.items() if k != "frames"},
                   "frames": {str(n): [list(d) for d in dets] for n, dets in sorted(t["frames"].items())}}
        for tid, t in tasks.items()}}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f)


def detect_export(
    export_path: str, dataset_dir: str, weights: str = "yolo11x.pt", imgsz: int = 1280,
    conf: float = 0.15, every: int = 1, classes: Iterable[int] = COCO_IDS,
) -> Dict[str, Dict[str, Any]]:
    import cv2
    from ultralytics import YOLO

    model = YOLO(weights)
    with open(export_path, "r", encoding="utf-8") as f:
        tasks = json.load(f)
    out: Dict[str, Dict[str, Any]] = {}
    for task, _ in kept_tasks(tasks):
        data = task.get("data", {})
        cap = cv2.VideoCapture(clip_path(dataset_dir, data.get("meta_path", "")))
        native_fps = cap.get(cv2.CAP_PROP_FPS) or float(data.get("fps") or 7.0)
        frames: List[Any] = []
        while True:
            ok, img = cap.read()
            if not ok:
                break
            frames.append(img)
        cap.release()
        if not frames:
            log.warning("Task %s: no frames decoded", task.get("id"))
            continue
        h, w = frames[0].shape[:2]
        idx = list(range(0, len(frames), max(1, every)))
        found: Dict[int, List[Tuple]] = {}
        for i in range(0, len(idx), 16):
            chunk = idx[i : i + 16]
            res = model.predict([frames[n] for n in chunk], imgsz=imgsz, conf=conf,
                                classes=list(classes), verbose=False)
            for n, r in zip(chunk, res):
                found[n] = [
                    (int(c), round(float(p), 4), x1 / w * 100, y1 / h * 100, (x2 - x1) / w * 100, (y2 - y1) / h * 100)
                    for (x1, y1, x2, y2), c, p in zip(r.boxes.xyxy.tolist(), r.boxes.cls.tolist(), r.boxes.conf.tolist())
                ]
        out[str(task["id"])] = {"clip": clip_id(task), "native_fps": native_fps,
                                "ls_fps": float(data.get("fps") or native_fps), "n_frames": len(frames),
                                "frames": found}
        log.info("Task %s: %d frames", task["id"], len(found))
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("export_path")
    p.add_argument("--dataset-dir", required=True, help="Folder with clips/ and meta/.")
    p.add_argument("--out", required=True)
    p.add_argument("--weights", default="yolo11x.pt")
    p.add_argument("--imgsz", type=int, default=1280)
    p.add_argument("--conf", type=float, default=0.15)
    p.add_argument("--every", type=int, default=1)
    a = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    tasks = detect_export(a.export_path, a.dataset_dir, a.weights, a.imgsz, a.conf, a.every)
    save(a.out, tasks, a.weights, a.imgsz)


if __name__ == "__main__":
    main()
