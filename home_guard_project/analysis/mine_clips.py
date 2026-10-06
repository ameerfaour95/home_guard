"""
Rank untagged clips by how much they would add to the YOLO training set, and
copy the best ones into a dataset_multi-style folder for Label Studio.

A clip scores for:

* classes the training set is short of (a big COCO model sees them at
  ``--min-conf`` or more; weight grows as the class gets rarer in ``--base``),
* night / infrared frames (the base set has almost none),
* small, far-away people (under 0.5% of the frame),
* clips the box or the owner already marked as false alarms (``false_positive``,
  ``owner_feedback`` kinds): hard-negative candidates.

Clips whose id is already in a tagged batch (``--skip-tagged DIR``, any
``*.mp4`` under it) are skipped.

Usage:
    py -m home_guard_project.analysis.mine_clips OUT_DIR --source DIR [--source DIR ...]
        --base YOLO_DIR [--skip-tagged DIR] [--top 150] [--per-camera 12] [--weights yolo11x.pt]
        [--cache MINE_CACHE.json]
    then: py -m home_guard_project.labeling --dataset-dir OUT_DIR
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import logging
import math
import os
import re
import shutil
from collections import Counter
from typing import Dict, List, Optional, Tuple

from home_guard_project.analysis.detections import COCO_IDS
from home_guard_project.analysis.external_sets import NAMES, is_gray

log = logging.getLogger("analysis.mine_clips")

COCO_TO_OURS = {coco: i for i, coco in enumerate(COCO_IDS)}
_KIND = re.compile(r"_\d{9,}_(?P<kind>[a-z_]+)$")
HARD_NEGATIVE_KINDS = {"false_positive", "fp", "owner_feedback"}


def class_weights(counts: Dict[int, int]) -> Dict[int, float]:
    """Rarer class -> bigger weight: log10(most common / this), capped at 3."""
    top = max(counts.values()) if counts else 1
    return {c: min(3.0, math.log10((top + 10) / (counts.get(c, 0) + 10))) for c in range(len(NAMES))}


def base_counts(base_dir: str) -> Dict[int, int]:
    counts: Counter = Counter()
    for path in glob.glob(os.path.join(base_dir, "labels", "**", "*.txt"), recursive=True):
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    counts[int(line.split()[0])] += 1
    return dict(counts)


def clip_kind(clip_id: str) -> str:
    """``back_door_1790944263_false_positive`` -> ``false_positive`` (the part after the timestamp)."""
    m = _KIND.search(clip_id)
    return m["kind"] if m else ""


def score_clip(summary: dict, weights: Dict[int, float], min_conf: float = 0.5) -> Tuple[float, List[str]]:
    """(score, reasons) from a clip summary {classes: {id: max conf}, night, small_person, kind}."""
    score, why = 0.0, []
    for c, conf in summary["classes"].items():
        c = int(c)
        if conf >= min_conf and weights.get(c, 0) >= 0.5:
            score += weights[c]
            why.append(NAMES[c])
    if summary["night"] >= 0.5:
        score += 2.0
        why.append("night")
    if summary["small_person"]:
        score += 1.0
        why.append("far_person")
    if summary["kind"] in HARD_NEGATIVE_KINDS:
        score += 1.5
        why.append(summary["kind"])
    return round(score, 3), why


def pick(rows: List[dict], top: int, per_camera: int) -> List[dict]:
    """Best-scoring clips first, at most ``per_camera`` from one camera, ``top`` in all."""
    taken: Counter = Counter()
    out = []
    for r in rows:
        if r["score"] <= 0 or taken[r["camera"]] >= per_camera:
            continue
        taken[r["camera"]] += 1
        out.append(r)
        if len(out) == top:
            break
    return out


def summarize(video: str, model, imgsz: int = 1280, conf: float = 0.4, max_frames: int = 15) -> Optional[dict]:
    """One frame a second (at most ``max_frames``): best confidence per class, night share, small people."""
    import cv2

    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 10
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    step = max(1, round(fps))
    idx = list(range(0, n_total or step * max_frames, step))[:max_frames]
    frames = []
    for i in idx:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, img = cap.read()
        if ok:
            frames.append(img)
    cap.release()
    if not frames:
        return None
    best: Dict[int, float] = {}
    small = 0
    for res in model.predict(frames, imgsz=imgsz, conf=conf, classes=list(COCO_IDS), verbose=False):
        for c, s, wh in zip(res.boxes.cls.tolist(), res.boxes.conf.tolist(), res.boxes.xywhn.tolist()):
            ours = COCO_TO_OURS[int(c)]
            best[ours] = max(best.get(ours, 0.0), round(float(s), 3))
            small += ours == 0 and s >= 0.5 and wh[2] * wh[3] < 0.005
    return {"classes": best, "night": round(sum(is_gray(f) for f in frames) / len(frames), 2),
            "small_person": small, "frames": len(frames)}


def find_clips(sources: List[str], skip: set) -> List[dict]:
    """Clips with a matching meta file, as {id, video, root, camera, date}; a clip in two sources counts once."""
    out, seen = [], set()
    for root in sources:
        for video in glob.glob(os.path.join(root, "clips", "*", "*", "*.mp4")):
            cid = os.path.splitext(os.path.basename(video))[0]
            date_dir = os.path.dirname(video)
            cam, date = os.path.basename(os.path.dirname(date_dir)), os.path.basename(date_dir)
            meta = os.path.join(root, "meta", cam, date, cid + ".meta.json")
            if cid in skip or cid in seen or not os.path.exists(meta):
                continue
            seen.add(cid)
            out.append({"id": cid, "video": video, "root": root, "camera": cam, "date": date})
    return out


def copy_clip(clip: dict, out_dir: str) -> None:
    """Clip, meta and any VLM crop into OUT_DIR with the dataset_multi layout."""
    rel = os.path.join(clip["camera"], clip["date"])
    for sub, pattern in (("clips", clip["id"] + ".mp4"), ("meta", clip["id"] + ".meta.json"),
                         ("vlm_crops", clip["id"] + "*")):
        for src in glob.glob(os.path.join(clip["root"], sub, rel, pattern)):
            dst_dir = os.path.join(out_dir, sub, rel)
            os.makedirs(dst_dir, exist_ok=True)
            shutil.copy2(src, dst_dir)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("out_dir")
    p.add_argument("--source", action="append", required=True, help="dataset_multi-style dir (repeatable).")
    p.add_argument("--base", required=True, help="YOLO dir (labels/ under it) whose class counts set the rarity weights.")
    p.add_argument("--skip-tagged", action="append", default=[], help="Dir of already tagged clips (repeatable).")
    p.add_argument("--top", type=int, default=150)
    p.add_argument("--per-camera", type=int, default=12,
                   help="At most this many clips from one camera (a parked truck can fill a batch otherwise).")
    p.add_argument("--weights", default="yolo11x.pt")
    p.add_argument("--imgsz", type=int, default=1280)
    p.add_argument("--cache", help="Per-clip summary cache JSON.")
    a = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    skip = {os.path.splitext(os.path.basename(v))[0]
            for d in a.skip_tagged for v in glob.glob(os.path.join(d, "**", "*.mp4"), recursive=True)}
    clips = find_clips(a.source, skip)
    log.info("%d untagged clips (%d tagged ids skipped)", len(clips), len(skip))
    cache: Dict[str, dict] = {}
    if a.cache and os.path.exists(a.cache):
        with open(a.cache, encoding="utf-8") as f:
            cache = json.load(f)
    todo = [c for c in clips if c["id"] not in cache]
    if todo:
        from ultralytics import YOLO

        model = YOLO(a.weights)
        for i, c in enumerate(todo, 1):
            s = summarize(c["video"], model, a.imgsz)
            if s:
                cache[c["id"]] = s
            if i % 20 == 0:
                log.info("detector: %d / %d clips", i, len(todo))
        if a.cache:
            with open(a.cache, "w", encoding="utf-8") as f:
                json.dump(cache, f)

    weights = class_weights(base_counts(a.base))
    log.info("class weights: %s", {NAMES[c]: round(w, 2) for c, w in weights.items()})
    rows = []
    for c in clips:
        s = cache.get(c["id"])
        if not s:
            continue
        s = {**s, "kind": clip_kind(c["id"])}
        score, why = score_clip(s, weights)
        rows.append({**c, "score": score, "why": "+".join(why), "night": s["night"],
                     "classes": ",".join(f"{NAMES[int(k)]}:{v}" for k, v in sorted(s["classes"].items(), key=lambda kv: int(kv[0])))})
    rows.sort(key=lambda r: -r["score"])
    os.makedirs(a.out_dir, exist_ok=True)
    picked = pick(rows, a.top, a.per_camera)
    for r in picked:
        copy_clip(r, a.out_dir)
    with open(os.path.join(a.out_dir, "mined.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["score", "why", "id", "camera", "date", "night", "classes", "root", "video"])
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in w.fieldnames})
    reasons = Counter(x for r in picked for x in r["why"].split("+") if x)
    log.info("picked %d of %d clips -> %s; reasons %s", len(picked), len(rows), a.out_dir, dict(reasons))


if __name__ == "__main__":
    main()
