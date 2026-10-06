"""
Bring outside YOLO datasets (Roboflow Universe exports) into our 9 classes, as
new sources of the unified YOLO dir (``home_guard_data/dataset/yolo``).

Per outside set:

1. Class names are mapped to ours (``CLASS_MAP``).  Things we do not detect
   (deer, raccoon, garbage bin...) are *ignored*: their boxes are dropped and
   the frame stays, so the model learns them as background.  Generic names that
   may hide one of our classes ("Animal", "vehicle") are *ambiguous*: the frame
   is left out.
2. Roboflow's augmented copies (``<stem>_jpg.rf.<hash>.jpg``) are dropped, one
   file per original image.
3. A big COCO model (YOLO11x at 1280, cached) checks every frame.  A detection
   of one of our classes at ``--min-conf`` or more that no box covers (IoU 0.3;
   car/bus/truck, bicycle/motorcycle and cat/dog count as one family; a box of
   an ignored class also covers it) means someone left an object untagged, and
   the frame is left out.  Same rule as ``review_frames``: a missing box teaches
   the model that the object is background, so a disputed frame is dropped,
   never patched with model boxes.

Kept frames go to ``images/ext_<set>/`` + ``labels/ext_<set>/`` and are listed
in ``external_frames.txt``.  ``--eval-set`` keeps one whole house out of that
list and writes it to ``newhouse_eval.txt``: a "house never seen" check for
later.  No train/val split is made here, and nothing outside ``ext_*`` and
these files is touched.

Usage:
    py -m home_guard_project.analysis.external_sets YOLO_DIR --set DIR [--set DIR ...]
        [--eval-set NAME] [--weights yolo11x.pt] [--cache DETECTIONS.json] [--batch 8]
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import re
from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import yaml

from home_guard_project.analysis.detections import COCO_IDS

log = logging.getLogger("analysis.external_sets")

NAMES = ["person", "bicycle", "car", "motorcycle", "bus", "truck", "bird", "cat", "dog"]
COCO_TO_OURS = {coco: i for i, coco in enumerate(COCO_IDS)}

# lower-cased outside class name -> our class name
CLASS_MAP = {
    "person": "person", "people": "person", "human": "person", "humans": "person",
    "pedestrian": "person", "man": "person", "woman": "person", "intruder": "person",
    "bicycle": "bicycle", "bike": "bicycle", "bike/bicycle": "bicycle", "cyclist": "bicycle",
    "car": "car", "honda civic": "car", "honda crv": "car", "sedan": "car", "suv": "car",
    "motorcycle": "motorcycle", "motorbike": "motorcycle", "motor": "motorcycle",
    "bus": "bus", "truck": "truck", "pickup": "truck",
    "bird": "bird", "chicken": "bird",
    "cat": "cat", "dog": "dog",
}
AMBIGUOUS = {"animal", "animals", "vehicle", "vehicles", "pet", "object", "objects"}
FAMILIES = [{1, 3}, {2, 4, 5}, {7, 8}]

_RF_STEM = re.compile(r"^(?P<stem>.+?)_(jpg|jpeg|png)\.rf\.[0-9a-f]+$", re.I)


def map_names(names: Sequence[str]) -> Dict[int, object]:
    """outside class id -> our id, ``"ignore"`` or ``"ambiguous"``."""
    out: Dict[int, object] = {}
    for i, name in enumerate(names):
        key = str(name).strip().lower()
        if key in CLASS_MAP:
            out[i] = NAMES.index(CLASS_MAP[key])
        elif key in AMBIGUOUS:
            out[i] = "ambiguous"
        else:
            out[i] = "ignore"
    return out


def original_stem(path: str) -> str:
    stem = os.path.splitext(os.path.basename(path))[0]
    m = _RF_STEM.match(stem)
    return m["stem"] if m else stem


def unique_images(paths: Iterable[str]) -> List[str]:
    """One image per original (Roboflow writes augmented copies with the same stem)."""
    seen: Dict[str, str] = {}
    for p in sorted(paths):
        seen.setdefault(original_stem(p), p)
    return sorted(seen.values())


def read_yolo(path: str) -> List[Tuple[int, float, float, float, float]]:
    """YOLO label rows as (cls, cx, cy, w, h); polygons become their bounding box."""
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path, encoding="utf-8") as f:
        for line in f:
            v = line.split()
            if len(v) < 5:
                continue
            c, xs = int(v[0]), [float(t) for t in v[1:]]
            if len(xs) == 4:
                rows.append((c, *xs))
            else:
                x, y = xs[0::2], xs[1::2]
                rows.append((c, (min(x) + max(x)) / 2, (min(y) + max(y)) / 2, max(x) - min(x), max(y) - min(y)))
    return rows


def label_path(image: str) -> str:
    """Ultralytics' rule: the last ``/images/`` in the path becomes ``/labels/``, extension ``.txt``."""
    norm = image.replace("\\", "/")
    head, sep, tail = norm.rpartition("/images/")
    return os.path.normpath(os.path.splitext(head + "/labels/" + tail if sep else norm)[0] + ".txt")


def _xyxy(cx: float, cy: float, w: float, h: float) -> Tuple[float, float, float, float]:
    return cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2


def iou(a: Sequence[float], b: Sequence[float]) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def _same_family(a: int, b: int) -> bool:
    return a == b or any(a in f and b in f for f in FAMILIES)


def check_frame(
    rows: Sequence[Tuple[int, float, float, float, float]], mapping: Dict[int, object],
    dets: Sequence[Sequence[float]], min_conf: float = 0.6, min_iou: float = 0.3,
) -> Tuple[Optional[str], List[Tuple[int, float, float, float, float]]]:
    """(reason the frame is left out or None, our label rows).

    ``dets`` rows are [our_class, conf, x1, y1, x2, y2] in 0..1 image units.
    """
    ours, ignored = [], []
    for c, cx, cy, w, h in rows:
        m = mapping.get(c, "ignore")
        if m == "ambiguous":
            return "ambiguous_class", []
        if m == "ignore":
            ignored.append(_xyxy(cx, cy, w, h))
        else:
            ours.append((int(m), cx, cy, w, h))
    for d in dets:
        cls, conf, box = int(d[0]), d[1], d[2:6]
        if conf < min_conf:
            continue
        covered = any(_same_family(cls, c) and iou(box, _xyxy(cx, cy, w, h)) >= min_iou
                      for c, cx, cy, w, h in ours)
        covered = covered or any(iou(box, b) >= min_iou for b in ignored)
        if not covered:
            return f"untagged_{NAMES[cls]}", ours
    return None, ours


def is_gray(img) -> bool:
    """True for infrared / black-and-white frames (no colour difference between channels)."""
    import numpy as np

    small = img[::4, ::4].astype(np.int16)
    b, g, r = small[..., 0], small[..., 1], small[..., 2]
    return float(np.abs(b - g).mean() + np.abs(g - r).mean()) < 4.0


def image_key(set_dir: str, path: str) -> str:
    """Cache key that survives moving the sets: ``<set>/<split>/images/<file>``."""
    return os.path.relpath(path, os.path.dirname(os.path.normpath(set_dir))).replace("\\", "/")


def detect(images: Dict[str, str], weights: str, imgsz: int = 1280, conf: float = 0.25,
           cache: Optional[Dict[str, list]] = None, cache_path: Optional[str] = None,
           batch: int = 8) -> Dict[str, list]:
    """{key: [[our_class, conf, x1, y1, x2, y2], ...]} for ``images`` ({key: path}) not cached yet.

    The cache file is rewritten every ~500 images, so a crash (e.g. GPU out of memory) loses little.
    """
    cache = cache if cache is not None else {}
    todo = [k for k in sorted(images) if k not in cache]
    if not todo:
        return cache
    from ultralytics import YOLO

    model = YOLO(weights)
    for i in range(0, len(todo), batch):
        keys = todo[i:i + batch]
        for key, res in zip(keys, model.predict([images[k] for k in keys], imgsz=imgsz, conf=conf,
                                                classes=list(COCO_IDS), verbose=False)):
            boxes = res.boxes
            cache[key] = [[COCO_TO_OURS[int(c)], round(float(s), 3), *[round(float(v), 4) for v in xy]]
                          for c, s, xy in zip(boxes.cls.tolist(), boxes.conf.tolist(), boxes.xyxyn.tolist())]
        done = min(i + batch, len(todo))
        if cache_path and (done % 496 < batch or done == len(todo)):
            save_cache(cache, cache_path)
            log.info("detector: %d / %d", done, len(todo))
    return cache


def save_cache(cache: Dict[str, list], path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(cache, f)
    os.replace(tmp, path)


def set_images(set_dir: str) -> List[str]:
    """One file per original image across the set's splits."""
    return unique_images(p for ext in ("jpg", "jpeg", "png")
                         for p in glob.glob(os.path.join(set_dir, "*", "images", f"*.{ext}")))


def import_set(set_dir: str, yolo_dir: str, dets: Dict[str, list], min_conf: float = 0.6,
               report: Optional[dict] = None) -> List[str]:
    """Write one outside set's kept frames as source ``ext_<name>`` of a unified YOLO dir.

    Returns the frames as ``images/ext_<name>/<file>.jpg`` (relative to ``yolo_dir``).
    """
    import cv2

    name = os.path.basename(os.path.normpath(set_dir))
    with open(os.path.join(set_dir, "data.yaml"), encoding="utf-8") as f:
        names = yaml.safe_load(f)["names"]
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names)]
    mapping = map_names(names)
    images = set_images(set_dir)
    src_name = f"ext_{name}"
    img_dir = os.path.join(yolo_dir, "images", src_name)
    lbl_dir = os.path.join(yolo_dir, "labels", src_name)
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lbl_dir, exist_ok=True)
    rep = {"images": len(images), "kept": 0, "left_out": Counter(), "boxes": Counter(),
           "background": 0, "gray": 0, "mapping": {str(n): (NAMES[m] if isinstance(m, int) else m)
                                                   for n, m in zip(names, mapping.values())}}
    kept = []
    for src in images:
        key = image_key(set_dir, src)
        if key not in dets:
            rep["left_out"]["not_checked"] += 1
            continue
        reason, rows = check_frame(read_yolo(label_path(src)), mapping, dets[key], min_conf)
        if reason:
            rep["left_out"][reason] += 1
            continue
        img = cv2.imread(src)
        if img is None:
            rep["left_out"]["unreadable"] += 1
            continue
        stem = original_stem(src)
        cv2.imwrite(os.path.join(img_dir, stem + ".jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        with open(os.path.join(lbl_dir, stem + ".txt"), "w", encoding="utf-8") as f:
            f.write("".join(f"{c} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n" for c, cx, cy, w, h in rows))
        for r in rows:
            rep["boxes"][NAMES[r[0]]] += 1
        rep["background"] += not rows
        rep["gray"] += is_gray(img)
        rep["kept"] += 1
        kept.append(f"images/{src_name}/{stem}.jpg")
    if report is not None:
        report[name] = rep
    return kept


def build(yolo_dir: str, set_dirs: List[str], dets: Dict[str, list], eval_set: Optional[str] = None,
          min_conf: float = 0.6) -> dict:
    """Import every set; list the frames in ``external_frames.txt`` (``eval_set`` in ``newhouse_eval.txt``).

    No train/val split is made here; the eval list is a house no training frame comes from.
    """
    report: dict = {"sets": {}}
    frames: List[str] = []
    newhouse: List[str] = []
    for d in set_dirs:
        kept = import_set(d, yolo_dir, dets, min_conf, report["sets"])
        (newhouse if os.path.basename(os.path.normpath(d)) == eval_set else frames).extend(kept)
    for name, items in (("external_frames.txt", frames), ("newhouse_eval.txt", newhouse)):
        if items:
            with open(os.path.join(yolo_dir, name), "w", encoding="utf-8") as f:
                f.write("\n".join(items) + "\n")
    report["external_frames"], report["newhouse_eval"] = len(frames), len(newhouse)
    with open(os.path.join(yolo_dir, "external_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=1, default=lambda o: dict(o))
    return report


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("yolo_dir", help="Unified YOLO dir (images/<source>/, labels/<source>/).")
    p.add_argument("--set", action="append", default=[], help="Outside YOLO export dir (repeatable).")
    p.add_argument("--eval-set", help="Name of one outside set kept out of training, listed in newhouse_eval.txt.")
    p.add_argument("--min-conf", type=float, default=0.6, help="Detector confidence that counts as an untagged object.")
    p.add_argument("--weights", default="yolo11x.pt")
    p.add_argument("--imgsz", type=int, default=1280)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--cache", help="Detector cache JSON (read if present, then updated).")
    a = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cache: Dict[str, list] = {}
    if a.cache and os.path.exists(a.cache):
        with open(a.cache, encoding="utf-8") as f:
            cache = json.load(f)
    images = {image_key(d, p): p for d in a.set for p in set_images(d)}
    cache = detect(images, a.weights, a.imgsz, cache=cache, cache_path=a.cache, batch=a.batch)
    rep = build(a.yolo_dir, a.set, cache, a.eval_set, a.min_conf)
    for name, r in rep["sets"].items():
        log.info("%s: kept %d of %d, left out %s, boxes %s", name, r["kept"], r["images"],
                 dict(r["left_out"]), dict(r["boxes"]))
    log.info("external frames %d, new-house eval %d", rep["external_frames"], rep["newhouse_eval"])


if __name__ == "__main__":
    main()
