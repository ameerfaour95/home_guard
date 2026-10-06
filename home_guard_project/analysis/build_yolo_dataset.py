"""
Assemble an Ultralytics YOLO dataset from analysis outputs + the clips.

Each analysis output dir holds ``yolo/labels/<camera>/<date>/<clip>_f<native>.txt``
(one file per native frame, empty = human-checked negative).  This tool decodes
the matching frames from ``<dataset-dir>/clips/<camera>/<date>/<clip>.mp4``,
keeps about ``--fps`` frames a second per clip, and splits train / val by clip
(never frames of one clip on both sides).

``--review FILE.review.json`` (from ``analysis.review_frames``, repeatable):

* ``hard_negatives``: frames where the deployed detector saw a person that
  annotators say is not there.  Always kept, and repeated
  ``--hard-negative-repeat`` times in ``train.txt`` so the fine-tuned model
  learns not to fire on them.
* ``excluded``: disputed frames (an object nobody tagged, a box held after its
  object left).  Never written.

Usage:
    py -m home_guard_project.analysis.build_yolo_dataset OUT_DIR \\
        --source ANALYSIS_OUT=DATASET_DIR [--source ...] \\
        [--review X.review.json ...] [--fps 3] [--val-share 0.15]
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import random
import re
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

import yaml

log = logging.getLogger("analysis.build_yolo_dataset")

_FRAME_RE = re.compile(r"^(?P<clip>.+)_f(?P<n>\d{4,})\.txt$")


def _collect(analysis_dir: str) -> Dict[Tuple[str, str, str], Dict[int, str]]:
    """{(camera, date, clip): {native_frame: label_path}}"""
    out: Dict[Tuple[str, str, str], Dict[int, str]] = defaultdict(dict)
    root = os.path.join(analysis_dir, "yolo", "labels")
    for path in glob.glob(os.path.join(root, "*", "*", "*.txt")):
        m = _FRAME_RE.match(os.path.basename(path))
        if not m:
            continue
        date_dir = os.path.dirname(path)
        cam, date = os.path.basename(os.path.dirname(date_dir)), os.path.basename(date_dir)
        out[(cam, date, m["clip"])][int(m["n"])] = path
    return out


def build(
    out_dir: str, sources: List[Tuple[str, str]], hard_negatives: Dict[str, Set[int]],
    excluded: Optional[Dict[str, Set[int]]] = None, fps: float = 3.0, val_share: float = 0.15, hn_repeat: int = 3, seed: int = 0,
) -> None:
    import cv2

    excluded = excluded or {}
    names = None
    clips = []
    for analysis_dir, dataset_dir in sources:
        dy = os.path.join(analysis_dir, "yolo", "data.yaml")
        with open(dy, encoding="utf-8") as f:
            these = yaml.safe_load(f)["names"]
        if names is not None and these != names:
            raise SystemExit(f"class names differ between sources: {dy}")
        names = these
        for key, frames in _collect(analysis_dir).items():
            clips.append((key, frames, dataset_dir, os.path.basename(os.path.normpath(analysis_dir))))

    rng = random.Random(seed)
    by_source: Dict[str, list] = defaultdict(list)
    for c in clips:
        by_source[c[3]].append(c)
    val_ids: Set[str] = set()
    for src, items in by_source.items():
        items = sorted(items, key=lambda c: c[0][2])
        rng.shuffle(items)
        val_ids.update(c[0][2] for c in items[: max(1, round(len(items) * val_share))])

    lists: Dict[str, List[str]] = {"train": [], "val": []}
    stats = defaultdict(int)
    for (cam, date, clip), frames, dataset_dir, src in clips:
        split = "val" if clip in val_ids else "train"
        video = os.path.join(dataset_dir, "clips", cam, date, f"{clip}.mp4")
        cap = cv2.VideoCapture(video)
        native_fps = cap.get(cv2.CAP_PROP_FPS) or fps
        step = max(1, round(native_fps / fps))
        hn = hard_negatives.get(clip, set())
        skip = excluded.get(clip, set())
        hn = hn - skip
        sampled = {n for n in frames if n % step == 0}
        stats["excluded_frames"] += len(sampled & skip)
        wanted = (sampled - skip) | (hn & set(frames))
        img_dir = os.path.join(out_dir, "images", split, cam)
        lbl_dir = os.path.join(out_dir, "labels", split, cam)
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(lbl_dir, exist_ok=True)
        n = 0
        while wanted:
            ok, img = cap.read()
            if not ok:
                break
            if n in wanted:
                wanted.discard(n)
                stem = f"{clip}_f{n:04d}"
                img_path = os.path.join(img_dir, stem + ".jpg")
                cv2.imwrite(img_path, img, [cv2.IMWRITE_JPEG_QUALITY, 95])
                with open(frames[n], encoding="utf-8") as f:
                    body = f.read()
                with open(os.path.join(lbl_dir, stem + ".txt"), "w", encoding="utf-8") as f:
                    f.write(body)
                rel = os.path.relpath(img_path, out_dir).replace("\\", "/")
                is_hn = n in hn
                lists[split].extend(["./" + rel] * (hn_repeat if is_hn and split == "train" else 1))
                stats[f"{split}_images"] += 1
                stats[f"{split}_empty"] += not body.strip()
                stats[f"{split}_hard_negative"] += is_hn
            n += 1
        cap.release()
        if wanted:
            log.warning("%s: %d labelled frames past the end of %s", clip, len(wanted), video)
            stats["frames_missing_from_video"] += len(wanted)

    for split, items in lists.items():
        with open(os.path.join(out_dir, f"{split}.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(items) + "\n")
    data = {"path": os.path.abspath(out_dir), "train": "train.txt", "val": "val.txt",
            "names": names, "nc": len(names)}
    with open(os.path.join(out_dir, "data.yaml"), "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)
    with open(os.path.join(out_dir, "val_clips.json"), "w", encoding="utf-8") as f:
        json.dump(sorted(val_ids), f, indent=1)
    log.info("Dataset -> %s: %s", out_dir, dict(stats))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("out_dir")
    p.add_argument("--source", action="append", required=True,
                   help="ANALYSIS_OUTPUT_DIR=DATASET_DIR (repeatable).")
    p.add_argument("--review", action="append", default=[],
                   help="analysis.review_frames output (repeatable).")
    p.add_argument("--hard-negative-repeat", type=int, default=3)
    p.add_argument("--fps", type=float, default=3.0, help="Frames kept per second of clip.")
    p.add_argument("--val-share", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    sources = [tuple(s.split("=", 1)) for s in a.source]
    hn: Dict[str, Set[int]] = {}
    excl: Dict[str, Set[int]] = {}
    for path in a.review:
        with open(path, encoding="utf-8") as f:
            rev = json.load(f)
        for row in rev.get("hard_negatives", []):
            hn.setdefault(row["clip"], set()).update(int(x) for x in row["frames"])
        for row in rev.get("excluded", []):
            excl.setdefault(row["clip"], set()).update(int(x) for x in row["frames"])
    build(a.out_dir, sources, hn, excl, a.fps, a.val_share, a.hard_negative_repeat, a.seed)


if __name__ == "__main__":
    main()
