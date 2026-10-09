"""
Calibrate the box's appearance ReID (box/reid.py, stage 3.2) on the human-tagged house clips.

The tagged tracks of a Label Studio export are true identities: one region is one person for the whole clip. From
them, with the box's own recipe (up to 3 looks per part, the biggest boxes standing in for the detector's best
scores, the EMA of their embeddings):

- **same person across a gap**: a region that disappears and comes back (a disabled keyframe), before vs after;
  and a region's first third vs its last third (later, elsewhere, another pose);
- **different people at the same camera**: two regions of one clip seen at the same time (surely two people);
- **different clips, same camera** (reported apart: the family walks past every day, so it is no clean negative).

It prints the ROC (AUC, the true-link rate at a few cosines), the lowest cosine with at most 1% wrong links, what
the veto would split, the box's re-link rule (``link`` + ``margin`` against every other person of the clip) and the
CPU time per crop. Nothing is written but the optional ``--out`` summary (no embeddings, no crops).

Usage:
    py -m home_guard_project.analysis.reid_calibrate --dataset-dir C:/Users/ameer/Ameer/home_guard_data/dataset
        --model <dir>/person-reidentification-retail-0277.xml [--device CPU] [--out summary.json]
"""

from __future__ import annotations

import argparse
import glob
import itertools
import json
import os
import statistics
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..box import reid
from .detections import clip_path, kept_tasks

PARTS = 3                 # looks per part, as reid.PER_TRACK
MIN_SPAN_SEC = 3.0        # a region this long gives a first-third / last-third pair
WRONG_LINKS = 0.01


def _segments(sequence: Sequence[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
    """The region's keyframes in runs: a disabled keyframe ends a run (Label Studio hides the box until the next)."""
    runs: List[List[Dict[str, Any]]] = [[]]
    for kf in sorted(sequence, key=lambda k: k.get("frame", 0)):
        runs[-1].append(kf)
        if not kf.get("enabled", True):
            runs.append([])
    return [r for r in runs if r]


def _box(kf: Dict[str, Any]) -> Tuple[float, float, float, float]:
    x, y, w, h = (float(kf[k]) / 100.0 for k in ("x", "y", "width", "height"))
    return (max(0.0, x), max(0.0, y), min(1.0, x + w), min(1.0, y + h))


def _read(path: str) -> Tuple[List[Any], float]:
    import cv2  # noqa: PLC0415

    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    return frames, float(fps)


class Clip:
    def __init__(self, camera: str, frames: List[Any], native_fps: float, ls_fps: float) -> None:
        self.camera, self.frames, self.native_fps, self.ls_fps = camera, frames, native_fps, ls_fps

    def crop(self, kf: Dict[str, Any]) -> Optional[Any]:
        n = int(round((int(kf.get("frame", 1)) - 1) * self.native_fps / self.ls_fps))
        if not 0 <= n < len(self.frames):
            return None
        return reid.crop_person(self.frames[n], _box(kf))


class Embedder:
    def __init__(self, model: str, device: str) -> None:
        self.inner = reid.Embedder(model, device)
        self.ms: List[float] = []

    def vector(self, clip: Clip, keyframes: Sequence[Dict[str, Any]]) -> Optional[List[float]]:
        """The box's recipe: the PARTS biggest looks, embedded, EMA in time order."""
        best = sorted(keyframes, key=lambda k: -float(k["width"]) * float(k["height"]))
        looks = []
        for kf in best:
            crop = clip.crop(kf)
            if crop is None:
                continue
            started = time.perf_counter()
            vec = self.inner.embed(crop)
            self.ms.append((time.perf_counter() - started) * 1000.0)
            looks.append((int(kf.get("frame", 0)), vec))
            if len(looks) >= PARTS:
                break
        return reid.ema(v for _, v in sorted(looks, key=lambda x: x[0])) if looks else None


def _span(run: Sequence[Dict[str, Any]], ls_fps: float) -> Tuple[float, float]:
    frames = [int(k.get("frame", 0)) for k in run]
    return min(frames) / ls_fps, max(frames) / ls_fps


def collect(dataset_dir: str, exports: Sequence[str], emb: Embedder) -> Dict[str, Any]:
    """Every region's parts as vectors, per clip."""
    clips = []
    for export in exports:
        with open(export, encoding="utf-8") as f:
            tasks = json.load(f)
        for task, ann in kept_tasks(tasks):
            meta = task.get("data", {}).get("meta_path", "")
            url = task.get("data", {}).get("video_url", "")
            path = clip_path(dataset_dir, meta) if meta else ""
            if not path or not os.path.isfile(path):
                name = os.path.basename(url.split("?", 1)[0])
                hits = glob.glob(os.path.join(dataset_dir, "clips", "*", name))
                path = hits[0] if hits else ""
            if not path:
                continue
            regions = [r for r in ann.get("result", []) if r.get("type") == "videorectangle"
                       and (r.get("value", {}).get("labels") or [""])[0] == "person"]
            if not regions:
                continue
            frames, native_fps = _read(path)
            if not frames:
                continue
            value = regions[0]["value"]
            ls_fps = float(value.get("framesCount") or len(frames)) / float(value.get("duration") or 1.0) or native_fps
            name = os.path.basename(path)
            clip = Clip(name.rsplit("_", 2)[0], frames, native_fps or ls_fps, ls_fps)
            people = []
            for r in regions:
                runs = _segments(r["value"].get("sequence") or [])
                every = [k for run in runs for k in run]
                whole = emb.vector(clip, every)
                if whole is None:
                    continue
                person: Dict[str, Any] = {"id": r.get("id"), "whole": whole, "span": _span(every, ls_fps),
                                          "gap_pairs": [], "early": None, "late": None}
                vecs = [(run, emb.vector(clip, run)) for run in runs]
                vecs = [(run, v) for run, v in vecs if v is not None]
                for (a, va), (b, vb) in zip(vecs, vecs[1:]):
                    person["gap_pairs"].append((va, vb, _span(b, ls_fps)[0] - _span(a, ls_fps)[1]))
                start, end = person["span"]
                if end - start >= MIN_SPAN_SEC:
                    third = (end - start) / 3.0
                    early = [k for k in every if int(k.get("frame", 0)) / ls_fps <= start + third]
                    late = [k for k in every if int(k.get("frame", 0)) / ls_fps >= end - third]
                    person["early"], person["late"] = emb.vector(clip, early), emb.vector(clip, late)
                people.append(person)
            clips.append({"name": name, "camera": clip.camera, "people": people})
            print(f"{name}: {len(people)} people", flush=True)
    return {"clips": clips}


def _overlap(a: Tuple[float, float], b: Tuple[float, float]) -> bool:
    return a[0] <= b[1] and b[0] <= a[1]


def roc(pos: Sequence[float], neg: Sequence[float]) -> Dict[str, Any]:
    cuts = [round(-0.2 + 0.005 * i, 3) for i in range(241)]
    rows = [(t, sum(p >= t for p in pos) / max(1, len(pos)), sum(n >= t for n in neg) / max(1, len(neg)))
            for t in cuts]
    pts = sorted([(fpr, tpr) for _, tpr, fpr in rows] + [(0.0, 0.0), (1.0, 1.0)])
    auc = sum((x2 - x1) * (y1 + y2) / 2 for (x1, y1), (x2, y2) in zip(pts, pts[1:]))
    safe = next((r for r in rows if r[2] <= WRONG_LINKS), None)
    at = {f"{t:.2f}": {"true_links": round(tpr, 3), "wrong_links": round(fpr, 3)}
          for t, tpr, fpr in rows if f"{t:.3f}" in ("0.350", "0.500", "0.600", "0.700", "0.750", "0.800")}
    return {"auc": round(auc, 4), "threshold_at_1pct_wrong": safe[0] if safe else None,
            "true_links_there": round(safe[1], 3) if safe else None, "at": at}


def relink_rule(clips: Sequence[Dict[str, Any]], link: float, margin: float) -> Dict[str, Any]:
    """The box's rule in clips with two or more people: each person's late part against every early part."""
    right = wrong = none = 0
    for c in clips:
        people = [p for p in c["people"] if p["early"] is not None and p["late"] is not None]
        if len(people) < 2:
            continue
        for q in people:
            scored = sorted(((reid.cosine(q["late"], g["early"]), g["id"]) for g in people), reverse=True)
            best, second = scored[0], scored[1]
            if best[0] >= link and best[0] - second[0] >= margin:
                right, wrong = right + (best[1] == q["id"]), wrong + (best[1] != q["id"])
            else:
                none += 1
    total = right + wrong + none
    return {"queries": total, "linked_right": right, "linked_wrong": wrong, "left_new": none,
            "wrong_rate": round(wrong / total, 4) if total else None}


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(prog="reid_calibrate", description=__doc__.split("\n\n")[0])
    p.add_argument("--dataset-dir", required=True)
    p.add_argument("--exports", nargs="*", default=None,
                   help="Label Studio exports (default: annotations/label_studio/ameer_house_batch_*.vehicle_fixed.json)")
    p.add_argument("--model", required=True, help="person-reidentification-retail-0277.xml")
    p.add_argument("--device", default="CPU")
    p.add_argument("--out", default="")
    args = p.parse_args(argv)
    exports = args.exports or sorted(glob.glob(os.path.join(args.dataset_dir, "annotations", "label_studio",
                                                            "ameer_house_batch_*.vehicle_fixed.json")))
    emb = Embedder(args.model, args.device)
    data = collect(args.dataset_dir, exports, emb)
    clips = data["clips"]
    gap_pos = [reid.cosine(a, b) for c in clips for p_ in c["people"] for a, b, _ in p_["gap_pairs"]]
    third_pos = [reid.cosine(p_["early"], p_["late"]) for c in clips for p_ in c["people"]
                 if p_["early"] is not None and p_["late"] is not None]
    co_neg = [reid.cosine(a["whole"], b["whole"]) for c in clips for a, b in itertools.combinations(c["people"], 2)
              if _overlap(a["span"], b["span"])]
    by_cam: Dict[str, List[Tuple[str, Dict[str, Any]]]] = {}
    for c in clips:
        for p_ in c["people"]:
            by_cam.setdefault(c["camera"], []).append((c["name"], p_))
    cross_neg = [reid.cosine(a["whole"], b["whole"]) for rows in by_cam.values()
                 for (ca, a), (cb, b) in itertools.combinations(rows, 2) if ca != cb]
    pos = gap_pos + third_pos
    ms = emb.ms[1:] or emb.ms
    q = lambda xs, f: round(sorted(xs)[int(f * (len(xs) - 1))], 3) if xs else None   # noqa: E731
    summary = {
        "clips": len(clips), "people": sum(len(c["people"]) for c in clips), "cameras": sorted(by_cam),
        "pairs": {"same_after_gap": len(gap_pos), "same_first_vs_last_third": len(third_pos),
                  "different_same_clip": len(co_neg), "different_clips_same_camera": len(cross_neg)},
        "cosine_median": {"same_after_gap": q(gap_pos, 0.5), "same_thirds": q(third_pos, 0.5),
                          "different_same_clip": q(co_neg, 0.5), "different_clips": q(cross_neg, 0.5)},
        "cosine_p05_same": q(pos, 0.05), "cosine_p99_different_same_clip": q(co_neg, 0.99),
        "roc_same_vs_different_same_clip": roc(pos, co_neg),
        "roc_same_vs_different_clips": roc(pos, cross_neg),
        "roc_gap_only_vs_different_same_clip": roc(gap_pos, co_neg),
        "veto_0.35": {"same_split": round(sum(x < reid.REID_VETO for x in pos) / max(1, len(pos)), 4),
                      "different_kept_apart": round(sum(x < reid.REID_VETO for x in co_neg) / max(1, len(co_neg)), 4)},
        "relink_rule_0.70_0.08": relink_rule(clips, reid.REID_LINK, reid.REID_MARGIN),
        "cpu_ms_per_crop": {"device": emb.inner.device, "crops": len(emb.ms), "mean": round(statistics.mean(ms), 2),
                            "p50": q(ms, 0.5), "p95": q(ms, 0.95)} if ms else {},
    }
    print(json.dumps(summary, indent=2))
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
