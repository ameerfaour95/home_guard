"""The crop A/B: one eval set, the same 5 moments of every clip, framed three ways.

    python -m home_guard_project.box.eval_crop_arms build --eval <eval_set_v2> --out <folder>
    python -m home_guard_project.box.eval_prompt run --dir <folder>/window --provider openrouter \
        --model qwen/qwen3.5-9b --tag crop-window-r1 --yes          # each arm twice: r1, r2
    python -m home_guard_project.box.eval_crop_arms compare --out <folder> --runs crop-follow-r1 crop-follow-r2 \
        --vs crop-window-r1 crop-window-r2

``build`` writes ``<out>/<arm>/manifest.jsonl`` (the eval set's rows, unchanged) and ``frames/`` for each arm:

* ``follow``: the old per-frame crop (vlm_crop policy ``follow``): it pans and zooms with the person.
* ``window``: one still window per clip (policy ``clip_window``, the default since 2026-10-09).
* ``whole``: the whole frame.

Every arm uses the box's own vlm_crop.crop_clip and the collector's detector (config models.yolo, trigger
classes) on the clip (or its annotated ``segment``), and the same 5 evenly spaced frames of it (the crop arms
pick them among the frames that have a crop, as the box does). The eval clips are sub-stream (704x576) or public
videos, not the 2592x1520 main stream the box crops, so the geometry is scaled as if they were: the crop's
minimum size shrinks by long side / 2592 and every arm's frames are enlarged by the same factor, then capped at
the box's vlm_max_side (1024). The arms then differ only in the framing. A clip with no trigger detection is
sent whole in both crop arms, as the box does (``whole_frame_fallback``).

``compare`` matches two arms clip by clip and counts a clip only when both runs of each arm agree, so run-to-run
noise (about +-6 alerts at temperature 0) is not read as a difference.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import os
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..data_collection import model_input, vlm_crop
from . import eval_prompt as ep

log = logging.getLogger("eval_crop_arms")

ARMS = ("follow", "window", "whole")
POLICIES = {"follow": vlm_crop.POLICY_FOLLOW, "window": vlm_crop.POLICY_CLIP_WINDOW}
MAIN_LONG_SIDE = 2592       # the box's main stream (ameer_week_0_1: 2592x1520)
MAX_SIDE = 1024             # box.yaml vlm_max_side


def read_clip(path: str) -> Tuple[List[Any], float]:
    """Every decodable frame of *path* and its fps."""
    import cv2  # noqa: PLC0415

    cap = cv2.VideoCapture(path)
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        frames = []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(frame)
        return frames, fps
    finally:
        cap.release()


def scale_of(frame: Any) -> float:
    """How much smaller than the box's main stream the clip is (1.0 for a clip as big or bigger)."""
    return min(1.0, max(frame.shape[:2]) / float(MAIN_LONG_SIDE))


def as_main_stream(frame: Any, scale: float) -> Any:
    """*frame* enlarged by 1 / *scale* (as if cut from the main stream), long side at most MAX_SIDE."""
    import cv2  # noqa: PLC0415

    h, w = frame.shape[:2]
    target = min(MAX_SIDE / float(max(h, w)), 1.0 / scale)
    if abs(target - 1.0) > 1e-6:
        frame = cv2.resize(frame, (max(1, round(w * target)), max(1, round(h * target))),
                           interpolation=cv2.INTER_CUBIC if target > 1 else cv2.INTER_AREA)
    return frame


class CachedDetector:
    """The collector's YOLO, asked once per frame of a clip: both crop arms see the very same boxes."""

    def __init__(self, model: Any, device: Optional[str] = None) -> None:
        self.model, self.device = model, device
        self.cache: Dict[int, Any] = {}
        self.index: Dict[int, int] = {}

    def start(self, frames: Sequence[Any]) -> None:
        self.cache, self.index = {}, {id(f): i for i, f in enumerate(frames)}

    def __call__(self, frame: Any, **kwargs: Any) -> Any:
        i = self.index.get(id(frame), -1)
        if i not in self.cache or i < 0:
            kw = dict(kwargs)
            if self.device:
                kw["device"] = self.device
            self.cache[i] = self.model(frame, **kw)
        return self.cache[i]


def render_arms(frames: Sequence[Any], detector: CachedDetector, base: vlm_crop.CropSettings,
                k: int = ep.FRAME_COUNT) -> Dict[str, Dict[str, Any]]:
    """{arm: {"frames": k pictures, "crops", "whole_frame", "fallback"}} for one clip's frames."""
    scale = scale_of(frames[0])
    detector.start(frames)
    out: Dict[str, Dict[str, Any]] = {}
    whole_picks = ep.sample_indices(len(frames), k)
    out["whole"] = {"frames": [as_main_stream(frames[i], scale) for i in whole_picks], "crops": None,
                    "whole_frame": True, "fallback": None, "picks": whole_picks}
    for arm, policy in POLICIES.items():
        settings = dataclasses.replace(base, policy=policy, min_size=max(2, round(base.min_size * scale)))
        r = vlm_crop.crop_clip(detector, settings, list(frames), 0.0, 1.0, list(frames), 0.0, name=arm)
        if r is None:
            out[arm] = dict(out["whole"], fallback="no_trigger_class_detection_or_usable_crop")
            continue
        usable = [i for i, c in enumerate(r.crops) if c is not None]
        picks = [usable[j] for j in ep.sample_indices(len(usable), k)]
        cut = [model_input.cut_crop(frames[i], r.crops[i], (r.width, r.height)) for i in picks]
        out[arm] = {"frames": [as_main_stream(f, scale) for f in cut], "crops": [list(r.crops[i]) for i in picks],
                    "whole_frame": bool(r.whole_frame), "fallback": None, "picks": picks}
    return out


def clip_path(row: Dict[str, Any], dataset_dir: str, out_dir: str) -> str:
    """The clip of a manifest row: ``clip`` under the dataset, else its ``s3_key`` under ``<out>/_s3`` (rows that
    ``eval_prompt add`` put in; download them there first)."""
    if row.get("clip"):
        return os.path.join(dataset_dir, row["clip"])
    return os.path.join(out_dir, "_s3", row["s3_key"])


def build(eval_dir: str, out_dir: str, dataset_dir: str, limit: Optional[int] = None,
          device: Optional[str] = None) -> Dict[str, int]:
    """Write the three arms of *eval_dir* into *out_dir*; clips already built are kept."""
    from ultralytics import YOLO  # noqa: PLC0415

    from ..data_collection.config import load_config  # noqa: PLC0415

    cfg = load_config()
    base = vlm_crop.settings_from_config(cfg)
    detector = CachedDetector(YOLO(cfg.YOLO_MODEL), device)
    with open(os.path.join(eval_dir, "manifest.jsonl"), encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    for arm in ARMS:
        os.makedirs(os.path.join(out_dir, arm, ep.FRAMES_DIR), exist_ok=True)
        with open(os.path.join(out_dir, arm, "manifest.jsonl"), "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    info_path = os.path.join(out_dir, "arms.jsonl")
    done = set()
    if os.path.exists(info_path):
        with open(info_path, encoding="utf-8") as f:
            done = {json.loads(line)["clip_id"] for line in f if line.strip()}
    counts = {"built": 0, "kept": len(done), "unreadable": 0}
    for row in rows[:limit] if limit else rows:
        cid = row["clip_id"]
        if cid in done:
            continue
        frames, fps = read_clip(clip_path(row, dataset_dir, out_dir))
        seg = row.get("segment")
        lo, hi = ep.segment_range(len(frames), fps, *(seg if seg else (None, None)))
        frames = frames[lo:hi]
        if len(frames) < 2:
            log.warning("%s: could not read the clip", cid)
            counts["unreadable"] += 1
            continue
        arms = render_arms(frames, detector, base)
        for arm, got in arms.items():
            for rel, frame in zip(ep.frame_paths(cid), got["frames"]):
                ep._write_jpeg(os.path.join(out_dir, arm, rel), frame)
        with open(info_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"clip_id": cid, "frames": len(frames), "scale": round(scale_of(frames[0]), 4),
                                **{arm: {k: v for k, v in got.items() if k != "frames"} for arm, got in arms.items()}},
                               ensure_ascii=False) + "\n")
        counts["built"] += 1
        log.info("built %s", cid)
    return counts


def _answers(out_dir: str, arm: str, tag: str) -> Dict[str, Dict[str, Any]]:
    """The last answer per clip of ``<out>/<arm>/results/<tag>.jsonl``."""
    path = os.path.join(out_dir, arm, "results", tag + ".jsonl")
    last: Dict[str, Dict[str, Any]] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                last[row["clip_id"]] = row
    return last


def _verdict(row: Dict[str, Any]) -> Optional[str]:
    """The model's label (alert / normal / empty); None for an error row."""
    return None if row.get("error") else (row.get("ai_label") or None)


def compare(out_dir: str, a: Tuple[str, Sequence[str]], b: Tuple[str, Sequence[str]]) -> Dict[str, Any]:
    """Clip by clip: where every run of arm *a* agrees, every run of arm *b* agrees, and they differ."""
    with open(os.path.join(out_dir, a[0], "manifest.jsonl"), encoding="utf-8") as f:
        truth = {r["clip_id"]: r for r in (json.loads(x) for x in f if x.strip())}
    runs_a = [_answers(out_dir, a[0], t) for t in a[1]]
    runs_b = [_answers(out_dir, b[0], t) for t in b[1]]
    result = {"a": a[0], "b": b[0], "a_better": [], "b_better": [], "both_stable": 0, "unstable": 0}
    for cid, row in truth.items():
        va = {_verdict(r[cid]) for r in runs_a if cid in r}
        vb = {_verdict(r[cid]) for r in runs_b if cid in r}
        if len(va) != 1 or len(vb) != 1 or None in va | vb:
            result["unstable"] += 1
            continue
        result["both_stable"] += 1
        (pa,), (pb,) = va, vb
        if (pa == "alert") == (pb == "alert"):
            continue
        right = "alert" if row["ours_label"] == "alert" else "no_alert"
        a_right = (pa == "alert") == (right == "alert")
        entry = {"clip_id": cid, "truth": row["ours_label"], a[0]: pa, b[0]: pb, "source": row.get("source"),
                 "category": row.get("category")}
        result["a_better" if a_right else "b_better"].append(entry)
    return result


def main(argv: Optional[Sequence[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    p = argparse.ArgumentParser(prog="eval_crop_arms", description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--eval", required=True, help="The eval set (manifest.jsonl).")
    b.add_argument("--out", required=True)
    b.add_argument("--dataset", default=None, help="Where the manifest's clip paths live (default "
                   "$HOMEGUARD_DATASET_DIR, else home_guard_data/dataset).")
    b.add_argument("--limit", type=int, default=None)
    b.add_argument("--device", default=None)
    c = sub.add_parser("compare")
    c.add_argument("--out", required=True)
    c.add_argument("--a", required=True, choices=ARMS)
    c.add_argument("--a-runs", nargs="+", required=True)
    c.add_argument("--b", required=True, choices=ARMS)
    c.add_argument("--b-runs", nargs="+", required=True)
    args = p.parse_args(argv)
    if args.cmd == "build":
        dataset = args.dataset or os.environ.get("HOMEGUARD_DATASET_DIR") or os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
            "home_guard_data", "dataset")
        print(json.dumps(build(args.eval, args.out, dataset, args.limit, args.device)))
    else:
        r = compare(args.out, (args.a, args.a_runs), (args.b, args.b_runs))
        print(json.dumps(r, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
