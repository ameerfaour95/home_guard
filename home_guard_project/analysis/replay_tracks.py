"""Replay saved alert clips through the box's detector and tracker, offline, to check the tracker against real video.

Fix tracker task 2.1 (the entity layer maps tracker tracks to P1/P2/CAR1; it needs recorded tracks to be judged).
For every clip under ``<root>/clips/<camera>/<date>/<stem>.mp4`` with its ``<root>/meta/.../<stem>.meta.json``:

1. Every frame (or every Nth, ``--every``) goes through the box's YOLO model with the box's thresholds
   (``inference.filter_by_thresholds``: person 0.8, everything else 0.7 by default), then through a fresh
   ``tracker.CameraTracker`` — exactly the code the box runs, fed at the clip's own timestamps
   (``clip_start_ts``..``clip_end_ts`` spread over the frames, as ``inference._prepare_alert`` does).
2. ``<stem>.tracks.json`` is written next to the clip: every track (id, kind, class, first and last seen, hits,
   confirmed, prev_id, returns) with its box at every look, plus the per-clip split statistics below.
3. ``--render`` draws the track ids on chosen clips (an mp4 and a contact sheet a person can check by eye).

Split statistics (what a person would call "one person became two tracks"):

- ``fragments``: a confirmed person track that starts within FRAGMENT_GAP_SEC of another person track's last look
  and within FRAGMENT_DISTANCE of its last foot point, without a box of the old one in between: the same walk,
  split.
- ``duplicates``: two person tracks seen in the same look with boxes overlapping more than DUPLICATE_IOU, for at
  least two looks: one person counted twice.
- ``flickers``: person tracks that never got confirmed (one look): noise the entity layer must ignore.
- ``excess_vs_eye``: confirmed person tracks minus the Eye's own people count for the clip (``alert.people`` in the
  meta, when present): a rough upper bound on over-segmentation.

The clips are the box's stored sub-stream (704x576, 7 fps) and each clip gets a fresh tracker: the box's tracker
runs continuously, so a person already tracked before the clip started is a new track here.

Usage (from the repo root, the project venv):

    python -m home_guard_project.analysis.replay_tracks --root C:/Users/ameer/Ameer/home_guard_data/eval/oct7_tracks
    python -m home_guard_project.analysis.replay_tracks --root ... --render ch3 --render-from 09:40 --render-to 11:00
"""

from __future__ import annotations

import argparse
import datetime as dt
import glob
import json
import math
import os
import statistics
import sys
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

FRAGMENT_GAP_SEC = 2.0
FRAGMENT_DISTANCE = 0.15
DUPLICATE_IOU = 0.5
COLOURS = [(66, 135, 245), (245, 66, 66), (66, 245, 117), (245, 197, 66), (197, 66, 245), (66, 233, 245),
           (245, 135, 66), (160, 160, 160)]


# ----------------------------------------------------------------------------
# Clips
# ----------------------------------------------------------------------------
def find_clips(root: str, camera: str = "") -> List[Tuple[str, str]]:
    """``(clip, meta)`` pairs under *root*, oldest first; a clip without its meta is skipped."""
    pairs = []
    for clip in glob.glob(os.path.join(root, "clips", "*", "*", "*.mp4")):
        rel = os.path.relpath(clip, os.path.join(root, "clips"))
        meta = os.path.join(root, "meta", os.path.splitext(rel)[0] + ".meta.json")
        if camera and camera not in os.path.basename(os.path.dirname(os.path.dirname(clip))):
            continue
        if os.path.exists(meta):
            pairs.append((clip, meta))
    return sorted(pairs, key=lambda p: os.path.basename(p[0]))


def frame_times(meta: Dict[str, Any], n: int) -> List[float]:
    """The frames' timestamps: the clip's start and end spread evenly, as the box spreads them."""
    start = float(meta.get("clip_start_ts") or 0.0)
    end = float(meta.get("clip_end_ts") or start + max(0, n - 1) / 7.0)
    if n <= 1:
        return [start] * n
    return [start + i * (end - start) / (n - 1) for i in range(n)]


def read_frames(path: str) -> List[Any]:
    import cv2  # noqa: PLC0415

    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    return frames


# ----------------------------------------------------------------------------
# Replay
# ----------------------------------------------------------------------------
class Detector:
    """The box's YOLO model with the box's per-type thresholds."""

    def __init__(self, weights: str, person: float, other: float, device: str = "") -> None:
        from ultralytics import YOLO  # noqa: PLC0415

        self.model = YOLO(weights)
        self.thresholds = {"person": person, "vehicle": other, "animal": other}
        self.other = other
        self.device = device

    def __call__(self, frame: Any) -> Any:
        from ..box.inference import detector_floor, filter_by_thresholds  # noqa: PLC0415

        kwargs = {"device": self.device} if self.device else {}
        raw = self.model.predict(frame, conf=detector_floor(self.thresholds, self.other), verbose=False, **kwargs)
        return filter_by_thresholds(raw[0], self.thresholds, self.other) if raw else None


def replay(frames: Sequence[Any], times: Sequence[float], detect: Any, camera: str = "",
           every: int = 1) -> Dict[str, Any]:
    """Run *frames* through *detect* and a fresh tracker; the tracks with a box at every look they were seen."""
    from ..box import scene_map as sm  # noqa: PLC0415
    from ..box.tracker import CameraTracker  # noqa: PLC0415

    tracker = CameraTracker(camera)
    seen: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    kinds: Dict[int, Tuple[str, int]] = {}
    looks: List[Dict[str, Any]] = []
    for i in range(0, len(frames), max(1, every)):
        frame, ts = frames[i], float(times[i])
        h, w = frame.shape[:2]
        dets = sm.detections_from_result(detect(frame), w, h)
        tracker.update(ts, dets)
        with tracker._lock:                       # the replay reads the tracker's own state after each look
            now = [t for t in tracker._active if t.last_seen == ts]
        looks.append({"frame": i, "ts": round(ts, 3), "detections": len(dets),
                      "tracks": [t.id for t in now]})
        for t in now:
            kinds[t.id] = (t.kind, t.cls)
            seen[t.id].append({"frame": i, "ts": round(ts, 3), "box": [round(v, 4) for v in t.box]})
    with tracker._lock:
        every_track = {t.id: t for t in tracker._history + tracker._active}
    tracks = []
    for tid in sorted(seen):
        t = every_track.get(tid)
        if t is None:                              # an unconfirmed track dropped on loss: rebuild what we saw
            boxes = seen[tid]
            tracks.append({"id": tid, "kind": kinds[tid][0], "cls": kinds[tid][1], "first_seen": boxes[0]["ts"], "last_seen": boxes[-1]["ts"],
                           "hits": len(boxes), "confirmed": False, "prev_id": None, "returns": 0, "boxes": boxes})
            continue
        tracks.append({"id": t.id, "kind": t.kind, "cls": t.cls, "first_seen": round(t.first_seen, 3),
                       "last_seen": round(t.last_seen, 3), "hits": t.hits, "confirmed": t.confirmed,
                       "prev_id": t.prev_id, "returns": t.returns, "boxes": seen[tid]})
    return {"looks": looks, "tracks": tracks}


# ----------------------------------------------------------------------------
# Statistics
# ----------------------------------------------------------------------------
def _iou(a: Sequence[float], b: Sequence[float]) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def _foot(box: Sequence[float]) -> Tuple[float, float]:
    return ((box[0] + box[2]) / 2.0, box[3])


def split_stats(result: Dict[str, Any], eye_people: Optional[int] = None) -> Dict[str, Any]:
    tracks = result["tracks"]
    people = [t for t in tracks if t["kind"] == "person" and t["confirmed"]]
    vehicles = [t for t in tracks if t["kind"] == "vehicle" and t["confirmed"]]
    flickers = [t for t in tracks if not t["confirmed"]]
    fragments = []
    for new in people:
        start = new["boxes"][0]
        for old in people:
            if old is new or old["last_seen"] >= new["first_seen"]:
                continue
            gap = new["first_seen"] - old["last_seen"]
            end = old["boxes"][-1]
            if gap <= FRAGMENT_GAP_SEC and math.dist(_foot(end["box"]), _foot(start["box"])) <= FRAGMENT_DISTANCE:
                fragments.append({"from": old["id"], "to": new["id"], "gap_sec": round(gap, 2),
                                  "linked": new["prev_id"] == old["id"]})
                break
    by_look: Dict[float, List[Tuple[int, List[float]]]] = defaultdict(list)
    for t in people:
        for b in t["boxes"]:
            by_look[b["ts"]].append((t["id"], b["box"]))
    overlap_looks: Dict[Tuple[int, int], int] = defaultdict(int)
    for items in by_look.values():
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                if _iou(items[i][1], items[j][1]) > DUPLICATE_IOU:
                    overlap_looks[tuple(sorted((items[i][0], items[j][0])))] += 1
    duplicates = [{"tracks": list(pair), "looks": n} for pair, n in overlap_looks.items() if n >= 2]
    most_at_once = max((len(v) for v in by_look.values()), default=0)
    out = {"person_tracks": len(people), "vehicle_tracks": len(vehicles), "flickers": len(flickers),
           "fragments": fragments, "duplicates": duplicates, "most_people_at_once": most_at_once,
           "returns_linked": sum(1 for t in people if t["prev_id"] is not None)}
    if eye_people is not None:
        out["eye_people"] = eye_people
        out["excess_vs_eye"] = len(people) - eye_people
    return out


def eye_people(meta: Dict[str, Any]) -> Optional[int]:
    alert = meta.get("alert") or {}
    for value in (alert.get("people"), (alert.get("vlm") or {}).get("people")):
        try:
            if value is not None:
                return int(value)
        except (TypeError, ValueError):
            continue
    return None


# ----------------------------------------------------------------------------
# Rendering
# ----------------------------------------------------------------------------
def render(frames: Sequence[Any], result: Dict[str, Any], out_mp4: str, out_sheet: str, fps: float = 7.0,
           sheet_every_sec: float = 1.0, title: str = "") -> None:
    """The clip with each track's box and id drawn (P<id>, V<id>; dashed for unconfirmed), and a contact sheet."""
    import cv2  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415

    boxes_at: Dict[int, List[Tuple[Dict[str, Any], Dict[str, Any]]]] = {lk["frame"]: [] for lk in result["looks"]}
    for t in result["tracks"]:
        for b in t["boxes"]:
            boxes_at.setdefault(b["frame"], []).append((t, b))
    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(out_mp4, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    drawn = []
    last: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
    for i, frame in enumerate(frames):
        img = frame.copy()
        here = boxes_at.get(i)
        if here is not None:
            last = here
        for t, b in (here if here is not None else last):
            x1, y1, x2, y2 = (int(b["box"][0] * w), int(b["box"][1] * h), int(b["box"][2] * w), int(b["box"][3] * h))
            colour = COLOURS[t["id"] % len(COLOURS)]
            tag = ("P" if t["kind"] == "person" else "V" if t["kind"] == "vehicle" else "?") + str(t["id"])
            if t["prev_id"]:
                tag += f"<{t['prev_id']}"
            thick = 2 if t["confirmed"] else 1
            cv2.rectangle(img, (x1, y1), (x2, y2), colour, thick)
            cv2.putText(img, tag, (x1, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 2)
        stamp = dt.datetime.fromtimestamp(result["looks"][0]["ts"]).strftime("%H:%M") if result["looks"] else ""
        cv2.putText(img, f"{title} f{i} {stamp}", (6, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        writer.write(img)
        drawn.append(img)
    writer.release()
    step = max(1, int(round(fps * sheet_every_sec)))
    picks = drawn[::step][:12]
    if picks:
        small = [cv2.resize(p, (w // 2, h // 2)) for p in picks]
        while len(small) % 4:
            small.append(np.zeros_like(small[0]))
        rows = [np.hstack(small[r:r + 4]) for r in range(0, len(small), 4)]
        cv2.imwrite(out_sheet, np.vstack(rows))


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def _local_hhmm(ts: float) -> str:
    return dt.datetime.fromtimestamp(ts).strftime("%H:%M")


def summarize(rows: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    rows = list(rows)
    with_people = [r for r in rows if r["stats"]["person_tracks"]]
    per_clip = [r["stats"]["person_tracks"] for r in rows]
    frag = sum(len(r["stats"]["fragments"]) for r in rows)
    dup = sum(len(r["stats"]["duplicates"]) for r in rows)
    excess = [r["stats"]["excess_vs_eye"] for r in rows if "excess_vs_eye" in r["stats"]]
    by_cam: Dict[str, Dict[str, Any]] = {}
    for r in rows:
        c = by_cam.setdefault(r["camera"], {"clips": 0, "person_tracks": 0, "fragments": 0, "duplicates": 0,
                                            "flickers": 0, "vehicle_tracks": 0})
        c["clips"] += 1
        for k in ("person_tracks", "flickers", "vehicle_tracks"):
            c[k] += r["stats"][k]
        c["fragments"] += len(r["stats"]["fragments"])
        c["duplicates"] += len(r["stats"]["duplicates"])
    return {
        "clips": len(rows), "clips_with_person_tracks": len(with_people),
        "person_tracks_per_clip": {"mean": round(statistics.mean(per_clip), 2) if per_clip else 0,
                                   "median": statistics.median(per_clip) if per_clip else 0,
                                   "max": max(per_clip, default=0)},
        "fragments": frag, "clips_with_fragments": sum(1 for r in rows if r["stats"]["fragments"]),
        "fragments_linked_as_return": sum(f["linked"] for r in rows for f in r["stats"]["fragments"]),
        "duplicates": dup, "clips_with_duplicates": sum(1 for r in rows if r["stats"]["duplicates"]),
        "flickers": sum(r["stats"]["flickers"] for r in rows),
        "excess_vs_eye": {"clips": len(excess), "exact": sum(1 for e in excess if e == 0),
                          "over": sum(1 for e in excess if e > 0), "under": sum(1 for e in excess if e < 0),
                          "sum_over": sum(e for e in excess if e > 0)},
        "by_camera": by_cam,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--root", required=True, help="Folder with clips/ and meta/ (a copy of a production_ prefix).")
    p.add_argument("--weights", default="yolo11s.pt")
    p.add_argument("--person", type=float, default=0.8, help="The box's person certainty (conf_person).")
    p.add_argument("--conf", type=float, default=0.7, help="The box's certainty for everything else (inference_conf).")
    p.add_argument("--every", type=int, default=1, help="Look at every Nth frame (1 = all 7 fps; 3 = about the box's pace).")
    p.add_argument("--device", default="")
    p.add_argument("--camera", default="", help="Only clips whose camera name contains this.")
    p.add_argument("--suffix", default="", help="Added to the output names (e.g. _every3), so runs do not overwrite.")
    p.add_argument("--render", default="", help="Render clips whose camera name contains this (e.g. ch3).")
    p.add_argument("--render-from", default="00:00")
    p.add_argument("--render-to", default="23:59")
    p.add_argument("--render-max", type=int, default=6)
    p.add_argument("--render-dir", default="", help="Default: <root>/renders")
    args = p.parse_args(argv)

    detect = Detector(args.weights, args.person, args.conf, args.device)
    render_dir = args.render_dir or os.path.join(args.root, "renders")
    rows, rendered = [], 0
    for clip, meta_path in find_clips(args.root, args.camera):
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
        camera = str(meta.get("camera_name") or os.path.basename(os.path.dirname(os.path.dirname(clip))))
        frames = read_frames(clip)
        if not frames:
            print(f"skip {clip}: no frames", file=sys.stderr)
            continue
        times = frame_times(meta, len(frames))
        result = replay(frames, times, detect, camera, args.every)
        stats = split_stats(result, eye_people(meta))
        stem = os.path.splitext(clip)[0]
        doc = {"clip": os.path.relpath(clip, args.root), "camera": camera, "start_local": _local_hhmm(times[0]),
               "params": {"weights": args.weights, "person": args.person, "conf": args.conf, "every": args.every,
                          "fresh_tracker_per_clip": True},
               "frames": len(frames), **result, "stats": stats}
        with open(stem + f".tracks{args.suffix}.json", "w", encoding="utf-8") as f:
            json.dump(doc, f, indent=1)
        rows.append({"clip": doc["clip"], "camera": camera, "start": doc["start_local"], "stats": stats})
        hhmm = _local_hhmm(times[0])
        if (args.render and args.render in camera and args.render_from <= hhmm <= args.render_to
                and rendered < args.render_max and stats["person_tracks"]):
            os.makedirs(render_dir, exist_ok=True)
            base = os.path.join(render_dir, os.path.basename(stem) + args.suffix)
            render(frames, result, base + ".mp4", base + ".jpg", title=f"{camera} {hhmm}")
            rendered += 1
        print(f"{camera} {hhmm} people={stats['person_tracks']} frag={len(stats['fragments'])} "
              f"dup={len(stats['duplicates'])} flick={stats['flickers']} eye={stats.get('eye_people')}")
    summary = summarize(rows)
    with open(os.path.join(args.root, f"summary{args.suffix}.json"), "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "clips": rows}, f, indent=1)
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
