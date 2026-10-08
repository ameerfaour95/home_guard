"""Replay saved alerts through the tracker, ``ground.ground_of`` and ``events.decide``: with and without the scene map.

Owner rule (2026-10-08): done means it runs on the box AND a replay verifies it. This takes alert metas and their
clips (a production folder, or a copy of one from S3), runs the detector over each clip at the live tracker's look
rate (cached in a looks file, so a second replay with other maps is quick), and decides every alert twice with the
same stage-2 policy (``events.EventBook`` plus the guard loop's appearance-only rule): once without a ground and
once with the camera's scene map. It reports how many messages the map would suppress or add, and why.

    python -m home_guard_project.box.ground_replay --raw <folder with meta/ and clips/> [--maps scene_maps.yaml]
           [--looks looks.json] [--dates 2026-10-07,2026-10-08] [--out replay.json]

Without ``--maps`` the box's own scene maps are used (``scene_map.load_scene_map``). What a replay cannot see:
clips recorded under a zone are blacked out outside it, so a person on what was black then is invisible here, and
the looks between two alerts are missing (the live loop keeps a session alive with them): both runs share that.
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
import logging
import os
import sys
import tempfile
from typing import Any, Callable, Dict, List, Optional, Sequence

from . import ground as gr
from . import scene_map as sm
from .alert_guards import appearance_only
from .events import EventBook
from .tracker import CameraTracker

log = logging.getLogger("box.ground_replay")

LOOK_FPS = 3.0                   # the live tracker sees about 1-3 looks a second
LABELS = ("normal", "suspicious", "escalation")


def load_metas(raw: str, dates: Sequence[str] = ()) -> List[Dict[str, Any]]:
    """The alert metas under *raw*/meta, oldest first, only *dates* (local, YYYY-MM-DD) when given."""
    rows = []
    for path in glob.glob(os.path.join(raw, "meta", "**", "*.meta.json"), recursive=True):
        with open(path, encoding="utf-8") as f:
            meta = json.load(f)
        if dates and str(meta.get("clip_start_local", ""))[:10] not in dates:
            continue
        rows.append(meta)
    return sorted(rows, key=lambda m: float(m.get("trigger_ts") or 0))


def stem_of(meta: Dict[str, Any]) -> str:
    return os.path.splitext(os.path.basename(str(meta.get("clip_path") or "")))[0]


def detect_looks(raw: str, metas: Sequence[Dict[str, Any]], looks: Dict[str, Any], detector: Any) -> Dict[str, Any]:
    """Fill *looks* ({stem: {camera, t0, t1, looks: [[ts, detections]]}}) for clips not in it yet."""
    import cv2  # noqa: PLC0415

    for meta in metas:
        stem = stem_of(meta)
        if stem in looks:
            continue
        found = glob.glob(os.path.join(raw, "clips", "**", f"{stem}.mp4"), recursive=True)
        if not found:
            continue
        cap = cv2.VideoCapture(found[0])
        frames = []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(frame)
        cap.release()
        t0, t1 = float(meta["clip_start_ts"]), float(meta["clip_end_ts"])
        step = max(1, int(round(len(frames) / max(1.0, (t1 - t0) * LOOK_FPS))))
        rows = []
        for k in range(0, len(frames), step):
            ts = t0 + (t1 - t0) * k / max(1, len(frames) - 1)
            h, w = frames[k].shape[:2]
            result = detector(frames[k])
            rows.append([ts, sm.detections_from_result(result[0] if result else None, w, h)])
        looks[stem] = {"camera": meta.get("camera_name"), "t0": t0, "t1": t1, "looks": rows}
    return looks


def _decide(book: EventBook, cam: str, ts: float, label: str, people: Any, summary: str, stem: str,
            where: Optional[Dict[str, Any]]) -> Any:
    """``inference._event_decision``: an alert without a label is the detector's, first in its event."""
    d = book.decide(cam, ts, label if label in LABELS else "normal", people or 0, summary, stem,
                    **({"ground": where} if where else {}))
    if label not in LABELS and not d.notify:
        session = book.session_of_alert(stem) or {}
        if session.get("reported_level", "none") == "none":
            d.notify, d.reason = True, "no label (the AI did not answer): first in this event"
    return d


def replay(metas: Sequence[Dict[str, Any]], looks: Dict[str, Any], scene_for: Callable[[str], Any]) -> List[Dict[str, Any]]:
    """Each alert decided without and with the ground; one row per alert."""
    base, grounded = EventBook(tempfile.mkdtemp()), EventBook(tempfile.mkdtemp())
    out = []
    for k, meta in enumerate(metas):
        cam, stem = str(meta["camera_name"]), stem_of(meta)
        alert = meta.get("alert") or {}
        ts = float(meta["trigger_ts"])
        label = str(alert.get("label") or "")
        why, reason = str(alert.get("why") or ""), str(alert.get("alert_reason") or "")
        summary, people = str(alert.get("summary") or ""), alert.get("people")
        if label == "suspicious" and appearance_only(f"{why} {reason}"):
            label = "normal"                     # the guard loop's appearance-only rule
        clip = looks.get(stem) or {"looks": [], "t0": ts, "t1": ts}
        scene = scene_for(cam)
        tracker = CameraTracker(cam)
        for t, dets in clip["looks"]:
            tracker.update(t, [tuple(d) for d in dets], scene_map=scene)
        tracks = tracker.tracks_between(clip["t0"], clip["t1"])
        g = gr.ground_of(tracks, scene)
        where = {} if g == gr.UNKNOWN else dict(g.record(), action=gr.is_action(f"{why} {reason}".strip() or summary))
        result = {}
        for name, book, arg in (("base", base, None), ("ground", grounded, where or None)):
            for t, dets in clip["looks"]:
                if t < ts:
                    book.activity(cam, t, people=sum(1 for d in dets if d[0] == 0),
                                  vehicles=sum(1 for d in dets if d[0] != 0))
            d = _decide(book, cam, ts, label, people, summary, stem, arg)
            if d.notify:
                sent = "suspicious" if arg and arg.get("entered") and label in ("normal", "") else (
                    label if label in LABELS else "normal")
                book.record_sent(d.session_id, sent, people or 0, ts, alert_id=stem, chat_id=-1, message_id=k)
            for t, dets in clip["looks"]:
                if t >= ts:
                    book.activity(cam, t, people=sum(1 for d in dets if d[0] == 0),
                                  vehicles=sum(1 for d in dets if d[0] != 0))
            result[name] = {"sent": d.notify, "reason": d.reason}
        out.append({"stem": stem, "camera": cam, "local": str(meta.get("clip_start_local", "")), "label": label,
                    "had_looks": stem in looks, "people_tracks": sum(1 for t in tracks if t.kind == "person"),
                    "ground": where, "base": result["base"], "with_ground": result["ground"],
                    "summary": summary, "why": why})
    return out


def summarize(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    per: Dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    for r in rows:
        c = per[r["camera"]]
        c["alerts"] += 1
        c["sent_without_map"] += r["base"]["sent"]
        c["sent_with_map"] += r["with_ground"]["sent"]
        c["suppressed"] += r["base"]["sent"] and not r["with_ground"]["sent"]
        c["added"] += r["with_ground"]["sent"] and not r["base"]["sent"]
        c["ground_" + (r["ground"].get("on") or "unknown")] += 1
    total: collections.Counter = collections.Counter()
    for c in per.values():
        total.update(c)
    return {"cameras": {k: dict(v) for k, v in sorted(per.items())}, "total": dict(total)}


def maps_from_file(path: str) -> Callable[[str], Any]:
    """Scene maps from a scene_maps.yaml-format file (``{scene_maps: {camera: entry}}``)."""
    from ..data_collection.zones import read_scene_maps  # noqa: PLC0415

    entries = read_scene_maps(path, strict=True)
    scenes = {cam: sm.SceneMap.from_dict(cam, entry) for cam, entry in entries.items()}
    return lambda cam: scenes.get(cam)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="ground_replay", description=__doc__.split("\n", 1)[0])
    parser.add_argument("--raw", required=True, help="a folder with meta/ and clips/ (production layout)")
    parser.add_argument("--maps", default="", help="scene maps in scene_maps.yaml format (default: the box's own)")
    parser.add_argument("--looks", default="", help="the detector looks cache (default: <raw>/ground_replay_looks.json)")
    parser.add_argument("--dates", default="", help="local dates to keep, comma separated")
    parser.add_argument("--out", default="", help="write every row here (JSON)")
    args = parser.parse_args(argv)
    metas = load_metas(args.raw, [d for d in args.dates.split(",") if d])
    looks_path = args.looks or os.path.join(args.raw, "ground_replay_looks.json")
    looks = {}
    if os.path.isfile(looks_path):
        with open(looks_path, encoding="utf-8") as f:
            looks = json.load(f)
    if any(stem_of(m) not in looks for m in metas):
        from ultralytics import YOLO  # noqa: PLC0415

        from . import paths  # noqa: PLC0415

        model = YOLO(paths.resolve_model(paths.DEFAULT_YOLO))
        looks = detect_looks(args.raw, metas, looks,
                             lambda frame: model.predict(frame, conf=0.35, classes=[0, 2, 3, 5, 7], imgsz=640,
                                                         verbose=False))
        with open(looks_path, "w", encoding="utf-8") as f:
            json.dump(looks, f)
    scene_for = maps_from_file(args.maps) if args.maps else sm.load_scene_map
    rows = replay(metas, looks, scene_for)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(rows, f, ensure_ascii=False, indent=1)
    sys.stdout.reconfigure(encoding="utf-8")
    print(json.dumps(summarize(rows), ensure_ascii=False, indent=1))
    for kind, test in (("suppressed", lambda r: r["base"]["sent"] and not r["with_ground"]["sent"]),
                       ("added", lambda r: r["with_ground"]["sent"] and not r["base"]["sent"])):
        for r in rows:
            if test(r):
                print(f"{kind}: {r['camera']} {r['local'][5:16]} {r['label'] or '-'} | {r['with_ground']['reason']} | "
                      f"{r['summary'][:100]}")
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    sys.exit(main())
