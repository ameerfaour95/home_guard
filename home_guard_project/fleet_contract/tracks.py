"""Box tracks and their interpolation (Label Studio VideoRectangle semantics, by time). Stdlib only.

A Track is one object: a COCO label plus keyframes sorted by time. A keyframe with enabled=False hides the box from
that keyframe until the next one. Between two keyframes the box moves linearly by TIME (never by frame count)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .classes import COCO_NAMES

EPS = 1e-9


@dataclass
class Keyframe:
    frame: int
    t_sec: float
    xyxy: list  # [x1, y1, x2, y2], normalised 0..1
    enabled: bool = True


@dataclass
class Track:
    track_id: str
    label: str
    keyframes: list = field(default_factory=list)
    source: str = "human"


def frame_time(frame_index: int, fps: float) -> float:
    return frame_index / fps


def frame_at(t_sec: float, fps: float) -> int:
    return int(round(t_sec * fps))


def box_at(track: Track, t_sec: float) -> Optional[list]:
    """The box at `t_sec`, or None when the object is not visible then."""
    kfs = track.keyframes
    if not kfs or t_sec < kfs[0].t_sec:
        return None
    prev = kfs[0]
    for nxt in kfs[1:]:
        if t_sec < nxt.t_sec:
            if not prev.enabled:
                return None
            span = nxt.t_sec - prev.t_sec
            w = (t_sec - prev.t_sec) / span if span > 0 else 0.0
            return [a + (b - a) * w for a, b in zip(prev.xyxy, nxt.xyxy)]
        prev = nxt
    return list(prev.xyxy) if prev.enabled else None  # at or after the last keyframe: held


def boxes_at(tracks: list, t_sec: float) -> list:
    """[(label, xyxy)] for every track visible at `t_sec`."""
    out = []
    for tr in tracks:
        b = box_at(tr, t_sec)
        if b is not None:
            out.append((tr.label, b))
    return out


def _iou(a, b) -> float:
    iw = min(a[2], b[2]) - max(a[0], b[0])
    ih = min(a[3], b[3]) - max(a[1], b[1])
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def _det(d):
    if isinstance(d, dict):
        return d["label"], list(d["xyxy"])
    return d[0], list(d[1])


def _sparse(points: list, tolerance: float) -> list:
    """Indices of `points` ([(frame, t, box)]) to keep so linear-in-time interpolation stays within `tolerance`."""
    if len(points) <= 2:
        return list(range(len(points)))
    keep, anchor = [0], 0
    for i in range(2, len(points)):
        a, b = points[anchor], points[i]
        span = b[1] - a[1]
        ok = True
        for m in range(anchor + 1, i):
            w = (points[m][1] - a[1]) / span if span > 0 else 0.0
            interp = [p + (q - p) * w for p, q in zip(a[2], b[2])]
            if max(abs(x - y) for x, y in zip(interp, points[m][2])) > tolerance + EPS:
                ok = False
                break
        if not ok:
            keep.append(i - 1)
            anchor = i - 1
    keep.append(len(points) - 1)
    return keep


def _centre_match(a, b, centre: float) -> bool:
    """Centres within `centre` (normalised) and widths/heights within a factor of 2: the same small object that moved
    by about its own size (IoU 0 for a far-away person)."""
    ax, ay = (a[0] + a[2]) / 2, (a[1] + a[3]) / 2
    bx, by = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
    if ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5 > centre + EPS:
        return False
    for sa, sb in ((a[2] - a[0], b[2] - b[0]), (a[3] - a[1], b[3] - b[1])):
        if sa <= 0 or sb <= 0 or not 0.5 <= sa / sb <= 2.0:
            return False
    return True


def _match(a, b, iou: float, centre: float):
    """(IoU, centre distance) when `b` continues `a`, else None."""
    score = _iou(a, b)
    if score < iou and not _centre_match(a, b, centre):
        return None
    dist = (((a[0] + a[2]) - (b[0] + b[2])) ** 2 + ((a[1] + a[3]) - (b[1] + b[3])) ** 2) ** 0.5 / 2
    return score, dist


def tracks_from_weak_labels(frames: list, iou: float = 0.2, tolerance: float = 0.02, max_gap: int = 3,
                            centre: float = 0.08, min_frames: int = 2) -> list:
    """Link YOLO weak detections into suggestion tracks.

    frames: [(frame_index, t_sec, boxes)] in time order, one entry per sampled frame; boxes: [(label, xyxy)] (or
    dicts with label/xyxy), normalised.

    Per frame and class, detections are assigned to live tracks greedily, best IoU first, then nearest centre; a
    detection continues a track when IoU >= `iou` with the track's last box, or its centre is within `centre` of it
    with a similar size. A track survives up to `max_gap` sampled frames without a detection (the box is
    interpolated across the gap); a longer absence -- or one that lasts to the end of the clip -- ends it with an
    enabled=False keyframe at the first missed frame. Fragments of one object (one ends, a matching one starts within
    the gap rule) are merged. Tracks seen on fewer than `min_frames` frames are dropped unless the clip has fewer
    than 4 sampled frames. Keyframes are kept only where linear interpolation would be off by more than `tolerance`
    (fraction of the frame)."""
    frames = list(frames)
    live: list = []
    done: list = []
    for pos, (frame, t, boxes) in enumerate(frames):
        dets = [_det(d) for d in boxes]
        pairs = []
        for ti, tr in enumerate(live):
            for di, (lab, b) in enumerate(dets):
                if lab != tr["label"]:
                    continue
                m = _match(tr["points"][-1][2], b, iou, centre)
                if m is not None:
                    pairs.append((-m[0], m[1], ti, di))
        pairs.sort()
        used_t, used_d = set(), set()
        for _, _, ti, di in pairs:
            if ti in used_t or di in used_d:
                continue
            used_t.add(ti)
            used_d.add(di)
            tr = live[ti]
            tr["points"].append((frame, t, dets[di][1]))
            tr["pos"].append(pos)
            tr["miss"] = None
        still = []
        for ti, tr in enumerate(live):
            if ti not in used_t:
                if tr["miss"] is None:
                    tr["miss"] = (pos, frame, t)
                if pos - tr["miss"][0] + 1 > max_gap:  # gone for longer than the gap rule: hidden from the first miss
                    done.append(tr)
                    continue
            still.append(tr)
        live = still
        for di, (lab, b) in enumerate(dets):
            if di not in used_d:
                live.append({"label": lab, "points": [(frame, t, b)], "pos": [pos], "miss": None})
    tracks = sorted(done + live, key=lambda x: x["pos"][0])
    tracks = _merge_fragments(tracks, iou, centre, max_gap)
    if len(frames) >= 4:
        tracks = [tr for tr in tracks if len(tr["points"]) >= min_frames]
    out = []
    for n, tr in enumerate(tracks, start=1):
        pts = tr["points"]
        kfs = [Keyframe(frame=pts[i][0], t_sec=pts[i][1], xyxy=list(pts[i][2]), enabled=True)
               for i in _sparse(pts, tolerance)]
        if tr["miss"] is not None:  # not seen again before it ended or the clip did
            _, frame, t = tr["miss"]
            kfs.append(Keyframe(frame=frame, t_sec=t, xyxy=list(pts[-1][2]), enabled=False))
        out.append(Track(track_id=f"s{n}", label=tr["label"], keyframes=kfs, source="suggestion"))
    return out


def _merge_fragments(tracks: list, iou: float, centre: float, max_gap: int) -> list:
    """Join a track that ended to one of the same class that starts after it, within `max_gap` missed sampled frames,
    with a matching first box (in start order; each fragment joins at most one predecessor)."""
    out: list = []
    for tr in tracks:
        target = None
        for prev in out:
            if prev["label"] != tr["label"] or prev["miss"] is None:
                continue
            gap = tr["pos"][0] - prev["pos"][-1] - 1
            if 0 <= gap <= max_gap and _match(prev["points"][-1][2], tr["points"][0][2], iou, centre) is not None:
                target = prev
                break
        if target is None:
            out.append(tr)
            continue
        target["points"] += tr["points"]
        target["pos"] += tr["pos"]
        target["miss"] = tr["miss"]
    return out


def validate_tracks(tracks: list, duration_sec: float) -> list:
    """Human-readable problems (empty list = valid)."""
    problems = []
    known = set(COCO_NAMES.values())
    for tr in tracks:
        who = f"track {tr.track_id}"
        if tr.label not in known:
            problems.append(f"{who}: unknown label {tr.label!r}")
        times = [k.t_sec for k in tr.keyframes]
        if any(b <= a for a, b in zip(times, times[1:])):
            problems.append(f"{who}: keyframe times must be sorted and unique")
        for k in tr.keyframes:
            if len(k.xyxy) != 4 or any(not 0.0 <= v <= 1.0 for v in k.xyxy):
                problems.append(f"{who}: box at frame {k.frame} must be four values within 0..1")
            elif not (k.xyxy[0] < k.xyxy[2] and k.xyxy[1] < k.xyxy[3]):
                problems.append(f"{who}: box at frame {k.frame} needs x1<x2 and y1<y2")
            if not 0.0 <= k.t_sec <= duration_sec + EPS:
                problems.append(f"{who}: keyframe at {k.t_sec}s is outside the clip (0..{duration_sec}s)")
    return problems
