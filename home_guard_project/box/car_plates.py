"""Car plates on the box: read them where they can be read, and name the cars the owner named.

Owner (2026-10-19, after three "suspicious" alerts about the electrician's parked car): "the car it said was
suspicious had a specific camera where the car number was clear; once it has the number and I told it the car is the
worker's, it should connect that number to the worker". The reader is plates.py (benchmark: docs/plates_design.md);
the owner's names are known_vehicles.py; the assistant binds them (assistant_v3/vehicles.py).

Where plates are read (all off the detection loop, in ONE worker thread, one job at a time, a queue of
``QUEUE_MAX`` where the oldest job is dropped):

- **Alerts** on the cameras in box.yaml ``plate_cameras`` (default none; this house: ch3 and ch6, the only two whose
  main stream shows plates at 50 px or more), by day only (``situation.phase_of``: not dark) and only from MAIN-stream
  pixels: on the detection loop :func:`alert_job` cuts each vehicle the tracker followed in the clip (parked ones too)
  out of the main-stream window at native pixels with plates.CROP_MARGIN (never resized), at most ``CUT_FRAMES``
  frames per vehicle, plus the same place in the sub stream at the nearest moment. The worker drops grey IR frames
  and frames whose crop disagrees with the sub stream (frame_check's test on the vehicle's own region: the main
  stream loses pieces on the box's Wi-Fi, 2026-10-12), keeps the best ``plate_frames`` (plates.best_frames), reads,
  votes (plates.vote: two reads within one digit) and matches the owner's cars (plates.match, within one digit).
- **The owner's live looks** (a live photo's ``_full.jpg``, a live video, both main stream) when he talks about the car
  in them (:meth:`PlateService.read_media`, waited for by the assistant): the worker finds the vehicles itself with
  its own copy of the box's detector (never the guard loop's), then the same steps.

What is kept, and where (the full plate never leaves the box):

- ``<events dir>/plates/reads.jsonl``: one line per job with each vehicle's voted plate IN FULL, for binding a name
  later ("this is Mansour's car", answering an alert minutes or hours after it). Lines older than ``KEEP_DAYS`` are
  dropped: a plate the owner never named is not kept beyond that.
- Everything else gets the masked plate only (plates.mask, "****875"): the clip's meta (which is uploaded), the event
  entity (``plate_masked``, ``plate_reads``, ``vehicle_id``, and the owner's name for it as ``owner_label``), logs,
  the alert text and the assistant's context. Night, the package missing, a camera not listed: nothing is read, and
  the reason is logged once.
"""
from __future__ import annotations

import collections
import json
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from . import plates

log = logging.getLogger("box.car_plates")

QUEUE_MAX = 4
CUT_FRAMES = 10            # main-stream frames cut per vehicle on the detection loop (spread over its time in view)
MAX_VEHICLES = 3           # per alert: the biggest ones (the closest plates)
MEDIA_FRAMES = 8           # frames of a live video looked at (spread)
JOB_SECONDS = 20.0         # a job longer than this is abandoned between vehicles
KEEP_DAYS = 3.0
RESULTS_KEPT = 200
VEHICLE_CLASSES = ("car", "truck", "bus", "motorcycle", "motorbike")
DETECT_CONF = 0.30
DETECT_IOU = 0.30          # one vehicle across a live video's frames: boxes overlapping this much
SINGLE_MARGINS = (0.10, 0.20, 0.30)   # a live photo: one vehicle cut three ways
READS_NAME = os.path.join("plates", "reads.jsonl")


# ----------------------------------------------------------------------------------------------------------------
# Settings (box.yaml)
# ----------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class PlateSettings:
    enabled: bool = True                  # box.yaml plates: off turns every read off (alerts and the owner's)
    cameras: Tuple[str, ...] = ()         # plate_cameras: alerts read only here
    device: str = "GPU"                   # plate_device: GPU (the iGPU, CPU fallback) | CPU
    frames: int = 5                       # plate_frames: best frames read per vehicle (benchmark: 5)

    def wants(self, camera: str) -> bool:
        return self.enabled and camera in self.cameras


def settings_of(box_settings: Optional[Mapping[str, Any]]) -> PlateSettings:
    s = box_settings if isinstance(box_settings, Mapping) else {}
    raw = s.get("plate_cameras") or ()
    cams = tuple(str(c).strip() for c in (raw if isinstance(raw, (list, tuple)) else str(raw).split(","))
                 if str(c).strip())
    enabled = str(s.get("plates", "on")).strip().lower() not in ("off", "false", "0", "no")
    device = str(s.get("plate_device") or "GPU").strip().upper()
    try:
        frames = max(1, min(10, int(s.get("plate_frames") or 5)))
    except (TypeError, ValueError):
        frames = 5
    return PlateSettings(enabled, cams, "CPU" if device == "CPU" else "GPU", frames)


def is_day(ts: float) -> bool:
    from .situation import phase_of  # noqa: PLC0415

    try:
        return not phase_of(float(ts))[1]
    except Exception:  # noqa: BLE001 - no sun table: the frames' own IR test decides
        return True


# ----------------------------------------------------------------------------------------------------------------
# Jobs
# ----------------------------------------------------------------------------------------------------------------
@dataclass
class Shot:
    """One vehicle's main-stream crops (with the vehicle box inside each), their pixel boxes in the main frame, and
    per crop the same place in the sub stream (None: no check)."""
    key: str                                   # the tracker key ("<id>@<first_seen>") or "v<n>" in a live look
    crops: List[plates.VehicleCrop] = field(default_factory=list)
    boxes: List[Tuple[float, float, float, float]] = field(default_factory=list)
    subs: List[Optional[np.ndarray]] = field(default_factory=list)
    frame_size: Optional[Tuple[int, int]] = None
    box: Optional[Tuple[float, float, float, float]] = None   # where it stood, normalised (stayed_vehicles.py)


@dataclass
class Job:
    key: str                                   # the alert id, or the media file read
    camera: str
    ts: float
    kind: str = "alert"                        # alert | media
    shots: List[Shot] = field(default_factory=list)
    media: str = ""                            # media: the file to read
    done: threading.Event = field(default_factory=threading.Event)
    result: Optional[Dict[str, Any]] = None


def _box_at(boxes: Sequence[Mapping[str, Any]], ts: float) -> Optional[Tuple[float, ...]]:
    from .describer import box_at  # noqa: PLC0415 - the describer's own interpolation, so ids and pixels agree

    return box_at(boxes, ts)


def _spread(items: Sequence[Any], n: int) -> List[Any]:
    if len(items) <= n:
        return list(items)
    return [items[round(i * (len(items) - 1) / (n - 1))] for i in range(n)] if n > 1 else [items[len(items) // 2]]


def _times(n: int, start: float, end: float) -> List[float]:
    if n <= 1:
        return [float(start)] * n
    return [float(start) + i * (float(end) - float(start)) / (n - 1) for i in range(n)]


def _sub_region(sub_clip: Sequence[Tuple[float, Any]], ts: float, box: Sequence[float],
                margin: float = plates.CROP_MARGIN) -> Optional[np.ndarray]:
    """The vehicle's padded place in the sub-stream frame nearest *ts* (a copy), for the damage check."""
    if not sub_clip:
        return None
    t, frame = min(sub_clip, key=lambda x: abs(float(x[0]) - ts))
    if not isinstance(frame, np.ndarray) or frame.ndim < 2 or abs(float(t) - ts) > 1.0:
        return None
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = (float(v) for v in box[:4])
    mx, my = (x2 - x1) * margin, (y2 - y1) * margin
    a, b = max(0, int((x1 - mx) * w)), max(0, int((y1 - my) * h))
    c, d = min(w, int(round((x2 + mx) * w))), min(h, int(round((y2 + my) * h)))
    if c - a < 4 or d - b < 4:
        return None
    return frame[b:d, a:c].copy()


def alert_job(alert_id: str, camera: str, ts: float, main: Any, sub_clip: Sequence[Tuple[float, Any]],
              tracks: Sequence[Mapping[str, Any]], max_vehicles: int = MAX_VEHICLES,
              cut_frames: int = CUT_FRAMES) -> Optional[Job]:
    """The plate job of one alert, cut on the detection loop (copies, so the window can go): every vehicle the
    tracker followed in the clip (parked ones too), the biggest *max_vehicles* first. *main* is the main-stream
    window ``(frames, start, end)``; *sub_clip* the sub stream's ``[(ts, frame)]``. None when nothing can be read."""
    try:
        frames, start, end = main[0], float(main[1]), float(main[2])
    except (TypeError, IndexError, ValueError):
        return None
    frames = [f for f in frames or () if isinstance(f, np.ndarray) and f.ndim == 3]
    if len(frames) < 1:
        return None
    times = _times(len(frames), start, end)
    h, w = frames[0].shape[:2]
    vehicles = []
    for t in tracks or ():
        if t.get("kind") != "vehicle" or not t.get("boxes"):
            continue
        sizes = [(float(b["box"][2]) - float(b["box"][0])) * (float(b["box"][3]) - float(b["box"][1]))
                 for b in t["boxes"] if b.get("box") is not None and len(b["box"]) == 4]
        if sizes:
            vehicles.append((max(sizes), t))
    vehicles.sort(key=lambda x: -x[0])
    from .entities import track_key  # noqa: PLC0415

    job = Job(alert_id, camera, float(ts))
    for _, t in vehicles[:max(0, max_vehicles)]:
        placed = [(i, b) for i, b in ((i, _box_at(t["boxes"], times[i])) for i in range(len(frames))) if b is not None]
        placed = [(i, b) for i, b in placed if (b[2] - b[0]) * w >= plates.MIN_PLATE_WIDTH]
        from .stayed_vehicles import median_box  # noqa: PLC0415

        shot = Shot(track_key(dict(t)), frame_size=(w, h), box=median_box([b for _, b in placed]))
        for i, b in _spread(placed, cut_frames):
            cut = plates.vehicle_crops([frames[i]], b, max_frames=1)
            if not cut:
                continue
            c = cut[0]
            shot.crops.append(plates.VehicleCrop(np.ascontiguousarray(c.image).copy(), c.inner))
            shot.boxes.append((b[0] * w, b[1] * h, b[2] * w, b[3] * h))
            shot.subs.append(_sub_region(sub_clip, times[i], b))
        if shot.crops:
            job.shots.append(shot)
    return job if job.shots else None


# ----------------------------------------------------------------------------------------------------------------
# The worker's steps
# ----------------------------------------------------------------------------------------------------------------
def damage(main_crop: np.ndarray, sub_crop: Optional[np.ndarray]) -> float:
    """How far a main-stream vehicle crop is from the sub stream's same place (frame_check's measure on one region,
    against one sub frame): 0 the same picture, frame_check.FRAME_ODD and up damaged. 0 without a sub crop."""
    if sub_crop is None:
        return 0.0
    from . import frame_check as fc  # noqa: PLC0415

    try:
        raw = fc._small(main_crop)
        env = fc._envelope(fc._small(sub_crop))
        if raw is None or env is None:
            return 0.0
        lo, hi, contrast = env
        if contrast >= fc.MIN_CONTRAST and float(raw.std()) < fc.FLAT * contrast:
            return 1.0                                        # a flat field where the sub shows a scene
        a = fc._norm(raw)
        off = np.maximum(np.maximum(lo - a, a - hi), 0.0)
        return float(np.mean(off > fc.CELL))
    except Exception:  # noqa: BLE001 - an odd crop is read as it is; the vote still needs two reads
        return 0.0


def read_shot(reader: Any, shot: Shot, k: int = 5) -> Tuple[Optional[str], int, Dict[str, int]]:
    """``(voted plate or None, reads behind it, counts)`` for one vehicle: IR and damaged crops out, the best *k*
    of the rest read, then plates.vote."""
    from .frame_check import FRAME_ODD  # noqa: PLC0415

    counts = {"crops": len(shot.crops), "ir": 0, "damaged": 0, "read": 0}
    keep: List[int] = []
    for i, c in enumerate(shot.crops):
        if plates.is_ir(c.image):
            counts["ir"] += 1
        elif damage(c.image, shot.subs[i] if i < len(shot.subs) else None) >= FRAME_ODD:
            counts["damaged"] += 1
        else:
            keep.append(i)
    if not keep:
        return None, 0, counts
    boxes = [shot.boxes[i] if i < len(shot.boxes) else (0.0, 0.0, 1.0, 1.0) for i in keep]
    best = plates.best_frames(boxes, [plates.sharpness(shot.crops[i].image) for i in keep], k=k,
                              frame_size=shot.frame_size)
    chosen = [shot.crops[keep[j]] for j in best]
    counts["read"] = len(chosen)
    reads = reader.read(chosen)
    plate = plates.vote(reads)
    agree = sum(1 for r in reads if plate and plates.levenshtein(r.digits, plate) <= 1)
    return plate, agree, counts


def _iou(a: Sequence[float], b: Sequence[float]) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def media_frames(path: str, max_frames: int = MEDIA_FRAMES) -> List[np.ndarray]:
    """The main-stream frames of a live look: a photo's ``_full.jpg`` (the photo itself when there is none), or up to
    *max_frames* frames spread over a live video."""
    import cv2  # noqa: PLC0415

    if path.lower().endswith((".jpg", ".jpeg", ".png")):
        root, ext = os.path.splitext(path)
        full = path if root.endswith("_full") else root + "_full" + ext
        frame = cv2.imread(full if os.path.exists(full) else path)
        return [frame] if frame is not None else []
    cap = cv2.VideoCapture(path)
    try:
        frames = []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(frame)
            if len(frames) > 400:
                break
        return _spread(frames, max_frames)
    finally:
        cap.release()


def media_shots(frames: Sequence[np.ndarray], detect: Callable[[np.ndarray], List[Tuple[float, ...]]]) -> List[Shot]:
    """Vehicles across a live look's frames, by box overlap (normalised boxes from *detect*), each with its crops.
    A single picture (a live photo) gives each vehicle three crops at different margins: the plate detector then sees
    three slightly different pictures, so plates.vote can still ask for two reads that agree."""
    margins = SINGLE_MARGINS if len(frames) == 1 else (plates.CROP_MARGIN,)
    groups: List[Dict[str, Any]] = []
    for f in frames:
        if not isinstance(f, np.ndarray) or f.ndim != 3:
            continue
        h, w = f.shape[:2]
        for box in detect(f):
            g = next((g for g in groups if _iou(g["last"], box) >= DETECT_IOU), None)
            if g is None:
                g = {"last": box, "shot": Shot(f"v{len(groups) + 1}", frame_size=(w, h))}
                groups.append(g)
            g["last"] = box
            g["shot"].box = tuple(float(v) for v in box)
            for margin in margins:
                cut = plates.vehicle_crops([f], box, margin=margin, max_frames=1)
                if cut:
                    g["shot"].crops.append(cut[0])
                    g["shot"].boxes.append((box[0] * w, box[1] * h, box[2] * w, box[3] * h))
                    g["shot"].subs.append(None)
    return [g["shot"] for g in groups if g["shot"].crops]


def yolo_vehicles(model: Any, predict_args: Optional[Mapping[str, Any]] = None) -> Callable[[np.ndarray], List[Tuple[float, ...]]]:
    """``detect(frame) -> [normalised x1, y1, x2, y2]`` of the vehicles an ultralytics model sees."""
    def detect(frame: np.ndarray) -> List[Tuple[float, ...]]:
        res = model(frame, conf=DETECT_CONF, verbose=False, **dict(predict_args or {}))
        r = res[0] if res else None
        if r is None or r.boxes is None:
            return []
        names = getattr(r, "names", None) or getattr(model, "names", {}) or {}
        out = []
        for box, cls in zip(r.boxes.xyxyn.cpu().numpy().tolist(), r.boxes.cls.cpu().numpy().tolist()):
            if str(names.get(int(cls), "")) in VEHICLE_CLASSES:
                out.append(tuple(float(v) for v in box))
        return out
    return detect


# ----------------------------------------------------------------------------------------------------------------
# The service
# ----------------------------------------------------------------------------------------------------------------
class PlateService:
    """The box's one plate worker. *reader* (plates.PlateReader) and *detector* (``() -> detect(frame)``, for live
    looks) load lazily in the worker; *store* (known_vehicles.KnownVehicles) names matches; *reads_path* is the
    local reads log."""

    def __init__(self, settings: PlateSettings, reader: Any = None, detector: Optional[Callable[[], Any]] = None,
                 store: Any = None, reads_path: str = "", now: Callable[[], float] = time.time,
                 start: bool = True, stayed: Any = None) -> None:
        self.settings = settings
        self.stayed = stayed            # stayed_vehicles.StayedVehicles: a named car keeps its name on its spot
        self.reader = reader if reader is not None else plates.PlateReader(device=settings.device)
        self._detector_factory = detector
        self._detect: Optional[Callable[[np.ndarray], List[Tuple[float, ...]]]] = None
        self.store = store
        self.reads_path = reads_path
        self._now = now
        self._lock = threading.Lock()
        self._wake = threading.Condition(self._lock)
        self._queue: "collections.deque[Job]" = collections.deque()
        self._results: "collections.OrderedDict[str, Dict[str, Any]]" = collections.OrderedDict()
        self._jobs: Dict[str, Job] = {}
        self._said: set = set()
        self._pruned = 0.0
        self._thread: Optional[threading.Thread] = None
        if start:
            self._thread = threading.Thread(target=self._run, name="plates", daemon=True)
            self._thread.start()

    # ---- queue ----
    def submit(self, job: Job, first: bool = False) -> bool:
        with self._lock:
            while len(self._queue) >= QUEUE_MAX:
                dropped = self._queue.pop() if first else self._queue.popleft()
                dropped.result = {"key": dropped.key, "skipped": "queue full"}
                dropped.done.set()
                log.info("plates: queue full, dropped %s", dropped.key)
            if first:
                self._queue.appendleft(job)
            else:
                self._queue.append(job)
            self._jobs[job.key] = job
            self._wake.notify()
        return True

    def result(self, key: str, wait: float = 0.0) -> Optional[Dict[str, Any]]:
        """The result of job *key* (waiting up to *wait* s while it is queued or running); else the reads log."""
        with self._lock:
            job = self._jobs.get(key)
            done = self._results.get(key)
        if done is not None:
            return done
        if job is not None and wait > 0 and job.done.wait(wait):
            return job.result
        if job is not None:
            return None
        return self._logged(key)

    def read_media(self, path: str, camera: str, wait: float = JOB_SECONDS) -> Optional[Dict[str, Any]]:
        """Read the cars in a live look the owner is talking about (first in the queue), waiting up to *wait* s."""
        if not self.settings.enabled:
            return {"key": path, "skipped": "plates off"}
        done = self.result(path)
        if done is not None and not done.get("skipped"):
            return done
        job = Job(path, camera, self._now(), kind="media", media=path)
        self.submit(job, first=True)
        return job.result if job.done.wait(wait) else None

    def run_once(self, job: Job) -> Dict[str, Any]:
        """Do one job now, in this thread (the worker's step; tests call it directly)."""
        started = time.monotonic()
        out: Dict[str, Any] = {"key": job.key, "camera": job.camera, "ts": job.ts, "kind": job.kind, "vehicles": {}}
        try:
            if not self.reader.available():
                out["skipped"] = "no plate reader (fast-alpr not installed or its models missing)"
            else:
                shots = job.shots
                if job.kind == "media":
                    detect = self._detector()
                    if detect is None:
                        out["skipped"] = "no vehicle detector"
                        shots = []
                    else:
                        shots = media_shots(media_frames(job.media), detect)
                for shot in shots:
                    if time.monotonic() - started > JOB_SECONDS:
                        out["cut_short"] = True
                        break
                    plate, agree, counts = read_shot(self.reader, shot, self.settings.frames)
                    v: Dict[str, Any] = {"plate": plate or "", "reads": agree, **counts}
                    if shot.box is not None:
                        v["box"] = [round(float(x), 4) for x in shot.box]
                    known = self.store.match(plate) if plate and self.store is not None else None
                    if known:
                        v.update(vehicle_id=str(known.get("id") or ""), label=str(known.get("label") or ""))
                    if plate and shot.box is not None and self.stayed is not None:
                        self.stayed.plate_read(job.camera, shot.box, plate, known,
                                               job.ts if job.kind == "alert" else self._now())
                    out["vehicles"][shot.key] = v
        except Exception as exc:  # noqa: BLE001 - a plate never stops anything
            log.warning("plates: %s failed: %s", job.key, exc)
            out["error"] = f"{type(exc).__name__}: {exc}"[:200]
        out["seconds"] = round(time.monotonic() - started, 2)
        self._remember(out)
        return out

    def _detector(self) -> Optional[Callable[[np.ndarray], List[Tuple[float, ...]]]]:
        if self._detect is None and self._detector_factory is not None:
            try:
                self._detect = self._detector_factory()
            except Exception as exc:  # noqa: BLE001
                log.warning("plates: no vehicle detector for live looks (%s)", exc)
                self._detector_factory = None
        return self._detect

    def _run(self) -> None:
        while True:
            with self._lock:
                while not self._queue:
                    self._wake.wait()
                job = self._queue.popleft()
            job.result = self.run_once(job)
            with self._lock:
                self._jobs.pop(job.key, None)
            job.done.set()

    # ---- results ----
    def _remember(self, out: Dict[str, Any]) -> None:
        with self._lock:
            self._results[out["key"]] = out
            while len(self._results) > RESULTS_KEPT:
                self._results.popitem(last=False)
        found = {k: v for k, v in out["vehicles"].items() if v.get("plate")}
        if out.get("skipped"):
            reason = str(out["skipped"])
            if reason not in self._said:
                self._said.add(reason)
                log.info("plates: nothing read (%s)", reason)
        else:
            log.info("plates: %s %s: %s (%.2f s)", out.get("camera"), out["key"],
                     ", ".join(f"{k} {plates.mask(v['plate'])} x{v['reads']}" + (f" = {v['label']}" if v.get('label')
                                                                                 else "") for k, v in found.items())
                     or "no plate", out.get("seconds", 0.0))
        if found and self.reads_path:
            self._log(out)

    def _log(self, out: Dict[str, Any]) -> None:
        line = {"ts": self._now(), "key": out["key"], "camera": out.get("camera", ""), "kind": out.get("kind", ""),
                "at": out.get("ts", 0.0), "vehicles": {k: {"plate": v["plate"], "reads": v["reads"],
                                                           "vehicle_id": v.get("vehicle_id", "")}
                                                       for k, v in out["vehicles"].items() if v.get("plate")}}
        try:
            os.makedirs(os.path.dirname(self.reads_path), exist_ok=True)
            with self._lock:
                with open(self.reads_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(line, ensure_ascii=False) + "\n")
                    f.flush()
                    os.fsync(f.fileno())
            if self._now() - self._pruned > 3600.0:
                self._pruned = self._now()
                self.prune()
        except OSError as exc:
            log.warning("plates: reads log not written: %s", exc)

    def _rows(self) -> List[Dict[str, Any]]:
        try:
            with open(self.reads_path, encoding="utf-8") as f:
                rows = []
                for raw in f:
                    try:
                        rows.append(json.loads(raw))
                    except ValueError:
                        continue
                return rows
        except OSError:
            return []

    def prune(self) -> None:
        """Drop reads older than KEEP_DAYS (the full plates of cars the owner never named)."""
        cutoff = self._now() - KEEP_DAYS * 86400.0
        with self._lock:
            rows = self._rows()
            keep = [r for r in rows if float(r.get("ts") or 0) >= cutoff]
            if len(keep) == len(rows):
                return
            tmp = self.reads_path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                for r in keep:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, self.reads_path)

    def _logged(self, key: str) -> Optional[Dict[str, Any]]:
        if not self.reads_path:
            return None
        with self._lock:
            rows = [r for r in self._rows() if r.get("key") == key]
        if not rows:
            return None
        r = rows[-1]
        return {"key": key, "camera": r.get("camera", ""), "ts": r.get("at", 0.0), "kind": r.get("kind", ""),
                "vehicles": {k: dict(v) for k, v in (r.get("vehicles") or {}).items()}, "from_log": True}


# ----------------------------------------------------------------------------------------------------------------
# What the rest of the box reads
# ----------------------------------------------------------------------------------------------------------------
def plates_found(result: Optional[Mapping[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """The vehicles of a result that have a plate, by key."""
    return {k: dict(v) for k, v in ((result or {}).get("vehicles") or {}).items() if v.get("plate")}


def meta_record(result: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    """The clip meta's ``plates`` (uploaded with the clip): masked plates only."""
    if not result:
        return {}
    out: Dict[str, Any] = {k: result[k] for k in ("skipped", "seconds", "error", "cut_short") if result.get(k)}
    out["vehicles"] = {k: {"plate": plates.mask(v.get("plate")), "reads": v.get("reads", 0),
                           **({"vehicle_id": v["vehicle_id"]} if v.get("vehicle_id") else {}),
                           **{c: v[c] for c in ("crops", "ir", "damaged", "read") if c in v}}
                       for k, v in (result.get("vehicles") or {}).items()}
    return out


def entity_fields(result: Optional[Mapping[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Per tracker key, what an event entity keeps of its plate (events.EventBook.note_vehicles): masked only."""
    out = {}
    for k, v in plates_found(result).items():
        fields = {"plate_masked": plates.mask(v["plate"]), "plate_reads": int(v.get("reads") or 0)}
        if v.get("vehicle_id"):
            fields.update(vehicle_id=str(v["vehicle_id"]), vehicle_label=str(v.get("label") or ""))
        out[k] = fields
    return out


def named_ids(result: Optional[Mapping[str, Any]], ids_by_key: Mapping[str, str]) -> Dict[str, str]:
    """CAR id in the alert text -> the owner's name for that car, for the vehicles that matched one he named."""
    out = {}
    for key, v in plates_found(result).items():
        car = ids_by_key.get(key)
        if car and v.get("label"):
            out[car] = str(v["label"])
    return out


def name_entities(entities: Sequence[Mapping[str, Any]], names: Mapping[str, str],
                  drawn: Sequence[str] = ()) -> List[Dict[str, Any]]:
    """The describer's per-id lines with the owner's name for a car he named in place of its look ("CAR1 · הרכב של
    החשמלאי מנסור: חונה בשביל הכניסה."). A named car the describer drew (*drawn*) but wrote nothing about gets
    its name alone. Other lines are untouched."""
    out = [dict(e) for e in entities]
    for e in out:
        name = names.get(str(e.get("id") or ""))
        if name:
            e["appearance"] = name
    have = {str(e.get("id") or "") for e in out}
    for car, name in names.items():
        if car not in have and car in drawn:
            out.append({"id": car, "appearance": name, "action": ""})
    return out


_SERVICE: Optional[PlateService] = None
_SERVICE_LOCK = threading.Lock()


def service(box_settings: Optional[Mapping[str, Any]] = None, events_dir: str = "",
            detector: Optional[Callable[[], Any]] = None) -> Optional[PlateService]:
    """The box's one plate service (the guard loop and the assistant share it), made on the first call with
    settings; None when plates are off in box.yaml or nobody started it yet."""
    global _SERVICE
    with _SERVICE_LOCK:
        if _SERVICE is None and box_settings is not None:
            settings = settings_of(box_settings)
            if not settings.enabled:
                log.info("plates: off (box.yaml plates: off)")
                return None
            if not events_dir:
                from . import paths  # noqa: PLC0415

                events_dir = os.path.join(paths.state_dir(), "events")
            from . import stayed_vehicles  # noqa: PLC0415
            from .known_vehicles import store_in  # noqa: PLC0415

            stayed = stayed_vehicles.store_in(events_dir, stayed_vehicles.empty_sec_of(box_settings))
            _SERVICE = PlateService(settings, detector=detector or default_detector(box_settings),
                                    store=store_in(events_dir), reads_path=os.path.join(events_dir, READS_NAME),
                                    stayed=stayed)
            log.info("plates: on for alerts at %s, by day; live looks on request; device %s",
                     ", ".join(settings.cameras) or "no camera", settings.device)
        return _SERVICE


def default_detector(box_settings: Mapping[str, Any]) -> Callable[[], Any]:
    """A loader of the worker's own copy of the box's detector (never the guard loop's instance)."""
    def load() -> Callable[[np.ndarray], List[Tuple[float, ...]]]:
        from . import paths  # noqa: PLC0415
        from .inference import AlertSettings, load_detector  # noqa: PLC0415

        s = AlertSettings.from_box_settings(dict(box_settings))
        model, device = load_detector(paths.resolve_model(s.model), s.device)
        return yolo_vehicles(model, {"device": device} if device else {})
    return load
