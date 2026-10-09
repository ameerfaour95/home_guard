"""Same-day appearance re-link (stage 3.2): is this new person the one we lost a while ago? By their CLOTHES only.

Owner (2026-10-08): keep P1 as P1 even after a longer disappearance. The tracker links a return only near where
someone was lost (tracker.RETURN_SEC, entities.REATTACH_SEC); a worker who walks off for five minutes and comes back
at the other end of the yard is a new "P3" and a new "1 more person arrived". An appearance embedding of the whole
person crop (what they wear) closes that gap, and is counter-evidence when geometry alone would merge two people.

Agreed privacy rules (fix plan, privacy section), enforced here:
- **Clothing only**: one embedding of the person's box (OpenVINO person-reidentification-retail-0277, 256x128 BGR
  in, 256 floats out). Never a face crop, never gait, never body measurements.
- **Same day only**: embeddings live in memory, never on disk, and are dropped when the event (session) holding them
  closes, after HISTORY_KEEP_SEC when no event ever held them, and after MAX_AGE_SEC (24 h) at the latest.
- **Never linked to a name**: the store is keyed by camera and tracker track key (``"<id>@<first_seen>"``) only;
  the owner's words stay in the event, not here.
- Used only to re-link a lost entity of the same (or rolled-over) session and as counter-evidence (entities.py), and
  to confirm a cross-camera hand-over (events.py, ``cross_camera``).

Cost budget (N150, YOLO on the iGPU at ~65-89 ms a frame): only CONFIRMED person tracks, at most PER_TRACK looks each
(the best detector scores), at most RATE_PER_SEC embeddings a second across all cameras, in a background thread so
the detection loop only copies a small crop. ~2 GFLOPs per crop.

box.yaml::

    reid: shadow            # off | shadow (the default: compute and log "would link P3->P1 (0.78)") | on
    reid_device: AUTO       # AUTO (graphics chip when there is one, else the CPU) | CPU | GPU
    reid_link: 0.70         # cosine to re-link a lost entity ...
    reid_margin: 0.08       # ... and better than the second candidate by this much
    reid_veto: 0.35         # geometry would re-attach, but the clothes match less than this: a new entity

Without the model files (``python -m home_guard_project.box.reid download``) ReID is simply off.
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import math
import os
import sys
import threading
import time
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

log = logging.getLogger("box.reid")

MODEL_NAME = "person-reidentification-retail-0277"
MODEL_FILE = MODEL_NAME + ".xml"
INPUT_H, INPUT_W = 256, 128
DIM = 256
MODES = ("off", "shadow", "on")
DEFAULT_MODE = "shadow"
DEVICES = ("AUTO", "CPU", "GPU")
REID_LINK = 0.70
REID_MARGIN = 0.08
REID_VETO = 0.35
EMA_RHO = 0.3                    # an entity's (and a track's) look: mean = (1 - rho) * mean + rho * new
PER_TRACK = 3                    # embeddings kept per track: its best-scored looks
MAX_ATTEMPTS = 2 * PER_TRACK     # a better look may replace a worse one, but a track costs at most this many
BETTER_BY = 0.05                 # a look replaces the worst kept one only when its score is this much higher
RATE_PER_SEC = 5.0               # embeddings a second across all cameras
MIN_CROP_PX = 48                 # a person box shorter than this (pixels) says nothing about clothes
MAX_AGE_SEC = 24 * 3600.0        # every embedding is gone after a day, whatever holds it
HISTORY_KEEP_SEC = 660.0         # a track no event ever held: as long as the tracker keeps it (HISTORY_SEC) + 1 min
MAX_TRACKS = 400                 # memory cap across cameras (oldest dropped first)
MAX_PENDING = 24                 # crops waiting for the embedder (a crop is 96 KB)
# Open Model Zoo storage (2023.0 release) and the sha384 checksums from the zoo's own model.yml.
OMZ_BASE = ("https://storage.openvinotoolkit.org/repositories/open_model_zoo/2023.0/models_bin/1/"
            "person-reidentification-retail-0277/")
OMZ_FILES = {
    "FP16": {
        MODEL_NAME + ".xml": (616140, "554b403108fea776cb54c971c4c8acfaaeaae3884b5945eefbcbad8b9ed3c84f"
                                      "0382cb72c666c02c07129826a668d79e"),
        MODEL_NAME + ".bin": (4178806, "0466aa71c44a14aa902a609c6500f4bc087cabbfa09c1e5bfb309f0a7e89fcdd"
                                       "9971e0359663e31803e3611bfa4d2353"),
    },
    "FP32": {
        MODEL_NAME + ".xml": (451948, "8b9bd45ee44c67f08b88c82c33cff1bb0f9e7e687c52741461577badbdb7cde1"
                                      "46e49fffedb1f813230cd47420af3470"),
        MODEL_NAME + ".bin": (8357492, "adf16b7292f3c3858bd642eed8e471072499872592e242ca2be75fd50100179f"
                                       "72a875b05c1907f56d3d970d751dcba0"),
    },
}


# ----------------------------------------------------------------------------
# Settings
# ----------------------------------------------------------------------------
def _mode(value: Any, default: str = DEFAULT_MODE) -> str:
    """A box.yaml off | shadow | on switch (YAML booleans too); anything else is *default*."""
    if value is False:
        return "off"
    if value is True:
        return "on"
    text = str(default if value is None else value).strip().lower()
    return text if text in MODES else default


def _number(value: Any, default: float, low: float, high: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if low <= number <= high else default


@dataclass(frozen=True)
class ReidSettings:
    mode: str = DEFAULT_MODE
    device: str = "AUTO"
    link: float = REID_LINK
    margin: float = REID_MARGIN
    veto: float = REID_VETO
    model: str = MODEL_FILE

    @property
    def enabled(self) -> bool:
        return self.mode != "off"

    @classmethod
    def from_box_settings(cls, s: Optional[Mapping[str, Any]]) -> "ReidSettings":
        g = (s or {}).get
        device = str(g("reid_device", "AUTO") or "AUTO").strip().upper()
        return cls(mode=_mode(g("reid", DEFAULT_MODE)), device=device if device in DEVICES else "AUTO",
                   link=_number(g("reid_link"), REID_LINK, 0.0, 1.0),
                   margin=_number(g("reid_margin"), REID_MARGIN, 0.0, 1.0),
                   veto=_number(g("reid_veto"), REID_VETO, -1.0, 1.0),
                   model=str(g("reid_model", MODEL_FILE) or MODEL_FILE))


# ----------------------------------------------------------------------------
# Vectors (pure Python on short lists; numpy arrays work as well)
# ----------------------------------------------------------------------------
def _unit(vec: Sequence[float]) -> List[float]:
    norm = math.sqrt(sum(float(x) * float(x) for x in vec))
    return [float(x) / norm for x in vec] if norm > 0 else [0.0 for _ in vec]


def cosine(a: Optional[Sequence[float]], b: Optional[Sequence[float]]) -> Optional[float]:
    """The cosine of two embeddings, None when either is missing."""
    if a is None or b is None or len(a) == 0 or len(a) != len(b):
        return None
    ua, ub = _unit(a), _unit(b)
    return float(sum(x * y for x, y in zip(ua, ub)))


def ema(vectors: Iterable[Sequence[float]], rho: float = EMA_RHO) -> Optional[List[float]]:
    """The running look of a series of embeddings in time order: ``mean = (1 - rho) * mean + rho * new``, unit length."""
    mean: Optional[List[float]] = None
    for vec in vectors:
        unit = _unit(vec)
        mean = unit if mean is None else _unit([(1.0 - rho) * m + rho * x for m, x in zip(mean, unit)])
    return mean


# ----------------------------------------------------------------------------
# The model
# ----------------------------------------------------------------------------
def crop_person(frame: Any, box: Sequence[float]) -> Any:
    """The person's box (normalised x1, y1, x2, y2) of *frame* resized to the model's 128x256, or None when it is too
    small to say anything about clothes. The whole box, never a face crop."""
    import cv2  # noqa: PLC0415

    height, width = frame.shape[:2]
    x1, y1 = max(0, int(float(box[0]) * width)), max(0, int(float(box[1]) * height))
    x2, y2 = min(width, int(math.ceil(float(box[2]) * width))), min(height, int(math.ceil(float(box[3]) * height)))
    if y2 - y1 < MIN_CROP_PX or x2 - x1 < MIN_CROP_PX // 4:
        return None
    return cv2.resize(frame[y1:y2, x1:x2], (INPUT_W, INPUT_H), interpolation=cv2.INTER_LINEAR)


def model_path(name: str = MODEL_FILE) -> str:
    """Where the model's .xml is: a bare name lives in the box's models folder (paths.resolve_model)."""
    from . import paths  # noqa: PLC0415

    return paths.resolve_model(name)


class Embedder:
    """person-reidentification-retail-0277 through OpenVINO. *device* AUTO / CPU / GPU; a device that does not
    compile falls back to the CPU. ``embed(crop)`` takes a 128x256 BGR crop (``crop_person``)."""

    def __init__(self, xml_path: str, device: str = "AUTO") -> None:
        import numpy as np  # noqa: PLC0415
        import openvino as ov  # noqa: PLC0415

        self._np = np
        core = ov.Core()
        model = core.read_model(xml_path)
        try:
            self._compiled = core.compile_model(model, device)
            self.device = device
        except Exception as exc:  # noqa: BLE001 - a missing graphics driver: the CPU does it
            if device == "CPU":
                raise
            log.info("ReID: %s not usable (%s); on the CPU", device, exc)
            self._compiled = core.compile_model(model, "CPU")
            self.device = "CPU"
        self._request = self._compiled.create_infer_request()
        self._lock = threading.Lock()

    def embed(self, crop: Any) -> List[float]:
        np = self._np
        if crop.shape[:2] != (INPUT_H, INPUT_W):
            import cv2  # noqa: PLC0415

            crop = cv2.resize(crop, (INPUT_W, INPUT_H))
        blob = np.ascontiguousarray(crop.transpose(2, 0, 1)[None].astype(np.float32))
        with self._lock:
            out = self._request.infer({0: blob})
        vec = np.asarray(next(iter(out.values()))).reshape(-1).astype(np.float32)
        norm = float(np.linalg.norm(vec))
        return (vec / norm).tolist() if norm > 0 else vec.tolist()


def load_embedder(settings: ReidSettings) -> Optional[Embedder]:
    """The embedder, or None (ReID off) when the model files are missing or OpenVINO cannot load them."""
    path = model_path(settings.model)
    if not (os.path.isfile(path) and os.path.isfile(os.path.splitext(path)[0] + ".bin")):
        log.info("ReID off: %s not found (python -m home_guard_project.box.reid download)", path)
        return None
    try:
        embedder = Embedder(path, settings.device)
    except Exception as exc:  # noqa: BLE001 - ReID only adds; the alerts go on without it
        log.warning("ReID off: the model did not load (%s)", exc)
        return None
    log.info("ReID %s: %s on %s (link %.2f, margin %.2f, veto %.2f)", settings.mode, os.path.basename(path),
             embedder.device, settings.link, settings.margin, settings.veto)
    return embedder


# ----------------------------------------------------------------------------
# The same-day store
# ----------------------------------------------------------------------------
@dataclass
class _Looks:
    camera: str
    key: str
    created: float
    last_seen: float
    embeds: List[Tuple[float, float, List[float]]] = field(default_factory=list)   # (ts, score, vector)
    pending: Optional[Tuple[float, float, Any]] = None                              # (score, ts, crop)
    attempts: int = 0
    held: bool = False               # an event (session) held it at least once

    def wants(self, score: float) -> bool:
        if self.attempts >= MAX_ATTEMPTS:
            return False
        if self.pending is not None and score <= self.pending[0]:
            return False
        if len(self.embeds) + (1 if self.pending is not None else 0) < PER_TRACK:
            return True
        return bool(self.embeds) and score > min(e[1] for e in self.embeds) + BETTER_BY

    def keep(self, ts: float, score: float, vec: List[float]) -> None:
        self.embeds.append((ts, score, vec))
        if len(self.embeds) > PER_TRACK:
            self.embeds.remove(min(self.embeds, key=lambda e: e[1]))
        self.embeds.sort(key=lambda e: e[0])

    def vector(self) -> Optional[List[float]]:
        return ema(e[2] for e in self.embeds) if self.embeds else None


def track_key(track_id: int, first_seen: float) -> str:
    """The tracker's track key, as entities.track_key."""
    return f"{int(track_id)}@{float(first_seen):.3f}"


class Reid:
    """The box's same-day appearance memory. The detection loop ``offer``s each look's confirmed people (a copy of
    the crop only when it is one of the track's best looks); a background thread (``start``) embeds them at most
    RATE_PER_SEC a second. The event book asks ``appearance(camera)`` (entities.Appearance) and ``cross_score``.
    Thread-safe; nothing is written to disk."""

    def __init__(self, settings: ReidSettings, embedder: Any = None, clock: Callable[[], float] = time.time,
                 rate_per_sec: float = RATE_PER_SEC) -> None:
        self.settings = settings
        self.mode = settings.mode
        self._embedder = embedder
        self._clock = clock
        self._rate = float(rate_per_sec)
        self._tokens = float(rate_per_sec)
        self._token_ts: Optional[float] = None
        self._lock = threading.Lock()
        self._looks: Dict[Tuple[str, str], _Looks] = {}
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.embedded = 0
        self.embed_ms = 0.0

    # -- the detection loop ---------------------------------------------------
    def offer(self, camera: str, ts: float, frame: Any, looks: Iterable[Mapping[str, Any]]) -> int:
        """One detector look's people (tracker.CameraTracker.person_looks): ``id, first_seen, box, conf,
        confirmed``. Keeps a crop of a confirmed track's look when it is among its best. Returns crops kept."""
        kept = 0
        with self._lock:
            for look in looks or ():
                if not look.get("confirmed"):
                    continue
                key = track_key(look["id"], look["first_seen"])
                rec = self._looks.get((camera, key))
                score = float(look.get("conf") or 0.0)
                if rec is None:
                    rec = self._looks[(camera, key)] = _Looks(camera, key, created=float(ts), last_seen=float(ts))
                rec.last_seen = max(rec.last_seen, float(ts))
                if not rec.wants(score):
                    continue
                try:
                    crop = crop_person(frame, look["box"])
                except Exception as exc:  # noqa: BLE001 - a bad box is skipped
                    log.debug("[%s] reid crop failed: %s", camera, exc)
                    crop = None
                if crop is None:
                    continue
                rec.pending = (score, float(ts), crop)
                kept += 1
            self._cap_pending()
        return kept

    def _cap_pending(self) -> None:
        waiting = [r for r in self._looks.values() if r.pending is not None]
        if len(waiting) > MAX_PENDING:
            # The tracks with the most embeddings already, then the oldest crops, wait no longer.
            for rec in sorted(waiting, key=lambda r: (-len(r.embeds), r.pending[1]))[:len(waiting) - MAX_PENDING]:
                rec.pending = None

    # -- the embedder thread ---------------------------------------------------
    def _take_token(self, now: float) -> bool:
        if self._token_ts is not None:
            self._tokens = min(self._rate, self._tokens + (now - self._token_ts) * self._rate)
        self._token_ts = now
        if self._tokens >= 1.0:
            self._tokens -= 1.0
            return True
        return False

    def step(self, now: Optional[float] = None) -> bool:
        """Embed one waiting crop (the track with the fewest embeddings first) if the rate allows. True when one
        was embedded."""
        if self._embedder is None:
            return False
        now = self._clock() if now is None else now
        with self._lock:
            waiting = [r for r in self._looks.values() if r.pending is not None]
            if not waiting or not self._take_token(now):
                return False
            rec = min(waiting, key=lambda r: (len(r.embeds), r.pending[1]))
            score, ts, crop = rec.pending
            rec.pending = None
            rec.attempts += 1
        started = time.perf_counter()
        try:
            vec = list(self._embedder.embed(crop))
        except Exception as exc:  # noqa: BLE001 - one failed crop is skipped
            log.debug("[%s] reid embed failed: %s", rec.camera, exc)
            return False
        ms = (time.perf_counter() - started) * 1000.0
        with self._lock:
            if self._looks.get((rec.camera, rec.key)) is rec:
                rec.keep(ts, score, vec)
            self.embedded += 1
            self.embed_ms = ms if self.embedded == 1 else 0.9 * self.embed_ms + 0.1 * ms
        return True

    def start(self) -> None:
        if self._thread is not None or self._embedder is None:
            return

        def loop() -> None:
            while not self._stop.is_set():
                try:
                    busy = self.step()
                except Exception as exc:  # noqa: BLE001 - the thread must outlive any one crop
                    log.debug("reid step failed: %s", exc)
                    busy = False
                if not busy:
                    self._stop.wait(0.05)

        self._thread = threading.Thread(target=loop, name="reid", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    # -- forgetting -------------------------------------------------------------
    def prune(self, now: float, held: Optional[Mapping[str, Set[str]]] = None) -> int:
        """Drop what may no longer be kept: a track an event held whose event closed, a track no event ever held
        after HISTORY_KEEP_SEC, anything after MAX_AGE_SEC. *held* is ``EventBook.open_track_keys()`` (camera ->
        track keys of its open session). Returns how many tracks were dropped."""
        held = held or {}
        dropped = 0
        with self._lock:
            for (camera, key), rec in list(self._looks.items()):
                now_held = key in held.get(camera, ())
                if now_held:
                    rec.held = True
                gone = (now - rec.created > MAX_AGE_SEC or (rec.held and not now_held)
                        or (not rec.held and now - rec.last_seen > HISTORY_KEEP_SEC))
                if gone:
                    del self._looks[(camera, key)]
                    dropped += 1
            if len(self._looks) > MAX_TRACKS:
                for k, _ in sorted(self._looks.items(), key=lambda kv: kv[1].last_seen)[:len(self._looks) - MAX_TRACKS]:
                    del self._looks[k]
                    dropped += 1
        return dropped

    def forget_all(self) -> None:
        with self._lock:
            self._looks.clear()

    # -- reading (the event book) ----------------------------------------------
    def track_vector(self, camera: str, key: str) -> Optional[List[float]]:
        with self._lock:
            rec = self._looks.get((camera, key))
            return rec.vector() if rec is not None else None

    def entity_vector(self, camera: str, keys: Iterable[str]) -> Optional[List[float]]:
        """An entity's look: the EMA (rho EMA_RHO) of its tracks' embeddings in time order."""
        with self._lock:
            rows = [e for k in keys for e in (self._looks.get((camera, k)).embeds if (camera, k) in self._looks else ())]
        return ema(v for _, _, v in sorted(rows, key=lambda e: e[0])) if rows else None

    def cross_score(self, camera_a: str, keys_a: Iterable[str], camera_b: str, keys_b: Iterable[str]) -> Optional[float]:
        """The cosine of an entity at *camera_a* and one at *camera_b*, None when either has no embedding yet."""
        return cosine(self.entity_vector(camera_a, keys_a), self.entity_vector(camera_b, keys_b))

    def appearance(self, camera: str) -> Any:
        """An entities.Appearance for *camera*'s session, or None with ReID off."""
        if self.mode == "off":
            return None
        from . import entities as ent  # noqa: PLC0415

        def score(track: Mapping[str, Any], entity: Mapping[str, Any]) -> Optional[float]:
            mine = self.track_vector(camera, ent.track_key(track))
            theirs = self.entity_vector(camera, [k for k in entity.get("tracks", ()) if k != ent.track_key(track)])
            return cosine(mine, theirs)

        s = self.settings
        return ent.Appearance(score=score, mode=self.mode, link=s.link, margin=s.margin, veto=s.veto)

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            return {"tracks": len(self._looks), "embeddings": sum(len(r.embeds) for r in self._looks.values()),
                    "pending": sum(1 for r in self._looks.values() if r.pending is not None),
                    "embedded": self.embedded, "embed_ms": round(self.embed_ms, 1)}


def start(box_settings: Optional[Mapping[str, Any]]) -> Optional[Reid]:
    """The box's ReID from box.yaml, its thread running; None when off or when the model cannot load. Never raises."""
    try:
        settings = ReidSettings.from_box_settings(box_settings)
        if not settings.enabled:
            log.info("ReID off (box.yaml reid: off)")
            return None
        embedder = load_embedder(settings)
        if embedder is None:
            return None
        reid = Reid(settings, embedder)
        reid.start()
        return reid
    except Exception as exc:  # noqa: BLE001
        log.warning("ReID not started: %s", exc)
        return None


# ----------------------------------------------------------------------------
# The downloader (run by hand on a machine with internet; never by the box itself)
# ----------------------------------------------------------------------------
def _sha384(path: str) -> str:
    h = hashlib.sha384()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(directory: str, precision: str = "FP16", opener: Callable[[str], Any] = urllib.request.urlopen) -> List[str]:
    """Download the model's IR files from the Open Model Zoo storage into *directory*, each checked against its
    size and sha384 before it is moved into place. Raises ValueError on a mismatch (nothing is left behind)."""
    files = OMZ_FILES[precision]
    os.makedirs(directory, exist_ok=True)
    done = []
    for name, (size, digest) in files.items():
        target = os.path.join(directory, name)
        if os.path.isfile(target) and os.path.getsize(target) == size and _sha384(target) == digest:
            done.append(target)
            continue
        tmp = target + ".part"
        with opener(f"{OMZ_BASE}{precision}/{name}") as response, open(tmp, "wb") as f:
            while True:
                chunk = response.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
        got_size, got = os.path.getsize(tmp), _sha384(tmp)
        if got_size != size or got != digest:
            os.remove(tmp)
            raise ValueError(f"{name}: checksum mismatch (size {got_size}, sha384 {got[:16]}...)")
        os.replace(tmp, target)
        done.append(target)
    return done


def main(argv: Optional[Sequence[str]] = None) -> int:
    """``python -m home_guard_project.box.reid download [--dir DIR] [--precision FP16|FP32]``."""
    p = argparse.ArgumentParser(prog="box reid", description="Same-day appearance ReID (clothing only).")
    sub = p.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("download", help="fetch person-reidentification-retail-0277 (checksum verified)")
    d.add_argument("--dir", default="", help="target folder (default: the box's models folder)")
    d.add_argument("--precision", choices=sorted(OMZ_FILES), default="FP16")
    args = p.parse_args(list(sys.argv[1:] if argv is None else argv))
    if args.cmd == "download":
        directory = args.dir or os.path.dirname(model_path())
        try:
            for path in download(directory, args.precision):
                print(f"ok  {path}")
        except (OSError, ValueError) as exc:
            print(f"failed: {exc}", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
