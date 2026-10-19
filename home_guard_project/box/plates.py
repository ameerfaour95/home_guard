"""Licence plates on the owner's cars: read them where they can be read, and only there.

Measured on our own cameras (``home_guard_data/checks/lpr_eval/REPORT.md``, 2026-10-06):

- The stored clips are the 704x576 sub-stream, where a plate is 10-25 px wide: never readable. Plates are read
  from the MAIN stream (``streams.MainStreamThread``, full resolution), cropped around the vehicle's box.
- On main-stream crops the fast-alpr parts read both plates checked by eye 13/13 at 54-79 px plate width; under
  about 50 px the OCR fails. A plate is kept only at ``MIN_PLATE_WIDTH`` (50) px or more.
- Benchmark on 416 hand-checked main-stream frames, Oct 3-9 (``docs/plates_design.md``): cct-s read the exact plate
  in 18 % of frames at 40-50 px, about 72 % at 50-80 px, 95 % at 80-100 px and 100 % from 100 px. Voting at 50 px
  matched more owner cars than at 60 px with no false owner match, so the floor is 50 px.
- Night IR blooms the plate: no reads at night (``is_ir`` tells a grey IR frame from a colour one).
- A white board and the camera's timestamp overlay were both "detected" as plates: a plate counts only when its
  centre lies inside the vehicle box it was found in.
- Repeated reads are not proof (a stable wrong cluster happened at a small size), so ``vote`` needs two reads that
  agree within one digit, and ``match`` compares with the owner's list only (same length, at most one digit off).

Israeli plates are digits only (7 or 8): the OCR is decoded over digits alone and separators are stripped.

The packages are optional at run time: without them every call degrades to "no plate" (``PlateReader.available()``
is False, logged once). pyproject pins ``fast-alpr[onnx]==0.4.0`` (open-image-models + fast-plate-ocr + onnxruntime
CPU). The two ONNX models then run on the box's own OpenVINO (pinned for the detector) on the N150's iGPU: measured
there at 24 ms (detector) and 28 ms (OCR) against 86 and 68 ms on all CPU threads. ``device`` "GPU" tries the iGPU,
then OpenVINO on the CPU, then onnxruntime on the CPU; "CPU" skips the iGPU. ``onnxruntime-openvino`` is NOT used: on
Windows it needs its own OpenVINO runtime beside the pinned one.

Models: plate detector ``yolo-v9-t-384-license-plate-end2end`` (16 ms on the laptop), OCR
``cct-s-v2-global-model`` (12 ms per plate). The first use downloads them to the packages' own cache.
"""

from __future__ import annotations

import logging
import re
import threading
from collections import Counter
from dataclasses import dataclass
from typing import Any, Callable, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

log = logging.getLogger("box.plates")

DETECTOR_MODEL = "yolo-v9-t-384-license-plate-end2end"
OCR_MODEL = "cct-s-v2-global-model"
DETECTOR_CONF = 0.25
MIN_PLATE_WIDTH = 50.0         # px on the main stream; 40-50 px read 18 %, 50-80 px about 72 % (benchmark)
PLATE_DIGITS = (7, 8)          # Israeli plates
MAX_FRAMES = 6                 # main-stream frames read per vehicle
CROP_MARGIN = 0.20             # around the vehicle box, so a plate on the bumper's edge is not cut
IR_SATURATION = 12.0           # mean HSV saturation under this: a grey (IR night) picture

PixelBox = Tuple[int, int, int, int]


@dataclass(frozen=True)
class PlateRead:
    digits: str
    confidence: float
    width_px: float


@dataclass(frozen=True)
class VehicleCrop:
    """A vehicle cut from a main-stream frame: the picture (with its margin) and the vehicle box inside it."""
    image: Any
    inner: PixelBox


# ----------------------------------------------------------------------------
# Digits, matching, voting
# ----------------------------------------------------------------------------
def digits_only(text: Any) -> str:
    """"12-345-67" -> "1234567"; anything that is not a digit goes."""
    return "".join(ch for ch in str(text or "") if "0" <= ch <= "9")


def valid_plate(digits: Any) -> bool:
    return isinstance(digits, str) and digits.isdigit() and digits.isascii() and len(digits) in PLATE_DIGITS


def mask(digits: Any) -> str:
    """A plate as replies show it: only the last 3 digits ("****567")."""
    digits = str(digits or "")
    return "*" * max(0, len(digits) - 3) + digits[-3:] if digits else ""


def levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def vote(reads: Sequence[PlateRead], min_agree: int = 2) -> Optional[str]:
    """The plate the reads agree on: the biggest group of reads within one digit of a candidate, at least
    *min_agree* strong; its most common exact text wins. None when no group is big enough."""
    reads = [r for r in reads if isinstance(r, PlateRead) and valid_plate(r.digits)]
    best: Optional[Tuple[int, float, str]] = None
    for cand in {r.digits for r in reads}:
        group = [r for r in reads if levenshtein(r.digits, cand) <= 1]
        if len(group) < min_agree:
            continue
        counts = Counter(r.digits for r in group)
        top = max(counts.items(), key=lambda kv: (kv[1], sum(r.confidence for r in group if r.digits == kv[0])))[0]
        key = (len(group), sum(r.confidence for r in group), top)
        if best is None or key[:2] > best[:2]:
            best = key
    return best[2] if best else None


def match(digits: Any, owner_vehicles: Iterable[Mapping[str, Any]]) -> Optional[Mapping[str, Any]]:
    """The owner vehicle whose plate is *digits*, or one digit off at the same length. An exact plate wins; two
    owner plates equally near is no match (it must not guess between the family's cars)."""
    digits = digits_only(digits)
    if not valid_plate(digits):
        return None
    near: List[Tuple[int, Mapping[str, Any]]] = []
    for v in owner_vehicles or ():
        plate = digits_only((v or {}).get("plate"))
        if plate and len(plate) == len(digits):
            d = sum(1 for x, y in zip(plate, digits) if x != y)
            if d <= 1:
                near.append((d, v))
    if not near:
        return None
    best = min(d for d, _ in near)
    winners = [v for d, v in near if d == best]
    return winners[0] if len(winners) == 1 else None


# ----------------------------------------------------------------------------
# Plates come from this reader only, never from a vision-language model
# ----------------------------------------------------------------------------
# Owner rule (2026-10-19): a plate is read by the plate-recognition tool here (detector + OCR, digits only) and kept
# as a number (a string of 7-8 digits); no VLM reads or guesses one - not the Eye, not the assistant's looks, not the
# describer. So a plate-like number in any VLM answer is blanked (:func:`scrub`, at inference.parse_vlm_json and
# inference.owner_guard), and a vision question about a plate is not asked (:func:`asks_plate`, brain/vision.py).
SCRUBBED = "••-•••-••"
_PLATE_LIKE = re.compile(r"(?<![\d:/])(?:\d{2,3}[-‐–.\s]\d{2,3}[-‐–.\s]\d{2,3}|\d{7,8})(?![\d:/])")
_PLATE_QUESTION = re.compile(
    r"licen[cs]e\s*plate|number\s*plate|plate\s*(?:number|no\b)|registration\s*(?:number|plate)|car(?:'s)?\s*number|"
    r"(?:read|what(?:'s| is))\s+(?:the\s+)?plate|לוחית|לוחיות|נומרטור|מספר\s+(?:ה|של\s+ה)?(?:רכב|אוטו|מכונית|טנדר|משאית)|"
    r"(?:רכב|אוטו|מכונית|טנדר|משאית)\S*\s+(?:\S+\s+)?מספר", re.IGNORECASE)


def scrub(value: Any) -> Any:
    """*value* (a VLM answer: text, or a dict / list of them) with every plate-like number blanked."""
    if isinstance(value, str):
        return _PLATE_LIKE.sub(lambda m: SCRUBBED if valid_plate(digits_only(m.group(0))) else m.group(0), value)
    if isinstance(value, dict):
        return {k: scrub(v) for k, v in value.items()}
    if isinstance(value, list):
        return [scrub(v) for v in value]
    return value


def asks_plate(question: Any) -> bool:
    """A question that asks a picture for a licence plate / a car's number (that is the plate reader's job)."""
    return bool(_PLATE_QUESTION.search(str(question or "")))


# ----------------------------------------------------------------------------
# Main-stream crops
# ----------------------------------------------------------------------------
def is_ir(frame: Any) -> bool:
    """A grey picture: the camera is in night IR mode, where plates bloom."""
    try:
        import cv2  # noqa: PLC0415

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        return float(hsv[..., 1].mean()) < IR_SATURATION
    except Exception:  # noqa: BLE001 - an odd frame is treated as unreadable
        return True


def sharpness(image: Any) -> float:
    """Variance of the Laplacian of the grey picture: higher is sharper (motion blur and focus both lower it)."""
    try:
        import cv2  # noqa: PLC0415

        return float(cv2.Laplacian(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())
    except Exception:  # noqa: BLE001 - no picture, no sharpness
        return 0.0


def _ranks(values: Sequence[float]) -> List[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    out = [0.0] * len(values)
    for pos, i in enumerate(order):
        out[i] = pos / max(1, len(values) - 1)
    return out


def best_frames(boxes: Sequence[Sequence[float]], sharp: Sequence[float], k: int = 5,
                frame_size: Optional[Tuple[int, int]] = None) -> List[int]:
    """The indices (in time order) of the *k* frames of one vehicle track most worth reading: the biggest box (the
    car closest), the sharpest crop (``sharpness``) and the stillest car (its box centre moved least to the frame
    before or after, in box widths), each as a rank so no single one dominates. A box touching the edge of a
    *frame_size* (w, h) picture comes last: its plate may be cut. Mismatched or empty input gives []."""
    n = len(boxes)
    if n == 0 or len(sharp) != n:
        return []
    try:
        bx = [tuple(float(v) for v in b[:4]) for b in boxes]
    except (TypeError, ValueError):
        return []
    area = [max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1]) for b in bx]
    centre = [((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0) for b in bx]
    motion = []
    for i, b in enumerate(bx):
        width = max(1.0, b[2] - b[0])
        near = [j for j in (i - 1, i + 1) if 0 <= j < n]
        motion.append(min((((centre[i][0] - centre[j][0]) ** 2 + (centre[i][1] - centre[j][1]) ** 2) ** 0.5 / width
                           for j in near), default=0.0))
    still = [-m for m in motion]
    edge = [0.0] * n
    if frame_size:
        fw, fh = frame_size
        edge = [1.0 if (b[0] <= 2 or b[1] <= 2 or b[2] >= fw - 2 or b[3] >= fh - 2) else 0.0 for b in bx]
    ranks = zip(_ranks(area), _ranks([float(v) for v in sharp]), _ranks(still), edge)
    score = [a + s + m - 10.0 * e for a, s, m, e in ranks]
    top = sorted(range(n), key=lambda i: (-score[i], i))[:max(0, int(k))]
    return sorted(top)


def _spread(items: Sequence[Any], n: int) -> List[Any]:
    if len(items) <= n:
        return list(items)
    if n <= 1:
        return [items[-1]]
    return [items[round(i * (len(items) - 1) / (n - 1))] for i in range(n)]


def vehicle_crops(frames: Sequence[Any], box: Sequence[float], margin: float = CROP_MARGIN,
                  max_frames: int = MAX_FRAMES, min_width: float = MIN_PLATE_WIDTH) -> List[VehicleCrop]:
    """Cut the vehicle (*box*: normalised x1, y1, x2, y2, as the sub-stream detector saw it; both streams show
    the same view) from at most *max_frames* main-stream frames spread over the list, with a margin. A vehicle
    narrower than a readable plate gives nothing."""
    try:
        x1, y1, x2, y2 = (float(v) for v in box[:4])
    except (TypeError, ValueError):
        return []
    if not (x2 > x1 and y2 > y1):
        return []
    out: List[VehicleCrop] = []
    for frame in _spread([f for f in frames or () if isinstance(f, np.ndarray) and f.ndim == 3], max_frames):
        h, w = frame.shape[:2]
        vx1, vy1, vx2, vy2 = x1 * w, y1 * h, x2 * w, y2 * h
        if vx2 - vx1 < min_width:
            continue
        mx, my = (vx2 - vx1) * margin, (vy2 - vy1) * margin
        cx1, cy1 = max(0, int(round(vx1 - mx))), max(0, int(round(vy1 - my)))
        cx2, cy2 = min(w, int(round(vx2 + mx))), min(h, int(round(vy2 + my)))
        if cx2 - cx1 < min_width or cy2 - cy1 < 8:
            continue
        inner = (int(round(vx1)) - cx1, int(round(vy1)) - cy1, int(round(vx2)) - cx1, int(round(vy2)) - cy1)
        out.append(VehicleCrop(frame[cy1:cy2, cx1:cx2], inner))
    return out


# ----------------------------------------------------------------------------
# The reader
# ----------------------------------------------------------------------------
class _OvSession:
    """An ONNX model compiled by OpenVINO, behind the calls the packages make of their onnxruntime session
    (``run(names, feeds)``, ``get_inputs()``, ``get_outputs()``): their pre- and post-processing stay theirs."""

    def __init__(self, session: Any, device: str) -> None:
        import openvino as ov  # noqa: PLC0415

        path = getattr(session, "_model_path", None)
        if not path:
            raise ValueError("the onnxruntime session has no model file")
        core = ov.Core()
        if device not in core.available_devices and not any(d.startswith(device) for d in core.available_devices):
            raise ValueError(f"no OpenVINO {device} here")
        self._session = session
        self._compiled = core.compile_model(core.read_model(path), device)
        self._request = self._compiled.create_infer_request()
        self._outputs = [o.name for o in session.get_outputs()]
        self.device = device

    def get_inputs(self) -> Any:
        return self._session.get_inputs()

    def get_outputs(self) -> Any:
        return self._session.get_outputs()

    def get_providers(self) -> List[str]:
        return [f"OpenVINO:{self.device}"]

    def run(self, names: Optional[Sequence[str]], feeds: Mapping[str, Any]) -> List[Any]:
        result = self._request.infer(dict(feeds))
        by_name = {}
        for port, value in result.items():
            for name in port.get_names():
                by_name[name] = value
        return [np.asarray(by_name[n]) for n in (names or self._outputs)]


def _on_openvino(session: Any, device: str) -> Tuple[Any, str]:
    """*session* moved to OpenVINO on *device* ("GPU": the iGPU, else OpenVINO's CPU), else kept on onnxruntime."""
    for dev in ([device, "CPU"] if device.upper() != "CPU" else ["CPU"]):
        try:
            return _OvSession(session, dev.upper()), f"openvino:{dev.upper()}"
        except Exception as exc:  # noqa: BLE001 - the next device, then onnxruntime
            log.info("Plate model not on OpenVINO %s (%s)", dev, exc)
    return session, "onnxruntime:CPU"


class _FastAlpr:
    """The fast-alpr parts behind two calls: ``detect(bgr) -> [(x1, y1, x2, y2, conf)]`` and
    ``ocr(bgr plate) -> (digits, confidence)``, the OCR decoded over the digits alone."""

    def __init__(self, detector_model: str, ocr_model: str, conf: float, device: str = "GPU") -> None:
        import open_image_models as oim  # noqa: PLC0415
        from fast_plate_ocr import LicensePlateRecognizer  # noqa: PLC0415

        make = getattr(oim, "create_detector", None)
        self.detector = (make(detector_model, conf_thresh=conf, providers=["CPUExecutionProvider"]) if make is not None
                         else oim.LicensePlateDetector(detection_model=detector_model, conf_thresh=conf))
        self.ocr_model = LicensePlateRecognizer(ocr_model, device="cpu")
        self.devices = {}
        if device:
            for name, owner in (("detector", self.detector), ("ocr", self.ocr_model)):
                if hasattr(owner, "model"):
                    owner.model, self.devices[name] = _on_openvino(owner.model, device)
        log.info("Plate reader on %s", self.devices or "onnxruntime")

    def detect(self, image: Any) -> List[Tuple[float, float, float, float, float]]:
        out = []
        for d in self.detector.predict(image):
            b = d.bounding_box
            out.append((float(b.x1), float(b.y1), float(b.x2), float(b.y2), float(d.confidence)))
        return out

    def ocr(self, plate: Any) -> Tuple[str, float]:
        import cv2  # noqa: PLC0415

        rec = self.ocr_model
        cfg = rec.config
        img = (cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY) if cfg.image_color_mode == "grayscale"
               else cv2.cvtColor(plate, cv2.COLOR_BGR2RGB))
        try:
            # Digits only: the network's own scores, restricted to 0-9 and the padding character.
            from fast_plate_ocr.core.process import preprocess_image  # noqa: PLC0415
            from fast_plate_ocr.inference.plate_recognizer import _load_image_from_source  # noqa: PLC0415

            x = preprocess_image(_load_image_from_source(img, cfg))
            raw = rec.model.run([rec.plate_output_name], {"input": x})[0]
            scores = np.asarray(raw).reshape(cfg.max_plate_slots, len(cfg.alphabet))
            keep = [cfg.alphabet.index(c) for c in "0123456789"] + [cfg.alphabet.index(cfg.pad_char)]
            digits, probs = "", []
            for row in scores[:, keep]:
                k = int(np.argmax(row))
                if k == 10:
                    break
                digits += "0123456789"[k]
                probs.append(float(row[k]))
            return digits, (float(np.mean(probs)) if probs else 0.0)
        except (ImportError, AttributeError, ValueError, IndexError):
            # Another package version: its plain text, separators and letters stripped.
            pred = rec.run_one(img, return_confidence=True)
            probs = [] if pred.char_probs is None else [float(p) for p in np.ravel(pred.char_probs)]
            text = digits_only(pred.plate)
            return text, (float(np.mean(probs[:len(text)])) if text and probs else 0.0)


def _default_loader(detector_model: str, ocr_model: str, conf: float, device: str = "GPU") -> Callable[[], Any]:
    return lambda: _FastAlpr(detector_model, ocr_model, conf, device)


class PlateReader:
    """Reads plates from vehicle crops. The models load lazily on the first read; without the packages (or with
    a broken install) ``available()`` is False and every read is empty."""

    def __init__(self, backend: Any = None, loader: Optional[Callable[[], Any]] = None,
                 detector_model: str = DETECTOR_MODEL, ocr_model: str = OCR_MODEL, conf: float = DETECTOR_CONF,
                 min_width: float = MIN_PLATE_WIDTH, device: str = "GPU") -> None:
        self._backend = backend
        self._loader = loader or _default_loader(detector_model, ocr_model, conf, device)
        self._tried = backend is not None
        self._lock = threading.Lock()
        self.min_width = float(min_width)

    def available(self) -> bool:
        with self._lock:
            if not self._tried:
                self._tried = True
                try:
                    self._backend = self._loader()
                except Exception as exc:  # noqa: BLE001 - no packages, no models: no plates, never a crash
                    log.info("Plate reading is off (%s)", exc)
                    self._backend = None
            return self._backend is not None

    def read(self, crops: Sequence[Any]) -> List[PlateRead]:
        """One read per plate found inside a vehicle, at least ``min_width`` px wide, with 7 or 8 digits."""
        if not self.available():
            return []
        reads: List[PlateRead] = []
        for item in crops or ():
            image, inner = (item.image, item.inner) if isinstance(item, VehicleCrop) else (item, None)
            if not isinstance(image, np.ndarray) or image.ndim != 3:
                continue
            h, w = image.shape[:2]
            ix1, iy1, ix2, iy2 = inner or (0, 0, w, h)
            try:
                boxes = self._backend.detect(image)
            except Exception as exc:  # noqa: BLE001
                log.warning("Plate detector failed: %s", exc)
                continue
            for x1, y1, x2, y2, _conf in boxes:
                width = float(x2) - float(x1)
                cx, cy = (float(x1) + float(x2)) / 2.0, (float(y1) + float(y2)) / 2.0
                if width < self.min_width or not (ix1 <= cx <= ix2 and iy1 <= cy <= iy2):
                    continue
                px1, py1 = max(0, int(round(x1))), max(0, int(round(y1)))
                px2, py2 = min(w, int(round(x2))), min(h, int(round(y2)))
                if px2 - px1 < 4 or py2 - py1 < 3:
                    continue
                try:
                    text, conf = self._backend.ocr(image[py1:py2, px1:px2])
                except Exception as exc:  # noqa: BLE001
                    log.warning("Plate OCR failed: %s", exc)
                    continue
                digits = digits_only(text)
                if valid_plate(digits):
                    reads.append(PlateRead(digits, round(float(conf), 4), round(width, 1)))
        return reads
