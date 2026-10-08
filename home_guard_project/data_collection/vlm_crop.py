"""The VLM crop: one pipeline for data collection and inference.

The AI must see exactly what the tagged training data shows, so both modes call
:func:`crop_clip` with the same settings (:func:`settings_from_config`): YOLO on
the sub-stream frames (about two looks a second), trigger boxes scaled to the
main stream, a padded square around their union, gaps interpolated, EMA
smoothing, every main-stream frame cropped to its own region and resized to the
median crop size. Moved here unchanged from data_collection.py (2026-10-03); a
golden-hash test pins the output. Cutting the frames and sampling them for the
model live in model_input.py, the one place that says what the model sees.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

if __package__:
    from . import model_input
else:
    import model_input      # script mode: run_collector.sh puts this dir on sys.path

log = logging.getLogger(__name__)

Crop = model_input.Crop


@dataclass(frozen=True)
class CropSettings:
    store_fps: float            # rate of the sub-stream frames (YOLO looks every round(store_fps / 2) frames)
    trigger_class_ids: Tuple[int, ...]
    conf: float
    imgsz: int
    padding: float
    min_size: int
    ema_alpha: float


def settings_from_config(cfg: Any) -> CropSettings:
    """The crop settings of a data_collection Config - the only place both modes take them from."""
    return CropSettings(
        store_fps=float(cfg.STORE_FPS),
        trigger_class_ids=tuple(int(c) for c in cfg.TRIGGER_CLASS_IDS),
        conf=float(cfg.YOLO_TRIGGER_CONF),
        imgsz=int(cfg.YOLO_IMGSZ),
        padding=float(cfg.CROP_PADDING),
        min_size=int(cfg.CROP_MIN_SIZE),
        ema_alpha=float(cfg.CROP_EMA_ALPHA),
    )


@dataclass
class CropResult:
    frames: List[np.ndarray]    # the cropped main-stream frames, all median_w x median_h
    first_crop: Crop            # the first frame's region, in main-stream pixels
    width: int
    height: int
    source: Tuple[int, int]     # main-stream (width, height)
    crops: List[Optional[Crop]] = field(default_factory=list)   # each main-stream frame's region (None: not cut)


def scale_boxes(
    boxes: Any,
    trigger_class_ids: List[int],
    src_h: int, src_w: int,
    dst_h: int, dst_w: int,
) -> List[Tuple[float, float, float, float]]:
    """Extract trigger-class xyxy boxes and scale from src to dst resolution."""
    if boxes is None or len(boxes) == 0:
        return []
    id_set = set(trigger_class_ids)
    sx = dst_w / max(1, src_w)
    sy = dst_h / max(1, src_h)
    scaled: List[Tuple[float, float, float, float]] = []
    for b in boxes:
        cid = int(b.cls.item()) if hasattr(b.cls, "item") else int(b.cls)
        if cid not in id_set:
            continue
        x1, y1, x2, y2 = b.xyxy[0].tolist()
        scaled.append((x1 * sx, y1 * sy, x2 * sx, y2 * sy))
    return scaled


def compute_trigger_crop(
    boxes_xyxy: List[Tuple[float, float, float, float]],
    frame_h: int,
    frame_w: int,
    padding: float = 0.3,
    min_size: int = 384,
) -> Optional[Crop]:
    """
    Compute a square crop region around the union of trigger-class bboxes.

    Returns (x1, y1, x2, y2) in pixel coordinates, or None if no boxes.
    """
    if not boxes_xyxy:
        return None

    ux1 = min(b[0] for b in boxes_xyxy)
    uy1 = min(b[1] for b in boxes_xyxy)
    ux2 = max(b[2] for b in boxes_xyxy)
    uy2 = max(b[3] for b in boxes_xyxy)

    bw = ux2 - ux1
    bh = uy2 - uy1
    pad_x = bw * padding
    pad_y = bh * padding

    cx1 = ux1 - pad_x
    cy1 = uy1 - pad_y
    cx2 = ux2 + pad_x
    cy2 = uy2 + pad_y

    cw = cx2 - cx1
    ch = cy2 - cy1
    side = max(cw, ch, float(min_size))

    center_x = (cx1 + cx2) / 2.0
    center_y = (cy1 + cy2) / 2.0

    cx1 = center_x - side / 2.0
    cy1 = center_y - side / 2.0
    cx2 = center_x + side / 2.0
    cy2 = center_y + side / 2.0

    if cx1 < 0:
        cx2 -= cx1
        cx1 = 0
    if cy1 < 0:
        cy2 -= cy1
        cy1 = 0
    if cx2 > frame_w:
        cx1 -= (cx2 - frame_w)
        cx2 = frame_w
    if cy2 > frame_h:
        cy1 -= (cy2 - frame_h)
        cy2 = frame_h

    cx1 = max(0, int(cx1))
    cy1 = max(0, int(cy1))
    cx2 = min(frame_w, int(cx2))
    cy2 = min(frame_h, int(cy2))

    if cx2 - cx1 < 2 or cy2 - cy1 < 2:
        return None
    return (cx1, cy1, cx2, cy2)


def smooth_crops(
    crops: List[Optional[Crop]],
    alpha: float,
) -> List[Optional[Crop]]:
    """Apply EMA smoothing to a sequence of crop regions for temporal stability."""
    if not crops:
        return []
    smoothed: List[Optional[Crop]] = []
    prev: Optional[Tuple[float, float, float, float]] = None
    for crop in crops:
        if crop is None:
            smoothed.append(prev if prev else None)
            continue
        if prev is None:
            prev = tuple(float(v) for v in crop)  # type: ignore[assignment]
            smoothed.append(crop)
        else:
            s = tuple(
                alpha * float(c) + (1.0 - alpha) * p
                for c, p in zip(crop, prev)
            )
            prev = s
            smoothed.append((int(s[0]), int(s[1]), int(s[2]), int(s[3])))
    return smoothed


def crop_clip(
    detector: Callable[..., Any],
    settings: CropSettings,
    sub_frames: List[np.ndarray],
    sub_start_ts: float,
    sub_end_ts: float,
    main_frames: List[np.ndarray],
    m_start: float,
    name: str = "",
) -> Optional[CropResult]:
    """Crop *main_frames* around the trigger-class detections found on *sub_frames*.

    YOLO runs on the lightweight sub-stream frames and the detections are scaled
    to main-stream coordinates. Returns None when there is nothing to crop.
    """
    if len(main_frames) < 2:
        log.warning("[%s] main-stream had too few frames for VLM crop", name)
        return None

    mh, mw = main_frames[0].shape[:2]

    # --- Align sub-stream frames to the main-stream time window ------------
    # The main-stream may cover a shorter (or equal) window than the sub-
    # stream.  Only use the sub-stream frames whose interpolated timestamps
    # fall within [m_start, m_end] so the crop matches what is actually
    # visible in the main-stream footage.
    sub_dur = sub_end_ts - sub_start_ts
    if sub_dur > 0 and m_start > sub_start_ts:
        overlap_ratio = (m_start - sub_start_ts) / sub_dur
        skip = int(overlap_ratio * len(sub_frames))
        aligned_sub = sub_frames[skip:]
    else:
        aligned_sub = sub_frames
    if not aligned_sub:
        aligned_sub = sub_frames

    # --- Sampled detection -> interpolate -> EMA smooth ---------------------
    # Run YOLO on ~2 fps worth of sub-stream frames (not every frame) to
    # keep processing fast for the live pipeline.  Crops for intermediate
    # frames are linearly interpolated, then the whole sequence is
    # EMA-smoothed for temporal stability.
    n_sub = len(aligned_sub)
    n_main = len(main_frames)

    sample_step = max(1, int(round(settings.store_fps / 2.0)))
    sampled_crops: Dict[int, Crop] = {}

    for i in range(0, n_sub, sample_step):
        sf = aligned_sub[i]
        if sf is None:
            continue
        sh, sw = sf.shape[:2]
        results = detector(
            sf, verbose=False,
            conf=settings.conf,
            imgsz=settings.imgsz,
        )
        scaled = scale_boxes(
            results[0].boxes, list(settings.trigger_class_ids),
            src_h=sh, src_w=sw, dst_h=mh, dst_w=mw,
        )
        crop = compute_trigger_crop(
            scaled, mh, mw,
            padding=settings.padding,
            min_size=settings.min_size,
        )
        if crop is not None:
            sampled_crops[i] = crop

    if not sampled_crops:
        log.warning("[%s] no trigger-class detections for VLM crop", name)
        return None

    # Linear interpolation to fill every sub-stream frame
    sorted_keys = sorted(sampled_crops.keys())
    raw_crops: List[Optional[Crop]] = []
    for i in range(n_sub):
        if i in sampled_crops:
            raw_crops.append(sampled_crops[i])
            continue
        prev_k = max((k for k in sorted_keys if k <= i), default=None)
        next_k = min((k for k in sorted_keys if k >= i), default=None)
        if prev_k is not None and next_k is not None and prev_k != next_k:
            t = (i - prev_k) / (next_k - prev_k)
            a, b = sampled_crops[prev_k], sampled_crops[next_k]
            raw_crops.append((
                int(a[0] + t * (b[0] - a[0])),
                int(a[1] + t * (b[1] - a[1])),
                int(a[2] + t * (b[2] - a[2])),
                int(a[3] + t * (b[3] - a[3])),
            ))
        elif prev_k is not None:
            raw_crops.append(sampled_crops[prev_k])
        elif next_k is not None:
            raw_crops.append(sampled_crops[next_k])
        else:
            raw_crops.append(None)

    smoothed_sub = smooth_crops(raw_crops, alpha=settings.ema_alpha)

    # Map sub-stream crops to main-stream frame count
    smoothed_main: List[Optional[Crop]] = []
    for mi in range(n_main):
        si = min(int(mi * n_sub / max(1, n_main)), n_sub - 1)
        smoothed_main.append(smoothed_sub[si])

    # Uniform output size from the median of smoothed crop dimensions
    median = model_input.median_crop_size(smoothed_main)
    if median is None:
        log.warning("[%s] no usable smoothed crops for VLM clip", name)
        return None
    median_w, median_h = median
    if median_w < 2 or median_h < 2:
        return None

    # --- Crop each main-stream frame with its own smoothed region -----------
    cropped_frames: List[np.ndarray] = []
    for frame, crop in zip(main_frames, smoothed_main):
        if crop is None:
            continue
        cropped = model_input.cut_crop(frame, crop, (median_w, median_h))
        if cropped is not None:
            cropped_frames.append(cropped)

    if len(cropped_frames) < 2:
        log.warning("[%s] no usable cropped frames for VLM clip", name)
        return None

    first_crop = next(c for c in smoothed_main if c is not None)
    return CropResult(frames=cropped_frames, first_crop=first_crop, width=median_w, height=median_h,
                      source=(mw, mh), crops=smoothed_main)


def crop_meta(result: CropResult, settings: CropSettings, rel_path: str, fps: float) -> Dict[str, Any]:
    """The meta.json "vlm_crop" block - the same fields in both modes."""
    x1, y1, x2, y2 = result.first_crop
    return {
        "enabled": True,
        "vlm_crop_path": rel_path,
        "crop_padding": settings.padding,
        "crop_min_size": settings.min_size,
        "source_resolution": [result.source[0], result.source[1]],
        "crop_region": [x1, y1, x2, y2],
        "crop_resolution": [result.width, result.height],
        "frames_written": len(result.frames),
        "fps": float(fps),
        "per_frame_tracking": True,
    }


def sample_for_vlm(frames: List[np.ndarray], fps: float, sample_fps: float) -> List[np.ndarray]:
    """Frames of the crop clip at *sample_fps* (config vlm.sample_fps), starting with the first."""
    return model_input.render_model_input(frames, {"fps": fps}, model_input.ModelInputConfig(sample_fps)).frames
