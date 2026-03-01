"""
Core logic for importing external videos into the dataset.

Probes each video, splits into CLIP_SECONDS chunks via ffmpeg, runs YOLO
detection, optionally generates VLM crops, and writes metadata compatible
with the data_collection pipeline output.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import tempfile
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO

from .config import COCO_NAMES, ImportConfig

log = logging.getLogger("video_import")

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mkv", ".mov", ".webm", ".flv", ".ts", ".m4v"}


# ─────────────────────────────────────────────────────────────────────────────
# ffmpeg / ffprobe helpers
# ─────────────────────────────────────────────────────────────────────────────

def _find_ffmpeg() -> str:
    """Locate ffmpeg, reusing labeling util if available, else PATH search."""
    try:
        from home_guard_project.labeling.utils.ffmpeg import detect_ffmpeg
        path = detect_ffmpeg()
        if path:
            return path
    except ImportError:
        pass
    which = shutil.which("ffmpeg")
    if which:
        return which
    raise FileNotFoundError(
        "ffmpeg not found. Install it or add it to PATH."
    )


def _find_ffprobe(ffmpeg_path: str) -> str:
    try:
        from home_guard_project.labeling.utils.ffmpeg import derive_ffprobe
        probe = derive_ffprobe(ffmpeg_path)
        if probe:
            return probe
    except ImportError:
        pass
    dirname = os.path.dirname(ffmpeg_path)
    basename = os.path.basename(ffmpeg_path).replace("ffmpeg", "ffprobe")
    candidate = os.path.join(dirname, basename)
    if os.path.isfile(candidate):
        return candidate
    which = shutil.which("ffprobe")
    if which:
        return which
    raise FileNotFoundError(
        "ffprobe not found. Install it alongside ffmpeg."
    )


def _probe_video(ffprobe: str, path: str) -> Dict[str, Any]:
    """Return duration, width, height, fps, codec for a video file."""
    cmd = [
        ffprobe, "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height,r_frame_rate,codec_name,duration",
        "-show_entries", "format=duration",
        "-of", "json",
        path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    info = json.loads(result.stdout)

    stream = (info.get("streams") or [{}])[0]
    fmt = info.get("format") or {}

    # Duration: prefer stream, fall back to format
    duration = stream.get("duration") or fmt.get("duration") or "0"
    duration = float(duration)

    # FPS from r_frame_rate fraction
    fps_str = stream.get("r_frame_rate", "30/1")
    if "/" in fps_str:
        num, den = fps_str.split("/")
        fps = float(num) / max(1.0, float(den))
    else:
        fps = float(fps_str)

    return {
        "duration": duration,
        "width": int(stream.get("width", 0)),
        "height": int(stream.get("height", 0)),
        "fps": fps,
        "codec": stream.get("codec_name", "unknown"),
    }


def _split_video(
    ffmpeg: str,
    src: str,
    clip_seconds: float,
    store_fps: float,
    duration: float,
    out_dir: str,
    prefix: str,
) -> List[Tuple[str, float, float]]:
    """
    Split *src* into segments of *clip_seconds*, re-encoding to H.264.

    Returns list of (output_path, start_offset, actual_duration).
    """
    os.makedirs(out_dir, exist_ok=True)
    segments: List[Tuple[str, float, float]] = []
    offset = 0.0
    idx = 0

    while offset < duration:
        remaining = duration - offset
        seg_dur = min(clip_seconds, remaining)
        if seg_dur < 1.0 and segments:
            break

        out_path = os.path.join(out_dir, f"{prefix}_seg{idx:04d}.mp4")
        cmd = [
            ffmpeg, "-y",
            "-ss", f"{offset:.3f}",
            "-i", src,
            "-t", f"{seg_dur:.3f}",
            "-c:v", "libx264", "-preset", "fast", "-crf", "23",
            "-r", str(store_fps),
            "-movflags", "+faststart",
            "-an",
            out_path,
        ]
        try:
            subprocess.run(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=True,
                timeout=120,
            )
            segments.append((out_path, offset, seg_dur))
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
            log.warning("Failed to split segment %d of %s: %s", idx, src, exc)
            if os.path.exists(out_path):
                os.remove(out_path)

        offset += clip_seconds
        idx += 1

    return segments


# ─────────────────────────────────────────────────────────────────────────────
# YOLO helpers
# ─────────────────────────────────────────────────────────────────────────────

def _xyxy_to_yolo_norm(
    x1: float, y1: float, x2: float, y2: float, w: int, h: int,
) -> Tuple[float, float, float, float]:
    """Convert pixel xyxy box to normalised YOLO centre-xywh."""
    x1 = max(0.0, min(float(x1), float(w - 1)))
    x2 = max(0.0, min(float(x2), float(w - 1)))
    y1 = max(0.0, min(float(y1), float(h - 1)))
    y2 = max(0.0, min(float(y2), float(h - 1)))
    bw, bh = max(0.0, x2 - x1), max(0.0, y2 - y1)
    xc, yc = x1 + bw / 2.0, y1 + bh / 2.0
    return xc / w, yc / h, bw / w, bh / h


def _summarize_yolo(results) -> Tuple[Dict[int, int], Dict[int, float]]:
    """Aggregate class counts and max confidence from YOLO results."""
    counts: Dict[int, int] = defaultdict(int)
    max_conf: Dict[int, float] = defaultdict(float)
    boxes = results[0].boxes
    if boxes is None or len(boxes) == 0:
        return {}, {}
    for b in boxes:
        cls_id = int(b.cls.item()) if hasattr(b.cls, "item") else int(b.cls)
        conf = float(b.conf.item()) if hasattr(b.conf, "item") else float(b.conf)
        counts[cls_id] += 1
        if conf > max_conf[cls_id]:
            max_conf[cls_id] = conf
    return dict(counts), dict(max_conf)


def _read_clip_frames(clip_path: str) -> List[np.ndarray]:
    """Read all frames from a clip file via OpenCV."""
    cap = cv2.VideoCapture(clip_path)
    frames: List[np.ndarray] = []
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(frame)
    finally:
        cap.release()
    return frames


def _export_yolo_frames(
    cfg: ImportConfig,
    detector: YOLO,
    frames: List[np.ndarray],
    camera_name: str,
    day: str,
    clip_id: str,
) -> Tuple[List[Dict[str, Any]], Dict[int, int], Dict[int, float]]:
    """
    Export sampled frames + YOLO labels.

    Returns (exported_list, aggregate_class_counts, aggregate_max_conf).
    """
    agg_counts: Dict[int, int] = defaultdict(int)
    agg_max_conf: Dict[int, float] = defaultdict(float)

    if not cfg.yolo_enabled or not frames:
        return [], dict(agg_counts), dict(agg_max_conf)

    base_dir = os.path.join(cfg.output_dir, "yolo")
    img_dir = os.path.join(base_dir, "images", camera_name, day)
    lbl_dir = os.path.join(base_dir, "labels", camera_name, day)
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lbl_dir, exist_ok=True)

    step = max(1, int(round(cfg.store_fps / max(0.1, cfg.yolo_fps))))
    jpeg_params = [int(cv2.IMWRITE_JPEG_QUALITY), int(cfg.yolo_jpeg_quality)]

    exported: List[Dict[str, Any]] = []
    for i in range(0, len(frames), step):
        frame = frames[i]
        if frame is None:
            continue
        h, w = frame.shape[:2]
        stem = f"{clip_id}_f{i:04d}"
        img_path = os.path.join(img_dir, f"{stem}.jpg")
        lbl_path = os.path.join(lbl_dir, f"{stem}.txt")

        if not cv2.imwrite(img_path, frame, jpeg_params):
            continue

        results = detector(
            frame, verbose=False,
            conf=float(cfg.yolo_conf),
            imgsz=cfg.yolo_imgsz,
        )

        counts, max_conf = _summarize_yolo(results)
        for cid, cnt in counts.items():
            agg_counts[cid] += cnt
        for cid, conf in max_conf.items():
            if conf > agg_max_conf.get(cid, 0.0):
                agg_max_conf[cid] = conf

        lines: List[str] = []
        boxes = results[0].boxes
        if boxes is not None and len(boxes) > 0:
            for b in boxes:
                cls_id = int(b.cls.item()) if hasattr(b.cls, "item") else int(b.cls)
                xyxy = b.xyxy[0].tolist()
                xc, yc, bw, bh = _xyxy_to_yolo_norm(*xyxy, w=w, h=h)
                lines.append(f"{cls_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")

        with open(lbl_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        exported.append({
            "frame_index": int(i),
            "approx_time_offset_sec": float(i / cfg.store_fps),
            "image_path": os.path.relpath(img_path, start=cfg.output_dir),
            "label_path": os.path.relpath(lbl_path, start=cfg.output_dir),
            "num_boxes": len(lines),
        })

    return exported, dict(agg_counts), dict(agg_max_conf)


# ─────────────────────────────────────────────────────────────────────────────
# VLM crop
# ─────────────────────────────────────────────────────────────────────────────

def _compute_trigger_crop(
    boxes_xyxy: List[Tuple[float, float, float, float]],
    frame_h: int,
    frame_w: int,
    padding: float = 0.3,
    min_size: int = 384,
) -> Optional[Tuple[int, int, int, int]]:
    """Compute a square crop region around the union of trigger-class bboxes."""
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


def _smooth_crops(
    crops: List[Optional[Tuple[int, int, int, int]]],
    alpha: float,
) -> List[Optional[Tuple[int, int, int, int]]]:
    """Apply EMA smoothing to a sequence of crop regions."""
    if not crops:
        return []
    smoothed: List[Optional[Tuple[int, int, int, int]]] = []
    prev: Optional[Tuple[float, float, float, float]] = None
    for crop in crops:
        if crop is None:
            smoothed.append(
                (int(prev[0]), int(prev[1]), int(prev[2]), int(prev[3]))
                if prev else None
            )
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


def _generate_vlm_crop(
    cfg: ImportConfig,
    detector: YOLO,
    frames: List[np.ndarray],
    camera_name: str,
    day: str,
    clip_id: str,
) -> Optional[Dict[str, Any]]:
    """
    Generate a VLM crop clip with per-frame tracking.

    Unlike the live data_collection pipeline (which samples ~5 frames and
    applies a single static crop), this runs YOLO on *every* frame,
    computes a per-frame crop region, applies EMA smoothing for temporal
    stability, and crops each frame with its own smoothed region.  The
    result is a crop that accurately follows subjects as they move.
    """
    if not cfg.vlm_crop_enabled or not frames:
        return None

    h, w = frames[0].shape[:2]
    trigger_set = set(cfg.trigger_class_ids)

    # --- Per-frame detection and crop computation ----------------------------
    raw_crops: List[Optional[Tuple[int, int, int, int]]] = []
    for frame in frames:
        if frame is None:
            raw_crops.append(None)
            continue
        results = detector(
            frame, verbose=False,
            conf=cfg.yolo_conf,
            imgsz=cfg.yolo_imgsz,
        )
        boxes = results[0].boxes
        frame_boxes: List[Tuple[float, float, float, float]] = []
        if boxes is not None and len(boxes) > 0:
            for b in boxes:
                cls_id = int(b.cls.item()) if hasattr(b.cls, "item") else int(b.cls)
                if cls_id in trigger_set:
                    frame_boxes.append(tuple(b.xyxy[0].tolist()))

        crop = _compute_trigger_crop(
            frame_boxes, h, w,
            padding=cfg.vlm_crop_padding,
            min_size=cfg.vlm_crop_min_size,
        )
        raw_crops.append(crop)

    if not any(c is not None for c in raw_crops):
        log.debug("[%s] no trigger-class detections for VLM crop in %s", camera_name, clip_id)
        return None

    # --- Temporal EMA smoothing ----------------------------------------------
    smoothed_crops = _smooth_crops(raw_crops, alpha=cfg.vlm_crop_ema_alpha)

    # --- Determine a uniform output size from the smoothed crops -------------
    # Use the median crop dimensions so the VideoWriter has a fixed frame size.
    valid_sizes = [
        (c[2] - c[0], c[3] - c[1])
        for c in smoothed_crops if c is not None
    ]
    if not valid_sizes:
        return None
    valid_sizes.sort()
    median_w, median_h = valid_sizes[len(valid_sizes) // 2]
    if median_w < 2 or median_h < 2:
        return None

    # --- Crop each frame with its smoothed region ----------------------------
    cropped_frames: List[np.ndarray] = []
    for frame, crop in zip(frames, smoothed_crops):
        if frame is None or crop is None:
            continue
        x1, y1, x2, y2 = crop
        cropped = frame[y1:y2, x1:x2]
        if cropped.size == 0:
            continue
        if cropped.shape[1] != median_w or cropped.shape[0] != median_h:
            cropped = cv2.resize(cropped, (median_w, median_h))
        cropped_frames.append(cropped)

    if len(cropped_frames) < 2:
        return None

    vlm_dir = os.path.join(cfg.output_dir, "vlm_crops", camera_name, day)
    os.makedirs(vlm_dir, exist_ok=True)

    fps = cfg.store_fps
    fd, tmp_path = tempfile.mkstemp(suffix=".mp4")
    os.close(fd)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(tmp_path, fourcc, float(fps), (median_w, median_h))
    if not writer.isOpened():
        os.remove(tmp_path)
        return None
    for f in cropped_frames:
        writer.write(f)
    writer.release()

    vlm_mp4 = os.path.join(vlm_dir, f"{clip_id}.mp4")
    shutil.move(tmp_path, vlm_mp4)

    # Representative crop region (first valid smoothed crop) for metadata
    first_crop = next(c for c in smoothed_crops if c is not None)
    x1, y1, x2, y2 = first_crop

    log.info("[%s] VLM crop saved: %s (%dx%d, %d frames, per-frame tracking)",
             camera_name, vlm_mp4, median_w, median_h, len(cropped_frames))

    return {
        "enabled": True,
        "vlm_crop_path": os.path.relpath(vlm_mp4, start=cfg.output_dir),
        "crop_padding": cfg.vlm_crop_padding,
        "crop_min_size": cfg.vlm_crop_min_size,
        "source_resolution": [w, h],
        "crop_region": [x1, y1, x2, y2],
        "crop_resolution": [median_w, median_h],
        "frames_written": len(cropped_frames),
        "fps": float(fps),
        "per_frame_tracking": True,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Timestamp helpers
# ─────────────────────────────────────────────────────────────────────────────

def _utc_iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def _local_iso(ts: float) -> str:
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")


# ─────────────────────────────────────────────────────────────────────────────
# Single-clip processing
# ─────────────────────────────────────────────────────────────────────────────

def _process_clip(
    cfg: ImportConfig,
    detector: YOLO,
    clip_path: str,
    clip_offset: float,
    clip_duration: float,
    clip_index: int,
    now_ts: float,
    source_name: str,
) -> Optional[Dict[str, Any]]:
    """
    Process one split clip: move to dataset tree, run YOLO export,
    generate VLM crop, and write metadata.

    Returns the metadata dict on success, None on failure.
    """
    day = datetime.now().strftime("%Y-%m-%d")
    clip_ts = now_ts + clip_index
    clip_id = f"{cfg.camera_name}_{int(clip_ts)}_{cfg.kind}"

    clips_dir = os.path.join(cfg.output_dir, "clips", cfg.camera_name, day)
    meta_dir = os.path.join(cfg.output_dir, "meta", cfg.camera_name, day)
    resp_dir = os.path.join(cfg.output_dir, "responses", cfg.camera_name, day)
    os.makedirs(clips_dir, exist_ok=True)
    os.makedirs(meta_dir, exist_ok=True)
    os.makedirs(resp_dir, exist_ok=True)

    final_mp4 = os.path.join(clips_dir, f"{clip_id}.mp4")
    shutil.move(clip_path, final_mp4)

    frames = _read_clip_frames(final_mp4)
    if not frames:
        log.warning("No frames read from %s — skipping", final_mp4)
        return None

    fh, fw = frames[0].shape[:2]
    start_ts = clip_ts
    end_ts = clip_ts + clip_duration

    # YOLO export
    exported, agg_counts, agg_max_conf = _export_yolo_frames(
        cfg, detector, frames, cfg.camera_name, day, clip_id,
    )

    trigger_detected = any(cid in agg_counts for cid in cfg.trigger_class_ids)
    trigger_conf_max = max(
        (agg_max_conf.get(cid, 0.0) for cid in cfg.trigger_class_ids),
        default=0.0,
    )

    meta: Dict[str, Any] = {
        "camera_name": cfg.camera_name,
        "kind": cfg.kind,
        "clip_path": os.path.relpath(final_mp4, start=cfg.output_dir),
        "source_file": source_name,

        "clip_start_ts": float(start_ts),
        "clip_end_ts": float(end_ts),
        "clip_start_utc": _utc_iso(start_ts),
        "clip_end_utc": _utc_iso(end_ts),
        "clip_start_local": _local_iso(start_ts),
        "clip_end_local": _local_iso(end_ts),

        "duration_sec": float(clip_duration),
        "frames_written": len(frames),
        "fps_estimated": float(cfg.store_fps),
        "vlm_sample_fps": 1,

        "buffer": {
            "store_fps": float(cfg.store_fps),
            "store_size": [fw, fh],
        },

        "yolo": {
            "class_counts": {
                COCO_NAMES[cid] if cid < len(COCO_NAMES) else str(cid): cnt
                for cid, cnt in agg_counts.items()
            },
            "class_max_conf": {
                COCO_NAMES[cid] if cid < len(COCO_NAMES) else str(cid): conf
                for cid, conf in agg_max_conf.items()
            },
            "trigger_classes": [
                COCO_NAMES[cid] for cid in cfg.trigger_class_ids
                if cid < len(COCO_NAMES)
            ],
            "trigger_detected": trigger_detected,
            "trigger_conf_max": float(trigger_conf_max),
            "detection_score": 0.0,
        },

        "roi": {
            "active": False,
            "polygon_normalized": None,
        },
    }

    if cfg.yolo_enabled and exported:
        meta["yolo_export"] = {
            "enabled": True,
            "export_fps": float(cfg.yolo_fps),
            "conf": float(cfg.yolo_conf),
            "images_root": "yolo/images",
            "labels_root": "yolo/labels",
            "exported_frames": exported,
        }

    # VLM crop
    vlm_crop_meta = _generate_vlm_crop(
        cfg, detector, frames, cfg.camera_name, day, clip_id,
    )
    if vlm_crop_meta is not None:
        meta["vlm_crop"] = vlm_crop_meta

    meta_path = os.path.join(meta_dir, f"{clip_id}.meta.json")
    raw_path = os.path.join(resp_dir, f"{clip_id}.model_raw.txt")

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    with open(raw_path, "w", encoding="utf-8") as f:
        pass  # empty placeholder

    log.info("Imported clip: %s (%d frames, %dx%d)", clip_id, len(frames), fw, fh)
    return meta


# ─────────────────────────────────────────────────────────────────────────────
# Collect input files
# ─────────────────────────────────────────────────────────────────────────────

def collect_videos(input_path: str) -> List[str]:
    """Collect video files from a file or directory path."""
    if os.path.isfile(input_path):
        ext = os.path.splitext(input_path)[1].lower()
        if ext in VIDEO_EXTENSIONS:
            return [input_path]
        log.warning("Not a recognized video file: %s", input_path)
        return []

    if not os.path.isdir(input_path):
        log.error("Input path does not exist: %s", input_path)
        return []

    videos: List[str] = []
    for entry in sorted(os.listdir(input_path)):
        full = os.path.join(input_path, entry)
        if os.path.isfile(full) and os.path.splitext(entry)[1].lower() in VIDEO_EXTENSIONS:
            videos.append(full)
    return videos


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def run(
    input_path: str,
    cfg: ImportConfig,
) -> Dict[str, int]:
    """
    Import external videos into the dataset.

    Returns counts: {imported, failed, skipped}.
    """
    videos = collect_videos(input_path)
    if not videos:
        log.error("No video files found at: %s", input_path)
        return {"imported": 0, "failed": 0, "skipped": 0}

    log.info("Found %d video(s) to import", len(videos))

    ffmpeg = _find_ffmpeg()
    ffprobe = _find_ffprobe(ffmpeg)
    log.info("Using ffmpeg: %s", ffmpeg)

    log.info("Loading YOLO model: %s", cfg.yolo_model)
    detector = YOLO(cfg.yolo_model)

    tmp_root = tempfile.mkdtemp(prefix="video_import_")
    imported = 0
    failed = 0
    skipped = 0

    try:
        for video_path in tqdm(videos, desc="Importing videos", unit="video"):
            video_name = os.path.basename(video_path)
            log.info("Processing: %s", video_name)

            try:
                info = _probe_video(ffprobe, video_path)
            except Exception as exc:
                log.error("Failed to probe %s: %s", video_name, exc)
                failed += 1
                continue

            duration = info["duration"]
            if duration < 0.5:
                log.warning("Skipping %s — too short (%.1fs)", video_name, duration)
                skipped += 1
                continue

            log.info(
                "  %s: %.1fs, %dx%d, %.1f fps, codec=%s",
                video_name, duration, info["width"], info["height"],
                info["fps"], info["codec"],
            )

            now_ts = time.time()
            stem = os.path.splitext(video_name)[0]
            seg_dir = os.path.join(tmp_root, stem)

            segments = _split_video(
                ffmpeg, video_path,
                clip_seconds=cfg.clip_seconds,
                store_fps=cfg.store_fps,
                duration=duration,
                out_dir=seg_dir,
                prefix=f"{cfg.camera_name}_{int(now_ts)}",
            )

            if not segments:
                log.warning("No segments produced for %s", video_name)
                failed += 1
                continue

            log.info("  Split into %d clip(s)", len(segments))

            for idx, (seg_path, seg_offset, seg_dur) in enumerate(segments):
                try:
                    meta = _process_clip(
                        cfg=cfg,
                        detector=detector,
                        clip_path=seg_path,
                        clip_offset=seg_offset,
                        clip_duration=seg_dur,
                        clip_index=idx,
                        now_ts=now_ts,
                        source_name=video_name,
                    )
                    if meta:
                        imported += 1
                    else:
                        failed += 1
                except Exception:
                    log.exception("Failed to process segment %d of %s", idx, video_name)
                    failed += 1
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)

    log.info(
        "Import complete: %d imported, %d failed, %d skipped",
        imported, failed, skipped,
    )
    return {"imported": imported, "failed": failed, "skipped": skipped}
