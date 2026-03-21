"""
Core logic for importing the UCA (UCF Crime Annotation) dataset.

Splits each source video into fixed-duration windows (default 10s), aggregates
overlapping annotation descriptions per window, runs YOLO detection, and writes
metadata compatible with the labeling pipeline.

Crash-resilient: a manifest file tracks processed clip IDs so interrupted runs
can be resumed by simply re-running the command.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
import zipfile
from collections import OrderedDict, defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO

from .config import COCO_NAMES, UCAConfig

log = logging.getLogger("uca_import")

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mkv", ".mov", ".webm", ".flv", ".ts", ".m4v"}

_ZIP_VIDEO_PREFIX = "UCF_Crimes/UCF_Crimes/Videos/"

_VLM_TEXT_SEPARATOR = "\n----\n"


# ─────────────────────────────────────────────────────────────────────────────
# Annotation parsing
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Segment:
    """One annotated temporal segment of a video."""
    video_name: str
    start_sec: float
    end_sec: float
    description: str
    split: str
    seg_index: int


@dataclass
class Window:
    """One fixed-duration time window of a source video."""
    video_name: str
    start_sec: float
    end_sec: float
    window_index: int
    total_windows: int
    split: str
    overlapping_annotations: List[Segment] = field(default_factory=list)
    aggregated_description: str = ""


def parse_annotations_json(json_bytes: bytes, split_name: str) -> List[Segment]:
    """Parse a UCA JSON annotation file into Segment objects."""
    data: Dict[str, Any] = json.loads(json_bytes)
    segments: List[Segment] = []
    for video_name, entry in data.items():
        timestamps = entry.get("timestamps", [])
        sentences = entry.get("sentences", [])
        for idx, (ts, sent) in enumerate(zip(timestamps, sentences)):
            if len(ts) < 2:
                continue
            segments.append(Segment(
                video_name=video_name,
                start_sec=float(ts[0]),
                end_sec=float(ts[1]),
                description=sent.strip(),
                split=split_name,
                seg_index=idx,
            ))
    return segments


def _parse_timestamp(ts: str) -> float:
    """Convert MM:SS.d or HH:MM:SS.d to seconds."""
    parts = ts.split(":")
    if len(parts) == 2:
        return float(parts[0]) * 60.0 + float(parts[1])
    if len(parts) == 3:
        return float(parts[0]) * 3600.0 + float(parts[1]) * 60.0 + float(parts[2])
    return float(ts)


def parse_annotations_txt(txt_bytes: bytes, split_name: str) -> List[Segment]:
    """Parse a UCA txt annotation file into Segment objects."""
    text = txt_bytes.decode("utf-8", errors="replace")
    segments: List[Segment] = []
    video_seg_counts: Dict[str, int] = defaultdict(int)

    for line in text.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        match = re.match(
            r"^(\S+)\s+(\d+:\d+(?:\.\d+)?)\s+(\d+:\d+(?:\.\d+)?)\s+##\s*(.*)",
            line,
        )
        if not match:
            log.debug("Skipping unparseable line: %s", line[:80])
            continue

        video_name = match.group(1)
        start_sec = _parse_timestamp(match.group(2))
        end_sec = _parse_timestamp(match.group(3))
        description = match.group(4).strip()

        idx = video_seg_counts[video_name]
        video_seg_counts[video_name] += 1

        segments.append(Segment(
            video_name=video_name,
            start_sec=start_sec,
            end_sec=end_sec,
            description=description,
            split=split_name,
            seg_index=idx,
        ))
    return segments


def load_annotations_from_zip(
    zf: zipfile.ZipFile,
    splits: List[str],
) -> List[Segment]:
    """Load annotations from the zip, preferring JSON over txt."""
    all_segments: List[Segment] = []
    names_in_zip = set(zf.namelist())

    for split in splits:
        json_name = f"{split}.json"
        txt_name = f"{split}.txt"

        if json_name in names_in_zip:
            log.info("Parsing %s (JSON)", json_name)
            segs = parse_annotations_json(zf.read(json_name), split)
        elif txt_name in names_in_zip:
            log.info("Parsing %s (txt fallback)", txt_name)
            segs = parse_annotations_txt(zf.read(txt_name), split)
        else:
            log.warning("No annotation file found for split %s", split)
            continue

        log.info("  %s: %d segments", split, len(segs))
        all_segments.extend(segs)

    return all_segments


# ─────────────────────────────────────────────────────────────────────────────
# Windowing
# ─────────────────────────────────────────────────────────────────────────────

def _generate_windows(
    duration: float, window_size: float = 10.0,
) -> List[Tuple[float, float]]:
    """
    Generate non-overlapping windows of *window_size* seconds.

    The last window slides back so it is a full *window_size* seconds.
    Videos shorter than *window_size* become a single window.
    """
    if duration <= 0:
        return []
    if duration <= window_size:
        return [(0.0, duration)]

    windows: List[Tuple[float, float]] = []
    start = 0.0
    while start + window_size <= duration:
        windows.append((start, start + window_size))
        start += window_size

    if start < duration and windows:
        last_start = max(0.0, duration - window_size)
        if last_start > windows[-1][0]:
            windows.append((last_start, duration))

    return windows


def _find_overlapping_annotations(
    segments: List[Segment],
    w_start: float,
    w_end: float,
) -> List[Segment]:
    """Return annotations whose time range overlaps [w_start, w_end)."""
    return [
        s for s in segments
        if s.start_sec < w_end and s.end_sec > w_start
    ]


# ─────────────────────────────────────────────────────────────────────────────
# ffmpeg / ffprobe helpers
# ─────────────────────────────────────────────────────────────────────────────

def _find_ffmpeg() -> str:
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
    raise FileNotFoundError("ffmpeg not found. Install it or add it to PATH.")


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
    raise FileNotFoundError("ffprobe not found. Install it alongside ffmpeg.")


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

    duration = stream.get("duration") or fmt.get("duration") or "0"
    duration = float(duration)

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


def _cut_window(
    ffmpeg: str,
    src: str,
    start_sec: float,
    end_sec: float,
    out_path: str,
    reencode: bool,
) -> bool:
    """Cut a time window from *src* and write to *out_path*."""
    duration = end_sec - start_sec
    if duration <= 0:
        return False

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    if reencode:
        cmd = [
            ffmpeg, "-y",
            "-ss", f"{start_sec:.3f}",
            "-i", src,
            "-t", f"{duration:.3f}",
            "-c:v", "libx264", "-crf", "18", "-preset", "slow",
            "-movflags", "+faststart",
            "-an",
            out_path,
        ]
    else:
        cmd = [
            ffmpeg, "-y",
            "-ss", f"{start_sec:.3f}",
            "-i", src,
            "-t", f"{duration:.3f}",
            "-c", "copy",
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
        return True
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        log.warning("Failed to cut window: %s", exc)
        if os.path.exists(out_path):
            os.remove(out_path)
        return False


# ─────────────────────────────────────────────────────────────────────────────
# YOLO helpers
# ─────────────────────────────────────────────────────────────────────────────

def _xyxy_to_yolo_norm(
    x1: float, y1: float, x2: float, y2: float, w: int, h: int,
) -> Tuple[float, float, float, float]:
    x1 = max(0.0, min(float(x1), float(w - 1)))
    x2 = max(0.0, min(float(x2), float(w - 1)))
    y1 = max(0.0, min(float(y1), float(h - 1)))
    y2 = max(0.0, min(float(y2), float(h - 1)))
    bw, bh = max(0.0, x2 - x1), max(0.0, y2 - y1)
    xc, yc = x1 + bw / 2.0, y1 + bh / 2.0
    return xc / w, yc / h, bw / w, bh / h


def _summarize_yolo(results) -> Tuple[Dict[int, int], Dict[int, float]]:
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
    cfg: UCAConfig,
    detector: YOLO,
    frames: List[np.ndarray],
    clip_fps: float,
    camera_name: str,
    sub_dir: str,
    clip_id: str,
) -> Tuple[List[Dict[str, Any]], Dict[int, int], Dict[int, float]]:
    agg_counts: Dict[int, int] = defaultdict(int)
    agg_max_conf: Dict[int, float] = defaultdict(float)

    if not cfg.yolo_enabled or not frames:
        return [], dict(agg_counts), dict(agg_max_conf)

    base_dir = os.path.join(cfg.output_dir, "yolo")
    img_dir = os.path.join(base_dir, "images", camera_name, sub_dir)
    lbl_dir = os.path.join(base_dir, "labels", camera_name, sub_dir)
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lbl_dir, exist_ok=True)

    step = max(1, int(round(clip_fps / max(0.1, cfg.yolo_fps))))
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
            "approx_time_offset_sec": float(i / clip_fps) if clip_fps > 0 else 0.0,
            "image_path": os.path.relpath(img_path, start=cfg.output_dir).replace("\\", "/"),
            "label_path": os.path.relpath(lbl_path, start=cfg.output_dir).replace("\\", "/"),
            "num_boxes": len(lines),
        })

    return exported, dict(agg_counts), dict(agg_max_conf)


# ─────────────────────────────────────────────────────────────────────────────
# Video discovery inside the zip
# ─────────────────────────────────────────────────────────────────────────────

def _build_video_lookup(zf: zipfile.ZipFile) -> Dict[str, str]:
    """Map video_name (stem) -> zip entry path for all video files."""
    lookup: Dict[str, str] = {}
    for entry in zf.namelist():
        if not entry.startswith(_ZIP_VIDEO_PREFIX):
            continue
        ext = os.path.splitext(entry)[1].lower()
        if ext not in VIDEO_EXTENSIONS:
            continue
        stem = os.path.splitext(os.path.basename(entry))[0]
        lookup[stem] = entry
    return lookup


def _extract_category(video_name: str) -> str:
    """Extract crime category from video name (e.g. 'Abuse039_x264' -> 'Abuse')."""
    match = re.match(r"^([A-Za-z]+)", video_name)
    return match.group(1) if match else "unknown"


# ─────────────────────────────────────────────────────────────────────────────
# Single-window processing
# ─────────────────────────────────────────────────────────────────────────────

def _process_window(
    cfg: UCAConfig,
    ffmpeg: str,
    ffprobe: str,
    detector: Optional[YOLO],
    extracted_video_path: str,
    win: Window,
    video_info: Dict[str, Any],
    skip_ids: Optional[set] = None,
) -> Optional[Dict[str, Any]]:
    """
    Process one time window: cut video, run YOLO, write metadata.

    Returns the metadata dict on success, None on skip/failure.
    """
    category = _extract_category(win.video_name)
    camera_name = category if cfg.camera_name_from == "category" else "uca"
    sub_dir = win.video_name
    clip_id = f"{category}_{win.video_name}_w{win.window_index:04d}"

    clips_dir = os.path.join(cfg.output_dir, "clips", camera_name, sub_dir)
    meta_dir = os.path.join(cfg.output_dir, "meta", camera_name, sub_dir)
    resp_dir = os.path.join(cfg.output_dir, "responses", camera_name, sub_dir)

    final_mp4 = os.path.join(clips_dir, f"{clip_id}.mp4")

    if (skip_ids and clip_id in skip_ids) or os.path.exists(final_mp4):
        log.debug("Skipping already-processed clip: %s", clip_id)
        return None

    os.makedirs(clips_dir, exist_ok=True)
    os.makedirs(meta_dir, exist_ok=True)
    os.makedirs(resp_dir, exist_ok=True)

    reencode = cfg.reencode
    if video_info.get("codec", "h264") != "h264" and not reencode:
        log.info("[%s] Source codec is %s — forcing re-encode for browser playback",
                 clip_id, video_info.get("codec"))
        reencode = True

    ok = _cut_window(
        ffmpeg, extracted_video_path,
        win.start_sec, win.end_sec,
        final_mp4, reencode,
    )
    if not ok:
        return None

    frames = _read_clip_frames(final_mp4)
    if not frames:
        log.warning("No frames read from %s — skipping", final_mp4)
        return None

    fh, fw = frames[0].shape[:2]
    clip_fps = video_info.get("fps", 30.0)
    duration = win.end_sec - win.start_sec

    # YOLO export
    exported: List[Dict[str, Any]] = []
    agg_counts: Dict[int, int] = {}
    agg_max_conf: Dict[int, float] = {}
    if detector is not None:
        exported, agg_counts, agg_max_conf = _export_yolo_frames(
            cfg, detector, frames, clip_fps, camera_name, sub_dir, clip_id,
        )

    trigger_detected = any(cid in agg_counts for cid in cfg.trigger_class_ids)
    trigger_conf_max = max(
        (agg_max_conf.get(cid, 0.0) for cid in cfg.trigger_class_ids),
        default=0.0,
    )

    meta: Dict[str, Any] = {
        "camera_name": camera_name,
        "kind": cfg.kind,
        "clip_path": os.path.relpath(final_mp4, start=cfg.output_dir).replace("\\", "/"),
        "source_file": f"{win.video_name}.mp4",
        "source_dataset": "uca",

        "uca_window": {
            "window_start_sec": win.start_sec,
            "window_end_sec": win.end_sec,
            "window_index": win.window_index,
            "total_windows": win.total_windows,
            "window_size_cfg": cfg.window_size_sec,
            "overlapping_annotations": [
                {
                    "start_sec": s.start_sec,
                    "end_sec": s.end_sec,
                    "description": s.description,
                    "segment_index": s.seg_index,
                }
                for s in win.overlapping_annotations
            ],
            "split": win.split,
        },

        "clip_start_ts": 0.0,
        "clip_end_ts": duration,
        "clip_start_local": f"{win.start_sec:.1f}s",
        "clip_end_local": f"{win.end_sec:.1f}s",

        "duration_sec": float(duration),
        "frames_written": len(frames),
        "fps_estimated": float(clip_fps),
        "vlm_sample_fps": 1,

        "model_response": win.aggregated_description,

        "buffer": {
            "store_fps": float(clip_fps),
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

    # Write VLM response text file
    raw_path = os.path.join(resp_dir, f"{clip_id}.model_raw.txt")
    meta["model_raw_text_path"] = os.path.relpath(
        raw_path, start=cfg.output_dir,
    ).replace("\\", "/")

    # Write metadata
    meta_path = os.path.join(meta_dir, f"{clip_id}.meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    with open(raw_path, "w", encoding="utf-8") as f:
        f.write(win.aggregated_description)

    log.info("Imported: %s (w%d/%d, %d frames, %dx%d, %.1fs)",
             clip_id, win.window_index, win.total_windows,
             len(frames), fw, fh, duration)
    return meta


# ─────────────────────────────────────────────────────────────────────────────
# Manifest helpers — tracks processed clip IDs across interrupted runs
# ─────────────────────────────────────────────────────────────────────────────

_MANIFEST_NAME = "_processed_manifest.json"


def _load_manifest(output_dir: str) -> set:
    """Load the set of already-processed clip IDs from the manifest file."""
    path = os.path.join(output_dir, _MANIFEST_NAME)
    if not os.path.isfile(path):
        return set()
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        entries = data.get("clips", [])
        return {e["clip_id"] for e in entries if isinstance(e, dict) and "clip_id" in e}
    except (json.JSONDecodeError, OSError, KeyError) as exc:
        log.warning("Could not load manifest %s: %s", path, exc)
        return set()


def _save_manifest(output_dir: str, clip_ids: set) -> None:
    """Write the manifest with all processed clip IDs (atomic via tmp+replace)."""
    path = os.path.join(output_dir, _MANIFEST_NAME)
    sorted_ids = sorted(clip_ids)
    data = {
        "total_processed": len(sorted_ids),
        "clips": [{"clip_id": cid} for cid in sorted_ids],
    }
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def reset_manifest(output_dir: str) -> None:
    """Delete the manifest file to force full reprocessing."""
    path = os.path.join(output_dir, _MANIFEST_NAME)
    if os.path.isfile(path):
        os.remove(path)
        log.info("Manifest deleted: %s", path)


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def run(
    zip_path: str,
    cfg: UCAConfig,
    extract_dir: Optional[str] = None,
) -> Dict[str, int]:
    """
    Import the UCA dataset from a zip file using fixed-duration windowing.

    Returns counts: {imported, failed, skipped, total_windows}.
    """
    if not os.path.isfile(zip_path):
        log.error("Zip file not found: %s", zip_path)
        return {"imported": 0, "failed": 0, "skipped": 0, "total_windows": 0}

    ffmpeg = _find_ffmpeg()
    ffprobe = _find_ffprobe(ffmpeg)
    log.info("Using ffmpeg: %s", ffmpeg)
    log.info("Window size: %.1fs", cfg.window_size_sec)

    detector: Optional[YOLO] = None
    if cfg.yolo_enabled:
        log.info("Loading YOLO model: %s", cfg.yolo_model)
        detector = YOLO(cfg.yolo_model)

    with zipfile.ZipFile(zip_path, "r") as zf:
        all_segments = load_annotations_from_zip(zf, cfg.splits)
        log.info("Total annotation segments: %d", len(all_segments))

        video_lookup = _build_video_lookup(zf)
        log.info("Found %d video files in zip", len(video_lookup))

        # Group annotations by video
        video_annotations: Dict[str, List[Segment]] = OrderedDict()
        for seg in all_segments:
            video_annotations.setdefault(seg.video_name, []).append(seg)

        # Also include videos that have no annotations (they exist in zip)
        for vname in video_lookup:
            if vname not in video_annotations:
                video_annotations[vname] = []

        if extract_dir is None:
            extract_dir = os.path.join(cfg.output_dir, "_uca_raw")

        os.makedirs(cfg.output_dir, exist_ok=True)

        skip_ids = _load_manifest(cfg.output_dir)
        if skip_ids:
            log.info("Resuming: manifest has %d clips already processed.", len(skip_ids))

        imported = 0
        failed = 0
        skipped = 0
        total_windows = 0

        video_names = list(video_annotations.keys())
        pbar = tqdm(video_names, desc="Processing videos", unit="video")

        excluded_set = {c.lower() for c in cfg.exclude_categories}
        if excluded_set:
            log.info("Excluding categories: %s", ", ".join(sorted(cfg.exclude_categories)))

        for video_name in pbar:
            annotations = video_annotations[video_name]
            split_name = annotations[0].split if annotations else "unknown"

            if video_name not in video_lookup:
                log.warning("Video not found in zip: %s", video_name)
                skipped += 1
                continue

            category = _extract_category(video_name)
            if excluded_set and category.lower() in excluded_set:
                continue

            pbar.set_postfix_str(video_name)

            zip_entry = video_lookup[video_name]
            video_dir = os.path.join(extract_dir, "videos")
            os.makedirs(video_dir, exist_ok=True)
            extracted_path = os.path.join(video_dir, os.path.basename(zip_entry))

            if not os.path.exists(extracted_path):
                try:
                    with zf.open(zip_entry) as src, open(extracted_path, "wb") as dst:
                        shutil.copyfileobj(src, dst)
                except Exception as exc:
                    log.error("Failed to extract %s: %s", video_name, exc)
                    failed += 1
                    continue

            try:
                video_info = _probe_video(ffprobe, extracted_path)
            except Exception as exc:
                log.error("Failed to probe %s: %s", video_name, exc)
                failed += 1
                continue

            vid_duration = video_info.get("duration", 0.0)
            if vid_duration <= 0:
                log.warning("Zero-duration video: %s — skipping", video_name)
                skipped += 1
                continue

            windows = _generate_windows(vid_duration, cfg.window_size_sec)
            total_windows += len(windows)

            # Fast-skip: if all windows for this video are already processed
            category = _extract_category(video_name)
            all_clip_ids = [
                f"{category}_{video_name}_w{i:04d}"
                for i in range(len(windows))
            ]
            if skip_ids and all(cid in skip_ids for cid in all_clip_ids):
                skipped += len(windows)
                try:
                    os.remove(extracted_path)
                except OSError:
                    pass
                continue

            new_in_video: List[str] = []
            for wi, (w_start, w_end) in enumerate(windows):
                overlapping = _find_overlapping_annotations(
                    annotations, w_start, w_end,
                )
                descriptions = [
                    s.description for s in overlapping if s.description.strip()
                ]
                aggregated = _VLM_TEXT_SEPARATOR.join(descriptions)

                win = Window(
                    video_name=video_name,
                    start_sec=w_start,
                    end_sec=w_end,
                    window_index=wi,
                    total_windows=len(windows),
                    split=split_name,
                    overlapping_annotations=overlapping,
                    aggregated_description=aggregated,
                )

                try:
                    meta = _process_window(
                        cfg, ffmpeg, ffprobe, detector,
                        extracted_path, win, video_info,
                        skip_ids=skip_ids,
                    )
                    if meta:
                        imported += 1
                        cid = f"{category}_{video_name}_w{wi:04d}"
                        new_in_video.append(cid)
                    else:
                        skipped += 1
                except Exception:
                    log.exception("Failed window %d of %s", wi, video_name)
                    failed += 1

            try:
                os.remove(extracted_path)
            except OSError:
                pass

            if new_in_video:
                skip_ids.update(new_in_video)
                _save_manifest(cfg.output_dir, skip_ids)

        pbar.close()

    try:
        shutil.rmtree(os.path.join(extract_dir, "videos"), ignore_errors=True)
    except OSError:
        pass

    log.info(
        "Import complete: %d imported, %d failed, %d skipped (%d total windows)",
        imported, failed, skipped, total_windows,
    )
    return {
        "imported": imported,
        "failed": failed,
        "skipped": skipped,
        "total_windows": total_windows,
    }
