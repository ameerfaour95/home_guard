"""
Core logic for importing the SmartHome-Bench dataset.

Downloads videos from YouTube, optionally trims them, splits into fixed-duration
windows (default 10s), runs YOLO detection, and writes metadata compatible with
the labeling pipeline.

Crash-resilient: a manifest file tracks processed clip IDs so interrupted runs
can be resumed by simply re-running the command.
"""

from __future__ import annotations

import csv
import json
import logging
import os
import re
import shutil
import subprocess
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from threading import Lock
from typing import Any, Dict, List, Optional, Set, Tuple

import cv2
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO

from .config import COCO_NAMES, SmartHomeConfig

log = logging.getLogger("smarthome_import")

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mkv", ".mov", ".webm", ".flv", ".ts", ".m4v"}


# ─────────────────────────────────────────────────────────────────────────────
# Annotation CSV parsing
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class VideoAnnotation:
    """Parsed row from Video_Annotation.csv."""
    title: str
    categories: List[str]
    anomaly_tag: str
    description: str
    reasoning: str


def _sanitize_camera_name(category: str) -> str:
    """Turn a human category name into a filesystem-safe camera_name."""
    return re.sub(r"[^A-Za-z0-9]+", "_", category.strip()).strip("_")


def parse_annotation_csv(csv_path: str) -> Dict[str, VideoAnnotation]:
    """Parse Video_Annotation.csv → {title: VideoAnnotation}."""
    annotations: Dict[str, VideoAnnotation] = {}
    with open(csv_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            title = row.get("Title", "").strip()
            if not title:
                continue
            raw_cat = row.get("Category", "").strip()
            cats = [c.strip() for c in raw_cat.split(",") if c.strip()]
            annotations[title] = VideoAnnotation(
                title=title,
                categories=cats,
                anomaly_tag=row.get("Anomaly Tag", "").strip(),
                description=row.get("Video Description", "").strip(),
                reasoning=row.get("Reasoning behind Normality/Anomalies", "").strip(),
            )
    return annotations


def parse_trim_csv(csv_path: str) -> Dict[str, float]:
    """Parse Trim_video_label.csv → {title: trim_seconds}.

    Positive = remove from end, negative = remove from beginning, 0 = no trim.
    """
    trims: Dict[str, float] = {}
    if not os.path.isfile(csv_path):
        return trims
    with open(csv_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            title = row.get("Title", "").strip()
            time_val = row.get("Time", "").strip()
            if title and time_val:
                try:
                    trims[title] = float(time_val)
                except ValueError:
                    pass
    return trims


def filter_by_categories(
    annotations: Dict[str, VideoAnnotation],
    selected: List[str],
) -> Dict[str, VideoAnnotation]:
    """Keep only videos whose category list intersects with *selected*."""
    selected_lower = {c.lower() for c in selected}
    filtered: Dict[str, VideoAnnotation] = {}
    for title, ann in annotations.items():
        ann_cats_lower = {c.lower() for c in ann.categories}
        if ann_cats_lower & selected_lower:
            filtered[title] = ann
    return filtered


def _pick_category(ann: VideoAnnotation, selected: List[str]) -> str:
    """Pick the first matching category from the video's categories."""
    selected_lower = {c.lower(): c for c in selected}
    for cat in ann.categories:
        if cat.lower() in selected_lower:
            return selected_lower[cat.lower()]
    return ann.categories[0] if ann.categories else "unknown"


# ─────────────────────────────────────────────────────────────────────────────
# Video URL CSV parsing
# ─────────────────────────────────────────────────────────────────────────────

def parse_url_csv(csv_path: str) -> Dict[str, str]:
    """Parse Video_url.csv → {title: youtube_url}. Skips privacy entries."""
    urls: Dict[str, str] = {}
    with open(csv_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            title_key = "Title" if "Title" in row else "title"
            url_key = "URL" if "URL" in row else "url"
            title = row.get(title_key, "").strip()
            url = row.get(url_key, "").strip()
            if not title or not url:
                continue
            if url.lower() == "privacy url" or not url.startswith("http"):
                continue
            urls[title] = url
    return urls


# ─────────────────────────────────────────────────────────────────────────────
# YouTube download
# ─────────────────────────────────────────────────────────────────────────────

def _check_ytdlp() -> bool:
    try:
        subprocess.run(["yt-dlp", "--version"], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def _download_one(
    title: str,
    url: str,
    output_dir: str,
    fmt: str,
    timeout: int,
    retries: int,
) -> Tuple[str, str]:
    """Download a single video. Returns (title, status)."""
    for ext in [".mp4", ".mkv", ".webm", ".flv"]:
        if os.path.isfile(os.path.join(output_dir, f"{title}{ext}")):
            return title, "exists"

    output_path = os.path.join(output_dir, f"{title}.%(ext)s")
    cmd = [
        "yt-dlp", "-o", output_path,
        "--no-playlist",
        "--format", fmt,
        "--no-warnings", "--quiet",
        "--retries", str(retries),
        url,
    ]
    try:
        subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=timeout)
        for ext in [".mp4", ".mkv", ".webm", ".flv"]:
            if os.path.isfile(os.path.join(output_dir, f"{title}{ext}")):
                return title, "ok"
        return title, "missing"
    except subprocess.TimeoutExpired:
        return title, "timeout"
    except subprocess.CalledProcessError:
        return title, "failed"


def download_videos(
    titles_urls: Dict[str, str],
    output_dir: str,
    cfg: SmartHomeConfig,
) -> Dict[str, str]:
    """Download all requested videos. Returns {title: status}."""
    os.makedirs(output_dir, exist_ok=True)
    results: Dict[str, str] = {}

    if not _check_ytdlp():
        log.error(
            "yt-dlp not found. Install it: pip install yt-dlp  "
            "or  uv pip install yt-dlp"
        )
        return {t: "no_ytdlp" for t in titles_urls}

    log.info("Downloading %d videos (max %d workers)...", len(titles_urls), cfg.dl_max_workers)

    lock = Lock()
    done = [0]

    def _task(item: Tuple[str, str]) -> Tuple[str, str]:
        title, url = item
        r = _download_one(title, url, output_dir, cfg.dl_format, cfg.dl_timeout_sec, cfg.dl_retries)
        with lock:
            done[0] += 1
            if r[1] == "ok":
                log.info("[%d/%d] Downloaded: %s", done[0], len(titles_urls), title)
            elif r[1] == "exists":
                log.debug("[%d/%d] Already exists: %s", done[0], len(titles_urls), title)
            else:
                log.warning("[%d/%d] %s: %s", done[0], len(titles_urls), r[1], title)
        return r

    with ThreadPoolExecutor(max_workers=cfg.dl_max_workers) as pool:
        futures = {pool.submit(_task, item): item for item in titles_urls.items()}
        for future in as_completed(futures):
            title, status = future.result()
            results[title] = status

    ok = sum(1 for s in results.values() if s in ("ok", "exists"))
    log.info("Download summary: %d ok/exists, %d failed out of %d",
             ok, len(results) - ok, len(results))
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Video trimming
# ─────────────────────────────────────────────────────────────────────────────

def _find_video_file(directory: str, title: str) -> Optional[str]:
    """Locate the downloaded video file for a given title."""
    for ext in VIDEO_EXTENSIONS:
        p = os.path.join(directory, f"{title}{ext}")
        if os.path.isfile(p):
            return p
    return None


def trim_video(
    ffmpeg: str,
    src: str,
    trim_sec: float,
    out_path: str,
    duration: float,
) -> bool:
    """Trim a video based on the Time value from the CSV.

    trim_sec > 0: remove last trim_sec seconds.
    trim_sec < 0: remove first |trim_sec| seconds.
    trim_sec == 0: copy as-is.
    """
    if trim_sec == 0.0:
        shutil.copy2(src, out_path)
        return True

    if trim_sec > 0:
        new_duration = max(0.1, duration - trim_sec)
        cmd = [ffmpeg, "-y", "-i", src, "-t", f"{new_duration:.3f}",
               "-c", "copy", "-an", out_path]
    else:
        start = abs(trim_sec)
        cmd = [ffmpeg, "-y", "-ss", f"{start:.3f}", "-i", src,
               "-c", "copy", "-an", out_path]

    try:
        subprocess.run(cmd, capture_output=True, check=True, timeout=120)
        return os.path.isfile(out_path)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        if os.path.exists(out_path):
            os.remove(out_path)
        return False


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
    cfg: SmartHomeConfig,
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
# VLM response text from SmartHome-Bench annotations
# ─────────────────────────────────────────────────────────────────────────────

def _build_vlm_text(ann: Optional[VideoAnnotation]) -> str:
    """Build pre-filled VLM description from SmartHome-Bench annotations.

    Combines the video description and anomaly reasoning so annotators can
    refine existing text instead of writing from scratch.
    """
    if ann is None:
        return ""
    parts: List[str] = []
    if ann.description:
        parts.append(ann.description)
    if ann.reasoning:
        parts.append(f"Reasoning: {ann.reasoning}")
    return "\n----\n".join(parts)


# ─────────────────────────────────────────────────────────────────────────────
# Single-window processing
# ─────────────────────────────────────────────────────────────────────────────

def _process_window(
    cfg: SmartHomeConfig,
    ffmpeg: str,
    detector: Optional[YOLO],
    video_path: str,
    title: str,
    ann: Optional[VideoAnnotation],
    camera_name: str,
    w_start: float,
    w_end: float,
    w_index: int,
    total_windows: int,
    video_info: Dict[str, Any],
    skip_ids: Optional[Set[str]] = None,
) -> Optional[Tuple[str, Dict[str, Any]]]:
    """Process one window: cut, YOLO, write metadata. Returns (clip_id, meta) or None."""
    sub_dir = title
    clip_id = f"{_sanitize_camera_name(camera_name)}_{title}_w{w_index:04d}"

    if skip_ids and clip_id in skip_ids:
        return None

    clips_dir = os.path.join(cfg.output_dir, "clips", camera_name, sub_dir)
    meta_dir = os.path.join(cfg.output_dir, "meta", camera_name, sub_dir)
    resp_dir = os.path.join(cfg.output_dir, "responses", camera_name, sub_dir)

    final_mp4 = os.path.join(clips_dir, f"{clip_id}.mp4")
    if os.path.exists(final_mp4):
        return None

    os.makedirs(clips_dir, exist_ok=True)
    os.makedirs(meta_dir, exist_ok=True)
    os.makedirs(resp_dir, exist_ok=True)

    reencode = cfg.reencode
    if video_info.get("codec", "h264") != "h264" and not reencode:
        log.info("[%s] Source codec is %s — forcing re-encode", clip_id, video_info.get("codec"))
        reencode = True

    ok = _cut_window(ffmpeg, video_path, w_start, w_end, final_mp4, reencode)
    if not ok:
        return None

    frames = _read_clip_frames(final_mp4)
    if not frames:
        log.warning("No frames read from %s — skipping", final_mp4)
        return None

    fh, fw = frames[0].shape[:2]
    clip_fps = video_info.get("fps", 30.0)
    duration = w_end - w_start

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
        "source_file": f"{title}.mp4",
        "source_dataset": "smarthome_bench",

        "smarthome_window": {
            "window_start_sec": w_start,
            "window_end_sec": w_end,
            "window_index": w_index,
            "total_windows": total_windows,
            "window_size_cfg": cfg.window_size_sec,
        },

        "smarthome_annotation": {
            "description": ann.description if ann else "",
            "anomaly_tag": ann.anomaly_tag if ann else "",
            "reasoning": ann.reasoning if ann else "",
            "original_categories": ann.categories if ann else [],
        },

        "clip_start_ts": 0.0,
        "clip_end_ts": duration,
        "clip_start_local": f"{w_start:.1f}s",
        "clip_end_local": f"{w_end:.1f}s",

        "duration_sec": float(duration),
        "frames_written": len(frames),
        "fps_estimated": float(clip_fps),
        "vlm_sample_fps": 1,

        "model_response": _build_vlm_text(ann),

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

    raw_path = os.path.join(resp_dir, f"{clip_id}.model_raw.txt")
    meta["model_raw_text_path"] = os.path.relpath(
        raw_path, start=cfg.output_dir,
    ).replace("\\", "/")

    meta_path = os.path.join(meta_dir, f"{clip_id}.meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    vlm_text = _build_vlm_text(ann)
    with open(raw_path, "w", encoding="utf-8") as f:
        f.write(vlm_text)

    log.info("Imported: %s (w%d/%d, %d frames, %dx%d, %.1fs)",
             clip_id, w_index, total_windows, len(frames), fw, fh, duration)
    return clip_id, meta


# ─────────────────────────────────────────────────────────────────────────────
# Manifest helpers
# ─────────────────────────────────────────────────────────────────────────────

_MANIFEST_NAME = "_processed_manifest.json"


def _load_manifest(output_dir: str) -> Set[str]:
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


def _save_manifest(output_dir: str, clip_ids: Set[str]) -> None:
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
    path = os.path.join(output_dir, _MANIFEST_NAME)
    if os.path.isfile(path):
        os.remove(path)
        log.info("Manifest deleted: %s", path)


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def run(
    repo_dir: str,
    cfg: SmartHomeConfig,
    skip_download: bool = False,
) -> Dict[str, int]:
    """
    Import the SmartHome-Bench dataset.

    Parameters
    ----------
    repo_dir
        Path to the cloned SmartHome-Bench-LLM repository (or directory
        containing Video_Annotation.csv and Video_url.csv under Videos/).
    cfg
        Pipeline configuration.
    skip_download
        If True, skip YouTube downloads (use already-downloaded videos).

    Returns
    -------
    dict with keys: imported, failed, skipped, total_windows, downloaded.
    """
    annotation_csv = os.path.join(repo_dir, "Videos", "Video_Annotation.csv")
    url_csv = os.path.join(repo_dir, "Videos", "Video_url.csv")
    trim_csv = os.path.join(repo_dir, "Videos", "Trim_Videos", "Trim_video_label.csv")

    if not os.path.isfile(annotation_csv):
        log.error("Video_Annotation.csv not found at %s", annotation_csv)
        return {"imported": 0, "failed": 0, "skipped": 0, "total_windows": 0, "downloaded": 0}

    ffmpeg = _find_ffmpeg()
    ffprobe = _find_ffprobe(ffmpeg)
    log.info("Using ffmpeg: %s", ffmpeg)
    log.info("Window size: %.1fs", cfg.window_size_sec)

    # ── Parse CSVs ────────────────────────────────────────────────────────
    all_annotations = parse_annotation_csv(annotation_csv)
    log.info("Parsed %d video annotations", len(all_annotations))

    filtered = filter_by_categories(all_annotations, cfg.categories)
    log.info("Filtered to %d videos in categories: %s",
             len(filtered), ", ".join(cfg.categories))

    urls = parse_url_csv(url_csv) if os.path.isfile(url_csv) else {}
    log.info("Parsed %d downloadable URLs", len(urls))

    trims = parse_trim_csv(trim_csv) if os.path.isfile(trim_csv) else {}
    log.info("Parsed %d trim entries", len(trims))

    # ── Download ──────────────────────────────────────────────────────────
    raw_dir = os.path.join(cfg.output_dir, "_smarthome_raw", "videos")
    trimmed_dir = os.path.join(cfg.output_dir, "_smarthome_raw", "trimmed")

    titles_to_download = {t: urls[t] for t in filtered if t in urls}
    titles_without_url = [t for t in filtered if t not in urls]
    if titles_without_url:
        log.info("%d videos have no downloadable URL (private)", len(titles_without_url))

    download_count = 0
    if not skip_download and titles_to_download:
        dl_results = download_videos(titles_to_download, raw_dir, cfg)
        download_count = sum(1 for s in dl_results.values() if s in ("ok", "exists"))
    elif skip_download:
        log.info("Skipping download (--skip-download)")
    else:
        log.info("No videos to download")

    # ── Load YOLO ─────────────────────────────────────────────────────────
    detector: Optional[YOLO] = None
    if cfg.yolo_enabled:
        log.info("Loading YOLO model: %s", cfg.yolo_model)
        detector = YOLO(cfg.yolo_model)

    os.makedirs(cfg.output_dir, exist_ok=True)
    os.makedirs(trimmed_dir, exist_ok=True)

    skip_ids = _load_manifest(cfg.output_dir)
    if skip_ids:
        log.info("Resuming: manifest has %d clips already processed.", len(skip_ids))

    imported = 0
    failed = 0
    skipped = 0
    total_windows = 0

    titles_sorted = sorted(filtered.keys())
    pbar = tqdm(titles_sorted, desc="Processing videos", unit="video")

    for title in pbar:
        ann = filtered[title]
        category = _pick_category(ann, cfg.categories)
        camera_name = _sanitize_camera_name(category) if cfg.camera_name_from == "category" else "smarthome"

        pbar.set_postfix_str(title[:30])

        video_path = _find_video_file(raw_dir, title)
        if video_path is None:
            log.debug("Video file not found for %s — skipping", title)
            skipped += 1
            continue

        # ── Trim if needed ────────────────────────────────────────────
        trim_val = trims.get(title, 0.0)
        if trim_val != 0.0:
            trimmed_path = os.path.join(trimmed_dir, f"{title}.mp4")
            if not os.path.exists(trimmed_path):
                try:
                    probe = _probe_video(ffprobe, video_path)
                except Exception as exc:
                    log.error("Failed to probe %s: %s", title, exc)
                    failed += 1
                    continue
                ok = trim_video(ffmpeg, video_path, trim_val, trimmed_path, probe["duration"])
                if not ok:
                    log.warning("Trim failed for %s — using original", title)
                else:
                    video_path = trimmed_path
            else:
                video_path = trimmed_path

        # ── Probe ─────────────────────────────────────────────────────
        try:
            video_info = _probe_video(ffprobe, video_path)
        except Exception as exc:
            log.error("Failed to probe %s: %s", title, exc)
            failed += 1
            continue

        vid_duration = video_info.get("duration", 0.0)
        if vid_duration <= 0:
            log.warning("Zero-duration video: %s — skipping", title)
            skipped += 1
            continue

        windows = _generate_windows(vid_duration, cfg.window_size_sec)
        total_windows += len(windows)

        sanitized = _sanitize_camera_name(camera_name)
        all_clip_ids = [f"{sanitized}_{title}_w{i:04d}" for i in range(len(windows))]
        if skip_ids and all(cid in skip_ids for cid in all_clip_ids):
            skipped += len(windows)
            continue

        new_in_video: List[str] = []
        for wi, (w_start, w_end) in enumerate(windows):
            try:
                result = _process_window(
                    cfg, ffmpeg, detector, video_path,
                    title, ann, camera_name,
                    w_start, w_end, wi, len(windows),
                    video_info, skip_ids=skip_ids,
                )
                if result:
                    clip_id, _ = result
                    imported += 1
                    new_in_video.append(clip_id)
                else:
                    skipped += 1
            except Exception:
                log.exception("Failed window %d of %s", wi, title)
                failed += 1

        if new_in_video:
            skip_ids.update(new_in_video)
            _save_manifest(cfg.output_dir, skip_ids)

    pbar.close()

    log.info(
        "Import complete: %d imported, %d failed, %d skipped (%d total windows), %d downloaded",
        imported, failed, skipped, total_windows, download_count,
    )
    return {
        "imported": imported,
        "failed": failed,
        "skipped": skipped,
        "total_windows": total_windows,
        "downloaded": download_count,
    }
