"""YOLO <-> Label Studio coordinate conversion, label parsing, and tracking."""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional, Tuple

from ..config import get_coco_labels, get_tracking_config

log = logging.getLogger("labeling.yolo")


# ---------------------------------------------------------------------------
# Coordinate helpers
# ---------------------------------------------------------------------------

def yolo_to_ls_rect(
    xc: float, yc: float, w: float, h: float,
) -> Dict[str, float]:
    """
    Convert YOLO normalised (0-1) centre-xywh to Label Studio
    percentage (0-100) top-left-xywh.
    """
    return {
        "x": max(0.0, (xc - w / 2.0)) * 100.0,
        "y": max(0.0, (yc - h / 2.0)) * 100.0,
        "width": w * 100.0,
        "height": h * 100.0,
    }


def _iou(a: Dict[str, float], b: Dict[str, float]) -> float:
    """
    Compute IoU between two boxes in Label Studio percentage coords
    (top-left x/y + width/height, range 0-100).
    """
    ax1, ay1 = a["x"], a["y"]
    ax2, ay2 = ax1 + a["width"], ay1 + a["height"]
    bx1, by1 = b["x"], b["y"]
    bx2, by2 = bx1 + b["width"], by1 + b["height"]

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter == 0.0:
        return 0.0

    area_a = a["width"] * a["height"]
    area_b = b["width"] * b["height"]
    union = area_a + area_b - inter
    if union <= 0.0:
        return 0.0
    return inter / union


# ---------------------------------------------------------------------------
# YOLO label file reader
# ---------------------------------------------------------------------------

def read_yolo_label_file(path: str) -> List[Dict[str, Any]]:
    """
    Parse a YOLO label ``.txt`` file.

    Returns list of dicts: ``{class_id, xc, yc, w, h}``.
    """
    detections: List[Dict[str, Any]] = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) < 5:
                    continue
                detections.append({
                    "class_id": int(parts[0]),
                    "xc": float(parts[1]),
                    "yc": float(parts[2]),
                    "w": float(parts[3]),
                    "h": float(parts[4]),
                })
    except FileNotFoundError:
        pass
    return detections


# ---------------------------------------------------------------------------
# IoU-based greedy tracker
# ---------------------------------------------------------------------------

_KeyFrame = Dict[str, Any]


def _make_keyframe(
    rect: Dict[str, float], frame: int, time_offset: float,
) -> _KeyFrame:
    return {
        "frame": frame,
        "x": round(rect["x"], 4),
        "y": round(rect["y"], 4),
        "width": round(rect["width"], 4),
        "height": round(rect["height"], 4),
        "time": round(time_offset, 4),
        "enabled": True,
        "rotation": 0,
    }


def _track_detections(
    frame_detections: List[Tuple[int, float, List[Dict[str, Any]]]],
    coco_labels: Dict[int, str],
    iou_threshold: float,
    all_processed_frames: Optional[Dict[int, float]] = None,
) -> List[Dict[str, Any]]:
    """
    Group per-frame YOLO detections into tracked objects using greedy IoU
    matching.

    Parameters
    ----------
    frame_detections
        List of ``(video_frame, time_offset, detections)`` tuples sorted
        by frame number.  Each detection dict has keys
        ``class_id, xc, yc, w, h``.
    coco_labels
        COCO class-id -> label name mapping.
    iou_threshold
        Minimum IoU to match a detection to an existing track.
    all_processed_frames
        Optional mapping of ``{video_frame: time_offset}`` for ALL frames
        that were processed by YOLO (including those with zero detections).
        When provided, tracks that end before the last processed frame get
        an ``enabled=False`` keyframe so the box disappears in Label Studio.

    Returns
    -------
    List of track dicts, each with:
        ``label_name``, ``class_id``, ``keyframes`` (list of LS keyframe
        dicts ready for the ``sequence`` field).
    """
    tracks: List[Dict[str, Any]] = []

    for video_frame, time_offset, detections in frame_detections:
        used_tracks: set[int] = set()
        unmatched_dets: List[Dict[str, Any]] = []

        scored: List[Tuple[float, int, int, Dict[str, float]]] = []
        for di, det in enumerate(detections):
            rect = yolo_to_ls_rect(det["xc"], det["yc"], det["w"], det["h"])
            for ti, trk in enumerate(tracks):
                if trk["class_id"] != det["class_id"]:
                    continue
                score = _iou(trk["last_rect"], rect)
                if score >= iou_threshold:
                    scored.append((score, ti, di, rect))

        scored.sort(key=lambda t: t[0], reverse=True)

        matched_dets: set[int] = set()
        for score, ti, di, rect in scored:
            if ti in used_tracks or di in matched_dets:
                continue
            det = detections[di]
            kf = _make_keyframe(rect, video_frame, time_offset)
            tracks[ti]["keyframes"].append(kf)
            tracks[ti]["last_rect"] = rect
            used_tracks.add(ti)
            matched_dets.add(di)

        for di, det in enumerate(detections):
            if di not in matched_dets:
                unmatched_dets.append(det)

        for det in unmatched_dets:
            rect = yolo_to_ls_rect(det["xc"], det["yc"], det["w"], det["h"])
            class_id = det["class_id"]
            label_name = coco_labels.get(class_id, f"class_{class_id}")
            kf = _make_keyframe(rect, video_frame, time_offset)
            tracks.append({
                "class_id": class_id,
                "label_name": label_name,
                "last_rect": rect,
                "keyframes": [kf],
            })

    # Add disabled keyframes so boxes disappear when the object leaves.
    if all_processed_frames and tracks:
        sorted_all = sorted(all_processed_frames.keys())
        max_frame = sorted_all[-1]
        for trk in tracks:
            last_kf = trk["keyframes"][-1]
            if last_kf["frame"] >= max_frame:
                continue
            next_frames = [f for f in sorted_all if f > last_kf["frame"]]
            if next_frames:
                disable_frame = next_frames[0]
                trk["keyframes"].append({
                    "frame": disable_frame,
                    "x": last_kf["x"],
                    "y": last_kf["y"],
                    "width": last_kf["width"],
                    "height": last_kf["height"],
                    "time": round(all_processed_frames[disable_frame], 4),
                    "enabled": False,
                    "rotation": 0,
                })

    return tracks


# ---------------------------------------------------------------------------
# Prediction builder
# ---------------------------------------------------------------------------

def build_predictions(
    meta: Dict[str, Any],
    dataset_dir: str,
) -> List[Dict[str, Any]]:
    """
    Build a Label Studio ``predictions`` list for a single task.

    When tracking is enabled (default), per-frame YOLO detections are merged
    into tracked objects via IoU matching so that each object produces a
    single ``VideoRectangle`` with multi-frame keypoints instead of N
    separate overlapping boxes.
    """
    coco_labels = get_coco_labels()
    tracking_cfg = get_tracking_config()
    tracking_enabled = tracking_cfg["enabled"]
    iou_threshold = tracking_cfg["iou_threshold"]

    results: List[Dict[str, Any]] = []
    result_idx = 0

    # --- YOLO box predictions (VideoRectangle) ---
    yolo_export = meta.get("yolo_export")
    if yolo_export and yolo_export.get("enabled"):
        exported_frames: List[Dict[str, Any]] = yolo_export.get("exported_frames", [])

        # Label Studio caps frameRate at 10 (see tasks.py).  YOLO frame_index
        # values reference the video's native FPS, so we must re-map them to
        # the LS frame space to keep keyframes synchronised with playback.
        native_fps = meta.get(
            "fps_estimated",
            meta.get("buffer", {}).get("store_fps", 10.0),
        )
        ls_fps = min(native_fps, 10.0)
        fps_ratio = ls_fps / max(native_fps, 1e-6)

        all_processed_frames: Dict[int, float] = {}
        frame_detections: List[Tuple[int, float, List[Dict[str, Any]]]] = []
        for ef in exported_frames:
            frame_index: int = ef.get("frame_index", 0)
            video_frame = round(frame_index * fps_ratio) + 1
            time_offset = ef.get("approx_time_offset_sec", 0.0)
            all_processed_frames[video_frame] = time_offset

            label_rel: str = ef.get("label_path", "")
            if not label_rel:
                continue

            label_path = os.path.join(dataset_dir, label_rel.replace("\\", "/"))
            detections = [
                d for d in read_yolo_label_file(label_path)
                if d["class_id"] in coco_labels
            ]
            if detections:
                frame_detections.append((video_frame, time_offset, detections))

        if tracking_enabled:
            tracks = _track_detections(
                frame_detections, coco_labels, iou_threshold,
                all_processed_frames=all_processed_frames,
            )
            for trk in tracks:
                region_id = f"bbox_{result_idx}"
                results.append({
                    "id": region_id,
                    "type": "videorectangle",
                    "from_name": "bbox",
                    "to_name": "video_full",
                    "value": {"sequence": trk["keyframes"]},
                })
                results.append({
                    "id": region_id,
                    "type": "labels",
                    "from_name": "label",
                    "to_name": "video_full",
                    "value": {
                        "labels": [trk["label_name"]],
                        "sequence": trk["keyframes"],
                    },
                })
                result_idx += 1
        else:
            for video_frame, time_offset, detections in frame_detections:
                for det in detections:
                    class_id = det["class_id"]
                    label_name = coco_labels.get(class_id, f"class_{class_id}")
                    rect = yolo_to_ls_rect(det["xc"], det["yc"], det["w"], det["h"])
                    kf = _make_keyframe(rect, video_frame, time_offset)
                    region_id = f"bbox_{result_idx}"

                    results.append({
                        "id": region_id,
                        "type": "videorectangle",
                        "from_name": "bbox",
                        "to_name": "video_full",
                        "value": {"sequence": [kf]},
                    })
                    results.append({
                        "id": region_id,
                        "type": "labels",
                        "from_name": "label",
                        "to_name": "video_full",
                        "value": {
                            "labels": [label_name],
                            "sequence": [kf],
                        },
                    })
                    result_idx += 1

    # --- VLM text prediction (TextArea) ---
    vlm_text = meta.get("model_response", "")
    if vlm_text:
        results.append({
            "id": f"vlm_text_{result_idx}",
            "type": "textarea",
            "from_name": "vlm_description",
            "to_name": "video_crop",
            "value": {
                "text": [vlm_text],
            },
        })

    return results
