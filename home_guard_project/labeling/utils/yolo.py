"""YOLO <-> Label Studio coordinate conversion and label parsing."""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List

from ..config import get_coco_labels

log = logging.getLogger("labeling.yolo")


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


def build_predictions(
    meta: Dict[str, Any],
    dataset_dir: str,
) -> List[Dict[str, Any]]:
    """
    Build a Label Studio ``predictions`` list for a single task.

    Includes:
      - VideoRectangle results from YOLO exported frames
      - TextArea result from VLM model_response
    """
    coco_labels = get_coco_labels()
    results: List[Dict[str, Any]] = []
    result_idx = 0

    # --- YOLO box predictions (VideoRectangle) ---
    yolo_export = meta.get("yolo_export")
    if yolo_export and yolo_export.get("enabled"):
        exported_frames: List[Dict[str, Any]] = yolo_export.get("exported_frames", [])
        for ef in exported_frames:
            frame_index: int = ef.get("frame_index", 0)
            label_rel: str = ef.get("label_path", "")
            if not label_rel:
                continue

            label_path = os.path.join(dataset_dir, label_rel.replace("\\", "/"))
            detections = read_yolo_label_file(label_path)

            for det in detections:
                class_id = det["class_id"]
                label_name = coco_labels.get(class_id)
                if label_name is None:
                    label_name = f"class_{class_id}"

                rect = yolo_to_ls_rect(det["xc"], det["yc"], det["w"], det["h"])
                video_frame = frame_index + 1
                time_offset = ef.get("approx_time_offset_sec", 0.0)
                region_id = f"bbox_{result_idx}"

                # VideoRectangle result (box coordinates)
                results.append({
                    "id": region_id,
                    "type": "videorectangle",
                    "from_name": "bbox",
                    "to_name": "video",
                    "value": {
                        "sequence": [
                            {
                                "frame": video_frame,
                                "x": round(rect["x"], 4),
                                "y": round(rect["y"], 4),
                                "width": round(rect["width"], 4),
                                "height": round(rect["height"], 4),
                                "time": round(time_offset, 4),
                                "enabled": True,
                                "rotation": 0,
                            }
                        ],
                    },
                })

                # Labels result (linked to the same region)
                results.append({
                    "id": f"label_{result_idx}",
                    "type": "labels",
                    "from_name": "label",
                    "to_name": "video",
                    "value": {
                        "labels": [label_name],
                        "sequence": [
                            {
                                "frame": video_frame,
                                "x": round(rect["x"], 4),
                                "y": round(rect["y"], 4),
                                "width": round(rect["width"], 4),
                                "height": round(rect["height"], 4),
                                "time": round(time_offset, 4),
                                "enabled": True,
                                "rotation": 0,
                            }
                        ],
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
            "to_name": "video",
            "value": {
                "text": [vlm_text],
            },
        })

    return results
