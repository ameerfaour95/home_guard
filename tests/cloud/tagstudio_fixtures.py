"""On-disk fixtures for the tagging studio: a unified dataset, an owner_feedback/ folder, eval results."""
from __future__ import annotations

import json
import os
from typing import Any, Dict, Iterable, List, Optional

VIDEO_BYTES = bytes(range(256)) * 40          # 10,240 bytes standing in for an mp4


def write(path: str, data: Any) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if isinstance(data, (bytes, bytearray)):
        with open(path, "wb") as f:
            f.write(data)
    else:
        with open(path, "w", encoding="utf-8") as f:
            f.write(data if isinstance(data, str) else json.dumps(data, ensure_ascii=False))
    return path


def write_jsonl(path: str, rows: Iterable[Dict[str, Any]]) -> str:
    return write(path, "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))


def dataset_row(clip_id: str, description: str = "A man walks past.", alert: Optional[bool] = False,
                source: str = "house", batch: str = "ameer_house_batch_1", camera: str = "front_side",
                crop: bool = True, **extra: Any) -> Dict[str, Any]:
    row = {"clip_id": clip_id, "source": source, "batch": batch, "camera": camera, "date": "2026-02-21",
           "duration_sec": 9.8, "clip": f"clips/{source}/{clip_id}.mp4",
           "crop": f"crops/{source}/{clip_id}.mp4" if crop else None, "meta": f"meta/{source}/{clip_id}.meta.json",
           "description": description, "alert": alert, "description_by": "human (Label Studio)",
           "num_persons": 1, "num_cars": 0, "needs_check": None}
    row.update(extra)
    return row


def make_dataset(root: str, rows: List[Dict[str, Any]], videos: bool = True) -> str:
    write_jsonl(os.path.join(root, "annotations", "clips.jsonl"), rows)
    write_jsonl(os.path.join(root, "vlm", "clips.jsonl"), [
        {"clip_id": r["clip_id"], "video": r["clip"], "crop": r["crop"],
         "video_s3_path": f"s3://bucket/home_guard_dataset/{r['clip']}",
         "vlm_crop_s3_path": f"s3://bucket/home_guard_dataset/{r['crop']}" if r["crop"] else "",
         "description": r["description"], "alert": r["alert"], "camera": r["camera"]} for r in rows])
    for r in rows:
        write(os.path.join(root, *r["meta"].split("/")), {"camera_name": r["camera"], "fps_estimated": 7.0})
        if videos:
            write(os.path.join(root, *r["clip"].split("/")), VIDEO_BYTES)
            if r["crop"]:
                write(os.path.join(root, *r["crop"].split("/")), VIDEO_BYTES[:5000])
    return root


def alert_meta(camera: str, stem: str, label: str = "normal", kind: str = "alert", ts: float = 1791091199.0,
               summary: str = "Two children play near a car.", **extra: Any) -> Dict[str, Any]:
    day = "2026-10-04"
    meta: Dict[str, Any] = {
        "camera_name": camera, "kind": kind, "clip_path": f"clips\\{camera}\\{day}\\{stem}.mp4",
        "clip_start_ts": ts - 4, "clip_end_ts": ts + 6, "clip_start_local": f"{day} 08:19:55", "duration_sec": 9.7,
        "fps_estimated": 7.0, "alert": {"raw_label": label, "label": label, "summary": summary, "people": 2},
        "model_response": {"summary": summary, "label": label, "raw_label": label},
        "teacher": {"model": "gpt-4o", "prompt_version": "2026-10-03.test"},
    }
    meta.update(extra)
    return meta


def feedback_record(alert_id: str, camera: str, time_utc: str, verdict: str, owner_label: str = "",
                    note: str = "", raw_text: str = "", owner_text: str = "") -> Dict[str, Any]:
    """A feedback file as box/feedback.py save_feedback writes it."""
    return {"time_utc": time_utc, "alert": {"alert_id": alert_id, "camera": camera, "label": "normal"},
            "verdict": verdict, "action": "none", "camera": None, "note": note, "source": "button",
            "raw_text": raw_text, "from": {"user_id": 1, "name": "Owner"}, "chat_id": "-1",
            "owner_label": owner_label, "owner_text": owner_text, "tagged_by": "Owner", "request_id": ""}


def put_owner_clip(dataset: str, prefix: str, meta: Dict[str, Any], stem: str, answers=(), video: bool = True) -> None:
    """A clip and its answers in <dataset>/owner_feedback/, in the box's layout."""
    root = os.path.join(dataset, "owner_feedback")
    camera, day = meta["camera_name"], meta["clip_start_local"][:10]
    write(os.path.join(root, prefix, "meta", camera, day, f"{stem}.meta.json"), meta)
    if video:
        write(os.path.join(root, prefix, "clips", camera, day, f"{stem}.mp4"), VIDEO_BYTES)
    for i, answer in enumerate(answers):
        write(os.path.join(root, "feedback", prefix, "feedback", camera, day, f"{stem}_{i}.feedback.json"), answer)
    write(os.path.join(root, "feedback_index.jsonl"), "")


def eval_results(results_dir: str, tag: str, model: str, answers: Dict[str, str], caught: float, flagged: float,
                 errors: int = 0, rows: Optional[int] = None) -> None:
    write_jsonl(os.path.join(results_dir, f"{tag}.jsonl"), [
        {"clip_id": cid, "ai_label": label, "ai_summary": f"{model} says {label}",
         "raw": json.dumps({"summary": "x", "label": label, "raw_label": label}), "error": "", "model": model}
        for cid, label in answers.items()])
    write(os.path.join(results_dir, f"{tag}.summary.json"),
          {"model": model, "rows": rows if rows is not None else len(answers), "errors": errors,
           "alerts_caught_ratio": caught, "normal_flagged_ratio": flagged})
