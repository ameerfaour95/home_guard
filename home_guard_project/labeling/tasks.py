"""Scan the dataset, build Label Studio tasks, and export them."""

from __future__ import annotations

import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm

from .config import get_all_label_names
from .utils.yolo import build_predictions

log = logging.getLogger("labeling.tasks")


# ---------------------------------------------------------------------------
# Freshness check — skip rebuilding when nothing changed
# ---------------------------------------------------------------------------

def tasks_are_fresh(dataset_dir: str) -> bool:
    """
    Return *True* if ``label_studio_tasks.json`` already exists and is
    newer than every ``.meta.json`` file in the dataset.

    This lets the pipeline skip the expensive task-generation step on
    re-runs when no new data has been collected.
    """
    tasks_path = os.path.join(dataset_dir, "label_studio_tasks.json")
    if not os.path.isfile(tasks_path):
        return False

    tasks_mtime = os.path.getmtime(tasks_path)

    meta_root = os.path.join(dataset_dir, "meta")
    if not os.path.isdir(meta_root):
        return False

    newest_meta = 0.0
    for dirpath, _dirs, filenames in os.walk(meta_root):
        for fname in filenames:
            if fname.endswith(".meta.json"):
                mtime = os.path.getmtime(os.path.join(dirpath, fname))
                if mtime > newest_meta:
                    newest_meta = mtime

    if newest_meta == 0.0:
        return False

    fresh = tasks_mtime >= newest_meta
    if fresh:
        log.info(
            "Tasks file is up-to-date (no new meta files since last build). Skipping."
        )
    return fresh


# ---------------------------------------------------------------------------
# Label Studio XML config template
# ---------------------------------------------------------------------------

_LABEL_CONFIG_XML = """\
<View>
  <Header value="$header_info"/>

  <Labels name="label" toName="video">
{label_tags}
  </Labels>

  <Video name="video" value="$video_url" frameRate="$fps"/>
  <VideoRectangle name="bbox" toName="video"/>

  <Header value="Scene Description (for VLM training)"/>
  <TextArea name="vlm_description" toName="video"
            rows="4" editable="true"
            placeholder="Describe what happens in the video..."/>
</View>
"""


def build_label_config() -> str:
    """Return the complete Label Studio XML labeling config."""
    tags = "\n".join(
        f'    <Label value="{name}"/>' for name in get_all_label_names()
    )
    return _LABEL_CONFIG_XML.format(label_tags=tags)


def write_config(output_path: str) -> None:
    """Write the Label Studio XML labeling config to disk."""
    config = build_label_config()
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(config)
    log.info("Labeling config written to %s", output_path)


# ---------------------------------------------------------------------------
# Meta-file iteration
# ---------------------------------------------------------------------------

def iter_meta_entries(
    dataset_dir: str,
    *,
    cameras: Optional[List[str]] = None,
    kinds: Optional[List[str]] = None,
    limit: Optional[int] = None,
):
    """
    Yield ``(meta_path, meta_dict)`` for meta files under
    ``dataset_dir/meta``, filtered by *cameras*/*kinds* and truncated by
    *limit*.
    """
    meta_root = os.path.join(dataset_dir, "meta")
    if not os.path.isdir(meta_root):
        return

    seen = 0
    for dirpath, _dirnames, filenames in os.walk(meta_root):
        for fname in sorted(filenames):
            if not fname.endswith(".meta.json"):
                continue

            meta_path = os.path.join(dirpath, fname)
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta: Dict[str, Any] = json.load(f)
            except (json.JSONDecodeError, OSError) as exc:
                log.warning("Skipping %s: %s", meta_path, exc)
                continue

            camera_name: str = meta.get("camera_name", "unknown")
            kind: str = meta.get("kind", "unknown")

            if cameras and camera_name not in cameras:
                continue
            if kinds and kind not in kinds:
                continue

            yield meta_path, meta

            seen += 1
            if limit and seen >= limit:
                return


def collect_clip_paths(
    dataset_dir: str,
    *,
    cameras: Optional[List[str]] = None,
    kinds: Optional[List[str]] = None,
    limit: Optional[int] = None,
) -> List[str]:
    """
    Collect clip absolute paths for the subset implied by
    *cameras*/*kinds*/*limit*.
    """
    clips: List[str] = []
    for _mp, meta in iter_meta_entries(dataset_dir, cameras=cameras, kinds=kinds, limit=limit):
        clip_rel = meta.get("clip_path", "").replace("\\", "/")
        if not clip_rel:
            continue
        clip_abs = os.path.join(dataset_dir, clip_rel)
        if os.path.isfile(clip_abs):
            clips.append(clip_abs)
    return clips


# ---------------------------------------------------------------------------
# Single-task builder (thread-safe)
# ---------------------------------------------------------------------------

def _build_single_task(
    meta_path: str,
    meta: Dict[str, Any],
    dataset_dir: str,
    include_predictions: bool,
    video_base_url: Optional[str],
) -> Optional[Dict[str, Any]]:
    """Build one Label Studio task dict from a single meta entry."""
    camera_name: str = meta.get("camera_name", "unknown")
    kind: str = meta.get("kind", "unknown")

    clip_rel = meta.get("clip_path", "").replace("\\", "/")
    if not clip_rel:
        return None

    clip_abs = os.path.join(dataset_dir, clip_rel)
    if not os.path.isfile(clip_abs):
        return None

    if video_base_url:
        video_url = f"{video_base_url.rstrip('/')}/{clip_rel}"
    else:
        video_url = f"/data/local-files/?d={clip_rel}"

    fps = meta.get("fps_estimated", meta.get("buffer", {}).get("store_fps", 10.0))
    start_local = meta.get("clip_start_local", "?")
    end_local = meta.get("clip_end_local", "?")
    end_display = end_local.split(" ")[-1] if " " in end_local else end_local
    header = f"{camera_name} | {kind} | {start_local} - {end_display}"

    task: Dict[str, Any] = {
        "data": {
            "video_url": video_url,
            "fps": fps,
            "header_info": header,
            "camera_name": camera_name,
            "kind": kind,
            "clip_start_local": start_local,
            "clip_end_local": end_local,
            "duration_sec": meta.get("duration_sec", 0),
            "meta_path": os.path.relpath(meta_path, start=dataset_dir).replace("\\", "/"),
        },
    }

    if include_predictions:
        preds = build_predictions(meta, dataset_dir)
        if preds:
            task["predictions"] = [
                {
                    "model_version": "yolov8n-weak-labels",
                    "result": preds,
                }
            ]

    return task


# ---------------------------------------------------------------------------
# Parallel task scanning
# ---------------------------------------------------------------------------

def scan_and_build_tasks(
    dataset_dir: str,
    *,
    cameras: Optional[List[str]] = None,
    kinds: Optional[List[str]] = None,
    limit: Optional[int] = None,
    include_predictions: bool = True,
    video_base_url: Optional[str] = None,
    workers: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Walk ``dataset_dir/meta/`` and build Label Studio tasks in parallel.

    Returns a list of task dicts ready for JSON export.
    """
    meta_root = os.path.join(dataset_dir, "meta")
    if not os.path.isdir(meta_root):
        log.error("Meta directory not found: %s", meta_root)
        return []

    log.info("Collecting meta entries...")
    entries: List[Tuple[str, Dict[str, Any]]] = list(
        iter_meta_entries(dataset_dir, cameras=cameras, kinds=kinds, limit=limit)
    )
    total = len(entries)
    if total == 0:
        log.warning("No meta entries found.")
        return []

    max_workers = workers or min(os.cpu_count() or 4, 8)
    log.info("Building %d tasks (%d workers)", total, max_workers)

    tasks: List[Dict[str, Any]] = []

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(
                _build_single_task,
                mp, m, dataset_dir, include_predictions, video_base_url,
            ): mp
            for mp, m in entries
        }

        for future in tqdm(
            as_completed(futures), total=total,
            desc="Building tasks", unit="task",
        ):
            try:
                task = future.result()
            except Exception as exc:
                log.warning("Task build failed: %s", exc)
                continue
            if task is not None:
                tasks.append(task)

    return tasks


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def export_tasks(tasks: List[Dict[str, Any]], output_path: str) -> None:
    """Write the tasks list as a JSON file."""
    log.info("Writing %d tasks to %s ...", len(tasks), output_path)
    t0 = time.monotonic()
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(tasks, f, ensure_ascii=False, indent=2)
    elapsed = time.monotonic() - t0
    log.info("Tasks written (%.1fs)", elapsed)
