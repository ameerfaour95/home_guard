"""Scan the dataset, build Label Studio tasks, and export them."""

from __future__ import annotations

import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm

from .config import S3StorageConfig, get_all_label_names, get_coco_labels
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
  <View style="display:flex; gap:16px;">
    <View style="flex:1;">
      <Header value="Full Frame — YOLO Re-tagging"/>
      <Labels name="label" toName="video_full">
{label_tags}
      </Labels>
      <Video name="video_full" value="$video_url" frameRate="$fps"/>
      <VideoRectangle name="bbox" toName="video_full"/>
    </View>
    <View style="flex:1;">
      <Header value="$vlm_crop_header"/>
      <Video name="video_crop" value="$vlm_crop_url" frameRate="$vlm_fps"/>
      <Header value="Scene Description (for VLM training)"/>
      <TextArea name="vlm_description" toName="video_crop"
                rows="4" editable="true"
                placeholder="Describe what happens in the video..."/>
    </View>
  </View>
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


def collect_vlm_crop_paths(
    dataset_dir: str,
    *,
    cameras: Optional[List[str]] = None,
    kinds: Optional[List[str]] = None,
    limit: Optional[int] = None,
) -> List[str]:
    """
    Collect VLM crop video absolute paths for the subset implied by
    *cameras*/*kinds*/*limit*.  Only returns paths that exist on disk.

    Falls back to inferring the VLM crop path from ``clip_path``
    (``clips/`` -> ``vlm_crops/``) when ``vlm_crop_path`` is absent.
    """
    paths: List[str] = []
    for _mp, meta in iter_meta_entries(dataset_dir, cameras=cameras, kinds=kinds, limit=limit):
        vlm_rel = meta.get("vlm_crop_path", "").replace("\\", "/")
        if not vlm_rel:
            clip_rel = meta.get("clip_path", "").replace("\\", "/")
            if clip_rel.startswith("clips/"):
                vlm_rel = "vlm_crops/" + clip_rel[len("clips/"):]
        if not vlm_rel:
            continue
        vlm_abs = os.path.join(dataset_dir, vlm_rel)
        if os.path.isfile(vlm_abs):
            paths.append(vlm_abs)
    return paths


# ---------------------------------------------------------------------------
# Single-task builder (thread-safe)
# ---------------------------------------------------------------------------

def _build_single_task(
    meta_path: str,
    meta: Dict[str, Any],
    dataset_dir: str,
    include_predictions: bool,
    video_base_url: Optional[str],
    s3_config: Optional[S3StorageConfig] = None,
    s3_vlm_keys: Optional[set] = None,
) -> Optional[Dict[str, Any]]:
    """Build one Label Studio task dict from a single meta entry."""
    camera_name: str = meta.get("camera_name", "unknown")
    kind: str = meta.get("kind", "unknown")

    clip_rel = meta.get("clip_path", "").replace("\\", "/")
    if not clip_rel:
        return None

    allowed_names = set(get_coco_labels().values())
    meta_classes = set(meta.get("yolo", {}).get("class_counts", {}).keys())
    if meta_classes and not (meta_classes & allowed_names):
        return None

    if s3_config is None:
        clip_abs = os.path.join(dataset_dir, clip_rel)
        if not os.path.isfile(clip_abs):
            return None

    if s3_config is not None:
        from .utils.s3 import presigned_url
        clip_key = f"{s3_config.prefix}/{clip_rel}"
        video_url = presigned_url(
            s3_config.bucket, clip_key,
            region=s3_config.region, expiry=s3_config.url_expiry_sec,
        )
    elif video_base_url:
        video_url = f"{video_base_url.rstrip('/')}/{clip_rel}"
    else:
        video_url = f"/data/local-files/?d={clip_rel}"

    fps_raw = meta.get("fps_estimated", meta.get("buffer", {}).get("store_fps", 10.0))
    fps = min(fps_raw, 10.0)
    start_local = meta.get("clip_start_local", "?")
    end_local = meta.get("clip_end_local", "?")
    end_display = end_local.split(" ")[-1] if " " in end_local else end_local
    header = f"{camera_name} | {kind} | {start_local} - {end_display}"

    vlm_crop_rel = meta.get("vlm_crop_path", "").replace("\\", "/")
    if not vlm_crop_rel and clip_rel.startswith("clips/"):
        vlm_crop_rel = "vlm_crops/" + clip_rel[len("clips/"):]

    if s3_config is not None:
        vlm_key = f"{s3_config.prefix}/{vlm_crop_rel}" if vlm_crop_rel else ""
        has_vlm_crop = bool(vlm_key) and (
            s3_vlm_keys is None or vlm_key in s3_vlm_keys
        )
    else:
        has_vlm_crop = bool(vlm_crop_rel) and os.path.isfile(
            os.path.join(dataset_dir, vlm_crop_rel)
        )

    if has_vlm_crop:
        if s3_config is not None:
            from .utils.s3 import presigned_url
            vlm_key = f"{s3_config.prefix}/{vlm_crop_rel}"
            vlm_crop_url = presigned_url(
                s3_config.bucket, vlm_key,
                region=s3_config.region, expiry=s3_config.url_expiry_sec,
            )
        elif video_base_url:
            vlm_abs = os.path.join(dataset_dir, vlm_crop_rel)
            mtime = int(os.path.getmtime(vlm_abs))
            vlm_crop_url = f"{video_base_url.rstrip('/')}/{vlm_crop_rel}?v={mtime}"
        else:
            vlm_crop_url = video_url
    else:
        vlm_crop_url = video_url

    vlm_fps = meta.get("main_stream", {}).get("store_fps",
              meta.get("vlm_crop", {}).get("fps", fps))

    vlm_crop_header = (
        "VLM Crop (high-res)"
        if has_vlm_crop
        else "No VLM Crop \u2014 showing full frame"
    )

    task: Dict[str, Any] = {
        "data": {
            "video_url": video_url,
            "vlm_crop_url": vlm_crop_url,
            "fps": fps,
            "vlm_fps": vlm_fps,
            "header_info": header,
            "vlm_crop_header": vlm_crop_header,
            "has_vlm_crop": has_vlm_crop,
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
    s3_config: Optional[S3StorageConfig] = None,
    workers: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Walk ``dataset_dir/meta/`` and build Label Studio tasks in parallel.

    When *s3_config* is provided, video URLs are S3 pre-signed URLs
    instead of local file-server URLs.

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

    s3_vlm_keys: Optional[set] = None
    if s3_config is not None:
        from .utils.s3 import list_s3_keys
        vlm_prefix = f"{s3_config.prefix}/vlm_crops/"
        s3_vlm_keys = set(list_s3_keys(
            s3_config.bucket, vlm_prefix, region=s3_config.region,
        ))
        log.info("Found %d VLM crop files on S3", len(s3_vlm_keys))

    log.info("Building %d tasks (%d workers)", total, max_workers)

    tasks: List[Dict[str, Any]] = []

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(
                _build_single_task,
                mp, m, dataset_dir, include_predictions, video_base_url,
                s3_config, s3_vlm_keys,
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

    skipped = total - len(tasks)
    if skipped:
        log.info(
            "Skipped %d clips with no relevant detections (allowed: %s)",
            skipped,
            ", ".join(sorted(set(get_coco_labels().values()))),
        )

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
