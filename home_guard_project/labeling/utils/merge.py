"""Merge annotations from a Label Studio export into freshly generated tasks."""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict, List, Optional

log = logging.getLogger("labeling.merge")


def _task_key(task: Dict[str, Any]) -> str:
    """Return a stable merge key for a task.

    Prefers ``data.meta_path`` (storage-mode agnostic) and falls back to
    ``data.video_url`` for backward compatibility with older exports that
    lack the ``meta_path`` field.
    """
    data = task.get("data", {})
    return data.get("meta_path", "") or data.get("video_url", "")


def merge_annotations(
    new_tasks_path: str,
    exported_path: str,
    output_path: Optional[str] = None,
) -> Dict[str, int]:
    """
    Merge annotations from a Label Studio JSON export into regenerated tasks.

    Matches tasks by ``data.meta_path`` (stable across storage modes) with
    a fallback to ``data.video_url`` for older exports.  Exported annotations
    are attached to the matching new task so that a clear-and-reimport cycle
    preserves all human work.

    Args:
        new_tasks_path: Path to the regenerated ``label_studio_tasks.json``.
        exported_path:  Path to the Label Studio JSON export.
        output_path:    Where to write the merged JSON.  Defaults to
                        overwriting *new_tasks_path*.

    Returns:
        A stats dict ``{total, with_annotations, new}``.
    """
    if output_path is None:
        output_path = new_tasks_path

    log.info("Merging annotations from %s into %s", exported_path, new_tasks_path)
    t0 = time.monotonic()

    with open(new_tasks_path, "r", encoding="utf-8") as f:
        new_tasks: List[Dict[str, Any]] = json.load(f)

    with open(exported_path, "r", encoding="utf-8") as f:
        exported: List[Dict[str, Any]] = json.load(f)

    annotation_lookup: Dict[str, List[Dict[str, Any]]] = {}
    for task in exported:
        key = _task_key(task)
        annotations = task.get("annotations", [])
        if key and annotations:
            annotation_lookup[key] = annotations

    log.info(
        "Export contains %d tasks (%d with annotations)",
        len(exported), len(annotation_lookup),
    )

    with_annotations = 0
    for task in new_tasks:
        key = _task_key(task)
        if key in annotation_lookup:
            task["annotations"] = annotation_lookup[key]
            with_annotations += 1

    new_count = len(new_tasks) - with_annotations
    stats = {
        "total": len(new_tasks),
        "with_annotations": with_annotations,
        "new": new_count,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(new_tasks, f, ensure_ascii=False, indent=2)

    elapsed = time.monotonic() - t0
    log.info(
        "Merge complete (%.1fs): %d total, %d with annotations restored, %d new",
        elapsed, stats["total"], stats["with_annotations"], stats["new"],
    )
    return stats
