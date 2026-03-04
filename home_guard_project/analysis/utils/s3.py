"""S3 path resolution from Label Studio export metadata."""

from __future__ import annotations

import os
import re


def meta_path_to_s3_clip(
    meta_path: str, bucket: str, prefix: str,
) -> str:
    """
    Derive the permanent S3 URI for the full-frame clip from meta_path.

    ``meta/front_side/2026-02-21/front_side_1771696897_trigger.meta.json``
    -> ``s3://bucket/prefix/clips/front_side/2026-02-21/front_side_1771696897_trigger.mp4``
    """
    rel = meta_path.replace("\\", "/")
    rel = re.sub(r"^meta/", "clips/", rel)
    rel = re.sub(r"\.meta\.json$", ".mp4", rel)
    return f"s3://{bucket}/{prefix}/{rel}"


def meta_path_to_s3_vlm_crop(
    meta_path: str, bucket: str, prefix: str,
) -> str:
    """
    Derive the permanent S3 URI for the VLM crop from meta_path.

    ``meta/front_side/2026-02-21/front_side_1771696897_trigger.meta.json``
    -> ``s3://bucket/prefix/vlm_crops/front_side/2026-02-21/front_side_1771696897_trigger.mp4``
    """
    rel = meta_path.replace("\\", "/")
    rel = re.sub(r"^meta/", "vlm_crops/", rel)
    rel = re.sub(r"\.meta\.json$", ".mp4", rel)
    return f"s3://{bucket}/{prefix}/{rel}"


def meta_path_to_clip_id(meta_path: str) -> str:
    """Extract the clip ID (filename without .meta.json) from meta_path."""
    basename = os.path.basename(meta_path)
    return re.sub(r"\.meta\.json$", "", basename)


def meta_path_to_camera_date(meta_path: str) -> tuple[str, str]:
    """Extract (camera_name, date_str) from meta_path like ``meta/cam/2026-02-21/...``."""
    parts = meta_path.replace("\\", "/").split("/")
    camera = parts[1] if len(parts) > 1 else "unknown"
    date = parts[2] if len(parts) > 2 else "unknown"
    return camera, date
