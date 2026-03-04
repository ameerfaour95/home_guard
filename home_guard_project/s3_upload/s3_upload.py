"""Core logic: clean up orphans, re-encode clips to H.264, then incrementally upload to S3."""

from __future__ import annotations

import json
import logging
import mimetypes
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, FrozenSet, List, Optional, Set, Tuple

import boto3
from botocore.exceptions import ClientError
from tqdm import tqdm

from home_guard_project.labeling.utils.cleanup import cleanup_orphans
from home_guard_project.labeling.utils.ffmpeg import reencode_videos

log = logging.getLogger("s3_upload")

MIME_OVERRIDES: Dict[str, str] = {
    ".mp4": "video/mp4",
    ".json": "application/json",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".txt": "text/plain",
    ".xml": "application/xml",
    ".png": "image/png",
}


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------

def _collect_mp4s(dataset_dir: str) -> List[str]:
    """Collect all .mp4 files under clips/ and vlm_crops/."""
    mp4s: List[str] = []
    for subdir in ("clips", "vlm_crops"):
        root = os.path.join(dataset_dir, subdir)
        if not os.path.isdir(root):
            continue
        for dirpath, _, filenames in os.walk(root):
            for f in filenames:
                if f.lower().endswith(".mp4"):
                    mp4s.append(os.path.join(dirpath, f))
    return mp4s


def _collect_all_files(dataset_dir: str) -> List[str]:
    """Walk the entire dataset dir and return all file paths (skip dotfiles)."""
    files: List[str] = []
    for dirpath, _, filenames in os.walk(dataset_dir):
        for f in filenames:
            if f.startswith("."):
                continue
            files.append(os.path.join(dirpath, f))
    return files


# ---------------------------------------------------------------------------
# Label-based relevance filtering
# ---------------------------------------------------------------------------

_CLIP_SUBDIRS = frozenset({"clips", "meta", "responses", "yolo", "vlm_crops"})


def _find_irrelevant_clip_stems(
    dataset_dir: str,
    allowed_labels: FrozenSet[str],
) -> Set[str]:
    """Return clip stems whose metadata contains ONLY classes outside *allowed_labels*."""
    irrelevant: Set[str] = set()
    meta_root = os.path.join(dataset_dir, "meta")
    if not os.path.isdir(meta_root):
        return irrelevant

    for dirpath, _, filenames in os.walk(meta_root):
        for fname in filenames:
            if not fname.endswith(".meta.json"):
                continue
            try:
                with open(os.path.join(dirpath, fname), "r", encoding="utf-8") as fh:
                    meta = json.load(fh)
            except (json.JSONDecodeError, OSError):
                continue

            class_counts = meta.get("yolo", {}).get("class_counts", {})
            if class_counts and not (set(class_counts.keys()) & allowed_labels):
                stem = fname[: -len(".meta.json")]
                irrelevant.add(stem)

    return irrelevant


def _clip_stem_from_filename(filename: str) -> str:
    """Extract the clip stem from a dataset filename.

    Handles double-extension (``.meta.json``), YOLO frame files
    (``clip_id_f0001.jpg``), and normal single-extension files.
    """
    if filename.endswith(".meta.json"):
        return filename[: -len(".meta.json")]

    name_no_ext = os.path.splitext(filename)[0]

    # YOLO frame files: *_fNNNN -> strip the suffix
    idx = name_no_ext.rfind("_f")
    if idx > 0 and name_no_ext[idx + 2 :].isdigit():
        return name_no_ext[:idx]

    return name_no_ext


def _filter_relevant_files(
    all_files: List[str],
    dataset_dir: str,
    irrelevant_stems: Set[str],
) -> List[str]:
    """Remove files belonging to irrelevant clips from *all_files*."""
    if not irrelevant_stems:
        return all_files

    kept: List[str] = []
    for fp in all_files:
        rel = os.path.relpath(fp, dataset_dir).replace("\\", "/")
        top_dir = rel.split("/", 1)[0]

        if top_dir not in _CLIP_SUBDIRS:
            kept.append(fp)
            continue

        stem = _clip_stem_from_filename(os.path.basename(fp))
        if stem in irrelevant_stems:
            continue

        kept.append(fp)

    return kept


# ---------------------------------------------------------------------------
# S3 helpers
# ---------------------------------------------------------------------------

def _content_type(filepath: str) -> str:
    ext = os.path.splitext(filepath)[1].lower()
    if ext in MIME_OVERRIDES:
        return MIME_OVERRIDES[ext]
    guess, _ = mimetypes.guess_type(filepath)
    return guess or "application/octet-stream"


def _s3_key(local_path: str, dataset_dir: str, prefix: str) -> str:
    rel = os.path.relpath(local_path, start=dataset_dir)
    return f"{prefix}/{rel}".replace("\\", "/")


def _remote_size(
    s3_client: "boto3.client",
    bucket: str,
    key: str,
) -> Optional[int]:
    """Return the size of an S3 object, or None if it doesn't exist."""
    try:
        resp = s3_client.head_object(Bucket=bucket, Key=key)
        return resp["ContentLength"]
    except ClientError:
        return None


def _upload_one(
    s3_client: "boto3.client",
    bucket: str,
    local_path: str,
    key: str,
    dry_run: bool,
    force: bool = False,
) -> str:
    """Upload a single file. Returns 'uploaded', 'skipped', or 'failed'."""
    if not force:
        local_size = os.path.getsize(local_path)
        remote = _remote_size(s3_client, bucket, key)

        if remote is not None and remote == local_size:
            return "skipped"

    if dry_run:
        return "uploaded"

    try:
        extra = {"ContentType": _content_type(local_path)}
        s3_client.upload_file(local_path, bucket, key, ExtraArgs=extra)
        return "uploaded"
    except Exception as exc:
        log.warning("Failed to upload %s: %s", key, exc)
        return "failed"


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run(
    dataset_dir: str,
    bucket: str,
    prefix: str,
    workers: int = 4,
    skip_reencode: bool = False,
    dry_run: bool = False,
    force: bool = False,
    no_cleanup: bool = False,
    allowed_labels: FrozenSet[str] = frozenset(),
) -> None:
    dataset_dir = os.path.abspath(dataset_dir)

    if not os.path.isdir(dataset_dir):
        log.error("Dataset directory does not exist: %s", dataset_dir)
        sys.exit(1)

    # -- Label-based relevance filter ------------------------------------
    irrelevant_stems: Set[str] = set()
    if allowed_labels:
        irrelevant_stems = _find_irrelevant_clip_stems(dataset_dir, allowed_labels)
        if irrelevant_stems:
            log.info(
                "Skipping %d clips with no relevant detections (allowed: %s)",
                len(irrelevant_stems),
                ", ".join(sorted(allowed_labels)),
            )
        else:
            log.info("All clips have at least one relevant detection.")

    # -- Orphan cleanup ---------------------------------------------------
    if not no_cleanup:
        log.info("Running orphan cleanup (clips/ is source of truth)...")
        stats = cleanup_orphans(dataset_dir)
        if stats.total_removed > 0:
            log.info(
                "Cleanup done: %d files removed (%d cascade clips, "
                "%d meta, %d responses, %d YOLO imgs, %d YOLO lbls, %d VLM crops)",
                stats.total_removed,
                stats.cascade_clips_removed,
                stats.meta_orphans,
                stats.response_orphans,
                stats.yolo_image_orphans,
                stats.yolo_label_orphans,
                stats.vlm_crop_orphans,
            )
        else:
            log.info("Dataset is clean — no orphans found.")
    else:
        log.info("Skipping orphan cleanup (--no-cleanup).")

    # -- Re-encode --------------------------------------------------------
    if not skip_reencode:
        mp4s = _collect_mp4s(dataset_dir)
        if mp4s:
            log.info("Found %d MP4 files to check / re-encode.", len(mp4s))
            stats = reencode_videos(mp4s, skip_reencoded=True)
            log.info(
                "Re-encode results: %d encoded, %d skipped, %d failed",
                stats["encoded"], stats["skipped"], stats["failed"],
            )
        else:
            log.info("No MP4 files found for re-encoding.")
    else:
        log.info("Skipping re-encode step (--skip-reencode).")

    # -- Collect all files ------------------------------------------------
    all_files = _collect_all_files(dataset_dir)
    if not all_files:
        log.warning("No files found in %s. Nothing to upload.", dataset_dir)
        return

    if irrelevant_stems:
        before = len(all_files)
        all_files = _filter_relevant_files(all_files, dataset_dir, irrelevant_stems)
        log.info("Label filter: %d -> %d files (removed %d)", before, len(all_files), before - len(all_files))

    log.info("Found %d files to sync to s3://%s/%s", len(all_files), bucket, prefix)

    if force:
        log.info("FORCE mode — all files will be (re-)uploaded.")
    if dry_run:
        log.info("DRY RUN — no files will be uploaded.")

    # -- Build (local_path, s3_key) pairs ---------------------------------
    pairs: List[Tuple[str, str]] = [
        (fp, _s3_key(fp, dataset_dir, prefix)) for fp in all_files
    ]

    # -- Upload -----------------------------------------------------------
    s3 = boto3.client("s3")

    uploaded = skipped = failed = 0

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_upload_one, s3, bucket, lp, key, dry_run, force): key
            for lp, key in pairs
        }
        pbar = tqdm(
            as_completed(futures),
            total=len(futures),
            desc="Dry-run check" if dry_run else "Uploading",
            unit="file",
        )
        for fut in pbar:
            result = fut.result()
            if result == "uploaded":
                uploaded += 1
            elif result == "skipped":
                skipped += 1
            else:
                failed += 1
            pbar.set_postfix(uploaded=uploaded, skipped=skipped, failed=failed)

    verb = "Would upload" if dry_run else "Uploaded"
    log.info(
        "Done. %s: %d  |  Skipped (same size): %d  |  Failed: %d",
        verb, uploaded, skipped, failed,
    )
