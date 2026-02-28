"""Core logic: re-encode clips to H.264, then incrementally upload to S3."""

from __future__ import annotations

import logging
import mimetypes
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple

import boto3
from botocore.exceptions import ClientError
from tqdm import tqdm

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
) -> None:
    dataset_dir = os.path.abspath(dataset_dir)

    if not os.path.isdir(dataset_dir):
        log.error("Dataset directory does not exist: %s", dataset_dir)
        sys.exit(1)

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
