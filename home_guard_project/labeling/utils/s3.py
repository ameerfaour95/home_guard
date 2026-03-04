"""S3 helpers: sync metadata from S3 and generate pre-signed URLs."""

from __future__ import annotations

import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple

import boto3
from botocore.exceptions import ClientError
from tqdm import tqdm

log = logging.getLogger("labeling.s3")


# ---------------------------------------------------------------------------
# Pre-signed URL generation
# ---------------------------------------------------------------------------

_s3_clients: dict = {}


def _get_client(region: str) -> "boto3.client":
    if region not in _s3_clients:
        from botocore.config import Config as BotoConfig
        _s3_clients[region] = boto3.client(
            "s3",
            region_name=region,
            config=BotoConfig(signature_version="s3v4"),
        )
    return _s3_clients[region]


def presigned_url(
    bucket: str,
    key: str,
    region: str = "us-east-1",
    expiry: int = 604800,
) -> str:
    """Generate a pre-signed GET URL for an S3 object."""
    client = _get_client(region)
    return client.generate_presigned_url(
        "get_object",
        Params={"Bucket": bucket, "Key": key},
        ExpiresIn=expiry,
    )


# ---------------------------------------------------------------------------
# Key listing
# ---------------------------------------------------------------------------

def list_s3_keys(
    bucket: str,
    prefix: str,
    region: str = "us-east-1",
) -> List[str]:
    """Return all object keys under *prefix* (skipping directory markers)."""
    client = _get_client(region)
    keys: List[str] = []
    for key, _size in _list_objects(client, bucket, prefix):
        keys.append(key)
    return keys


# ---------------------------------------------------------------------------
# Metadata sync: download meta/ and yolo/labels/ from S3
# ---------------------------------------------------------------------------

def _list_objects(
    client: "boto3.client", bucket: str, prefix: str,
) -> List[Tuple[str, int]]:
    """List all ``(key, size)`` pairs under *prefix*, skipping directory markers."""
    items: List[Tuple[str, int]] = []
    paginator = client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.endswith("/"):
                continue
            items.append((key, obj["Size"]))
    return items


def _download_one(
    client: "boto3.client",
    bucket: str,
    key: str,
    local_path: str,
) -> str:
    """Download a single S3 object. Returns 'ok' or 'failed'."""
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    try:
        client.download_file(bucket, key, local_path)
        return "ok"
    except ClientError as exc:
        log.warning("Failed to download s3://%s/%s: %s", bucket, key, exc)
        return "failed"


def sync_metadata_from_s3(
    bucket: str,
    prefix: str,
    local_dir: str,
    region: str = "us-east-1",
    workers: int = 8,
) -> Tuple[int, int]:
    """
    Download ``meta/`` and ``yolo/labels/`` from S3 to *local_dir*.

    Only downloads files that are missing locally or differ in size
    (sizes come from the S3 listing — no extra HEAD calls).
    Returns ``(downloaded, skipped)``.
    """
    client = _get_client(region)

    subdirs = [
        f"{prefix}/meta/",
        f"{prefix}/yolo/labels/",
    ]

    all_objects: List[Tuple[str, int]] = []
    for sub in subdirs:
        all_objects.extend(_list_objects(client, bucket, sub))

    if not all_objects:
        log.warning("No metadata found on S3 at s3://%s/%s", bucket, prefix)
        return (0, 0)

    to_download: List[Tuple[str, str]] = []
    skipped = 0

    for key, remote_size in all_objects:
        rel = key[len(prefix):].lstrip("/")
        local_path = os.path.join(local_dir, rel)

        if os.path.isfile(local_path):
            try:
                if os.path.getsize(local_path) == remote_size:
                    skipped += 1
                    continue
            except OSError:
                pass
        to_download.append((key, local_path))

    if not to_download:
        log.info("All %d metadata files are up-to-date.", skipped)
        return (0, skipped)

    log.info(
        "Syncing %d metadata files from S3 (%d already up-to-date).",
        len(to_download), skipped,
    )

    downloaded = 0
    failed = 0

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_download_one, client, bucket, key, lp): key
            for key, lp in to_download
        }
        pbar = tqdm(
            as_completed(futures),
            total=len(futures),
            desc="Syncing metadata",
            unit="file",
        )
        for fut in pbar:
            result = fut.result()
            if result == "ok":
                downloaded += 1
            else:
                failed += 1
            pbar.set_postfix(downloaded=downloaded, failed=failed)

    if failed:
        log.warning("%d metadata files failed to download.", failed)

    return (downloaded, skipped)
