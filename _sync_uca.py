#!/usr/bin/env python
"""Resume-aware S3 upload for large datasets.

Lists remote objects in bulk first (fast), then uploads only missing/changed
files with many threads and a tqdm progress bar.
"""

import argparse
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import boto3
from tqdm import tqdm

MIME = {
    ".mp4": "video/mp4", ".json": "application/json",
    ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
    ".txt": "text/plain", ".xml": "application/xml", ".png": "image/png",
}


def list_remote(s3, bucket, prefix):
    """Batch-list all S3 objects → {key: size}."""
    inv = {}
    paginator = s3.get_paginator("list_objects_v2")
    pages = paginator.paginate(Bucket=bucket, Prefix=prefix + "/")
    for page in tqdm(pages, desc="Listing S3", unit="page", dynamic_ncols=True):
        for obj in page.get("Contents", []):
            inv[obj["Key"]] = obj["Size"]
    return inv


def collect_local(local_dir):
    """Walk local dir → [(abs_path, rel_path)] with progress."""
    files = []
    for root, _, names in os.walk(local_dir):
        for f in names:
            if f.startswith("."):
                continue
            full = os.path.join(root, f)
            rel = os.path.relpath(full, local_dir).replace("\\", "/")
            files.append((full, rel))
    return files


def main():
    ap = argparse.ArgumentParser(description="Resume-aware S3 sync")
    ap.add_argument("local_dir")
    ap.add_argument("s3_url", help="s3://bucket/prefix")
    ap.add_argument("--workers", type=int, default=32)
    args = ap.parse_args()

    local_dir = os.path.abspath(args.local_dir)
    parts = args.s3_url.replace("s3://", "").split("/", 1)
    bucket = parts[0]
    prefix = parts[1].rstrip("/") if len(parts) > 1 else ""

    print(f"Local:   {local_dir}")
    print(f"Remote:  s3://{bucket}/{prefix}/")
    print(f"Workers: {args.workers}\n")

    s3 = boto3.client("s3")

    # Phase 1: list what's already on S3
    print("Phase 1: Listing existing S3 objects...")
    remote = list_remote(s3, bucket, prefix)
    print(f"  Found {len(remote):,} objects on S3\n")

    # Phase 2: scan local files
    print("Phase 2: Scanning local files...")
    t0 = time.time()
    local_files = collect_local(local_dir)
    print(f"  Found {len(local_files):,} local files ({time.time()-t0:.0f}s)\n")

    # Phase 3: diff
    print("Phase 3: Comparing...")
    to_upload = []
    skipped = 0
    for full, rel in local_files:
        key = f"{prefix}/{rel}" if prefix else rel
        local_size = os.path.getsize(full)
        remote_size = remote.get(key)
        if remote_size is not None and remote_size == local_size:
            skipped += 1
        else:
            to_upload.append((full, key))
    print(f"  Skip: {skipped:,}  Upload: {len(to_upload):,}\n")

    if not to_upload:
        print("Everything already on S3. Done.")
        return

    # Phase 4: upload
    print("Phase 4: Uploading...")
    errors = []
    start = time.time()

    def upload(item):
        full, key = item
        ext = os.path.splitext(full)[1].lower()
        ct = MIME.get(ext, "application/octet-stream")
        s3_client = boto3.client("s3")
        s3_client.upload_file(full, bucket, key, ExtraArgs={"ContentType": ct})

    with tqdm(total=len(to_upload), desc="Uploading", unit="file",
              dynamic_ncols=True,
              bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} "
                         "[{elapsed}<{remaining}, {rate_fmt}]") as pbar:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futs = {pool.submit(upload, f): f for f in to_upload}
            for fut in as_completed(futs):
                try:
                    fut.result()
                except Exception as e:
                    _, key = futs[fut]
                    errors.append((key, str(e)))
                    if len(errors) <= 5:
                        tqdm.write(f"  ERROR: {key}: {e}")
                pbar.update(1)

    elapsed = time.time() - start
    print(f"\nDone in {elapsed/60:.1f} min")
    print(f"  Uploaded: {len(to_upload)-len(errors):,}")
    print(f"  Skipped:  {skipped:,}")
    print(f"  Errors:   {len(errors):,}")


if __name__ == "__main__":
    main()
