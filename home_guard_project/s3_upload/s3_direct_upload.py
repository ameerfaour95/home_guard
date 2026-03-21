#!/usr/bin/env python
"""Direct boto3 upload for large directories where `aws s3 sync` chokes.

Skips the expensive comparison step — just uploads everything.
For small text files, re-uploading duplicates is cheaper than comparing 400K files.
"""

import argparse
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import boto3
from tqdm import tqdm


def collect_files(local_dir: str) -> list[tuple[str, str]]:
    """Return list of (absolute_path, relative_path) for all files."""
    result = []
    for root, _, files in os.walk(local_dir):
        for fname in files:
            full = os.path.join(root, fname)
            rel = os.path.relpath(full, local_dir).replace("\\", "/")
            result.append((full, rel))
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description="Direct S3 upload with progress")
    ap.add_argument("local_dir", help="Local directory to upload")
    ap.add_argument("s3_url", help="S3 destination (s3://bucket/prefix)")
    ap.add_argument("--workers", type=int, default=32,
                    help="Concurrent upload threads (default 32)")
    args = ap.parse_args()

    local_dir = os.path.abspath(args.local_dir)
    parts = args.s3_url.replace("s3://", "").split("/", 1)
    bucket = parts[0]
    prefix = parts[1].rstrip("/") if len(parts) > 1 else ""

    print(f"Local:   {local_dir}")
    print(f"Bucket:  {bucket}")
    print(f"Prefix:  {prefix}")
    print(f"Workers: {args.workers}")
    print()

    print("Scanning files...")
    files = collect_files(local_dir)
    print(f"Found {len(files):,} files to upload\n")

    if not files:
        print("Nothing to upload.")
        return

    session = boto3.Session()
    errors: list[tuple[str, str]] = []
    start = time.time()

    def upload_one(item: tuple[str, str]) -> None:
        full, rel = item
        key = f"{prefix}/{rel}" if prefix else rel
        client = session.client("s3")
        client.upload_file(full, bucket, key)

    with tqdm(total=len(files), desc="Uploading", unit="file",
              bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} "
                         "[{elapsed}<{remaining}, {rate_fmt}]") as pbar:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(upload_one, f): f for f in files}
            for fut in as_completed(futures):
                try:
                    fut.result()
                except Exception as e:
                    _, rel = futures[fut]
                    errors.append((rel, str(e)))
                    if len(errors) <= 5:
                        tqdm.write(f"  ERROR: {rel}: {e}")
                pbar.update(1)

    elapsed = time.time() - start
    print(f"\nDone in {elapsed / 60:.1f} minutes")
    print(f"  Uploaded: {len(files) - len(errors):,}")
    print(f"  Errors:   {len(errors):,}")
    if errors:
        for rel, err in errors[:10]:
            print(f"    {rel}: {err}")


if __name__ == "__main__":
    main()
