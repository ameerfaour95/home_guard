#!/usr/bin/env python
"""Upload dataset directories to S3 with parallel per-category syncs + progress.

Uses `aws s3 sync` per category directory for efficient delta uploads,
runs multiple categories in parallel, and shows progress via tqdm.
"""

import argparse
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm


def discover_sync_tasks(local_dir: str, subdirs: list[str]) -> list[tuple[str, str, str, int]]:
    """Build (local_path, s3_suffix, label, file_count) for each category."""
    tasks: list[tuple[str, str, str, int]] = []

    for top_dir in subdirs:
        local_top = os.path.join(local_dir, top_dir.replace("/", os.sep))
        if not os.path.isdir(local_top):
            continue

        for cat in sorted(os.listdir(local_top)):
            cat_path = os.path.join(local_top, cat)
            if not os.path.isdir(cat_path):
                continue
            n_files = sum(len(files) for _, _, files in os.walk(cat_path))
            if n_files == 0:
                continue
            suffix = f"{top_dir}/{cat}"
            tasks.append((cat_path, suffix, suffix, n_files))

    return tasks


def run_sync(local_path: str, s3_url: str) -> tuple[int, int]:
    """Run `aws s3 sync` and count 'upload:' lines."""
    cmd = [sys.executable, "-m", "awscli", "s3", "sync",
           local_path, s3_url, "--no-progress"]
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, bufsize=1,
    )
    uploaded = 0
    for line in proc.stdout:
        if line.strip().startswith("upload:"):
            uploaded += 1
    proc.wait()
    if proc.returncode != 0:
        err = proc.stderr.read().strip()
        if err:
            tqdm.write(f"    STDERR: {err[:300]}")
    else:
        proc.stderr.read()
    return proc.returncode, uploaded


_lock = threading.Lock()


def main() -> None:
    ap = argparse.ArgumentParser(description="S3 sync with progress")
    ap.add_argument("local_dir", help="Local dataset directory")
    ap.add_argument("s3_url", help="S3 destination (e.g. s3://bucket/prefix)")
    ap.add_argument("--subdirs", nargs="*",
                    default=["clips", "meta", "responses",
                             "yolo/images", "yolo/labels"],
                    help="Subdirectories to sync")
    ap.add_argument("--parallel", type=int, default=4,
                    help="Number of parallel sync processes (default 4)")
    args = ap.parse_args()

    local_dir = os.path.abspath(args.local_dir)
    s3_base = args.s3_url.rstrip("/")

    print(f"Local:    {local_dir}")
    print(f"Remote:   {s3_base}")
    print(f"Subdirs:  {args.subdirs}")
    print(f"Parallel: {args.parallel}")
    print()

    print("Scanning local files...")
    tasks = discover_sync_tasks(local_dir, args.subdirs)
    total_files = sum(t[3] for t in tasks)
    print(f"Found {len(tasks)} category-directories with {total_files:,} total files\n")

    grand_uploaded = 0
    grand_skipped = 0
    errors = 0
    start_all = time.time()

    file_bar = tqdm(total=total_files, desc="Files", unit="file", position=0,
                    bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} "
                               "[{elapsed}<{remaining}, {rate_fmt}]")
    dir_bar = tqdm(total=len(tasks), desc="Dirs ", unit="dir", position=1,
                   bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}")

    def sync_task(item: tuple[str, str, str, int]) -> tuple[str, int, int, int, float]:
        local_path, suffix, label, n_files = item
        s3_url = f"{s3_base}/{suffix}/"
        t0 = time.time()
        rc, uploaded = run_sync(local_path, s3_url)
        elapsed = time.time() - t0
        return label, n_files, uploaded, rc, elapsed

    with ThreadPoolExecutor(max_workers=args.parallel) as pool:
        futures = {pool.submit(sync_task, t): t for t in tasks}

        for fut in as_completed(futures):
            label, n_files, uploaded, rc, elapsed = fut.result()
            skipped = n_files - uploaded

            with _lock:
                grand_uploaded += uploaded
                grand_skipped += skipped
                if rc != 0:
                    errors += 1

                if uploaded > 0:
                    tqdm.write(
                        f"  {label}: {uploaded:,} uploaded, "
                        f"{skipped:,} skipped ({elapsed:.0f}s)")

                file_bar.update(n_files)
                dir_bar.update(1)

    file_bar.close()
    dir_bar.close()

    # Upload manifest separately
    manifest = os.path.join(local_dir, "_processed_manifest.json")
    if os.path.isfile(manifest):
        print("Uploading manifest...")
        cmd = [sys.executable, "-m", "awscli", "s3", "cp",
               manifest, f"{s3_base}/_processed_manifest.json"]
        subprocess.run(cmd, capture_output=True, text=True)

    total_time = time.time() - start_all
    print(f"\n{'=' * 60}")
    print(f"Complete in {total_time / 60:.1f} minutes")
    print(f"  Uploaded:  {grand_uploaded:,} files")
    print(f"  Skipped:   {grand_skipped:,} files (already on S3)")
    print(f"  Errors:    {errors}")
    print(f"  Total:     {total_files:,} files")


if __name__ == "__main__":
    main()
