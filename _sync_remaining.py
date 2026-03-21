"""Sync the remaining yolo/labels/Normal directory to S3 in small batches."""

import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm

BUCKET = "s3://security-camera-project-v1/dataset_uca"
LOCAL = os.path.abspath("./dataset_uca")
WORKERS = 8


def sync_one(local_path: str, s3_url: str) -> tuple[int, int]:
    cmd = [sys.executable, "-m", "awscli", "s3", "sync",
           local_path, s3_url, "--no-progress"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            text=True, bufsize=1)
    uploaded = 0
    for line in proc.stdout:
        if line.strip().startswith("upload:"):
            uploaded += 1
    proc.wait()
    proc.stderr.read()
    return proc.returncode, uploaded


def main():
    target = os.path.join(LOCAL, "yolo", "labels", "Normal")
    subdirs = sorted(d for d in os.listdir(target)
                     if os.path.isdir(os.path.join(target, d)))
    print(f"yolo/labels/Normal: {len(subdirs)} video subdirectories")

    total_uploaded = 0

    def do_sync(name: str) -> tuple[str, int]:
        lp = os.path.join(target, name)
        s3 = f"{BUCKET}/yolo/labels/Normal/{name}/"
        _, up = sync_one(lp, s3)
        return name, up

    with tqdm(total=len(subdirs), desc="Normal subdirs", unit="dir") as bar:
        with ThreadPoolExecutor(max_workers=WORKERS) as pool:
            futs = {pool.submit(do_sync, d): d for d in subdirs}
            for f in as_completed(futs):
                name, up = f.result()
                total_uploaded += up
                bar.update(1)

    print(f"\nDone — uploaded {total_uploaded:,} new files from yolo/labels/Normal")

    # Also upload the manifest
    manifest = os.path.join(LOCAL, "_processed_manifest.json")
    if os.path.isfile(manifest):
        print("Uploading manifest...")
        subprocess.run([sys.executable, "-m", "awscli", "s3", "cp",
                        manifest, f"{BUCKET}/_processed_manifest.json"],
                       capture_output=True, text=True)
        print("Manifest uploaded.")


if __name__ == "__main__":
    main()
