"""
CLI for the collector box.

Usage:
    python -m home_guard_project.box upload      # move finished clips to the outbox, sync to S3, heartbeat
    python -m home_guard_project.box heartbeat   # write the status JSON to S3
    python -m home_guard_project.box status      # print the status JSON locally (no network)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from typing import Any, Callable, Tuple

from .boxconfig import (
    ALIVE_FILE,
    LIVE_DIR,
    OUTBOX_DIR,
    BoxConfig,
    BoxConfigError,
    load_box_config,
    s3_prefix,
)
from .heartbeat import build_heartbeat, put_heartbeat
from .outbox import move_finished_clips

log = logging.getLogger("box")


def _has_files(root: str) -> bool:
    return any(filenames for _, _, filenames in os.walk(root))


def run_upload(
    cfg: BoxConfig,
    live_dir: str,
    outbox_dir: str,
    bucket: str,
    workers: int,
    uploader: Callable[..., Any],
) -> Tuple[int, int]:
    """Move finished clips to the outbox and upload it. Returns ``(clips_moved, files_moved)``."""
    os.makedirs(outbox_dir, exist_ok=True)
    moved = move_finished_clips(live_dir, outbox_dir, cfg.min_age_minutes * 60)
    log.info("Moved %d finished clip(s) (%d files) to the outbox.", *moved)

    if not _has_files(outbox_dir):
        log.info("Outbox is empty — nothing to upload.")
        return moved

    # Orphan cleanup and the label filter are off: every clip the box saved is
    # uploaded, and what to keep is decided at tagging time.
    uploader(
        dataset_dir=outbox_dir,
        bucket=bucket,
        prefix=s3_prefix(cfg.site),
        workers=workers,
        skip_reencode=False,
        no_cleanup=True,
        allowed_labels=frozenset(),
        delete_local=True,
    )
    return moved


def main() -> None:
    parser = argparse.ArgumentParser(description="Collector box: upload, heartbeat, status.")
    parser.add_argument("command", choices=["upload", "heartbeat", "status"])
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(name)-12s  %(levelname)-8s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    try:
        cfg = load_box_config()
    except BoxConfigError as exc:
        log.error("%s", exc)
        sys.exit(1)

    if args.command == "status":
        print(json.dumps(build_heartbeat(cfg.site, LIVE_DIR, OUTBOX_DIR, ALIVE_FILE), indent=2))
        return

    from home_guard_project.s3_upload.config import load_config as load_s3_config

    s3_cfg = load_s3_config()

    if args.command == "upload":
        from home_guard_project.s3_upload.s3_upload import run as s3_run

        run_upload(cfg, LIVE_DIR, OUTBOX_DIR, s3_cfg.bucket, s3_cfg.workers, uploader=s3_run)

    key = put_heartbeat(
        build_heartbeat(cfg.site, LIVE_DIR, OUTBOX_DIR, ALIVE_FILE),
        s3_cfg.bucket,
        s3_prefix(cfg.site),
    )
    log.info("Heartbeat written to s3://%s/%s", s3_cfg.bucket, key)


if __name__ == "__main__":
    main()
