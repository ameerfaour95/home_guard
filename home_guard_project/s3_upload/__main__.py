"""
CLI entry point for the S3 upload pipeline.

Usage:
    python -m home_guard_project.s3_upload [DATASET_DIR] [--skip-reencode] [--dry-run] [--force]
"""

from __future__ import annotations

import argparse
import logging
import sys

from .config import load_config
from .s3_upload import run


def main() -> None:
    cfg = load_config()

    parser = argparse.ArgumentParser(
        description="Re-encode clips to H.264 and upload dataset to S3.",
    )
    parser.add_argument(
        "dataset_dir",
        nargs="?",
        default=cfg.dataset_dir,
        help=f"Path to the dataset_multi directory (default: {cfg.dataset_dir}).",
    )
    parser.add_argument(
        "--bucket",
        default=cfg.bucket,
        help=f"S3 bucket name (default: {cfg.bucket}).",
    )
    parser.add_argument(
        "--prefix",
        default=cfg.prefix,
        help=f"S3 key prefix (default: {cfg.prefix}).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=cfg.workers,
        help=f"Parallel upload threads (default: {cfg.workers}).",
    )
    parser.add_argument(
        "--skip-reencode",
        action="store_true",
        help="Skip the H.264 re-encoding step.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be uploaded without actually uploading.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-upload all files even if they already exist on S3 with the same size.",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(name)-12s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    run(
        dataset_dir=args.dataset_dir,
        bucket=args.bucket,
        prefix=args.prefix,
        workers=args.workers,
        skip_reencode=args.skip_reencode,
        dry_run=args.dry_run,
        force=args.force,
    )


if __name__ == "__main__":
    main()
