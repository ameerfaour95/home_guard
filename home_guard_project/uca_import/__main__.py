"""
CLI entry point for the UCA dataset import pipeline.

Usage:
    python -m home_guard_project.uca_import --zip "path/to/UCA Dataset.zip"
"""

from __future__ import annotations

import argparse
import logging
import sys

from .config import load_config
from .uca_import import reset_manifest, run


def main() -> None:
    cfg = load_config()

    parser = argparse.ArgumentParser(
        description="Import the UCA (UCF Crime Annotation) dataset into the "
                    "labeling-compatible dataset format using fixed-duration "
                    "windowing.",
    )
    parser.add_argument(
        "--zip",
        required=True,
        help="Path to the UCA Dataset zip file.",
    )
    parser.add_argument(
        "--output-dir",
        default=cfg.output_dir,
        help=f"Dataset output directory (default: {cfg.output_dir}).",
    )
    parser.add_argument(
        "--camera-name-from",
        choices=["category", "uca"],
        default=cfg.camera_name_from,
        help='How to derive camera_name: "category" uses the crime type, '
             '"uca" uses a flat name (default: %(default)s).',
    )
    parser.add_argument(
        "--reencode",
        action="store_true",
        default=cfg.reencode,
        help="Re-encode clips at CRF 18 for frame-accurate cuts "
             "(default: stream copy, zero quality loss).",
    )
    parser.add_argument(
        "--no-yolo",
        action="store_true",
        help="Disable YOLO export.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=cfg.splits,
        help="Annotation splits to import (default: %(default)s).",
    )
    parser.add_argument(
        "--extract-dir",
        default=None,
        help="Directory for temporarily extracting videos from zip "
             "(default: <output-dir>/_uca_raw).",
    )
    parser.add_argument(
        "--window-size",
        type=float,
        default=cfg.window_size_sec,
        help=f"Fixed window duration in seconds (default: {cfg.window_size_sec}).",
    )
    parser.add_argument(
        "--exclude-categories",
        nargs="+",
        default=cfg.exclude_categories,
        help="Skip videos whose category matches these names "
             '(e.g. --exclude-categories Normal). Case-sensitive.',
    )
    parser.add_argument(
        "--reset-manifest",
        action="store_true",
        help="Delete the progress manifest to force full reprocessing. "
             "Normally the pipeline resumes from where it left off.",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(name)-18s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    from dataclasses import replace
    overrides = {}
    if args.output_dir != cfg.output_dir:
        overrides["output_dir"] = args.output_dir
    if args.camera_name_from != cfg.camera_name_from:
        overrides["camera_name_from"] = args.camera_name_from
    if args.reencode and not cfg.reencode:
        overrides["reencode"] = True
    if args.no_yolo:
        overrides["yolo_enabled"] = False
    if args.splits != cfg.splits:
        overrides["splits"] = args.splits
    if args.window_size != cfg.window_size_sec:
        overrides["window_size_sec"] = args.window_size
    if args.exclude_categories != cfg.exclude_categories:
        overrides["exclude_categories"] = args.exclude_categories

    if overrides:
        cfg = replace(cfg, **overrides)

    if args.reset_manifest:
        reset_manifest(cfg.output_dir)

    result = run(
        zip_path=args.zip,
        cfg=cfg,
        extract_dir=args.extract_dir,
    )

    total = result["imported"] + result["failed"]
    if total == 0 and result["skipped"] == 0:
        print("No windows processed.")
        sys.exit(1)

    print(
        f"\nDone: {result['imported']} imported, "
        f"{result['failed']} failed, "
        f"{result['skipped']} skipped "
        f"(of {result['total_windows']} total windows)"
    )

    if result["failed"] > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
