"""
CLI entry point for the SmartHome-Bench dataset import pipeline.

Usage:
    python -m home_guard_project.smarthome_import --repo /path/to/SmartHome-Bench-LLM
"""

from __future__ import annotations

import argparse
import logging
import sys

from .config import load_config
from .smarthome_import import reset_manifest, run


def main() -> None:
    cfg = load_config()

    parser = argparse.ArgumentParser(
        description="Import the SmartHome-Bench dataset into the "
                    "labeling-compatible dataset format using fixed-duration "
                    "windowing and YOLO detection.",
    )
    parser.add_argument(
        "--repo",
        required=True,
        help="Path to the cloned SmartHome-Bench-LLM repository.",
    )
    parser.add_argument(
        "--output-dir",
        default=cfg.output_dir,
        help=f"Dataset output directory (default: {cfg.output_dir}).",
    )
    parser.add_argument(
        "--camera-name-from",
        choices=["category", "smarthome"],
        default=cfg.camera_name_from,
        help='How to derive camera_name: "category" uses the video category, '
             '"smarthome" uses a flat name (default: %(default)s).',
    )
    parser.add_argument(
        "--reencode",
        action="store_true",
        default=cfg.reencode,
        help="Re-encode clips at CRF 18 for frame-accurate cuts "
             "(default: stream copy).",
    )
    parser.add_argument(
        "--no-yolo",
        action="store_true",
        help="Disable YOLO export.",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip YouTube download (use already-downloaded videos).",
    )
    parser.add_argument(
        "--window-size",
        type=float,
        default=cfg.window_size_sec,
        help=f"Fixed window duration in seconds (default: {cfg.window_size_sec}).",
    )
    parser.add_argument(
        "--categories",
        nargs="+",
        default=cfg.categories,
        help="Categories to include (default: %(default)s).",
    )
    parser.add_argument(
        "--reset-manifest",
        action="store_true",
        help="Delete the progress manifest to force full reprocessing.",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(name)-22s  %(levelname)-8s  %(message)s",
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
    if args.window_size != cfg.window_size_sec:
        overrides["window_size_sec"] = args.window_size
    if args.categories != cfg.categories:
        overrides["categories"] = args.categories

    if overrides:
        cfg = replace(cfg, **overrides)

    if args.reset_manifest:
        reset_manifest(cfg.output_dir)

    result = run(
        repo_dir=args.repo,
        cfg=cfg,
        skip_download=args.skip_download,
    )

    total = result["imported"] + result["failed"]
    if total == 0 and result["skipped"] == 0:
        print("No windows processed.")
        sys.exit(1)

    print(
        f"\nDone: {result['imported']} imported, "
        f"{result['failed']} failed, "
        f"{result['skipped']} skipped "
        f"(of {result['total_windows']} total windows), "
        f"{result['downloaded']} downloaded"
    )

    if result["failed"] > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
