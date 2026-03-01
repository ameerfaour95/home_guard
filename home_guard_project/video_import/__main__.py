"""
CLI entry point for the external video import pipeline.

Usage:
    python -m home_guard_project.video_import <INPUT_PATH> [options]

INPUT_PATH can be a single video file or a directory of videos.
"""

from __future__ import annotations

import argparse
import logging
import sys

from .config import load_config
from .video_import import run


def main() -> None:
    cfg = load_config()

    parser = argparse.ArgumentParser(
        description="Import external videos into the security camera dataset.",
    )
    parser.add_argument(
        "input_path",
        help="Path to a video file or directory of videos to import.",
    )
    parser.add_argument(
        "--camera-name",
        default=cfg.camera_name,
        help=f"Camera name for imported clips (default: {cfg.camera_name}).",
    )
    parser.add_argument(
        "--output-dir",
        default=cfg.output_dir,
        help=f"Dataset output directory (default: {cfg.output_dir}).",
    )
    parser.add_argument(
        "--clip-seconds",
        type=float,
        default=cfg.clip_seconds,
        help=f"Split videos into chunks of this duration (default: {cfg.clip_seconds}s).",
    )
    parser.add_argument(
        "--store-fps",
        type=float,
        default=cfg.store_fps,
        help=f"Output FPS for clips (default: {cfg.store_fps}).",
    )
    parser.add_argument(
        "--no-yolo",
        action="store_true",
        help="Disable YOLO export.",
    )
    parser.add_argument(
        "--no-vlm-crop",
        action="store_true",
        help="Disable VLM crop generation.",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(name)-18s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    from dataclasses import replace
    overrides = {}
    if args.camera_name != cfg.camera_name:
        overrides["camera_name"] = args.camera_name
    if args.output_dir != cfg.output_dir:
        overrides["output_dir"] = args.output_dir
    if args.clip_seconds != cfg.clip_seconds:
        overrides["clip_seconds"] = args.clip_seconds
    if args.store_fps != cfg.store_fps:
        overrides["store_fps"] = args.store_fps
    if args.no_yolo:
        overrides["yolo_enabled"] = False
    if args.no_vlm_crop:
        overrides["vlm_crop_enabled"] = False

    if overrides:
        cfg = replace(cfg, **overrides)

    result = run(input_path=args.input_path, cfg=cfg)

    if result["failed"] > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
