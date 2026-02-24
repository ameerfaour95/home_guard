"""
CLI entry point for the labeling pipeline.

Usage:
    python -m home_guard_project.labeling
    python -m home_guard_project.labeling --dataset-dir ./dataset_multi --limit 50
    python -m home_guard_project.labeling --reencode
    python -m home_guard_project.labeling --serve
    python -m home_guard_project.labeling --merge exported.json
    python -m home_guard_project.labeling --cleanup-only
    python -m home_guard_project.labeling --cleanup-only --dry-run
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from typing import List, Optional

from .config import get_default_dataset_dir, get_file_server_port
from .tasks import (
    collect_clip_paths,
    collect_vlm_crop_paths,
    export_tasks,
    scan_and_build_tasks,
    tasks_are_fresh,
    write_config,
)
from .utils.cleanup import cleanup_orphans
from .utils.ffmpeg import reencode_videos
from .utils.file_server import start_file_server
from .utils.merge import merge_annotations

log = logging.getLogger("labeling")


def setup_logging(verbose: bool = False) -> None:
    """Configure structured logging with timestamps."""
    level = logging.DEBUG if verbose else logging.INFO
    fmt = "%(asctime)s  %(levelname)-5s  %(message)s"
    datefmt = "%H:%M:%S"
    logging.basicConfig(level=level, format=fmt, datefmt=datefmt, stream=sys.stderr)


def main() -> None:
    default_port = get_file_server_port()
    default_dataset = get_default_dataset_dir()

    parser = argparse.ArgumentParser(
        description="Generate Label Studio config and tasks from security camera dataset.",
    )
    parser.add_argument(
        "--dataset-dir",
        default=default_dataset,
        help=f"Path to dataset_multi directory (default: {default_dataset})",
    )
    parser.add_argument(
        "--reencode",
        action="store_true",
        help="Re-encode MP4 clips from mp4v to H.264 (requires ffmpeg)",
    )
    parser.add_argument(
        "--ffmpeg",
        default=None,
        help="Optional path to ffmpeg executable",
    )
    parser.add_argument(
        "--cameras",
        default=None,
        help="Comma-separated camera names to include (e.g. main_door,back_door)",
    )
    parser.add_argument(
        "--kinds",
        default=None,
        help="Comma-separated clip kinds to include (e.g. trigger,random)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of tasks to generate (for testing)",
    )
    parser.add_argument(
        "--no-predictions",
        action="store_true",
        help="Skip generating pre-annotations (YOLO boxes + VLM text)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of parallel workers (default: auto = CPU count, cap 8)",
    )
    parser.add_argument(
        "--serve",
        action="store_true",
        help="Start a CORS-enabled file server for video files",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=default_port,
        help=f"Port for the file server (default: {default_port})",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-encoding and task rebuild even if tasks are up-to-date",
    )
    parser.add_argument(
        "--merge",
        metavar="EXPORTED_JSON",
        default=None,
        help="Merge annotations from a Label Studio JSON export into the "
             "generated tasks file.",
    )
    parser.add_argument(
        "--no-cleanup",
        action="store_true",
        help="Skip orphan cleanup (removal of meta/yolo files with no matching clip)",
    )
    parser.add_argument(
        "--cleanup-only",
        action="store_true",
        help="Only run orphan cleanup, then exit",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="With --cleanup-only: show what would be removed without deleting",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable debug-level logging",
    )

    args = parser.parse_args()
    setup_logging(verbose=args.verbose)
    t_start = time.monotonic()

    dataset_dir: str = args.dataset_dir
    if not os.path.isdir(dataset_dir):
        log.error("Dataset directory not found: %s", dataset_dir)
        sys.exit(1)

    # --serve: start the file server and block
    if args.serve:
        start_file_server(dataset_dir, port=args.port, background=False)
        return

    # --cleanup-only: run orphan cleanup and exit
    if args.cleanup_only:
        stats = cleanup_orphans(dataset_dir, dry_run=args.dry_run)
        print()
        print(stats.summary())
        if stats.total_removed == 0:
            log.info("Dataset is clean — no orphans found.")
        elif args.dry_run:
            log.info("Re-run without --dry-run to delete %d orphan files.", stats.total_removed)
        else:
            log.info("Removed %d orphan files in %.1fs", stats.total_removed, time.monotonic() - t_start)
        return

    # --merge only (no generation): merge and exit
    if args.merge and not args.reencode:
        tasks_path = os.path.join(dataset_dir, "label_studio_tasks.json")
        if not os.path.isfile(tasks_path):
            log.error("Tasks file not found: %s — generate tasks first.", tasks_path)
            sys.exit(1)
        if not os.path.isfile(args.merge):
            log.error("Export file not found: %s", args.merge)
            sys.exit(1)
        merge_annotations(tasks_path, args.merge)
        log.info("Done in %.1fs", time.monotonic() - t_start)
        return

    # Parse filters
    camera_filter: Optional[List[str]] = None
    if args.cameras:
        camera_filter = [c.strip() for c in args.cameras.split(",") if c.strip()]

    kind_filter: Optional[List[str]] = None
    if args.kinds:
        kind_filter = [k.strip() for k in args.kinds.split(",") if k.strip()]

    video_base_url = f"http://localhost:{args.port}"

    # Step 0a: Orphan cleanup (remove meta/response/yolo for deleted clips)
    if not args.no_cleanup:
        stats = cleanup_orphans(dataset_dir)
        if stats.total_removed > 0:
            log.info("Cleanup: removed %d orphan files", stats.total_removed)

    # Step 0b: Check if tasks are already up-to-date
    if not args.force and not args.limit and tasks_are_fresh(dataset_dir):
        log.info("Nothing to do (%.1fs). Use --force to rebuild anyway.",
                 time.monotonic() - t_start)
        return

    # Step 1: Write labeling config
    config_path = os.path.join(dataset_dir, "label_studio_config.xml")
    write_config(config_path)

    # Step 2: Optionally re-encode videos (clips + VLM crops)
    if args.reencode:
        clips = collect_clip_paths(
            dataset_dir,
            cameras=camera_filter,
            kinds=kind_filter,
            limit=args.limit,
        )
        vlm_crops = collect_vlm_crop_paths(
            dataset_dir,
            cameras=camera_filter,
            kinds=kind_filter,
            limit=args.limit,
        )
        reencode_videos(
            clips + vlm_crops,
            ffmpeg_path=args.ffmpeg,
            workers=args.workers,
        )

    # Step 3: Scan and build tasks
    tasks = scan_and_build_tasks(
        dataset_dir,
        cameras=camera_filter,
        kinds=kind_filter,
        limit=args.limit,
        include_predictions=not args.no_predictions,
        video_base_url=video_base_url,
        workers=args.workers,
    )
    log.info("Built %d tasks", len(tasks))

    if not tasks:
        log.warning("No tasks generated. Check filters and dataset directory.")
        sys.exit(0)

    # Step 4: Export tasks JSON
    tasks_path = os.path.join(dataset_dir, "label_studio_tasks.json")
    export_tasks(tasks, tasks_path)

    # Step 4b: Merge annotations from a previous export (if provided)
    if args.merge:
        if os.path.isfile(args.merge):
            merge_annotations(tasks_path, args.merge)
        else:
            log.warning("Export file not found: %s — skipping merge.", args.merge)

    log.info("All done in %.1fs", time.monotonic() - t_start)


if __name__ == "__main__":
    main()
