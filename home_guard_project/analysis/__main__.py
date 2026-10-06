"""
CLI entry point for the label export analysis pipeline.

Usage:
    py -m home_guard_project.analysis path/to/export.json [--output-dir ./analysis_output]
"""

from __future__ import annotations

import argparse
import logging
import sys

from .analyze import run
from .config import load_config


def main() -> None:
    cfg = load_config()

    parser = argparse.ArgumentParser(
        description=(
            "Analyze a Label Studio JSON export: generate summary report, "
            "Excel workbook, YOLO training labels, and VLM fine-tuning JSONL."
        ),
    )
    parser.add_argument(
        "export_path",
        help="Path to the Label Studio JSON export file.",
    )
    parser.add_argument(
        "--output-dir",
        default=cfg.output_dir,
        help=f"Output directory (default: {cfg.output_dir}).",
    )
    parser.add_argument(
        "--no-report",
        action="store_true",
        help="Skip summary report generation.",
    )
    parser.add_argument(
        "--no-excel",
        action="store_true",
        help="Skip Excel workbook generation.",
    )
    parser.add_argument(
        "--no-yolo",
        action="store_true",
        help="Skip YOLO training data export.",
    )
    parser.add_argument(
        "--no-vlm",
        action="store_true",
        help="Skip VLM fine-tuning JSONL export.",
    )
    parser.add_argument(
        "--dataset-dir",
        default=None,
        help=(
            "Path to the dataset directory (clips' meta for native fps; with "
            "--purge-dropped, its files of dropped tasks are deleted too)."
        ),
    )
    parser.add_argument(
        "--purge-dropped",
        action="store_true",
        help=(
            "Delete the S3 objects (and --dataset-dir files) of [delete]-marked "
            "and untagged tasks. Off by default: false-trigger clips are hard "
            "negatives for YOLO training."
        ),
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(name)-12s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    run(
        export_path=args.export_path,
        cfg=cfg,
        output_dir=args.output_dir,
        dataset_dir=args.dataset_dir,
        skip_report=args.no_report,
        skip_excel=args.no_excel,
        skip_yolo=args.no_yolo,
        skip_vlm=args.no_vlm,
        purge_dropped=args.purge_dropped,
    )


if __name__ == "__main__":
    main()
