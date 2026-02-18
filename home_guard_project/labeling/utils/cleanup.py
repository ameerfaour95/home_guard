"""
Orphan cleanup utility for dataset_multi.

Walks the dataset and removes metadata, VLM responses, and YOLO exports
whose corresponding video clip no longer exists.

Can be used standalone:
    python -m home_guard_project.labeling.utils.cleanup ./dataset_multi

Or imported:
    from home_guard_project.labeling.utils.cleanup import cleanup_orphans
    stats = cleanup_orphans("./dataset_multi")
"""
from __future__ import annotations

import argparse
import logging
import os
from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple

log = logging.getLogger(__name__)


@dataclass
class CleanupStats:
    """Counters for what was found and removed."""

    clips_found: int = 0
    meta_orphans: int = 0
    response_orphans: int = 0
    yolo_image_orphans: int = 0
    yolo_label_orphans: int = 0
    empty_dirs_removed: int = 0
    errors: list = field(default_factory=list)

    @property
    def total_removed(self) -> int:
        return (
            self.meta_orphans
            + self.response_orphans
            + self.yolo_image_orphans
            + self.yolo_label_orphans
        )

    def summary(self) -> str:
        lines = [
            f"Clips present       : {self.clips_found}",
            f"Orphan meta removed : {self.meta_orphans}",
            f"Orphan resp removed : {self.response_orphans}",
            f"Orphan YOLO imgs    : {self.yolo_image_orphans}",
            f"Orphan YOLO lbls    : {self.yolo_label_orphans}",
            f"Empty dirs removed  : {self.empty_dirs_removed}",
        ]
        if self.errors:
            lines.append(f"Errors              : {len(self.errors)}")
        return "\n".join(lines)


def _collect_clip_ids(clips_dir: str) -> Set[str]:
    """Return set of clip_ids (filename without .mp4) that exist on disk."""
    ids: Set[str] = set()
    if not os.path.isdir(clips_dir):
        return ids
    for dirpath, _, filenames in os.walk(clips_dir):
        for fname in filenames:
            if fname.lower().endswith(".mp4"):
                ids.add(os.path.splitext(fname)[0])
    return ids


def _find_orphans_in_tree(
    root: str,
    suffix: str,
    clip_ids: Set[str],
    prefix_match: bool = False,
) -> List[str]:
    """
    Walk *root* and return paths of files ending with *suffix* whose clip_id
    is NOT in *clip_ids*.
    """
    orphans: List[str] = []
    if not os.path.isdir(root):
        return orphans

    for dirpath, _, filenames in os.walk(root):
        for fname in filenames:
            if not fname.lower().endswith(suffix):
                continue

            if prefix_match:
                stem = os.path.splitext(fname)[0]
                parts = stem.rsplit("_f", 1)
                candidate_id = parts[0] if len(parts) == 2 and parts[1].isdigit() else stem
            else:
                candidate_id = fname
                for ext in (".meta.json", ".model_raw.txt"):
                    if candidate_id.endswith(ext):
                        candidate_id = candidate_id[: -len(ext)]
                        break

            if candidate_id not in clip_ids:
                orphans.append(os.path.join(dirpath, fname))

    return orphans


def _prune_empty_dirs(root: str, stats: CleanupStats, dry_run: bool) -> None:
    """Remove empty directories bottom-up (won't remove *root* itself)."""
    if not os.path.isdir(root):
        return
    for dirpath, dirnames, filenames in os.walk(root, topdown=False):
        if dirpath == root:
            continue
        if not dirnames and not filenames:
            try:
                if dry_run:
                    log.debug("[DRY-RUN] Would remove empty dir: %s", dirpath)
                else:
                    os.rmdir(dirpath)
                    log.debug("Removed empty dir: %s", dirpath)
                stats.empty_dirs_removed += 1
            except OSError:
                pass


def cleanup_orphans(
    dataset_dir: str,
    dry_run: bool = False,
) -> CleanupStats:
    """
    Scan ``dataset_dir`` and remove orphaned artifacts.

    An artifact is "orphaned" when no corresponding ``.mp4`` clip exists
    under ``clips/``.

    Parameters
    ----------
    dataset_dir : str
        Path to the dataset root (e.g. ``./dataset_multi``).
    dry_run : bool
        If True, only log what would be removed without deleting.

    Returns
    -------
    CleanupStats
        Summary of what was (or would be) removed.
    """
    from tqdm import tqdm

    stats = CleanupStats()

    clips_dir = os.path.join(dataset_dir, "clips")
    meta_dir = os.path.join(dataset_dir, "meta")
    resp_dir = os.path.join(dataset_dir, "responses")
    yolo_img_dir = os.path.join(dataset_dir, "yolo", "images")
    yolo_lbl_dir = os.path.join(dataset_dir, "yolo", "labels")

    clip_ids = _collect_clip_ids(clips_dir)
    stats.clips_found = len(clip_ids)
    log.info("Found %d clips in %s", len(clip_ids), clips_dir)

    if not clip_ids:
        log.warning("No clips found — aborting cleanup to prevent accidental mass deletion.")
        return stats

    # Phase 1: Scan for orphans (fast — just file listing)
    category_orphans: List[Tuple[str, str, List[str]]] = [
        ("meta_orphans", "Scanning meta/", _find_orphans_in_tree(meta_dir, ".meta.json", clip_ids)),
        ("response_orphans", "Scanning responses/", _find_orphans_in_tree(resp_dir, ".model_raw.txt", clip_ids)),
        ("yolo_image_orphans", "Scanning yolo/images/", _find_orphans_in_tree(yolo_img_dir, ".jpg", clip_ids, prefix_match=True)),
        ("yolo_label_orphans", "Scanning yolo/labels/", _find_orphans_in_tree(yolo_lbl_dir, ".txt", clip_ids, prefix_match=True)),
    ]

    all_orphans: List[str] = []
    for attr, _desc, paths in category_orphans:
        setattr(stats, attr, len(paths))
        all_orphans.extend(paths)

    if not all_orphans:
        return stats

    # Phase 2: Delete with progress bar
    if dry_run:
        for path in all_orphans:
            log.info("[DRY-RUN] Would remove: %s", path)
    else:
        failed = 0
        with tqdm(
            all_orphans,
            desc="Removing orphans",
            unit="file",
            ncols=80,
            disable=None,
        ) as pbar:
            for filepath in pbar:
                try:
                    os.remove(filepath)
                except OSError as exc:
                    stats.errors.append(str(exc))
                    failed += 1

        if failed:
            log.warning("%d files could not be removed.", failed)

    # Phase 3: Clean up empty directories left behind
    for d in (meta_dir, resp_dir, yolo_img_dir, yolo_lbl_dir):
        _prune_empty_dirs(d, stats, dry_run)

    return stats


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Remove orphaned meta/response/YOLO files whose clips have been deleted.",
    )
    parser.add_argument(
        "dataset_dir",
        nargs="?",
        default="./dataset_multi",
        help="Path to the dataset root (default: ./dataset_multi)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print what would be removed, without deleting anything.",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Show every file removed (DEBUG level logging).",
    )
    args = parser.parse_args()

    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s  %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )

    if not os.path.isdir(args.dataset_dir):
        log.error("Dataset directory not found: %s", args.dataset_dir)
        raise SystemExit(1)

    mode = "DRY-RUN" if args.dry_run else "LIVE"
    log.info("Orphan cleanup [%s] for: %s", mode, os.path.abspath(args.dataset_dir))

    stats = cleanup_orphans(args.dataset_dir, dry_run=args.dry_run)

    print()
    print(stats.summary())

    if stats.total_removed == 0:
        print("\nDataset is clean — no orphans found.")
    elif args.dry_run:
        print(f"\nRe-run without --dry-run to delete {stats.total_removed} orphan files.")
    else:
        print(f"\nDone. Removed {stats.total_removed} orphan files.")


if __name__ == "__main__":
    main()
