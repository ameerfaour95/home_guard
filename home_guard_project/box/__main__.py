"""
CLI for the collector box.

Usage:
    python -m home_guard_project.box upload           # move finished clips to the outbox, sync to S3, heartbeat
    python -m home_guard_project.box heartbeat        # write the status JSON to S3
    python -m home_guard_project.box status           # print the status JSON locally (no network)
    python -m home_guard_project.box mode             # print the box's mode (read by run_collector.sh)
    python -m home_guard_project.box set-site house2  # name the house; clips then go to s3://<bucket>/dataset_house2/
    python -m home_guard_project.box set-option show_cameras true   # camera windows on the box's own screen
    python -m home_guard_project.box set-option alert_start_hour 22 # the options are listed in boxconfig.py
    python -m home_guard_project.box set-option telegram_chat_ids=-1001234567,987654
    python -m home_guard_project.box get-option show_cameras        # prints the stored value
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from typing import Any, Callable, Sequence, Tuple

from .boxconfig import (
    ALIVE_FILE,
    LIVE_DIR,
    MODE_INFERENCE,
    OUTBOX_DIR,
    PRODUCTION_LIVE_DIR,
    PRODUCTION_OUTBOX_DIR,
    BoxConfig,
    BoxConfigError,
    get_option,
    load_box_config,
    production_prefix,
    s3_prefix,
    set_option,
    set_site,
)
from .heartbeat import build_heartbeat, put_heartbeat
from .outbox import ORPHAN_AGE_SEC, move_finished_clips, move_orphans

log = logging.getLogger("box")


def _has_files(root: str) -> bool:
    return any(filenames for _, _, filenames in os.walk(root))


def _site_dirs(outbox_dir: str) -> list[str]:
    """Site names that have a folder in the outbox. Each site's clips wait in their own folder."""
    if not os.path.isdir(outbox_dir):
        return []
    return sorted(d for d in os.listdir(outbox_dir) if os.path.isdir(os.path.join(outbox_dir, d)))


def run_upload(
    cfg: BoxConfig,
    live_dir: str,
    outbox_dir: str,
    bucket: str,
    workers: int,
    uploader: Callable[..., Any],
    prefix_for: Callable[[str], str] = s3_prefix,
) -> Tuple[int, int]:
    """Move finished clips to the outbox and upload it. Returns ``(clips_moved, files_moved)``.

    The outbox has one folder per site, uploaded to that site's own S3 folder
    (named by *prefix_for*), so clips saved before a box changed house still go
    where they belong.
    """
    os.makedirs(outbox_dir, exist_ok=True)
    site_outbox = os.path.join(outbox_dir, cfg.site)
    moved = move_finished_clips(live_dir, site_outbox, cfg.min_age_minutes * 60)
    log.info("Moved %d finished clip(s) (%d files) to the outbox.", *moved)
    # A clip cut short by a restart has no meta; it is uploaded too, not left on the box.
    orphans = move_orphans(live_dir, site_outbox, ORPHAN_AGE_SEC)
    if orphans:
        log.info("Moved %d file(s) of interrupted clips (no meta) to the outbox.", orphans)

    uploaded_any = False
    for site in _site_dirs(outbox_dir):
        site_dir = os.path.join(outbox_dir, site)
        if not _has_files(site_dir):
            continue
        uploaded_any = True
        # Orphan cleanup and the label filter are off: every clip the box saved is
        # uploaded, and what to keep is decided at tagging time.
        uploader(
            dataset_dir=site_dir,
            bucket=bucket,
            prefix=prefix_for(site),
            workers=workers,
            skip_reencode=False,
            no_cleanup=True,
            allowed_labels=frozenset(),
            delete_local=True,
        )
    if not uploaded_any:
        log.info("Outbox is empty — nothing to upload.")
    return moved


def change_site(
    new_site: str,
    live_dir: str,
    outbox_dir: str,
    box_yaml: str,
    also: Sequence[Tuple[str, str]] = (),
) -> Tuple[int, int]:
    """Rename the site. Clips already saved are set aside under the old name first.

    *also* lists further ``(live_dir, outbox_dir)`` pairs to set aside the same way.
    Returns the clips and files moved, over all pairs.
    """
    clips = files = 0
    try:
        old_site = load_box_config(box_yaml).site
    except BoxConfigError:
        old_site = None
    if old_site and old_site != new_site:
        for live, outbox in ((live_dir, outbox_dir), *also):
            moved = move_finished_clips(live, os.path.join(outbox, old_site), 0)
            clips, files = clips + moved[0], files + moved[1]
    set_site(new_site, box_yaml)
    return clips, files


def split_option(values: list[str]) -> Tuple[str, str]:
    """``KEY VALUE`` or ``KEY=VALUE`` -> ``(key, value)``.

    The one-token form carries values the command line would misread as a
    flag, such as a chat id list that starts with a minus sign.
    """
    if len(values) == 2:
        return values[0], values[1]
    if len(values) == 1 and "=" in values[0]:
        key, _, value = values[0].partition("=")
        return key, value
    raise BoxConfigError("usage: set-option KEY VALUE  (or KEY=VALUE)")


def _clip_dirs(mode: str) -> Tuple[str, str]:
    """The live and outbox folders the box is filling in *mode*, for the status report."""
    if mode == MODE_INFERENCE:
        return PRODUCTION_LIVE_DIR, PRODUCTION_OUTBOX_DIR
    return LIVE_DIR, OUTBOX_DIR


def _shown(value: Any) -> str:
    """An option value as the command line prints it: true/false, a number, text, or nothing."""
    if isinstance(value, bool):
        return "true" if value else "false"
    return "" if value is None else str(value)


def main() -> None:
    parser = argparse.ArgumentParser(description="Collector box: upload, heartbeat, status, mode, settings.")
    parser.add_argument("command", choices=["upload", "heartbeat", "status", "mode", "set-site", "set-option", "get-option"])
    parser.add_argument("values", nargs="*", help="set-site NAME | set-option KEY VALUE (or KEY=VALUE) | get-option KEY")
    args = parser.parse_args()
    args.value = args.values[0] if args.values else None

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(name)-12s  %(levelname)-8s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if args.command in ("set-option", "get-option"):
        try:
            if args.command == "get-option":
                if len(args.values) != 1:
                    raise BoxConfigError("usage: get-option KEY")
                print(_shown(get_option(args.values[0])))
            else:
                key, value = split_option(args.values)
                print(f"{key} set to {_shown(set_option(key, value))}")
        except BoxConfigError as exc:
            log.error("%s", exc)
            sys.exit(1)
        return

    if args.command == "set-site":
        from .boxconfig import BOX_YAML

        try:
            if not args.value:
                raise BoxConfigError("set-site needs a site name, e.g. set-site house2")
            clips, _ = change_site(
                args.value, LIVE_DIR, OUTBOX_DIR, BOX_YAML,
                also=[(PRODUCTION_LIVE_DIR, PRODUCTION_OUTBOX_DIR)],
            )
        except BoxConfigError as exc:
            log.error("%s", exc)
            sys.exit(1)
        if clips:
            log.info("Set aside %d clip(s) saved under the previous site name.", clips)
        print(f"site set to {args.value}; clips go to the S3 folder {s3_prefix(args.value)}/")
        return

    try:
        cfg = load_box_config()
    except BoxConfigError as exc:
        log.error("%s", exc)
        sys.exit(1)

    if args.command == "mode":
        print(cfg.mode)
        return

    if args.command == "status":
        print(json.dumps(build_heartbeat(cfg.site, *_clip_dirs(cfg.mode), ALIVE_FILE, mode=cfg.mode), indent=2))
        return

    from home_guard_project.s3_upload.config import load_config as load_s3_config

    s3_cfg = load_s3_config()

    if args.command == "upload":
        from home_guard_project.s3_upload.s3_upload import run as s3_run

        run_upload(cfg, LIVE_DIR, OUTBOX_DIR, s3_cfg.bucket, s3_cfg.workers, uploader=s3_run)
        # Clips saved in inference mode go to the folder that expires after two weeks.
        run_upload(
            cfg, PRODUCTION_LIVE_DIR, PRODUCTION_OUTBOX_DIR, s3_cfg.bucket, s3_cfg.workers,
            uploader=s3_run, prefix_for=production_prefix,
        )

    key = put_heartbeat(
        build_heartbeat(cfg.site, *_clip_dirs(cfg.mode), ALIVE_FILE, mode=cfg.mode),
        s3_cfg.bucket,
        s3_prefix(cfg.site),
    )
    log.info("Heartbeat written to s3://%s/%s", s3_cfg.bucket, key)


if __name__ == "__main__":
    main()
