"""Admin Center management CLI: python -m home_guard_project.cloud.manage <command>."""
from __future__ import annotations

import argparse
import secrets
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import select

HERE = Path(__file__).resolve().parent
ROLES = ("admin", "support", "labeler")


def alembic_config(db_url: str):
    """The Alembic configuration for `db_url`. `manage init-db` is the supported entry point; a bare
    `alembic upgrade head` has no database URL (migrations/env.py reads it from this config's attributes)."""
    from alembic.config import Config

    cfg = Config(str(HERE / "alembic.ini"))
    cfg.set_main_option("script_location", str(HERE / "migrations").replace("%", "%%"))
    cfg.attributes["url"] = db_url
    return cfg


def run_migrations(db_url: str) -> None:
    from alembic import command

    command.upgrade(alembic_config(db_url), "head")


class ConfigError(Exception):
    pass


def _db_url() -> str:
    """One resolver for every command: HG_CLOUD_DB_URL only."""
    import os

    url = os.environ.get("HG_CLOUD_DB_URL")
    if not url:
        raise ConfigError("HG_CLOUD_DB_URL is not set (the database to use)")
    return url


def _engine():
    from .db import make_engine

    return make_engine(_db_url())


def _optional(module: str):
    """Import a module that a later task adds; None only if that very module is missing."""
    import importlib

    name = f"{__package__}.{module}"
    try:
        return importlib.import_module(name)
    except ModuleNotFoundError as e:
        if e.name != name:
            raise
        return None


def cmd_init_db(args) -> int:
    run_migrations(_db_url())
    print("database is at head")
    return 0


def cmd_create_staff(args) -> int:
    import pyotp
    from . import auth
    from .db import session_scope
    from .models import Staff

    if args.name.strip().lower().startswith("system:"):
        print("staff names cannot start with 'system:' (reserved for automatic actors)", file=sys.stderr)
        return 1
    password = secrets.token_urlsafe(16)
    totp_secret = pyotp.random_base32()
    engine = _engine()
    with session_scope(engine) as s:
        if s.scalars(select(Staff).where(Staff.email == args.email)).first():
            print(f"staff {args.email} already exists", file=sys.stderr)
            return 1
        s.add(Staff(email=args.email, name=args.name, role=args.role,
                    password_hash=auth.hash_password(password), totp_secret=totp_secret))
    uri = pyotp.TOTP(totp_secret).provisioning_uri(name=args.email, issuer_name="Home Guard Admin")
    print(f"Created {args.role} {args.email}. Shown once, store it now.")
    print(f"password: {password}")
    print(f"totp: {uri}")
    return 0


def cmd_enroll(args) -> int:
    from .db import session_scope
    from .models import Customer, Device

    engine = _engine()
    with session_scope(engine) as s:
        if s.scalars(select(Device).where(Device.site == args.site)).first():
            print(f"site {args.site} is already enrolled", file=sys.stderr)
            return 1
        cust = s.scalars(select(Customer).where(Customer.name == args.customer)).first()
        if cust is None:
            cust = Customer(name=args.customer)
            s.add(cust)
            s.flush()
        dev = Device(device_id=str(uuid.uuid4()), site=args.site, tailscale_host=args.tailscale_host or "",
                     customer_id=cust.id, enrolled_at=datetime.now(timezone.utc))
        s.add(dev)
        s.flush()
        from . import redact

        redact.remember(s, dev)
        print(f"enrolled device_id={dev.device_id} site={dev.site} customer={cust.name} (id {cust.id})")
    return 0


def _s3():
    import os

    import boto3

    from .s3 import S3

    client = boto3.client("s3", region_name=os.environ.get("HG_CLOUD_REGION", "us-east-1"))
    return S3(client, os.environ.get("HG_CLOUD_BUCKET", "security-camera-project-v1"))


def cmd_index_once(args) -> int:
    """One index pass, under the same lock as the server's indexer loop (never both at once)."""
    from . import indexer
    from .db import session_scope
    from .loops import loop_lock

    engine = _engine()
    with loop_lock(engine, "indexer") as got:
        if not got:
            print("the server is already indexing; try again in a few minutes")
            return 0
        s3 = _s3()
        with session_scope(engine) as s:
            results = indexer.index_all(s, s3, full_scan=args.full)
    for site, stats in results.items():
        print(f"{site}: {stats}")
    return 0


def cmd_discover_once(args) -> int:
    """One discovery pass (boxes that registered themselves), under the same lock as the server's loop."""
    from . import discovery
    from .db import session_scope
    from .loops import loop_lock

    engine = _engine()
    with loop_lock(engine, "discovery") as got:
        if not got:
            print("the server is already discovering boxes; try again in a few minutes")
            return 0
        s3 = _s3()
        with session_scope(engine) as s:
            results = discovery.discover(s, s3)
    for site, what in results.items():
        print(f"{site}: {what}")
    if not results:
        print("nothing new")
    return 0


def cmd_media_once(args) -> int:
    """One media pass, under the same lock as the server's media loop."""
    from . import media
    from .db import session_scope
    from .loops import loop_lock

    engine = _engine()
    with loop_lock(engine, "media") as got:
        if not got:
            print("the server is already making media; try again in a few minutes")
            return 0
        s3 = _s3()
        with session_scope(engine) as s:
            n = media.process_pending(s, s3, limit=args.limit)
    print(f"media: {n} event(s) gained artifacts")
    return 0


def cmd_media_retry(args) -> int:
    """Clear media problems (quarantined or given-up clips) so the media loop tries them again; `--event ID`
    clears only that event's."""
    from . import media
    from .db import session_scope

    with session_scope(_engine()) as s:
        n = media.clear_problems(s, event_id=args.event)
    print(f"cleared {n} media problem(s); the media loop retries them on its next pass")
    return 0


def cmd_redact_backfill(args) -> int:
    """Fill the labelers' redacted search text; `--all` recomputes it (after a camera or customer rename)."""
    from . import redact
    from .db import session_scope
    from .models import Device

    engine = _engine()
    with session_scope(engine) as s:
        for device in s.scalars(select(Device).order_by(Device.id)).all():
            n = redact.backfill(s, device, everything=args.all)
            print(f"{device.site}: {n} event(s) redacted")
    return 0


def cmd_export_download(args) -> int:
    """Download a finished training export (without its _private/ folder) with the server's S3 credentials and
    make its YOLO data.yaml point at the absolute local folder, ready for `yolo detect train`."""
    from . import studio
    from .db import session_scope
    from .models import Export

    with session_scope(_engine()) as s:
        export = s.get(Export, args.export_id) if 1 <= args.export_id <= 2 ** 31 - 1 else None
        if export is None:
            print(f"export {args.export_id} not found", file=sys.stderr)
            return 1
        if export.state not in ("ready", "partial"):
            print(f"export {args.export_id} is {export.state}, not finished", file=sys.stderr)
            return 1
        if studio.consent_withdrawn(s, export):
            print(f"export {args.export_id}: a household has withdrawn training consent since it was made; "
                  "not downloaded", file=sys.stderr)
            return 1
        prefix = export.s3_prefix
    dest = Path(args.dest)
    n = studio.download_export(_s3(), prefix, dest)
    print(f"downloaded {n} file(s) into {dest.resolve()} (see README.txt there)")
    return 0


def cmd_serve(args) -> int:
    import uvicorn

    if _optional("app") is None:
        print("serve: not available yet")
        return 0
    uvicorn.run("home_guard_project.cloud.app:create_app_from_env", factory=True, host="0.0.0.0", port=args.port)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="manage")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("init-db").set_defaults(fn=cmd_init_db)
    c = sub.add_parser("create-staff")
    c.add_argument("--email", required=True)
    c.add_argument("--name", required=True)
    c.add_argument("--role", required=True, choices=ROLES)
    c.set_defaults(fn=cmd_create_staff)
    e = sub.add_parser("enroll")
    e.add_argument("--customer", required=True)
    e.add_argument("--site", required=True)
    e.add_argument("--tailscale-host", default="")
    e.set_defaults(fn=cmd_enroll)
    i = sub.add_parser("index-once")
    i.add_argument("--full", action="store_true", help="full scan instead of incremental")
    i.set_defaults(fn=cmd_index_once)
    sub.add_parser("discover-once", help="enrol boxes that registered themselves in S3").set_defaults(
        fn=cmd_discover_once)
    mo = sub.add_parser("media-once")
    mo.add_argument("--limit", type=int, default=50)
    mo.set_defaults(fn=cmd_media_once)
    mr = sub.add_parser("media-retry", help="retry clips whose media failed (all, or one event)")
    mr.add_argument("--event", type=int, default=None, help="only this event id")
    mr.set_defaults(fn=cmd_media_retry)
    r = sub.add_parser("redact-backfill")
    r.add_argument("--all", action="store_true", help="recompute every event, not only missing ones")
    r.set_defaults(fn=cmd_redact_backfill)
    d = sub.add_parser("export-download", help="download a training export, ready to train on")
    d.add_argument("export_id", type=int)
    d.add_argument("--dest", required=True, help="local folder to download into")
    d.set_defaults(fn=cmd_export_download)
    s = sub.add_parser("serve")
    s.add_argument("--port", type=int, default=8600)
    s.set_defaults(fn=cmd_serve)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.fn(args)
    except ConfigError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
