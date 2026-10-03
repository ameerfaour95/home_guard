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


def run_migrations(db_url: str) -> None:
    from alembic import command
    from alembic.config import Config

    cfg = Config(str(HERE / "alembic.ini"))
    cfg.set_main_option("script_location", str(HERE / "migrations").replace("%", "%%"))
    cfg.attributes["url"] = db_url
    command.upgrade(cfg, "head")


class ConfigError(Exception):
    pass


def _db_url() -> str:
    """One resolver for every command: HG_CLOUD_DB_URL, else the full settings."""
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
        print(f"enrolled device_id={dev.device_id} site={dev.site} customer={cust.name} (id {cust.id})")
    return 0


def cmd_index_once(args) -> int:
    indexer = _optional("indexer")  # Task 8
    if indexer is None:
        print("index-once: not available yet")
        return 0
    return indexer.run_once(_engine())


def cmd_serve(args) -> int:
    import uvicorn

    if _optional("app") is None:
        print("serve: not available yet")
        return 0
    uvicorn.run("home_guard_project.cloud.app:create_app", factory=True, host="0.0.0.0", port=args.port)
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
    sub.add_parser("index-once").set_defaults(fn=cmd_index_once)
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
