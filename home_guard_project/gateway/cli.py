"""The gateway's command line: run the server, and issue or revoke box tokens.

    python -m home_guard_project.gateway serve       [--config gateway.yaml]
    python -m home_guard_project.gateway add-box     house2 [--cap 0.50] [--note "Faour family"] [--rotate]
    python -m home_guard_project.gateway revoke-box  house2
    python -m home_guard_project.gateway set-cap     house2 0.75        (or "default": the config's cap)
    python -m home_guard_project.gateway list-boxes
    python -m home_guard_project.gateway report      [--days 7] [--box house2]
    python -m home_guard_project.gateway prune       --keep-days 90
    python -m home_guard_project.gateway new-admin-token

The config path comes from ``--config``, else ``HOMEGUARD_GATEWAY_CONFIG``, else
``gateway.yaml``. A token is printed once and only its hash is stored: copy it to the
box's ``api_key.env`` as ``HOMEGUARD_BOX_TOKEN`` right away.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import sys
from typing import List, Optional, TextIO

from .config import ConfigError, load_config
from .store import Store, StoreError, new_token

log = logging.getLogger("gateway.cli")

_BOX_RE = re.compile(r"^[a-z0-9_\-]{1,64}$")


def _cap(value: str) -> Optional[float]:
    if value.strip().lower() in ("default", "none"):
        return None
    cap = float(value)
    if cap < 0:
        raise argparse.ArgumentTypeError("a cap must not be negative")
    return cap


def _box(value: str) -> str:
    if not _BOX_RE.match(value):
        raise argparse.ArgumentTypeError("a box id is lowercase letters, digits, '_' or '-' (the box's site name)")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m home_guard_project.gateway",
                                     description="Home Guard API gateway: model calls for the boxes.")
    parser.add_argument("--config", default=os.environ.get("HOMEGUARD_GATEWAY_CONFIG", "gateway.yaml"))
    sub = parser.add_subparsers(dest="command", required=True)
    serve = sub.add_parser("serve", help="Run the server.")
    serve.add_argument("--host")
    serve.add_argument("--port", type=int)
    add = sub.add_parser("add-box", help="Issue a token for a box (printed once).")
    add.add_argument("box", type=_box)
    add.add_argument("--cap", type=_cap, default=None, help="This box's daily $ cap (default: the config's).")
    add.add_argument("--note", default="")
    add.add_argument("--rotate", action="store_true", help="Replace the box's token (the old one stops working).")
    rev = sub.add_parser("revoke-box", help="Stop a box's token from working.")
    rev.add_argument("box", type=_box)
    cap = sub.add_parser("set-cap", help="Set a box's own daily $ cap ('default' to use the config's).")
    cap.add_argument("box", type=_box)
    cap.add_argument("cap", type=_cap)
    sub.add_parser("list-boxes", help="Every box, its cap and today's spend.")
    rep = sub.add_parser("report", help="Cost per box and UTC day.")
    rep.add_argument("--days", type=int, default=7)
    rep.add_argument("--box", type=_box)
    prune = sub.add_parser("prune", help="Delete call rows older than N days (daily totals stay).")
    prune.add_argument("--keep-days", type=int, required=True)
    sub.add_parser("new-admin-token", help="Make an admin token; prints it and the hash for the config.")
    return parser


def main(argv: Optional[List[str]] = None, out: TextIO = sys.stdout) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s %(message)s")
    if args.command == "new-admin-token":
        token = new_token().replace("hgb_", "hga_", 1)
        print(f"admin token (keep it; shown once): {token}", file=out)
        print(f"admin_token_sha256: {hashlib.sha256(token.encode('utf-8')).hexdigest()}", file=out)
        return 0
    try:
        cfg = load_config(args.config)
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2
    store = Store(cfg.db_path)
    try:
        if args.command == "serve":
            return _serve(cfg, store, args.host, args.port)
        if args.command == "add-box":
            token = store.add_box(args.box, args.cap, args.note, rotate=args.rotate)
            print(f"HOMEGUARD_BOX_TOKEN={token}", file=out)
            print(f"(shown once; put it in {args.box}'s api_key.env)", file=out)
        elif args.command == "revoke-box":
            store.revoke_box(args.box)
            print(f"{args.box}: revoked", file=out)
        elif args.command == "set-cap":
            store.set_cap(args.box, args.cap)
            print(f"{args.box}: daily cap {'from the config' if args.cap is None else f'${args.cap:.2f}'}", file=out)
        elif args.command == "list-boxes":
            for b in store.list_boxes():
                state = f"revoked {b['revoked_at']}" if b["revoked_at"] else "active"
                own = "config" if b["daily_cap_usd"] is None else f"${b['daily_cap_usd']:.2f}"
                print(f"{b['box_id']:<24} {state:<30} cap={own:<8} today=${b['spent_usd']:.4f} {b['note']}", file=out)
        elif args.command == "report":
            print(json.dumps(store.costs(args.days, args.box), indent=1), file=out)
        elif args.command == "prune":
            print(f"deleted {store.prune(args.keep_days)} call rows", file=out)
    except StoreError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    finally:
        if args.command != "serve":
            store.close()
    return 0


def _serve(cfg, store: Store, host: Optional[str], port: Optional[int]) -> int:
    from .server import Gateway, make_server  # noqa: PLC0415 - the token commands do not need httpx

    for alias in cfg.aliases.values():
        for up in alias.upstreams:
            try:
                up.resolve(cfg.env)
            except Exception as exc:  # noqa: BLE001 - warn now, not at the first box's call
                log.warning("alias %s: upstream %s is not usable yet: %s", alias.name, up.name, exc)
    if not cfg.admin_token_sha256:
        log.warning("no admin_token_sha256: the /admin endpoints are off")
    server = make_server(Gateway(cfg, store), host, port)
    log.info("gateway listening on %s:%d with %d aliases", *server.server_address[:2], len(cfg.aliases))
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        store.close()
    return 0
