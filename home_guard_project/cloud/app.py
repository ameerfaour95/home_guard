from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import FastAPI
from sqlalchemy.orm import sessionmaker

from .db import make_engine
from .routes import audit, auth, customers, events, fleet, media, studio
from .settings import Settings

log = logging.getLogger(__name__)

LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s %(message)s"


def configure_logging() -> None:
    """Server logging: INFO with timestamps on the root logger, so loop failures ("loop indexer iteration failed")
    and progress appear in the server log. Idempotent. Nothing here logs presigned URLs or secrets."""
    root = logging.getLogger()
    if not any(getattr(h, "_home_guard", False) for h in root.handlers):
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(LOG_FORMAT))
        handler._home_guard = True
        root.addHandler(handler)
    root.setLevel(logging.INFO)


def sweep_exports(app: FastAPI) -> int:
    """Startup: exports whose worker is gone (no heartbeat for 5 minutes, or queued for 30) are marked failed;
    the periodic exports loop keeps doing this while the API runs."""
    from .studio import sweep_stale_exports

    try:
        session = app.state.sessionmaker()
        try:
            n = sweep_stale_exports(session, app.state.clock())
            session.commit()
            return n
        finally:
            session.close()
    except Exception as e:  # noqa: BLE001 -- a database that is not reachable yet must not stop the API
        log.warning("startup export sweep skipped: %s", type(e).__name__)
        return 0


EXPORT_WORKERS = 2  # export builds running at once; more requests wait in the executor's queue


def _export_executor() -> ThreadPoolExecutor:
    return ThreadPoolExecutor(max_workers=EXPORT_WORKERS, thread_name_prefix="training-export")


@asynccontextmanager
async def _lifespan(app: FastAPI):
    if getattr(app.state.export_executor, "_shutdown", False):  # a restart of the same app (tests)
        app.state.export_executor = _export_executor()
    sweep_exports(app)
    running = None
    if app.state.settings.run_loops and app.state.s3 is not None:
        from .loops import build_loops

        running = build_loops(app.state.sessionmaker, app.state.s3, engine=app.state.engine)
        running.start()
    try:
        yield
    finally:
        if running is not None:
            running.stop(5.0)
        # queued jobs are dropped (their exports stay queued and the sweep fails them); running ones finish
        app.state.export_executor.shutdown(wait=False, cancel_futures=True)


def create_app(settings: Settings, s3=None, init_db: bool = True) -> FastAPI:
    app = FastAPI(title="Home Guard Admin Center", version="1", lifespan=_lifespan)
    app.state.settings = settings
    app.state.s3 = s3
    app.state.clock = lambda: datetime.now(timezone.utc)  # tests inject a fixed clock
    app.state.engine = make_engine(settings.db_url)  # lazy: connects on first use
    app.state.sessionmaker = sessionmaker(app.state.engine, expire_on_commit=False)
    app.state.export_executor = _export_executor()
    app.state.export_runner = None  # None: the bounded executor; tests inject a synchronous runner
    if init_db:
        from .manage import run_migrations

        run_migrations(settings.db_url)
    for module in (auth, fleet, customers, events, media, studio, audit):
        app.include_router(module.router, prefix="/v1")
    return app


def create_app_from_env() -> FastAPI:
    """uvicorn factory: settings from HG_CLOUD_*, S3 from the ambient AWS credentials."""
    import boto3

    from .s3 import S3

    configure_logging()
    settings = Settings.from_env()
    s3 = S3(boto3.client("s3", region_name=settings.region), settings.bucket)
    return create_app(settings, s3=s3, init_db=False)
