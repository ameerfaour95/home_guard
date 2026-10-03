from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import FastAPI
from sqlalchemy.orm import sessionmaker

from .db import make_engine
from .routes import audit, auth, customers, events, fleet, media, studio
from .settings import Settings

log = logging.getLogger(__name__)


def sweep_exports(app: FastAPI) -> int:
    """Startup: exports a previous process left queued or running for over an hour are marked failed."""
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


@asynccontextmanager
async def _lifespan(app: FastAPI):
    sweep_exports(app)
    running = None
    if app.state.settings.run_loops and app.state.s3 is not None:
        from .loops import build_loops

        running = build_loops(app.state.sessionmaker, app.state.s3)
        running.start()
    try:
        yield
    finally:
        if running is not None:
            running.stop(5.0)


def create_app(settings: Settings, s3=None, init_db: bool = True) -> FastAPI:
    app = FastAPI(title="Home Guard Admin Center", version="1", lifespan=_lifespan)
    app.state.settings = settings
    app.state.s3 = s3
    app.state.clock = lambda: datetime.now(timezone.utc)  # tests inject a fixed clock
    app.state.engine = make_engine(settings.db_url)  # lazy: connects on first use
    app.state.sessionmaker = sessionmaker(app.state.engine, expire_on_commit=False)
    app.state.export_runner = studio.default_export_runner  # starts a thread; tests inject a synchronous runner
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

    settings = Settings.from_env()
    s3 = S3(boto3.client("s3", region_name=settings.region), settings.bucket)
    return create_app(settings, s3=s3, init_db=False)
