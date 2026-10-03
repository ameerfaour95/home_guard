from __future__ import annotations

from datetime import datetime, timezone

from fastapi import FastAPI
from sqlalchemy.orm import sessionmaker

from .db import make_engine
from .routes import audit, auth, customers, events, fleet, media, studio
from .settings import Settings


def create_app(settings: Settings, s3=None, init_db: bool = True) -> FastAPI:
    app = FastAPI(title="Home Guard Admin Center", version="1")
    app.state.settings = settings
    app.state.s3 = s3
    app.state.clock = lambda: datetime.now(timezone.utc)  # tests inject a fixed clock
    app.state.engine = make_engine(settings.db_url)  # lazy: connects on first use
    app.state.sessionmaker = sessionmaker(app.state.engine, expire_on_commit=False)
    if init_db:
        from .manage import run_migrations

        run_migrations(settings.db_url)
    for module in (auth, fleet, customers, events, media, studio, audit):
        app.include_router(module.router, prefix="/v1")
    return app
