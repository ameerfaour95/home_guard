from __future__ import annotations

from fastapi import FastAPI

from .routes import audit, auth, customers, events, fleet, media, studio
from .settings import Settings


def create_app(settings: Settings, s3=None, init_db: bool = True) -> FastAPI:
    app = FastAPI(title="Home Guard Admin Center", version="1")
    app.state.settings = settings
    app.state.s3 = s3
    if init_db:
        pass  # DB engine/session wiring arrives with the DB task
    for module in (auth, fleet, customers, events, media, studio, audit):
        app.include_router(module.router, prefix="/v1")
    return app
