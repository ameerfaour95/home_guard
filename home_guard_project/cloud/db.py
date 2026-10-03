from __future__ import annotations

from contextlib import contextmanager

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker


def make_engine(url: str) -> Engine:
    if url.startswith("postgresql://"):
        url = "postgresql+psycopg://" + url[len("postgresql://"):]
    if url.startswith("sqlite"):  # tests only: SQLite pools take no size arguments
        return create_engine(url, pool_pre_ping=True)
    return create_engine(url, pool_pre_ping=True, pool_size=10, max_overflow=10)


@contextmanager
def session_scope(engine: Engine):
    """Commit on success, roll back on error."""
    session: Session = sessionmaker(engine, expire_on_commit=False)()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
