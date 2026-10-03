"""Background loops: each kind in its own daemon thread with its own DB sessions.

A Postgres session advisory lock per loop kind keeps two API processes from running the same job at once.
Exceptions are logged and never kill a loop; stop() returns within its bound (a job in flight is abandoned
as a daemon thread, its transaction rolls back).
"""
from __future__ import annotations

import logging
import random
import threading
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Callable, Iterator, Optional

from sqlalchemy import text
from sqlalchemy.orm import sessionmaker  # noqa: F401  (re-exported for callers)

log = logging.getLogger(__name__)

LOCK_BASE = 0x48470000  # "HG" namespace for advisory-lock keys
INDEXER_LOCK, MEDIA_LOCK, NOTICES_LOCK = LOCK_BASE + 1, LOCK_BASE + 2, LOCK_BASE + 3
EXPORTS_LOCK = LOCK_BASE + 4
DISCOVERY_LOCK = LOCK_BASE + 5
LOCKS = {"indexer": INDEXER_LOCK, "media": MEDIA_LOCK, "notices": NOTICES_LOCK, "exports": EXPORTS_LOCK,
         "discovery": DISCOVERY_LOCK}


@contextmanager
def loop_lock(engine, kind) -> Iterator[bool]:
    """The session advisory lock of one loop kind ("indexer", "media", ... or a raw key), held on its own
    connection for the duration of the block; yields False (and holds nothing) when someone else has it.

    The background loops and the management CLI (`index-once`, `media-once`) take the same lock, so a manual run
    never races the server's loop."""
    key = LOCKS[kind] if isinstance(kind, str) else int(kind)
    conn = engine.connect()
    try:
        got = bool(conn.execute(text("SELECT pg_try_advisory_lock(:k)"), {"k": key}).scalar())
        conn.commit()
        try:
            yield got
        finally:
            if got:
                try:
                    conn.execute(text("SELECT pg_advisory_unlock(:k)"), {"k": key})
                    conn.commit()
                except Exception:  # noqa: BLE001 -- a broken connection drops its session lock when closed
                    conn.invalidate()
    finally:
        conn.close()


class Loop:
    def __init__(self, name: str, interval: float, job: Callable, sm, s3, lock_key: int, jitter: float = 0.1,
                 engine=None):
        self.engine = engine if engine is not None else sm.kw["bind"]
        self.name, self.interval, self.job, self.sm, self.s3 = name, interval, job, sm, s3
        self.lock_key, self.jitter = lock_key, jitter
        self.runs = 0
        self.skipped = 0
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name=f"hg-loop-{self.name}", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout)

    def alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def _run(self) -> None:
        if self._stop.wait(random.uniform(0, min(self.interval, 5.0))):
            return
        while not self._stop.is_set():
            try:
                self.run_once()
            except Exception:  # noqa: BLE001
                log.exception("loop %s iteration failed", self.name)
            if self._stop.wait(self.interval * random.uniform(1 - self.jitter, 1 + self.jitter)):
                return

    def run_once(self) -> bool:
        """One locked attempt; False when another process holds the lock."""
        with loop_lock(self.engine, self.lock_key) as got:
            if not got:
                self.skipped += 1
                return False
            session = self.sm()
            try:
                self.job(session, self.s3)
                session.commit()
            except Exception:
                session.rollback()
                raise
            finally:
                session.close()
                self.runs += 1
            return True


class Loops:
    def __init__(self, loops: list[Loop]):
        self.loops = loops

    def start(self) -> None:
        for lp in self.loops:
            lp.start()

    def stop(self, timeout: float = 5.0) -> None:
        deadline = time.monotonic() + timeout
        for lp in self.loops:
            lp._stop.set()
        for lp in self.loops:
            lp.stop(max(0.0, deadline - time.monotonic()))


def _index_job(session, s3):
    from . import indexer

    indexer.index_all(session, s3)  # the indexer schedules its own full scans


def _media_job(session, s3):
    from . import media

    media.process_pending(session, s3, limit=50)


def _notices_job(session, s3):
    from . import audit

    audit.retry_pending_notices(session, s3)


def _exports_job(session, s3):
    from . import studio, tagging

    studio.sweep_stale_exports(session, datetime.now(timezone.utc))
    tagging.sweep_stale_publishes(session, datetime.now(timezone.utc))


def _discovery_job(session, s3):
    from . import discovery

    discovery.discover(session, s3)


def build_loops(sm, s3, engine=None) -> Loops:
    """The background loops; `engine` (the app's) defaults to the sessionmaker's bind."""
    from . import studio

    engine = engine if engine is not None else sm.kw["bind"]
    built = [
        Loop("indexer", 120, _index_job, sm, s3, INDEXER_LOCK, engine=engine),
        Loop("media", 60, _media_job, sm, s3, MEDIA_LOCK, engine=engine),
        Loop("notices", 300, _notices_job, sm, s3, NOTICES_LOCK, engine=engine),
        Loop("discovery", 300, _discovery_job, sm, s3, DISCOVERY_LOCK, engine=engine),
    ]
    if hasattr(studio, "sweep_stale_exports"):
        built.append(Loop("exports", 60, _exports_job, sm, s3, EXPORTS_LOCK, engine=engine))
    return Loops(built)
