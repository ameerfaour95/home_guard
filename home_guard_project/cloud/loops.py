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
from typing import Callable, Optional

from sqlalchemy import text
from sqlalchemy.orm import sessionmaker  # noqa: F401  (re-exported for callers)

log = logging.getLogger(__name__)

LOCK_BASE = 0x48470000  # "HG" namespace for advisory-lock keys
INDEXER_LOCK, MEDIA_LOCK, NOTICES_LOCK = LOCK_BASE + 1, LOCK_BASE + 2, LOCK_BASE + 3
FULL_SCAN_EVERY = 30 * 60  # seconds, handled inside the indexer job


class Loop:
    def __init__(self, name: str, interval: float, job: Callable, sm, s3, lock_key: int, jitter: float = 0.1):
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
        conn = self.sm.kw["bind"].connect()
        try:
            got = conn.execute(text("SELECT pg_try_advisory_lock(:k)"), {"k": self.lock_key}).scalar()
            conn.commit()
            if not got:
                self.skipped += 1
                return False
            try:
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
            finally:
                try:
                    conn.execute(text("SELECT pg_advisory_unlock(:k)"), {"k": self.lock_key})
                    conn.commit()
                except Exception:  # noqa: BLE001 -- a broken connection drops its session lock when closed
                    conn.invalidate()
            return True
        finally:
            conn.close()


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


def _index_job_factory(clock=time.monotonic):
    state = {"last_full": clock()}

    def job(session, s3):
        from . import indexer

        full = clock() - state["last_full"] >= FULL_SCAN_EVERY
        indexer.index_all(session, s3, full_scan=full)
        if full:
            state["last_full"] = clock()

    return job


def _media_job(session, s3):
    from . import media

    media.process_pending(session, s3, limit=50)


def _notices_job(session, s3):
    from . import audit

    audit.retry_pending_notices(session, s3)


def build_loops(sm, s3) -> Loops:
    return Loops([
        Loop("indexer", 120, _index_job_factory(), sm, s3, INDEXER_LOCK),
        Loop("media", 60, _media_job, sm, s3, MEDIA_LOCK),
        Loop("notices", 300, _notices_job, sm, s3, NOTICES_LOCK),
    ])
