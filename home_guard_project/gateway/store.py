"""The gateway's SQLite file: box tokens (hashes only), every upstream call, and each box's daily spend.

``calls`` has one row per upstream attempt (failed ones too, at no cost), so a
provider's error rate can be read from it. ``spend`` is the running total per
UTC day and box that the caps check, one row read per request.
"""
from __future__ import annotations

import hashlib
import secrets
import sqlite3
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

TOKEN_PREFIX = "hgb_"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS boxes (
    box_id        TEXT PRIMARY KEY,
    token_sha256  TEXT NOT NULL UNIQUE,
    created_at    TEXT NOT NULL,
    revoked_at    TEXT,
    daily_cap_usd REAL,
    note          TEXT NOT NULL DEFAULT ''
);
CREATE TABLE IF NOT EXISTS calls (
    id                INTEGER PRIMARY KEY,
    ts                REAL NOT NULL,
    day               TEXT NOT NULL,
    box_id            TEXT NOT NULL,
    endpoint          TEXT NOT NULL,
    alias             TEXT NOT NULL,
    provider          TEXT NOT NULL,
    model             TEXT NOT NULL,
    prompt_tokens     INTEGER NOT NULL DEFAULT 0,
    completion_tokens INTEGER NOT NULL DEFAULT 0,
    cost_usd          REAL NOT NULL DEFAULT 0,
    latency_ms        INTEGER NOT NULL DEFAULT 0,
    status            INTEGER NOT NULL,
    error             TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS calls_day_box ON calls (day, box_id);
CREATE TABLE IF NOT EXISTS spend (
    day      TEXT NOT NULL,
    box_id   TEXT NOT NULL,
    cost_usd REAL NOT NULL DEFAULT 0,
    PRIMARY KEY (day, box_id)
);
"""


class StoreError(Exception):
    """A box is unknown, or already has a token."""


def hash_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def new_token() -> str:
    return TOKEN_PREFIX + secrets.token_urlsafe(32)


def utc_day(ts: Optional[float] = None) -> str:
    return datetime.fromtimestamp(time.time() if ts is None else ts, timezone.utc).strftime("%Y-%m-%d")


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


@dataclass(frozen=True)
class Box:
    box_id: str
    daily_cap_usd: Optional[float]


@dataclass
class Call:
    box_id: str
    endpoint: str
    alias: str
    provider: str
    model: str
    status: int
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    latency_ms: int = 0
    error: str = ""
    ts: float = 0.0


class Store:
    """One connection shared by the server's threads, behind a lock (a few calls a second at most)."""

    def __init__(self, path: str) -> None:
        self._db = sqlite3.connect(path, check_same_thread=False, timeout=30)
        self._db.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        with self._lock:
            if path != ":memory:":
                self._db.execute("PRAGMA journal_mode=WAL")
            self._db.executescript(_SCHEMA)
            self._db.commit()

    def close(self) -> None:
        with self._lock:
            self._db.close()

    # --- tokens -------------------------------------------------------------------------------
    def add_box(self, box_id: str, daily_cap_usd: Optional[float] = None, note: str = "",
                rotate: bool = False) -> str:
        """A new token for *box_id*, returned once (only its hash is kept). An existing box needs
        *rotate*, which replaces its token (and un-revokes it)."""
        token = new_token()
        with self._lock, self._db:
            row = self._db.execute("SELECT box_id FROM boxes WHERE box_id = ?", (box_id,)).fetchone()
            if row is not None and not rotate:
                raise StoreError(f"box {box_id!r} already has a token; use --rotate to replace it")
            if row is None:
                self._db.execute("INSERT INTO boxes (box_id, token_sha256, created_at, daily_cap_usd, note) "
                                 "VALUES (?, ?, ?, ?, ?)", (box_id, hash_token(token), _now_iso(), daily_cap_usd, note))
            else:
                self._db.execute("UPDATE boxes SET token_sha256 = ?, revoked_at = NULL WHERE box_id = ?",
                                 (hash_token(token), box_id))
        return token

    def revoke_box(self, box_id: str) -> None:
        with self._lock, self._db:
            cur = self._db.execute("UPDATE boxes SET revoked_at = ? WHERE box_id = ? AND revoked_at IS NULL",
                                   (_now_iso(), box_id))
            if cur.rowcount == 0:
                raise StoreError(f"no active box {box_id!r}")

    def set_cap(self, box_id: str, daily_cap_usd: Optional[float]) -> None:
        with self._lock, self._db:
            cur = self._db.execute("UPDATE boxes SET daily_cap_usd = ? WHERE box_id = ?", (daily_cap_usd, box_id))
            if cur.rowcount == 0:
                raise StoreError(f"no box {box_id!r}")

    def box_for_token(self, token: str) -> Optional[Box]:
        if not token:
            return None
        with self._lock:
            row = self._db.execute("SELECT box_id, daily_cap_usd FROM boxes WHERE token_sha256 = ? "
                                   "AND revoked_at IS NULL", (hash_token(token),)).fetchone()
        return Box(row["box_id"], row["daily_cap_usd"]) if row else None

    def list_boxes(self, day: Optional[str] = None) -> List[Dict[str, Any]]:
        """Every box (never its hash) with what it spent on *day* (today, UTC)."""
        with self._lock:
            rows = self._db.execute(
                "SELECT b.box_id, b.created_at, b.revoked_at, b.daily_cap_usd, b.note, "
                "COALESCE(s.cost_usd, 0) AS spent_usd FROM boxes b "
                "LEFT JOIN spend s ON s.box_id = b.box_id AND s.day = ? ORDER BY b.box_id",
                (day or utc_day(),)).fetchall()
        return [dict(r) for r in rows]

    # --- metering -----------------------------------------------------------------------------
    def record(self, call: Call) -> None:
        ts = call.ts or time.time()
        day = utc_day(ts)
        with self._lock, self._db:
            self._db.execute(
                "INSERT INTO calls (ts, day, box_id, endpoint, alias, provider, model, prompt_tokens, "
                "completion_tokens, cost_usd, latency_ms, status, error) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (ts, day, call.box_id, call.endpoint, call.alias, call.provider, call.model, call.prompt_tokens,
                 call.completion_tokens, call.cost_usd, call.latency_ms, call.status, call.error[:300]))
            if call.cost_usd:
                self._db.execute(
                    "INSERT INTO spend (day, box_id, cost_usd) VALUES (?, ?, ?) "
                    "ON CONFLICT (day, box_id) DO UPDATE SET cost_usd = cost_usd + excluded.cost_usd",
                    (day, call.box_id, call.cost_usd))

    def spent(self, box_id: str, day: Optional[str] = None) -> float:
        with self._lock:
            row = self._db.execute("SELECT cost_usd FROM spend WHERE day = ? AND box_id = ?",
                                   (day or utc_day(), box_id)).fetchone()
        return float(row["cost_usd"]) if row else 0.0

    def fleet_spent(self, day: Optional[str] = None) -> float:
        with self._lock:
            row = self._db.execute("SELECT COALESCE(SUM(cost_usd), 0) AS total FROM spend WHERE day = ?",
                                   (day or utc_day(),)).fetchone()
        return float(row["total"])

    def costs(self, days: int = 7, box_id: Optional[str] = None, today: Optional[str] = None) -> List[Dict[str, Any]]:
        """Per box and UTC day, newest first: calls answered, failed attempts, tokens, dollars, mean latency."""
        end = datetime.strptime(today or utc_day(), "%Y-%m-%d").replace(tzinfo=timezone.utc)
        first = utc_day(end.timestamp() - (max(1, days) - 1) * 86400)
        sql = ("SELECT day, box_id, SUM(status = 200) AS calls, SUM(status <> 200) AS failed, "
               "SUM(prompt_tokens) AS prompt_tokens, SUM(completion_tokens) AS completion_tokens, "
               "ROUND(SUM(cost_usd), 6) AS cost_usd, "
               "CAST(AVG(CASE WHEN status = 200 THEN latency_ms END) AS INTEGER) AS mean_latency_ms "
               "FROM calls WHERE day >= ?")
        args: List[Any] = [first]
        if box_id:
            sql += " AND box_id = ?"
            args.append(box_id)
        sql += " GROUP BY day, box_id ORDER BY day DESC, cost_usd DESC"
        with self._lock:
            return [dict(r) for r in self._db.execute(sql, args).fetchall()]

    def prune(self, keep_days: int) -> int:
        """Drop call rows older than *keep_days* (the daily totals stay). Returns how many."""
        cutoff = utc_day(time.time() - keep_days * 86400)
        with self._lock, self._db:
            return self._db.execute("DELETE FROM calls WHERE day < ?", (cutoff,)).rowcount
