"""Proof of every action the assistant takes.

A tool that sends, pauses, turns off or saves something writes a receipt before
it returns, and the reply's confirmations are rendered from receipts only - so
"sent" or "turned off" can never appear unless it happened. Receipts are kept
as append-only JSON lines, one file per day; an update (a camera change that
the restarted box confirmed) is a new line with the same id. The idempotency
key stops a retried tool call from acting twice.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import math
import os
import threading
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

log = logging.getLogger("box.brain.receipts")

DONE = "done"
REQUESTED = "requested"
FAILED = "failed"
UNDONE = "undone"

ACTING_TOOLS = frozenset({
    "send_media", "check_camera", "record_clip", "pause_alerts", "resume_alerts",
    "set_camera_active", "record_verdict", "set_alias", "change_setting", "set_alert_types", "set_sensitivity",
    "house_state", "house_expect", "house_cancel", "mark_known", "camera_fact",
})


@dataclass
class Receipt:
    id: str
    turn: str
    tool: str
    status: str
    target: str = ""
    detail: Dict[str, Any] = field(default_factory=dict)
    reason: str = ""
    ts: float = 0.0
    key: str = ""

    def summary(self) -> str:
        text = f"{self.id} {self.tool}{' ' + self.target if self.target else ''} {self.status}"
        return f"{text} ({self.reason})" if self.reason else text

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ReceiptBook:
    """Issues and keeps receipts. Never raises: a disk error is logged and the receipt still returned."""

    def __init__(self, directory: str, now: Callable[[], float] = time.time) -> None:
        self._dir = directory
        self._now = now
        self._lock = threading.Lock()
        self._by_key: Dict[str, Receipt] = {}
        self._counters: Dict[str, int] = {}

    def find(self, key: str) -> Optional[Receipt]:
        return self._by_key.get(key) if key else None

    def issue(self, turn: str, tool: str, status: str, target: str = "", detail: Optional[Dict[str, Any]] = None,
              reason: str = "", key: str = "") -> Receipt:
        with self._lock:
            if key and key in self._by_key:
                return self._by_key[key]
            n = self._counters.get(turn, 0) + 1
            self._counters[turn] = n
            receipt = Receipt(id=f"R{n}", turn=turn, tool=tool, status=status, target=target,
                              detail=dict(detail or {}), reason=reason, ts=self._now(), key=key)
            if key:
                self._by_key[key] = receipt
            self._append(receipt)
            return receipt

    def update(self, receipt: Receipt, status: str, detail: Optional[Dict[str, Any]] = None,
               reason: Optional[str] = None) -> Receipt:
        with self._lock:
            receipt.status = status
            if detail:
                receipt.detail.update(detail)
            if reason is not None:
                receipt.reason = reason
            receipt.ts = self._now()
            self._append(receipt)
            return receipt

    def open_receipts(self, tool: str, days: int = 2) -> List[Receipt]:
        """Receipts of *tool* still ``requested`` in the last *days* day files (read from disk: survives a restart)."""
        latest: Dict[Tuple[str, str], Receipt] = {}
        today = dt.datetime.fromtimestamp(self._now()).date()
        for back in range(days - 1, -1, -1):
            path = self._path(today - dt.timedelta(days=back))
            try:
                with open(path, encoding="utf-8", errors="replace") as f:
                    for line in f:
                        try:
                            r = Receipt(**json.loads(line))
                        except (ValueError, TypeError):
                            continue
                        latest[(r.turn, r.id)] = r
            except OSError:
                continue
        return [r for r in latest.values() if r.tool == tool and r.status == REQUESTED]

    def turn_receipts(self, turn: str, days: int = 2) -> List[Receipt]:
        """Latest states for this turn on disk. Skip malformed records; never raises."""
        latest: Dict[str, Receipt] = {}
        warned = False
        try:
            if not isinstance(turn, str) or type(days) is not int or days < 0:
                raise ValueError("invalid turn or day count")
            today = dt.datetime.fromtimestamp(self._now()).date()
            for back in range(days - 1, -1, -1):
                try:
                    with open(self._path(today - dt.timedelta(days=back)), encoding="utf-8",
                              errors="replace") as f:
                        for line in f:
                            try:
                                r = Receipt(**json.loads(line))
                                if (not all(isinstance(v, str) for v in
                                            (r.id, r.turn, r.tool, r.status, r.target, r.reason, r.key))
                                        or not r.id.startswith("R") or not r.id[1:].isdigit()
                                        or not isinstance(r.detail, dict) or not math.isfinite(float(r.ts))):
                                    raise ValueError("invalid receipt")
                                json.dumps(r.to_dict(), allow_nan=False)
                                if r.turn == turn:
                                    latest[r.id] = r
                            except (ValueError, TypeError, OverflowError):
                                if not warned:
                                    log.warning("Skipped malformed turn receipt")
                                    warned = True
                except OSError:
                    continue
            return sorted(latest.values(), key=lambda r: int(r.id[1:]))
        except Exception as exc:
            log.warning("Could not read turn receipts: %s", exc)
            return []

    def later(self, turn: str, receipt_id: str, days: int = 2) -> List[Receipt]:
        """Latest states of the receipts first written after receipt *receipt_id* of *turn* (file order, so
        equal timestamps still sort right). Empty when that receipt is not on disk. Never raises."""
        try:
            if not isinstance(turn, str) or not isinstance(receipt_id, str) or type(days) is not int or days < 0:
                raise ValueError("invalid turn, receipt or day count")
            order: List[Tuple[str, str]] = []
            latest: Dict[Tuple[str, str], Receipt] = {}
            today = dt.datetime.fromtimestamp(self._now()).date()
            for back in range(days - 1, -1, -1):
                try:
                    with open(self._path(today - dt.timedelta(days=back)), encoding="utf-8",
                              errors="replace") as f:
                        for line in f:
                            try:
                                r = Receipt(**json.loads(line))
                                if (not all(isinstance(v, str) for v in (r.id, r.turn, r.tool, r.status, r.target))
                                        or not isinstance(r.detail, dict)):
                                    raise ValueError("invalid receipt")
                            except (ValueError, TypeError):
                                continue
                            key = (r.turn, r.id)
                            if key not in latest:
                                order.append(key)
                            latest[key] = r
                except OSError:
                    continue
            mine = (turn, receipt_id)
            if mine not in latest:
                return []
            return [latest[k] for k in order[order.index(mine) + 1:]]
        except Exception as exc:
            log.warning("Could not read later receipts: %s", exc)
            return []

    def _path(self, day: dt.date) -> str:
        return os.path.join(self._dir, f"{day.isoformat()}.jsonl")

    def _append(self, receipt: Receipt) -> None:
        try:
            os.makedirs(self._dir, exist_ok=True)
            path = self._path(dt.datetime.fromtimestamp(receipt.ts).date())
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(receipt.to_dict(), ensure_ascii=False, default=str) + "\n")
        except Exception as exc:  # never raise: OSError, TypeError, ValueError, OverflowError
            log.warning("Receipt %s not written to disk: %s", receipt.summary(), exc)
