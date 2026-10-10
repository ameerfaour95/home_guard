"""Stop, start and restart the box's program without administrator rights.

The program runs from a scheduled task with elevated rights, so the app on the
desktop (an ordinary process) can neither change the task nor end its
processes. It asks instead, with two flag files that the runner
(run_collector.sh) looks at every couple of seconds:

    logs/collector.stop      while it exists, the runner keeps the program stopped
    logs/collector.restart   the runner restarts the program once and deletes the flag

A stop lasts until ``start``: it survives a reboot, and the heartbeat's
self-heal leaves a stopped box alone.

A stop also records its time in ``logs/owner_stop.json`` (system_notices), which
the start does not clear: the next inference start can then tell a box the owner
stopped from one that died, and says nothing about the owner's own stop
(owner decision 2026-10-10).
"""

from __future__ import annotations

import os

from .boxconfig import LOG_DIR

STOP_FLAG = "collector.stop"
RESTART_FLAG = "collector.restart"


def _flag(name: str, log_dir: str) -> str:
    return os.path.join(log_dir, name)


def _touch(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="ascii"):
        pass


def stop(log_dir: str = LOG_DIR) -> None:
    """Ask the runner to stop the program and keep it stopped, and record when (system_notices)."""
    from .system_notices import record_owner_stop  # noqa: PLC0415

    record_owner_stop(log_dir)
    _touch(_flag(STOP_FLAG, log_dir))


def start(log_dir: str = LOG_DIR) -> None:
    """Let the runner start the program again."""
    for name in (STOP_FLAG, RESTART_FLAG):
        try:
            os.remove(_flag(name, log_dir))
        except FileNotFoundError:
            pass


def is_stopped(log_dir: str = LOG_DIR) -> bool:
    return os.path.exists(_flag(STOP_FLAG, log_dir))


def request_restart(log_dir: str = LOG_DIR) -> None:
    """Ask the runner to restart the program once, so it picks up changed settings. No effect while stopped."""
    if not is_stopped(log_dir):
        _touch(_flag(RESTART_FLAG, log_dir))
