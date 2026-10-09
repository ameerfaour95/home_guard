"""Push owner notices to the box over Tailscale SSH: the box never reads S3, so the fleet/ object is the cloud's
record and this is how the box learns of a notice.

The command is fleet_contract.notices.notice_ssh_argv (NOTICE_CLI, the one template). A notice waits in
delivery_state "pending" (audit.owner_notice sets it on every new or rewritten body) until the box answers:
- delivered: exit 0 and a JSON reply without "error" (the box replaces by id, so a rewrite re-pushes safely);
- failed: exit 1 with {"error": ...}, or a body over MAX_B64. Permanent for that body (delivery_body_sha): a later
  rewrite with a new body is a new attempt;
- anything else (ssh exit 255, a timeout, no tailscale host) stays pending. The next attempt waits for a newer
  heartbeat from the box (delivery_heartbeat_at), so an unreachable box is not hammered; after 7 days of waiting the
  notice is given up (gave_up);
- the box's SSH host key changed (it is pinned on first contact, notices.NOTICE_SSH_OPTIONS): failed, with
  last_delivery_error "host key changed for <host>", never a silent retry. The box's Fleet row warns
  (host_key_changed_boxes) until a delivery to it gets through again.
Every attempt and the give-up are audited ("notice_delivery"); the detail never holds the notice text. The
last failure's reason is kept on the row (last_delivery_error).

The target is the box's live row (boxes.current_device_for: box_id first, then hosts), so a notice about an old site
name reaches the box under its current name: its tailscale_host and ssh_user.
"""
from __future__ import annotations

import logging
import subprocess
from datetime import datetime, timedelta, timezone
from typing import Callable, Iterable, Optional

from sqlalchemy import select, text
from sqlalchemy.orm import Session

from home_guard_project.fleet_contract import notices

from . import audit
from .boxes import current_device_for
from .db import session_scope
from .models import Device, OwnerNotice

log = logging.getLogger(__name__)

GIVE_UP_AFTER = timedelta(days=7)
SSH_TIMEOUT = 60  # seconds for the whole call; ConnectTimeout=15 already ends an unreachable box sooner
ACTOR = "notice delivery"  # audit_log.staff_name of the attempts
ERROR_MAX = 200

# argv -> (exit code, or None when it timed out / could not start; stdout; stderr)
Runner = Callable[[list[str]], tuple[Optional[int], str, str]]


HOST_KEY_WARNING = ("Notices can't reach the box: its SSH host key changed. Check the box was reinstalled, then "
                    "reset its key (remove its line from ~/.homeguard/box_known_hosts)")


def ssh_run(argv: list[str]) -> tuple[Optional[int], str, str]:
    """The real runner: ssh with the laptop's box key (~/.ssh/homeguard_box), never prompting (BatchMode)."""
    try:  # the dedicated known_hosts file (notices.notice_known_hosts), made on first use
        hosts = notices.notice_known_hosts()
        hosts.parent.mkdir(parents=True, exist_ok=True)
        hosts.touch(exist_ok=True)
    except OSError:
        log.warning("could not create %s; ssh will try to create it", notices.notice_known_hosts())
    try:
        r = subprocess.run(argv, capture_output=True, timeout=SSH_TIMEOUT, stdin=subprocess.DEVNULL)
    except subprocess.TimeoutExpired:
        return None, "", "timed out"
    except OSError as e:  # no ssh on this machine
        return None, "", type(e).__name__
    return r.returncode, r.stdout.decode("utf-8", "replace"), r.stderr.decode("utf-8", "replace")


def _short(text: str) -> str:
    lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
    return lines[-1][:ERROR_MAX] if lines else ""


def _record(session: Session, row: OwnerNotice, box: Device, now: datetime, outcome: str, **detail) -> None:
    if outcome != "delivered":
        row.last_delivery_error = (f"{outcome}: {detail['error']}" if detail.get("error") else outcome)[:255]
    audit.record(session, None, "notice_delivery", target=f"owner_notice/{row.id}", reason=outcome,
                 customer_id=box.customer_id, device_id=box.device_id, ts=now, staff_name=ACTOR,
                 detail={"notice_id": row.id, "ok": outcome == "delivered", "outcome": outcome,
                         "attempt": row.delivery_attempts or 0,
                         **{k: v for k, v in detail.items() if v is not None and v != ""}})


def _due(row: OwnerNotice, box: Device) -> bool:
    """Never tried, or the box sent a heartbeat newer than the one seen at the last attempt."""
    if not row.delivery_attempts:
        return True
    return box.last_heartbeat_at is not None and (
        row.delivery_heartbeat_at is None or box.last_heartbeat_at > row.delivery_heartbeat_at)


def deliver_notice(session: Session, row: OwnerNotice, runner: Runner, now: datetime, force: bool = False) -> str:
    """Push one pending notice to its box (the one place NOTICE_CLI is sent). Returns the outcome: delivered |
    rejected | too_long | retry | no_target | gave_up | waiting (not due: no newer heartbeat since the last try)."""
    device = session.get(Device, row.device_pk)
    box = current_device_for(session, device)
    if row.delivery_since is not None and now - row.delivery_since > GIVE_UP_AFTER:
        row.delivery_state = "gave_up"
        _record(session, row, box, now, "gave_up", since=row.delivery_since.isoformat())
        return "gave_up"
    if not force and not _due(row, box):
        return "waiting"
    body = audit.notice_body(session, row, device)
    row.delivery_body_sha = audit.notice_hash(body)
    row.delivery_attempts = (row.delivery_attempts or 0) + 1
    row.delivery_heartbeat_at = box.last_heartbeat_at
    host = (box.tailscale_host or "").strip()
    if len(notices.notice_b64(body)) > notices.MAX_B64:
        row.delivery_state = "failed"
        _record(session, row, box, now, "too_long", size=len(notices.notice_b64(body)), limit=notices.MAX_B64)
        return "too_long"
    if not host:
        _record(session, row, box, now, "no_target", error="the box has no tailscale host")
        return "no_target"
    code, out, err = runner(notices.notice_ssh_argv(box.ssh_user, host, body))
    if notices.notice_host_key_changed(code, err):
        row.delivery_state = "failed"
        _record(session, row, box, now, "host_key_changed", host=host, exit=code,
                error=f"host key changed for {host}")
        row.last_delivery_error = f"host key changed for {host}"[:255]
        log.warning("owner notice %s: the SSH host key of %s changed; not retried", row.id, host)
        return "host_key_changed"
    outcome, reply = notices.notice_reply(code, out)
    error = str(reply.get("error", ""))[:ERROR_MAX]
    if outcome == "delivered":
        row.delivery_state = "delivered"
        _record(session, row, box, now, outcome, host=host, exit=code, result=str(reply.get("result", ""))[:32])
    elif outcome == "rejected":
        row.delivery_state = "failed"
        _record(session, row, box, now, outcome, host=host, exit=code, error=error)
    else:
        _record(session, row, box, now, outcome, host=host, exit=code, error=error or _short(err) or _short(out))
    return outcome


def deliver_pending(session: Session, now: Optional[datetime] = None, runner: Optional[Runner] = None,
                    notice_ids: Optional[Iterable[int]] = None) -> int:
    """Try the pending notices (all, or *notice_ids* right after a view: then without waiting for a heartbeat);
    returns how many were delivered. A box that did not answer is not tried again in the same pass."""
    now = now or datetime.now(timezone.utc)
    runner = runner or ssh_run
    q = select(OwnerNotice).where(OwnerNotice.delivery_state == "pending")
    if notice_ids is not None:
        q = q.where(OwnerNotice.id.in_(list(notice_ids)))
    unreachable: set[int] = set()
    done = 0
    for row in session.scalars(q.order_by(OwnerNotice.id).with_for_update(skip_locked=True)).all():
        box = current_device_for(session, session.get(Device, row.device_pk))
        if box.id in unreachable:
            continue
        outcome = deliver_notice(session, row, runner, now, force=notice_ids is not None)
        if outcome == "delivered":
            done += 1
        elif outcome in ("retry", "no_target", "host_key_changed"):
            unreachable.add(box.id)
        session.flush()
    return done


def deliver_now(engine, notice_ids: list[int], runner: Runner, clock: Callable[[], datetime]) -> None:
    """Background task after the staff request committed: push these notices now. Never raises (the loop retries)."""
    try:
        with session_scope(engine) as session:
            deliver_pending(session, clock(), runner, notice_ids=notice_ids)
    except Exception:  # noqa: BLE001
        log.exception("owner notices %s: immediate delivery failed; the notices loop retries", notice_ids)


def schedule(request, background_tasks, rows: Iterable[Optional[OwnerNotice]]) -> None:
    """After the request's commit, push the notices of *rows* that wait for delivery. Only a server that runs the
    loops (HG_CLOUD_RUN_LOOPS: the one on the tailnet with the box key) has app.state.notice_runner; elsewhere the
    notice waits for that server's notices loop."""
    runner = getattr(request.app.state, "notice_runner", None)
    ids = [r.id for r in rows if r is not None and r.delivery_state == "pending"]
    if runner is not None and ids:
        background_tasks.add_task(deliver_now, request.app.state.engine, ids, runner, request.app.state.clock)


def host_key_changed_boxes(session: Session) -> set[str]:
    """device_ids of the boxes whose newest push that reached ssh's host check ended in "host_key_changed" (a later
    delivered or box-rejected push means the key was reset). Audited attempts are under the box's live row."""
    return {device_id for device_id, outcome in session.execute(text(
        "SELECT DISTINCT ON (device_id) device_id, reason FROM audit_log "
        "WHERE action = 'notice_delivery' AND reason IN ('delivered', 'rejected', 'host_key_changed') "
        "AND device_id IS NOT NULL ORDER BY device_id, id DESC")).all() if outcome == "host_key_changed"}
