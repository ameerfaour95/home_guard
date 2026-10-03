# home_guard_project/box/brain/render.py
"""The Telegram message for one turn: the model's facts, then one code-written line per receipt."""

from __future__ import annotations

import logging
import math
from typing import Sequence

from .i18n import TEMPLATES, t
from .receipts import DONE, FAILED, REQUESTED, Receipt

log = logging.getLogger("box.brain.render")


def receipt_line(receipt: Receipt, lang: str, retention_days: float = 14.0) -> str:
    """Render a receipt, or skip malformed input and log once; never raise into the poll loop."""
    try:
        return _receipt_line(receipt, lang, retention_days)
    except Exception:
        log.warning("Skipped malformed receipt while rendering")
        return ""


def _nonnegative_number(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (str, int, float)):
        raise ValueError("Invalid number")
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise ValueError("Invalid number")
    return number


def _receipt_line(receipt: Receipt, lang: str, retention_days: float) -> str:
    if not isinstance(receipt, Receipt) or not isinstance(lang, str):
        raise ValueError("Invalid receipt or language")
    if not all(isinstance(value, str) for value in (receipt.tool, receipt.status, receipt.target, receipt.reason)):
        raise ValueError("Invalid receipt fields")
    if receipt.status not in (DONE, FAILED, REQUESTED):
        raise ValueError("Invalid receipt status")
    if receipt.status == REQUESTED and receipt.tool != "set_camera_active":
        raise ValueError("Only a pending camera change has a requested template")
    d = receipt.detail
    if not isinstance(d, dict):
        raise ValueError("Invalid receipt detail")
    if receipt.status == FAILED:
        reason_key = f"reason_{receipt.reason}"
        if reason_key in TEMPLATES:
            reason = t(reason_key, lang, days=int(_nonnegative_number(retention_days)))
        else:
            reason = receipt.reason or t("reason_error", lang)
        what_key = f"what_{receipt.tool}"
        what = t(what_key, lang) if what_key in TEMPLATES else receipt.tool
        return t("failed", lang, what=what, reason=reason)
    # Inspect only fields used by the renderer; extra detail such as `by` is ignored.
    for key in ("camera", "kind", "bounds", "until", "verdict", "alias"):
        if key in d and not isinstance(d[key], str):
            raise ValueError("Invalid receipt text")
    camera = d.get("camera") or receipt.target
    if receipt.tool == "send_media":
        if d.get("kind") == "photo":
            return t("sent_photo", lang, camera=camera)
        return t("sent_video", lang, bounds=d.get("bounds", ""))
    if receipt.tool == "check_camera":
        return t("sent_photo", lang, camera=camera)
    if receipt.tool == "record_clip":
        if "seconds" in d:
            _nonnegative_number(d["seconds"])
        return t("sent_live_clip", lang, seconds=d.get("seconds", ""), camera=camera)
    if receipt.tool == "pause_alerts":
        if d.get("camera"):
            return t("paused_camera", lang, camera=d["camera"], until=d.get("until", ""))
        return t("paused_all", lang, until=d.get("until", ""))
    if receipt.tool == "resume_alerts":
        return t("resumed_camera", lang, camera=d["camera"]) if d.get("camera") else t("resumed_all", lang)
    if receipt.tool == "set_camera_active":
        if not isinstance(d.get("active"), bool):
            raise ValueError("Invalid camera state")
        state = "on" if d.get("active") else "off"
        suffix = "requested" if receipt.status == REQUESTED else "done"
        return t(f"camera_{state}_{suffix}", lang, camera=camera)
    if receipt.tool == "record_verdict":
        return t("verdict_saved", lang, verdict=t(f"verdict_{d.get('verdict')}", lang))
    if receipt.tool == "set_alias":
        return t("alias_saved", lang, alias=d.get("alias", ""), camera=camera)
    return f"✓ {receipt.tool}"


def render_reply(answer: str, receipts: Sequence[Receipt], lang: str, retention_days: float = 14.0) -> str:
    """Facts followed by receipt lines; skip malformed entries and log at most once per call."""
    malformed = not isinstance(answer, str)
    text = answer.strip() if isinstance(answer, str) else ""
    lines = [text] if text else []
    if not isinstance(receipts, Sequence) or isinstance(receipts, (str, bytes, bytearray)):
        receipts = ()
        malformed = True
    for receipt in receipts:
        try:
            lines.append(_receipt_line(receipt, lang, retention_days))
        except Exception:
            malformed = True
    if malformed:
        log.warning("Skipped malformed input while rendering reply")
    return "\n".join(lines)
