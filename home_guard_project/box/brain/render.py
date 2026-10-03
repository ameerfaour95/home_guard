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


def undo_what(tool: str, detail: dict, target: str, lang: str) -> str:
    """What an Undo was about, for its "changed since" and "could not undo" lines."""
    detail = detail if isinstance(detail, dict) else {}
    camera = detail.get("camera") if isinstance(detail.get("camera"), str) else ""
    if tool == "pause_alerts":
        return t("undo_what_pause_camera", lang, camera=camera) if camera else t("undo_what_pause_all", lang)
    if tool == "set_camera_active":
        return t("undo_what_camera", lang, camera=camera or target)
    if tool == "change_setting":
        key = f"setting_{detail.get('setting')}"
        return t(key, lang) if key in TEMPLATES else str(detail.get("setting") or target)
    if tool in ("set_alert_types", "set_sensitivity"):
        return t(f"undo_what_{tool[4:]}", lang, where=camera or t("the_house", lang))
    key = f"what_{tool}"
    return t(key, lang) if key in TEMPLATES else str(tool)


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
        undo_of = d.get("undo_of")
        if undo_of:
            if not isinstance(undo_of, str):
                raise ValueError("Invalid undo receipt")
            return t("undo_failed", lang, what=undo_what(undo_of, d, receipt.target, lang), reason=reason)
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
    if receipt.tool == "resume_alerts" and d.get("undo_of"):
        # An undone pause: say what still holds alerts back, so "back on" is only written when it is true.
        still = d.get("still_until") or ""
        if not isinstance(still, str):
            raise ValueError("Invalid undo receipt")
        if still:
            if d.get("camera"):
                return t("undo_camera_still_paused", lang, camera=d["camera"], until=still)
            return t("undo_all_still_paused", lang, until=still)
        pauses = d.get("still_pauses") or []
        if not isinstance(pauses, list) or not all(
                isinstance(p, list) and len(p) == 2 and all(isinstance(x, str) for x in p) for p in pauses):
            raise ValueError("Invalid undo receipt")
        if pauses and not d.get("camera"):
            return t("undo_back_on_except", lang,
                     pauses=", ".join(t("pause_until_item", lang, camera=c, until=u) for c, u in pauses))
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
    if receipt.tool == "change_setting":
        for key in ("setting", "old", "new"):
            if key in d and not isinstance(d[key], str):
                raise ValueError("Invalid setting receipt text")
        return t("setting_changed", lang, setting=t(f"setting_{d.get('setting')}", lang), old=d.get("old", ""),
                 new=d.get("new", ""))
    if receipt.tool in ("set_alert_types", "set_sensitivity"):
        house = not d.get("camera")
        where = t("the_house", lang) if house else d["camera"]
        if house and where:
            where = where[:1].upper() + where[1:]
        name = lambda kind: t(f"type_{kind}", lang) if f"type_{kind}" in TEMPLATES else kind  # noqa: E731
        order = {"person": 0, "vehicle": 1, "animal": 2}
        if receipt.tool == "set_alert_types":
            if any(not isinstance(d.get(key), list) or not all(isinstance(k, str) for k in d[key])
                   for key in ("old", "new")):
                raise ValueError("invalid alert types receipt")
            show = lambda types: ", ".join(name(k) for k in sorted(types or [], key=lambda k: order.get(k, 3)))  # noqa: E731
            return t("alert_types_changed_house" if house else "alert_types_changed", lang, camera=where,
                     old=show(d.get("old")), new=show(d.get("new")))
        for key in ("old", "new"):
            if not isinstance(d.get(key), dict):
                raise ValueError("invalid sensitivity receipt")
            for kind, value in d[key].items():
                if not isinstance(kind, str) or not 0.05 <= _nonnegative_number(value) <= 0.95:
                    raise ValueError("invalid sensitivity receipt value")
        old, new = d["old"], d["new"]
        kinds = sorted(set(old) | set(new), key=lambda k: (order.get(k, 3), k))
        changed = [k for k in kinds if k in old and k in new and old[k] != new[k]] or kinds
        pct = lambda vals, k: round(float(vals.get(k, new.get(k, old.get(k)))) * 100)  # noqa: E731
        changes = ", ".join(t("sensitivity_change", lang, kind=name(k), old=pct(old, k), new=pct(new, k))
                            for k in changed)
        return t("sensitivity_changed", lang, camera=where, changes=changes)
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
