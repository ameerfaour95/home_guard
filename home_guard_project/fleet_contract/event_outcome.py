"""What the box decided about a clip, beyond the model's alert command: the event layer's real outcome.

Since the box groups clips into events (box/events.py, owner 2026-10-08: "events, not triggers"), a clip's meta
says what happened to it in ``alert`` (the box's decision record; some fields also at the top level):

- ``event``: the event decision ``{notify, session_id, reason, reply_to, new_people, known_text, entities, fresh,
  unmarked, counted_by}`` -- one ongoing activity per camera is one session; only a change reaches the owner;
- ``sent`` (false + ``not_sent_reason`` when the event layer held it back) and ``dispatch`` (what Telegram said);
- optionally ``downgraded`` ("appearance only"), ``investigation`` (the lingering investigator: ``lowered``),
  ``second_look`` (a red re-checked: ``answered``, ``confirmed``, ``class``, ``what_it_is``), ``raised`` ("rare
  here", "came onto our ground"), ``ground`` (scene map) and ``baseline`` (what is usual at this camera:
  ``mode``, ``rarity``, ``would_raise``, ``raise``, ``text_en``; mode "shadow" only records what it would do).

``decision_of`` keeps those fields (small, JSON) for the index; ``outcome`` turns them into what the admin shows in
the AI-decision column. Never "No alert" for a clip the event layer kept: that hides why the owner was not told.
"""

from __future__ import annotations

from typing import Any, Optional

VERSION = 1
_TEXT_MAX = 300


def _dict(value: Any) -> dict:
    return value if isinstance(value, dict) else {}


def _text(value: Any, limit: int = _TEXT_MAX) -> str:
    return value.strip()[:limit] if isinstance(value, str) else ""


def _bool(value: Any) -> Optional[bool]:
    return value if isinstance(value, bool) else None


def decision_of(meta: Any) -> dict:
    """The box's decision record of a clip meta, reduced to the fields the admin shows and filters on.

    Always a dict with ``v`` (so an indexed clip whose meta predates events is told apart from one not parsed yet)."""
    meta = _dict(meta)
    alert = _dict(meta.get("alert"))

    def pick(name):  # the decision record first, then the meta's top level (input_meta copies)
        return alert.get(name) if name in alert else meta.get(name)
    event = _dict(pick("event"))
    dispatch = _dict(alert.get("dispatch"))
    out: dict[str, Any] = {"v": VERSION}
    if event:
        out["event"] = {"notify": _bool(event.get("notify")), "session_id": _text(event.get("session_id"), 64),
                        "reason": _text(event.get("reason")), "known_text": _text(event.get("known_text"), 200),
                        "new_people": event.get("new_people") if isinstance(event.get("new_people"), int) else 0,
                        "replied": bool(event.get("reply_to"))}
    sent = _bool(alert.get("sent"))
    out["sent"] = sent if sent is not None else _bool(dispatch.get("sent"))
    out["not_sent_reason"] = _text(alert.get("not_sent_reason")) or (
        _text(dispatch.get("reason")) if dispatch.get("sent") is False else "")
    out["muted"] = bool(alert.get("muted"))
    out["false_positive"] = bool(alert.get("false_positive"))
    out["command"] = _text(alert.get("alert_command"), 32)
    for name in ("downgraded", "raised", "investigator"):
        if _text(pick(name)):
            out[name] = _text(pick(name), 120)
    look = _dict(pick("second_look"))
    if look:
        out["second_look"] = {"answered": bool(look.get("answered")), "confirmed": _bool(look.get("confirmed")),
                              "class": _text(look.get("class"), 60), "what_it_is": _text(look.get("what_it_is"), 200)}
    found = _dict(pick("investigation"))
    if found:
        out["investigation"] = {"verdict": _text(found.get("verdict"), 60), "lowered": bool(found.get("lowered"))}
    usual = _dict(pick("baseline"))
    if usual:
        out["baseline"] = {"mode": _text(usual.get("mode"), 16), "rarity": _text(usual.get("rarity"), 32),
                           "would_raise": bool(usual.get("would_raise")), "raise": bool(usual.get("raise")),
                           "text_en": _text(usual.get("text_en"), 200)}
    where = _dict(pick("ground"))
    if where:
        out["ground"] = {k: where[k] for k in ("on", "entered", "off_our_ground") if k in where
                         and isinstance(where[k], (str, bool))}
    return out


def session_id(decision: Any) -> str:
    return _text(_dict(_dict(decision).get("event")).get("session_id"), 64)


def would_raise(decision: Any) -> bool:
    """The baseline in shadow mode would have raised this clip ("rare for this camera") but did not."""
    usual = _dict(_dict(decision).get("baseline"))
    return bool(usual.get("would_raise")) and not usual.get("raise")


def outcome(decision: Any, private: bool = True) -> tuple[str, str]:
    """(code, text) of what happened to the clip, for the AI-decision column. ``private`` False leaves out the owner's
    own words (labelers): "Not sent: owner said known" without what he said. Codes: sent, undelivered, held, known,
    not_ours, muted, dismissed, none ('' when the meta predates the event layer and says nothing)."""
    d = _dict(decision)
    event = _dict(d.get("event"))
    steps = []
    if d.get("downgraded"):
        steps.append(f"Lowered: {d['downgraded']}")
    if _dict(d.get("investigation")).get("lowered"):
        steps.append("Lowered: short visit (investigator)")
    look = _dict(d.get("second_look"))
    if look.get("answered") and look.get("confirmed") is False:
        what = look.get("what_it_is")
        steps.append(f"Second look: not {_article(look.get('class') or 'that')}" + (f" ({what})" if what else ""))
    if d.get("raised"):
        steps.append(f"Raised: {d['raised']}")
    reason = event.get("reason") or d.get("not_sent_reason") or ""
    if d.get("muted"):
        code, text = "muted", "Not sent: camera muted by the owner"
    elif d.get("false_positive"):
        code, text = "dismissed", "Dismissed: the AI saw nothing to alert on"
    elif event and event.get("notify") is False or d.get("sent") is False and reason:
        known = event.get("known_text")
        if known or "owner said who is here" in reason:
            code = "known"
            text = "Not sent: owner said known" + (f" ({known})" if known and private else "")
        elif "not ours" in reason:
            code, text = "not_ours", "Not sent: nothing done on our ground"
        elif reason.startswith("normal"):
            code, text = "held", "Kept in the event, not sent (normal)"
        elif "already reported" in reason:
            code, text = "held", "Kept in the event, not sent (already reported" + (
                ", nobody new)" if "nobody new" in reason or "same people" in reason else ")")
        elif event:
            code, text = "held", f"Kept in the event, not sent ({reason})" if reason else "Kept in the event, not sent"
        else:
            code, text = "undelivered", f"Not delivered ({reason})"
    elif d.get("sent") is True:
        code, text = "sent", "Sent"
        if reason.startswith("normal, but rare here"):
            text = "Sent: rare for this camera"
        elif event.get("replied") and event.get("new_people"):
            text = f"Sent: {event['new_people']} more people"
    elif d.get("sent") is False:
        code, text = "undelivered", "Not delivered"
    else:
        code, text = "none", ""
    parts = steps + ([text] if text else []) + (["Would raise: rare for this camera"] if would_raise(d) else [])
    return (code if parts else ""), "  ·  ".join(parts)


def _article(word: str) -> str:
    word = str(word).replace("_", " ").strip()
    return f"an {word}" if word[:1].lower() in "aeiou" else f"a {word}"
