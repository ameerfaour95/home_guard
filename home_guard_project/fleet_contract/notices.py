"""Owner notices: what staff looked at, written by the cloud to fleet/<device_id>/notices/<time>_<id>.json.

Each notice is ``{"schema_version": 1, "id", "kind", "staff_name", "cameras", "from_utc", "to_utc", "message"}``.
``kind`` is data a consumer (the box, the owner's app) branches on; ``message`` is only the English sentence for
display. A consumer never parses the message to learn what was viewed.
"""

from __future__ import annotations

# recording: staff opened a recording (clip, picture) of these cameras; chat: staff opened the owner's Telegram
# conversation with the assistant (the Admin Center's Chat tab, or a picture sent in it).
NOTICE_KINDS = ("recording", "chat")


def check_kind(kind: str) -> str:
    """*kind* when it is a notice kind, else ValueError (a new kind is added here first, then used)."""
    if kind not in NOTICE_KINDS:
        raise ValueError(f"unknown notice kind {kind!r}; known: {', '.join(NOTICE_KINDS)}")
    return kind


def notice_message(kind: str, cameras: list[str], when: str) -> str:
    """The display sentence for a notice of *kind* (cameras already in the owner's names, *when* "HH:MM" or a range)."""
    check_kind(kind)
    if kind == "chat":
        return f"Home Guard support viewed your chat with the assistant ({when})"
    names = "" if not cameras else cameras[0] if len(cameras) == 1 else ", ".join(cameras[:-1]) + " and " + cameras[-1]
    return f"Home Guard support viewed recordings{' from ' + names if names else ''} ({when})"
