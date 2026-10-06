"""The one media consent decision: may this role open a household's media for this purpose?

Used by every route that hands out media -- artifact access (clips, filmstrips, any artifact), the event
thumbnail redirect, and the thumbnail links in event lists -- so a thumbnail (a frame of the recording) is never
easier to get than the recording itself.

Labelers see only households that agreed to training use, and only for training (their visibility filter answers
404 before this decision runs). Support never opens training data. Recordings need recordings consent; training
use needs training consent.
"""
from __future__ import annotations

from typing import Optional

ROLE_CANNOT_TRAIN = "Your role cannot open training data"
ROLE_CANNOT = "Your role cannot do this"
NO_TRAINING_CONSENT = "This customer withdrew consent for training use"
NO_RECORDINGS_CONSENT = "This customer withdrew consent for recordings access"


def media_refusal(role: str, purpose: str, consent_recordings: bool, consent_training: bool) -> Optional[str]:
    """None when `role` may open the media for `purpose`, else the reason it may not."""
    if purpose == "training":
        if role not in ("labeler", "admin"):
            return ROLE_CANNOT_TRAIN
        return None if consent_training else NO_TRAINING_CONSENT
    if role == "labeler":
        return ROLE_CANNOT
    return None if consent_recordings else NO_RECORDINGS_CONSENT


def thumbnail_purpose(role: str) -> str:
    """Thumbnails carry no purpose of their own: labelers look at them to label (training), staff to review."""
    return "training" if role == "labeler" else "review"


def may_see_thumbnail(role: str, consent_recordings: bool, consent_training: bool) -> bool:
    return media_refusal(role, thumbnail_purpose(role), consent_recordings, consent_training) is None
