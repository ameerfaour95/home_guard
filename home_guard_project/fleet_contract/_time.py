"""UTC parsing and normalization shared by the legacy contract readers."""

from datetime import datetime, timezone
from typing import Optional


def normalize_utc(value: datetime) -> datetime:
    """Treat naive datetimes as UTC and convert aware datetimes to UTC."""
    return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value.astimezone(timezone.utc)


def parse_utc(value) -> Optional[datetime]:
    """Read legacy ISO strings; malformed or non-string values stay unknown."""
    if not isinstance(value, str):
        return None
    try:
        return normalize_utc(datetime.fromisoformat(value))
    except (ValueError, OverflowError):
        return None
