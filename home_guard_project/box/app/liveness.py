"""Pure clock rules shared by live badges and relative timestamps."""
import math

from .strings import tr


def relative_time(stamp, now):
    try:
        age = now-float(stamp)
        if not math.isfinite(age): return tr("rel_unknown")
        age = max(0,age)
    except (TypeError, ValueError): return tr("rel_unknown")
    if age < 2: return tr("rel_now")
    if age < 60: return tr("rel_seconds", n=int(age))
    if age < 3600: return tr("rel_minutes", n=int(age//60))
    if age < 86400: return tr("rel_hours", n=int(age//3600))
    return tr("rel_days", n=int(age//86400))


def frame_health(stamp, now, started):
    age = max(0, now-(stamp if stamp is not None else started))
    if stamp is not None and age <= 3: return tr("live"), False
    if stamp is None and age <= 3: return tr("tile_connecting"), False
    return tr("tile_reconnecting", seconds=int(age)), True
