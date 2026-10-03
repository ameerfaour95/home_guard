"""Pure clock rules shared by live badges and relative timestamps."""
import math


def relative_time(stamp, now):
    try:
        age = now-float(stamp)
        if not math.isfinite(age): return "Unknown"
        age = max(0,age)
    except (TypeError, ValueError): return "Unknown"
    if age < 2: return "now"
    if age < 60: return f"{int(age)} s ago"
    if age < 3600: return f"{int(age//60)} min ago"
    if age < 86400: return f"{int(age//3600)} h ago"
    return f"{int(age//86400)} d ago"


def frame_health(stamp, now, started):
    age = max(0, now-(stamp if stamp is not None else started))
    if stamp is not None and age <= 3: return "LIVE", False
    if stamp is None and age <= 3: return "Connecting…", False
    return f"Reconnecting… {int(age)} s", True
