from datetime import datetime, timezone
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError


def utcnow():
    return datetime.now(timezone.utc)


def age(value, now=None):
    if value is None:
        return 'Not reported'
    seconds = max(0, int(((now or utcnow()) - value).total_seconds()))
    if seconds < 60:
        return 'Just now'
    if seconds < 3600:
        return f'{seconds // 60} min ago'
    if seconds < 86400:
        return f'{seconds // 3600} h ago'
    return f'{seconds // 86400} d ago'


def local_time(value, zone):
    if value is None:
        return 'Not reported'
    try:
        tz = ZoneInfo(zone)
    except ZoneInfoNotFoundError:
        tz = timezone.utc
    return value.astimezone(tz).strftime('%d %b %Y, %H:%M %Z')


def site_name(site):
    return site.replace('_', ' ').title()


def mode_name(mode):
    return {'inference': 'Security', 'data_collection': 'Data collection'}.get(mode, mode or 'Not reported')
