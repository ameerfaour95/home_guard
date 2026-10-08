from datetime import datetime, timezone
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
from home_guard_project.fleet_contract.camera_names import channel_of, display_name as box_display_name, family_names


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


def humanise(value):
    text = str(value or '').replace('_', ' ')
    return text[:1].upper()+text[1:]


def export_warning(value):
    import re
    words = {'vlm':'VLM', 'yolo':'YOLO', 'train':'training', 'val':'validation', 'site+day':'household/day'}
    return re.sub(r'\b(?:vlm|yolo|train|val|site\+day)\b', lambda match:words[match[0]], value)


# Owner names the cloud knows (camera id -> the family's names, oldest first), filled when a customer's cameras load.
# Matched by exact id only: the box's channel fallback for renamed sites must not borrow another house's ch6.
KNOWN_NAMES = {}


def remember_names(names):
    """Record owner names for camera ids: ``{camera_id: name}``; an empty name is ignored."""
    for camera, name in (names or {}).items():
        if camera and name and name.strip():
            KNOWN_NAMES[camera] = [name.strip()]


def owner_camera_name(camera, aliases=None):
    """What the admin shows for a camera id: the box's display_name (owner alias, else "Camera N"), never the id."""
    aliases = aliases if aliases is not None else ({camera: KNOWN_NAMES[camera]} if camera in KNOWN_NAMES else {})
    if family_names(camera, aliases) or channel_of(camera):
        return box_display_name(camera, 'en', aliases)
    return humanise(camera)


def isolate(text):
    """*text* inside Unicode isolates (FSI ... PDI) when it holds right-to-left letters, so a Hebrew camera name in an
    English line ("כניסה ראשית  ·  A person at the door", "#102 · פרגולה") neither flips the line nor reorders
    the numbers around it. Plain text is returned as it is."""
    return f'\u2068{text}\u2069' if any('\u0590' <= ch <= '\u08ff' for ch in str(text)) else text


def camera_name(value, display_name=None):
    raw = value if isinstance(value, str) else value.camera
    if raw.startswith('cam-'):
        return raw
    name = display_name or getattr(value, 'display_name', None)
    if name:
        return isolate(name)
    if '/' in raw:
        site, camera = raw.split('/', 1)
        return f'{humanise(site)} / {isolate(owner_camera_name(camera))}'
    return isolate(owner_camera_name(raw))


def delivery_text(dispatch):
    if not dispatch or dispatch.sent is None:
        return 'Delivery not recorded'
    if dispatch.sent:
        return 'Sent to the owner on Telegram'
    reason = dispatch.detail.get('reason') or dispatch.detail.get('error') or 'Reason not recorded'
    return f'Not delivered ({humanise(reason)})'
