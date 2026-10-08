"""What a camera is called on screen: the box's name for it, never its id (owner rule, 2026-10-08).

The box is the only source of names (the family sets them over Telegram; ``scene_interview names --json``).
The app keeps the last answer here; a camera the box did not name reads "Camera 2 of 5" by its place in the
box's list, or just "Camera".
"""

_names = {}          # camera id -> the box's name for it
_order = []          # the box's cameras, in the order the app lists them


def set_names(names, order=()):
    """The box's names (``{id: name}``, empty names dropped) and the cameras' order on screen."""
    global _names, _order
    _names = {str(k): str(v) for k, v in dict(names or {}).items() if v}
    _order = [str(c) for c in order] or list(_names)


def set_order(order):
    global _order
    _order = [str(c) for c in order]


def known(camera):
    return str(camera) in _names


def shown(camera):
    """The name to show for camera *camera* (an id)."""
    from .scene_editor import fallback_name
    camera = str(camera or "")
    if camera in _names:
        return _names[camera]
    position = (_order.index(camera) + 1, len(_order)) if camera in _order else None
    return fallback_name(position)
