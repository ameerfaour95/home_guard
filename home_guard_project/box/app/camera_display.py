"""What a camera is called on screen: the box's name for it, never its id (owner rule, 2026-10-08).

The box is the only source of names (the family sets them over Telegram; ``scene_interview names --json``).
The app keeps the last answer here; a camera the box did not name reads "Camera 2 of 5" by its place in the
box's list, or just "Camera".
"""

import weakref

_names = {}          # camera id -> the box's name for it
_order = []          # the box's cameras, in the order the app lists them
_listeners = []      # weak references to callables told when the names change (GUI thread only)


def subscribe(callback):
    """Call *callback()* whenever the names change (a renamed camera: tiles, the feed, titles follow)."""
    ref = weakref.WeakMethod(callback) if hasattr(callback, "__self__") else (lambda: callback)
    _listeners.append(ref)


def set_names(names, order=()):
    """The box's names (``{id: name}``, empty names dropped) and the cameras' order on screen. Call it on the GUI
    thread: the screens showing names are refreshed."""
    global _names, _order
    _names = {str(k): str(v) for k, v in dict(names or {}).items() if v}
    _order = [str(c) for c in order] or list(_names)
    for ref in list(_listeners):
        callback = ref()
        if callback is None:
            _listeners.remove(ref)
            continue
        try:
            callback()
        except RuntimeError:          # a closed window's widgets are gone
            _listeners.remove(ref)


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
