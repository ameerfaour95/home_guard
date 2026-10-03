"""On-demand live view: grab a current frame from a camera and describe it.

The owner asks the assistant "what's happening at the front door now?" - the
saved-alert tools can't answer that, because there may be no saved clip of
"now". This grabs a current frame from that camera's stream and sends it to the
vision model for a short factual description, so the assistant can speak about
the present, not only about saved alerts.

Reuses the camera list and frame grab from find_cameras, and the OS trust store
(like the other OpenAI callers) so it works where TLS is intercepted. Never
raises: every failure comes back as ``{"error": ...}``.
"""

from __future__ import annotations

import base64
import logging
import os
import ssl
import time
import uuid
from typing import Any, Callable, Dict, Optional, Tuple

log = logging.getLogger("box.live_view")

VISION_MODEL = "gpt-4o"      # vision-capable; the text agent stays on gpt-4o-mini
VISION_PROMPT = (
    "You are a home security assistant looking at one still frame from a camera. In one or two short, "
    "factual sentences, say what is visible right now - any people, vehicles or animals and what they "
    "appear to be doing. If nothing notable is there, say the view looks clear."
)


def _camera_url(camera: str, cameras_path: str) -> Optional[str]:
    try:
        from .find_cameras import _read_cameras_raw  # noqa: PLC0415 - heavy import kept lazy

        data = _read_cameras_raw(cameras_path)
        if not isinstance(data, dict):
            raise ValueError("camera file must contain a mapping")
        cams = data.get("cameras", {})
        if not isinstance(cams, dict) or not isinstance(camera, str):
            raise ValueError("invalid camera mapping or name")
        url = cams.get(camera)
        if url is not None and not isinstance(url, str):
            raise ValueError("camera URL must be a string")
        return url or None
    except Exception as exc:  # noqa: BLE001 - configuration must not stop the poll loop
        log.warning("Live-view camera lookup failed: %s", exc)
        return None


def _describe(image_path: str, api_key: str, model: str = VISION_MODEL, timeout: float = 30.0) -> Optional[str]:
    """One vision call describing the frame, or None on any failure."""
    import httpx  # noqa: PLC0415

    try:
        with open(image_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("ascii")
        payload = {
            "model": model,
            "temperature": 0,
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": VISION_PROMPT},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
            ]}],
        }
        resp = httpx.post("https://api.openai.com/v1/chat/completions", json=payload, timeout=timeout,
                          headers={"Authorization": f"Bearer {api_key}"}, verify=ssl.create_default_context())
        resp.raise_for_status()
        text = (resp.json()["choices"][0]["message"]["content"] or "").strip()
        return text or None
    except Exception as exc:  # noqa: BLE001 - offline, TLS, a refusal: the caller reports a clean error
        log.warning("Live-view describe failed: %s", exc)
        return None


def _mask_in_place(image_path: str, polygon: Any) -> bool:
    """Black out everything outside *polygon* in the saved picture. False on any failure."""
    try:
        import cv2  # noqa: PLC0415

        from ..data_collection.zones import ZoneMask  # noqa: PLC0415

        img = cv2.imread(image_path)
        if img is None:
            return False
        masked = ZoneMask(polygon).apply(img)
        return bool(cv2.imwrite(image_path, masked, [cv2.IMWRITE_JPEG_QUALITY, 90]))
    except Exception as exc:  # noqa: BLE001 - fail closed: the caller drops the picture
        log.warning("Live-view mask failed: %s", exc)
        return False


def _remove_quietly(path: str) -> None:
    try:
        os.remove(path)
    except OSError:
        pass


def grab_masked(camera: str, cameras_path: str, out_dir: str, now: Callable[[], float] = time.time,
                grab: Optional[Callable[[str, str], bool]] = None,
                zones_path: Optional[str] = None) -> Dict[str, Any]:
    """A current picture from *camera*, masked to its watch zone: ``{"camera", "image"}`` or ``{"error"}``.

    The unmasked grab only ever exists under a temporary name; if the zone cannot
    be applied the picture is dropped (fail closed).
    """
    temp_path = None
    try:
        url = _camera_url(camera, cameras_path)
        if not url:
            return {"error": f"no camera named {camera!r} is configured"}
        if grab is None:
            from .find_cameras import _grab_snapshot as grab  # noqa: PLC0415
        try:
            os.makedirs(out_dir, exist_ok=True)
        except OSError as exc:
            return {"error": f"could not prepare a place for the picture ({exc})"}
        # Independent captures must never share an unmasked temporary file.
        image_path = os.path.join(out_dir, f"{camera}_{int(now())}_{uuid.uuid4().hex[:8]}.jpg")
        temp_path = image_path + ".tmp.jpg"
        polygon, readable = strict_zone(camera, zones_path)
        if not readable:
            return {"error": f"could not prepare the picture from {camera} right now"}
        if not grab(url, temp_path):
            return {"error": f"could not get a picture from {camera} right now (is it online?)"}
        if polygon and not _mask_in_place(temp_path, polygon):
            return {"error": f"could not prepare the picture from {camera} right now"}
        os.replace(temp_path, image_path)
        return {"camera": camera, "image": image_path}
    except Exception as exc:  # noqa: BLE001 - never expose an unmasked picture after a failure
        log.warning("Live-view capture failed: %s", exc)
        return {"error": f"could not prepare the picture from {camera} right now"}
    finally:
        if temp_path is not None:
            _remove_quietly(temp_path)


def strict_zone(camera: str, zones_path: Optional[str] = None) -> Tuple[Optional[Any], bool]:
    """``(polygon or None, readable)`` for *camera*. Unlike ``zones.load_zones``, a zone file that exists but
    cannot be read, or an invalid polygon for this camera, is reported as unreadable, so a capture fails
    closed instead of going out unmasked. No file, or no zone for this camera, is ``(None, True)``."""
    try:
        import yaml  # noqa: PLC0415

        from ..data_collection.zones import ZONES_PATH, validate_points  # noqa: PLC0415

        path = ZONES_PATH if zones_path is None else zones_path
        try:
            with open(path, encoding="utf-8") as f:
                data = yaml.safe_load(f)
        except FileNotFoundError:
            return None, True
        if data is None:
            data = {}
        if not isinstance(data, dict):
            raise ValueError("zone file must contain a mapping")
        entries = data.get("zones", {})
        if not isinstance(entries, dict):
            raise ValueError("zones must contain a mapping")
        raw = {str(k): v for k, v in entries.items()}.get(camera)
        if raw is None:
            return None, True
        return validate_points(raw), True
    except Exception as exc:  # noqa: BLE001 - imports, encoding, types and I/O all fail closed
        log.warning("Live-view zone handling failed: %s", exc)
        return None, False


def look_now(camera: str, cameras_path: str, env: Dict[str, str], out_dir: str,
             now: Callable[[], float] = time.time,
             grab: Optional[Callable[[str, str], bool]] = None,
             describe: Optional[Callable[[str, str], Optional[str]]] = None,
             zones_path: Optional[str] = None) -> Dict[str, Any]:
    """Grab a current frame from *camera* and describe it.

    Returns ``{"camera", "description", "image"}`` on success, or ``{"error": ...}``.
    *grab* and *describe* are injectable for testing.
    """
    url = _camera_url(camera, cameras_path)
    if not url:
        return {"error": f"no camera named {camera!r} is configured"}
    api_key = env.get("OPENAI_API_KEY", "") if isinstance(env, dict) else ""
    if not api_key:
        return {"error": "live view needs the vision model, which is not configured on this box"}
    describe = describe or (lambda path, key: _describe(path, key))
    shot = grab_masked(camera, cameras_path, out_dir, now=now, grab=grab, zones_path=zones_path)
    if shot.get("error"):
        return shot
    try:
        text = describe(shot["image"], api_key)
        if not isinstance(text, str):
            text = None
    except Exception as exc:  # noqa: BLE001
        log.warning("Live-view describe failed: %s", exc)
        text = None
    if not text:
        return {"error": f"got a picture from {camera} but could not describe it"}
    return {"camera": camera, "description": text, "image": shot["image"]}


def make_look_now(cameras_path: str, env: Optional[Dict[str, str]] = None,
                  out_dir: str = "") -> Optional[Callable[[str], Dict[str, Any]]]:
    """A ``look_now(camera)`` callable for the agent, or None when it can't run (no key / no dir)."""
    if env is not None and not isinstance(env, dict):
        log.warning("Live-view environment must be a mapping")
        return None
    env = dict(os.environ if env is None else env)
    if not env.get("OPENAI_API_KEY") or not out_dir:
        return None
    return lambda camera: look_now(camera, cameras_path, env, out_dir)
