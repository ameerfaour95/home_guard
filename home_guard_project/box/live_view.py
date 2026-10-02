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
from typing import Any, Callable, Dict, Optional

log = logging.getLogger("box.live_view")

VISION_MODEL = "gpt-4o"      # vision-capable; the text agent stays on gpt-4o-mini
VISION_PROMPT = (
    "You are a home security assistant looking at one still frame from a camera. In one or two short, "
    "factual sentences, say what is visible right now - any people, vehicles or animals and what they "
    "appear to be doing. If nothing notable is there, say the view looks clear."
)


def _camera_url(camera: str, cameras_path: str) -> Optional[str]:
    from .find_cameras import _read_cameras_raw  # noqa: PLC0415 - heavy import kept lazy

    data = _read_cameras_raw(cameras_path) or {}
    cams = data.get("cameras") if isinstance(data.get("cameras"), dict) else {}
    url = cams.get(camera)
    return str(url) if url else None


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


def look_now(camera: str, cameras_path: str, env: Dict[str, str], out_dir: str,
             now: Callable[[], float] = time.time,
             grab: Optional[Callable[[str, str], bool]] = None,
             describe: Optional[Callable[[str, str], Optional[str]]] = None) -> Dict[str, Any]:
    """Grab a current frame from *camera* and describe it.

    Returns ``{"camera", "description", "image"}`` on success, or ``{"error": ...}``.
    *grab* and *describe* are injectable for testing.
    """
    url = _camera_url(camera, cameras_path)
    if not url:
        return {"error": f"no camera named {camera!r} is configured"}
    api_key = env.get("OPENAI_API_KEY", "")
    if not api_key:
        return {"error": "live view needs the vision model, which is not configured on this box"}
    if grab is None:
        from .find_cameras import _grab_snapshot as grab  # noqa: PLC0415
    describe = describe or (lambda path, key: _describe(path, key))

    try:
        os.makedirs(out_dir, exist_ok=True)
    except OSError as exc:
        return {"error": f"could not prepare a place for the picture ({exc})"}
    image_path = os.path.join(out_dir, f"{camera}_{int(now())}.jpg")
    if not grab(url, image_path):
        return {"error": f"could not get a picture from {camera} right now (is it online?)"}
    text = describe(image_path, api_key)
    if not text:
        return {"error": f"got a picture from {camera} but could not describe it"}
    return {"camera": camera, "description": text, "image": image_path}


def make_look_now(cameras_path: str, env: Optional[Dict[str, str]] = None,
                  out_dir: str = "") -> Optional[Callable[[str], Dict[str, Any]]]:
    """A ``look_now(camera)`` callable for the agent, or None when it can't run (no key / no dir)."""
    env = dict(os.environ if env is None else env)
    if not env.get("OPENAI_API_KEY") or not out_dir:
        return None
    return lambda camera: look_now(camera, cameras_path, env, out_dir)
