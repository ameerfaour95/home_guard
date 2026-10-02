"""Owner notifications for a collector box in inference mode.

Sends a WhatsApp message for ``[send_message]`` and places a phone call for
``[call_owner]`` through Twilio's REST API, using only the standard library
(no new dependency on the box). Delivery is dry-run-safe: nothing goes out
unless credentials are configured and ``dry_run`` is off, so a misconfigured
box is silent rather than wrong.

Secrets come from the environment (loaded from api_key.env by the caller):
``TWILIO_ACCOUNT_SID`` and ``TWILIO_AUTH_TOKEN``. The phone numbers and sender
are non-secret box settings. No credential is ever logged.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, Optional

log = logging.getLogger("box.notify")

_API_ROOT = "https://api.twilio.com/2010-04-01/Accounts"


@dataclass(frozen=True)
class NotifyConfig:
    """Everything the notifier needs. Build it with :func:`load_notify_config`.

    Twilio REST auth is either the Account SID + Auth Token, or an API Key
    (``SK...``) + its secret. Either way the Account SID (``AC...``) is still
    required because it is part of the request URL.
    """

    account_sid: str = ""        # "AC..." - always required (URL path)
    auth_token: str = ""         # Account Auth Token (if not using an API key)
    api_key_sid: str = ""        # "SK..." - API key, used instead of the Auth Token
    api_key_secret: str = ""
    whatsapp_from: str = ""      # e.g. "whatsapp:+14155238886"
    owner_whatsapp: str = ""     # e.g. "whatsapp:+972501234567"
    voice_from: str = ""         # e.g. "+14155238886" (for [call_owner])
    owner_phone: str = ""        # e.g. "+972501234567"
    dry_run: bool = False

    @property
    def _auth(self) -> tuple:
        """(username, password) for HTTP basic auth: API key if set, else Account SID."""
        if self.api_key_sid and self.api_key_secret:
            return (self.api_key_sid, self.api_key_secret)
        return (self.account_sid, self.auth_token)

    @property
    def _has_auth(self) -> bool:
        user, pw = self._auth
        return bool(user and pw)

    @property
    def can_message(self) -> bool:
        return bool(self.account_sid and self._has_auth and self.whatsapp_from and self.owner_whatsapp)

    @property
    def can_call(self) -> bool:
        return bool(self.account_sid and self._has_auth and self.voice_from and self.owner_phone)


def load_notify_config(settings: Dict[str, Any], env: Optional[Dict[str, str]] = None) -> NotifyConfig:
    """Build a config from box settings (non-secret) + environment (secrets).

    ``settings`` is the flat box.yaml dict. ``env`` defaults to ``os.environ``.
    ``notify_dry_run: true`` in settings forces dry-run even when creds exist.
    """
    env = dict(os.environ if env is None else env)
    return NotifyConfig(
        account_sid=env.get("TWILIO_ACCOUNT_SID", ""),
        auth_token=env.get("TWILIO_AUTH_TOKEN", ""),
        api_key_sid=env.get("TWILIO_API_KEY_SID", ""),
        api_key_secret=env.get("TWILIO_API_KEY_SECRET", ""),
        whatsapp_from=str(settings.get("twilio_whatsapp_from", "")),
        owner_whatsapp=str(settings.get("owner_whatsapp", "")),
        voice_from=str(settings.get("twilio_voice_from", "")),
        owner_phone=str(settings.get("owner_phone", "")),
        dry_run=bool(settings.get("notify_dry_run", False)),
    )


def _http_post(url: str, fields: Dict[str, str], auth_user: str, auth_pass: str, timeout: float = 15.0) -> Dict[str, Any]:
    """POST form fields to Twilio with basic auth. Returns the parsed response.

    Isolated so tests can patch it without touching the network.
    """
    data = urllib.parse.urlencode(fields).encode("utf-8")
    auth = base64.b64encode(f"{auth_user}:{auth_pass}".encode("utf-8")).decode("ascii")
    req = urllib.request.Request(url, data=data, method="POST")
    req.add_header("Authorization", f"Basic {auth}")
    req.add_header("Content-Type", "application/x-www-form-urlencoded")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = resp.read().decode("utf-8", errors="replace")
    try:
        return json.loads(body)
    except ValueError:
        return {"raw": body}


def send_whatsapp(cfg: NotifyConfig, body: str) -> Dict[str, Any]:
    """Send a WhatsApp message. Never raises; returns a small status dict."""
    if cfg.dry_run or not cfg.can_message:
        reason = "dry_run" if cfg.dry_run else "not_configured"
        log.info("WhatsApp not sent (%s): %s", reason, body)
        return {"sent": False, "reason": reason}
    url = f"{_API_ROOT}/{cfg.account_sid}/Messages.json"
    fields = {"From": cfg.whatsapp_from, "To": cfg.owner_whatsapp, "Body": body}
    user, pw = cfg._auth
    try:
        resp = _http_post(url, fields, user, pw)
        log.info("WhatsApp sent (sid=%s)", resp.get("sid", "?"))
        return {"sent": True, "sid": resp.get("sid")}
    except (urllib.error.URLError, OSError) as exc:
        log.warning("WhatsApp send failed: %s", exc)
        return {"sent": False, "reason": "error", "error": str(exc)}


def call_owner(cfg: NotifyConfig, say_text: str) -> Dict[str, Any]:
    """Place a phone call that speaks *say_text*. Never raises."""
    if cfg.dry_run or not cfg.can_call:
        reason = "dry_run" if cfg.dry_run else "not_configured"
        log.info("Call not placed (%s): %s", reason, say_text)
        return {"called": False, "reason": reason}
    url = f"{_API_ROOT}/{cfg.account_sid}/Calls.json"
    safe = say_text.replace("&", "and").replace("<", "").replace(">", "")
    twiml = f"<Response><Say>{safe}</Say></Response>"
    fields = {"From": cfg.voice_from, "To": cfg.owner_phone, "Twiml": twiml}
    user, pw = cfg._auth
    try:
        resp = _http_post(url, fields, user, pw)
        log.info("Call placed (sid=%s)", resp.get("sid", "?"))
        return {"called": True, "sid": resp.get("sid")}
    except (urllib.error.URLError, OSError) as exc:
        log.warning("Call failed: %s", exc)
        return {"called": False, "reason": "error", "error": str(exc)}


def notify(cfg: NotifyConfig, command: str, summary: str = "", reason: str = "") -> Dict[str, Any]:
    """Dispatch on an alert_command. Returns what was attempted.

    ``[send_message]`` -> WhatsApp. ``[call_owner]`` -> a call AND a WhatsApp
    with the details; if calls are not configured, the WhatsApp is the fallback.
    ``[none]`` (or anything else) -> nothing.
    """
    detail = summary if not reason else f"{summary} - {reason}"
    detail = detail.strip() or "activity detected"
    if command == "[send_message]":
        return {"command": command, "whatsapp": send_whatsapp(cfg, f"Home Guard: {detail}")}
    if command == "[call_owner]":
        result: Dict[str, Any] = {"command": command}
        result["whatsapp"] = send_whatsapp(cfg, f"Home Guard ALERT: {detail}")
        if cfg.can_call and not cfg.dry_run:
            result["call"] = call_owner(cfg, f"Home Guard alert. {detail}")
        else:
            log.info("call_owner: calls not configured, WhatsApp used as fallback")
            result["call"] = {"called": False, "reason": "fallback_to_whatsapp"}
        return result
    return {"command": command, "sent": False}
