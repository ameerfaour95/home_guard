"""Who this box belongs to: the owner, what they agreed to, and who installed it.

The setup program records it once (``box register``); the box then publishes it to
``dataset_<site>/_status/registration.json`` next to the heartbeat, so the customer
appears in the Admin Center without anyone typing them in.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import logging
import os
import re
import socket
import subprocess
import tempfile
from typing import Any, Dict, Optional

from . import paths
from .boxconfig import BOX_YAML, PROJECT_ROOT, load_box_config

log = logging.getLogger("box.registration")

REGISTRATION_PATH = paths.registration_json()
PUBLISHED_PATH = paths.registration_published()
STATUS_KEY = "_status/registration.json"
SCHEMA_VERSION = 1

_SITE_RE = re.compile(r"^[a-z0-9_]+$")
_CONTROL_RE = re.compile(r"[\x00-\x1f\x7f]")
_CONSENTS = ("live", "recordings", "training")


class RegistrationError(Exception):
    """The registration details are missing or invalid."""


def _now_utc() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _text(name: str, value: Any, max_len: int) -> str:
    if not isinstance(value, str):
        raise RegistrationError(f"{name} must be text")
    if len(value.strip()) > max_len or _CONTROL_RE.search(value):
        raise RegistrationError(f"{name} must be at most {max_len} characters, on one line")
    return value.strip()


def _app_version() -> str:
    """Short git commit of this checkout, or empty when git or the checkout is missing."""
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=PROJECT_ROOT, capture_output=True,
                             text=True, timeout=10)
        return out.stdout.strip() if out.returncode == 0 else ""
    except (OSError, subprocess.SubprocessError):
        return ""


def load_registration(path: str = REGISTRATION_PATH) -> Optional[Dict[str, Any]]:
    """The saved registration, or None when it is missing or unreadable (a warning is logged)."""
    if not os.path.isfile(path):
        return None
    try:
        with open(path, encoding="utf-8") as f:
            reg = json.load(f)
    except (OSError, ValueError):
        log.warning("Could not read the registration; ignoring it.")
        return None
    if (not isinstance(reg, dict) or reg.get("schema_version") != SCHEMA_VERSION
            or not isinstance(reg.get("site"), str) or not _SITE_RE.fullmatch(reg["site"])
            or not isinstance(reg.get("owner_name"), str) or not 1 <= len(reg["owner_name"].strip()) <= 120
            or _CONTROL_RE.search(reg["owner_name"])
            or not isinstance(reg.get("consent"), dict)
            or any(type(reg["consent"].get(k)) is not bool for k in _CONSENTS)):
        log.warning("Invalid registration; ignoring it.")
        return None
    return reg


def _save(path: str, reg: Dict[str, Any]) -> None:
    _atomic_text(path, json.dumps(reg, indent=2, ensure_ascii=False) + "\n")


def _atomic_text(path: str, text: str) -> None:
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    descriptor, tmp = tempfile.mkstemp(prefix="registration-", suffix=".tmp", dir=directory)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp): os.remove(tmp)


def write_registration(path: str = REGISTRATION_PATH, box_yaml: str = BOX_YAML, **fields: Any) -> Dict[str, Any]:
    """Validate *fields* and save them. Fields not given keep their saved value.

    Fields: site, owner_name, owner_phone, installer, tailscale_host, box_host,
    app_version, consent_live / consent_recordings / consent_training (true or false).
    A changed consent, or a new installer, stamps ``consent.recorded_utc``.
    """
    unknown = set(fields) - {"site", "owner_name", "owner_phone", "installer", "tailscale_host", "box_host",
                             "app_version", "consent_live", "consent_recordings", "consent_training"}
    if unknown:
        raise RegistrationError(f"unknown field(s): {', '.join(sorted(unknown))}")
    old = load_registration(path) or {}
    box_site = load_box_config(box_yaml).site
    site = fields.get("site") or old.get("site") or box_site
    if not isinstance(site, str) or not _SITE_RE.fullmatch(site):
        raise RegistrationError("site must be lowercase letters, digits or underscores")
    if site != box_site:
        raise RegistrationError(f"site {site!r} does not match this box's site {box_site!r}")

    name = fields.get("owner_name")
    name = _text("owner_name", name if name is not None else old.get("owner_name") or site, 120)
    if not name:
        raise RegistrationError("owner_name must be 1 to 120 characters")

    old_consent = old.get("consent") or {}
    consent = {k: bool(old_consent.get(k, False)) for k in _CONSENTS}
    for k in _CONSENTS:
        if f"consent_{k}" in fields:
            if not isinstance(fields[f"consent_{k}"], bool):
                raise RegistrationError(f"consent_{k} must be true or false")
            consent[k] = fields[f"consent_{k}"]
    installer = _text("installer", fields.get("installer") if fields.get("installer") is not None
                      else old_consent.get("recorded_by", ""), 120)
    changed = any(consent[k] != bool(old_consent.get(k, False)) for k in _CONSENTS) \
        or installer != old_consent.get("recorded_by", "") or "recorded_utc" not in old_consent
    consent["recorded_utc"] = _now_utc() if changed else old_consent["recorded_utc"]
    consent["recorded_by"] = installer

    def pick(key: str, default: str) -> str:
        value = fields.get(key)
        return _text(key, value if value is not None else old.get(key, default), 200)

    reg = {
        "schema_version": SCHEMA_VERSION,
        "site": site,
        "owner_name": name,
        "owner_phone": pick("owner_phone", ""),
        "consent": consent,
        "installer": installer,
        "installed_utc": old.get("installed_utc") or _now_utc(),
        "tailscale_host": pick("tailscale_host", ""),
        "box_host": pick("box_host", socket.gethostname()),
        "app_version": pick("app_version", _app_version()),
    }
    _save(path, reg)
    return reg


def update_site(path: str, site: str) -> Optional[Dict[str, Any]]:
    """Point a saved registration at a new site (``set-site``). None when the box is not registered."""
    reg = load_registration(path)
    if reg is None:
        return None
    if not isinstance(site, str) or not _SITE_RE.fullmatch(site):
        raise RegistrationError("site must be lowercase letters, digits or underscores")
    if reg["site"] == site:
        return reg
    reg["site"] = site
    _save(path, reg)
    return reg


def register_from_json(src: str, path: str = REGISTRATION_PATH, box_yaml: str = BOX_YAML) -> Dict[str, Any]:
    """Register from a JSON file of the answers, then delete it (it holds a name and phone).

    Missing consents mean no; a missing owner name means the site name.
    """
    try:
        with open(src, encoding="utf-8-sig") as f:
            answers = json.load(f)
    except ValueError as exc:
        raise RegistrationError("the answers file is not valid JSON") from exc
    finally:
        try:
            os.remove(src)
        except OSError:
            pass
    if not isinstance(answers, dict):
        raise RegistrationError("the answers file must hold a JSON object")
    fields: Dict[str, Any] = {"owner_name": answers.get("owner_name") or load_box_config(box_yaml).site}
    for key in ("owner_phone", "installer", "tailscale_host"):
        fields[key] = answers.get(key) or ""
    for k in _CONSENTS:
        value = answers.get(f"consent_{k}", False)
        fields[f"consent_{k}"] = value if isinstance(value, bool) else False
    return write_registration(path, box_yaml=box_yaml, **fields)


def registration_summary(reg: Dict[str, Any]) -> str:
    """One line for the console; never include owner details."""
    c = reg.get("consent", {})
    yes = lambda k: "yes" if c.get(k) else "no"  # noqa: E731
    return (f"registered {reg.get('site')}; consent: live {yes('live')}, "
            f"recordings {yes('recordings')}, training {yes('training')}")


def put_registration(reg: Dict[str, Any], bucket: str, prefix: str, s3_client: Any = None,
                     published_path: str = PUBLISHED_PATH) -> Optional[str]:
    """Write *reg* to ``<prefix>/_status/registration.json`` unless it is already there.

    The hash of the last published content, bucket and key is kept in *published_path*,
    so the hourly upload sends the file only after it changes. Returns the key, or None when skipped.
    """
    body = json.dumps(reg, indent=2, ensure_ascii=False, sort_keys=True).encode("utf-8")
    key = f"{prefix}/{STATUS_KEY}"
    digest = hashlib.sha256(bucket.encode("utf-8") + b"\0" + key.encode("utf-8") + b"\0" + body).hexdigest()
    try:
        with open(published_path, encoding="utf-8") as f:
            if f.read().strip() == digest:
                return None
    except OSError:
        pass
    if s3_client is None:
        import boto3

        s3_client = boto3.client("s3")
    s3_client.put_object(Bucket=bucket, Key=key, Body=body, ContentType="application/json")
    _atomic_text(published_path, digest)
    return key
