"""Stable pseudonyms that replace customer identity for labelers.

`customer-<6 hex>` and `cam-<6 hex>` are HMAC-SHA256 digests under a key derived from the JWT secret, so they are
stable across requests and restarts but cannot be reversed or recomputed without the server secret.
"""
from __future__ import annotations

import hashlib
import hmac

_DOMAIN = b"home-guard-admin/pseudonym/v1"


def _key(secret: str) -> bytes:
    """A sub-key for pseudonyms only, so the digests never act as JWT-secret oracles."""
    return hmac.new(secret.encode("utf-8"), _DOMAIN, hashlib.sha256).digest()


def _short(secret: str, msg: str) -> str:
    return hmac.new(_key(secret), msg.encode("utf-8"), hashlib.sha256).hexdigest()[:6]


def customer(secret: str, customer_id: int) -> str:
    """Pseudonym of a customer; also used as the site name a labeler sees."""
    return "customer-" + _short(secret, f"customer:{int(customer_id)}")


def camera(secret: str, site: str, camera_name: str) -> str:
    """Pseudonym of one camera of one site (the same camera name on two sites gets two pseudonyms)."""
    return "cam-" + _short(secret, f"camera:{site}\x00{camera_name}")


def staff(secret: str, staff_id: int) -> str:
    """Pseudonym of a staff member (who labeled a clip), for training exports that never name anyone."""
    return "labeler-" + _short(secret, f"staff:{int(staff_id)}")
