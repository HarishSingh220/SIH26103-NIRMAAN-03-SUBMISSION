"""
JWT token utilities using only Python stdlib + the already-installed
`cryptography` package (no python-jose / PyJWT dependency required).

Implements HS256 (HMAC-SHA256) signed JWTs.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import secrets
import time
from typing import Optional

# ──────────────────────────────────────────────────────────────────────────────
# Secret key  (read from env; generate a random one if not set)
# ──────────────────────────────────────────────────────────────────────────────

SECRET_KEY: str = os.environ.get("SECRET_KEY", secrets.token_hex(32))
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_SECONDS: int = int(
    os.environ.get("ACCESS_TOKEN_EXPIRE_SECONDS", str(60 * 60 * 24 * 7))  # 7 days default
)


# ──────────────────────────────────────────────────────────────────────────────
# Low-level JWT helpers
# ──────────────────────────────────────────────────────────────────────────────

def _b64url_encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _b64url_decode(s: str) -> bytes:
    padding = 4 - len(s) % 4
    if padding != 4:
        s += "=" * padding
    return base64.urlsafe_b64decode(s)


def _sign(header_payload: str, key: str) -> str:
    sig = hmac.new(
        key.encode("utf-8"),
        header_payload.encode("utf-8"),
        hashlib.sha256,
    ).digest()
    return _b64url_encode(sig)


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def create_access_token(
    user_id: int,
    expire_seconds: Optional[int] = None,
) -> str:
    """Create a signed JWT carrying `sub=<user_id>`."""
    exp = int(time.time()) + (expire_seconds or ACCESS_TOKEN_EXPIRE_SECONDS)
    header = _b64url_encode(json.dumps({"alg": ALGORITHM, "typ": "JWT"}).encode())
    payload = _b64url_encode(
        json.dumps({"sub": str(user_id), "exp": exp, "iat": int(time.time())}).encode()
    )
    header_payload = f"{header}.{payload}"
    sig = _sign(header_payload, SECRET_KEY)
    return f"{header_payload}.{sig}"


def decode_access_token(token: str) -> Optional[dict]:
    """Decode + verify a JWT. Returns payload dict or None if invalid/expired."""
    try:
        parts = token.split(".")
        if len(parts) != 3:
            return None
        header_payload = f"{parts[0]}.{parts[1]}"
        expected_sig = _sign(header_payload, SECRET_KEY)
        if not secrets.compare_digest(expected_sig, parts[2]):
            return None
        payload = json.loads(_b64url_decode(parts[1]))
        if payload.get("exp", 0) < int(time.time()):
            return None  # expired
        return payload
    except Exception:
        return None
