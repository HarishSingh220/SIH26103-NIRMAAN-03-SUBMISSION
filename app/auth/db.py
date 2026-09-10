"""
SQLite-backed user storage for NIRMAAN auth.

Uses only Python stdlib (sqlite3, hashlib, secrets) and the already-installed
`cryptography` package (for HMAC-SHA256 password hashing). No passlib or
bcrypt dependency required.
"""
from __future__ import annotations

import hashlib
import logging
import os
import secrets
import sqlite3
from pathlib import Path
from typing import Optional

from app.common.paths import PROJECT_ROOT

logger = logging.getLogger("paimana.api")

# ──────────────────────────────────────────────────────────────────────────────
# Database location
# ──────────────────────────────────────────────────────────────────────────────

_DB_PATH: str = os.environ.get(
    "DATABASE_URL",
    str(PROJECT_ROOT / "nirmaan_auth.db"),
)

# ──────────────────────────────────────────────────────────────────────────────
# Schema
# ──────────────────────────────────────────────────────────────────────────────

_CREATE_USERS = """
CREATE TABLE IF NOT EXISTS users (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    full_name   TEXT    NOT NULL,
    email       TEXT    NOT NULL UNIQUE,
    username    TEXT    NOT NULL UNIQUE,
    hashed_pw   TEXT    NOT NULL,
    pw_salt     TEXT    NOT NULL,
    created_at  TEXT    NOT NULL DEFAULT (datetime('now'))
);
"""


def _get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(_DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """Create tables if they don't already exist. Safe to call on every startup."""
    with _get_conn() as conn:
        conn.execute(_CREATE_USERS)
        conn.commit()
    logger.info("Auth DB initialised at %s", _DB_PATH)


# ──────────────────────────────────────────────────────────────────────────────
# Password hashing  (PBKDF2-HMAC-SHA256 via stdlib, 260 000 iterations)
# ──────────────────────────────────────────────────────────────────────────────

_ITERATIONS = 260_000
_HASH_ALG = "sha256"


def _hash_password(password: str, salt: Optional[str] = None) -> tuple[str, str]:
    """Returns (hashed_hex, salt_hex). Pass an existing salt to verify."""
    if salt is None:
        salt = secrets.token_hex(32)
    dk = hashlib.pbkdf2_hmac(
        _HASH_ALG,
        password.encode("utf-8"),
        salt.encode("utf-8"),
        _ITERATIONS,
    )
    return dk.hex(), salt


def verify_password(password: str, hashed_hex: str, salt_hex: str) -> bool:
    computed, _ = _hash_password(password, salt_hex)
    return secrets.compare_digest(computed, hashed_hex)


# ──────────────────────────────────────────────────────────────────────────────
# User CRUD
# ──────────────────────────────────────────────────────────────────────────────

class UserExistsError(ValueError):
    """Raised when email or username is already taken."""


def create_user(
    full_name: str,
    email: str,
    password: str,
) -> dict:
    """Create a new user. Raises UserExistsError if email is already taken."""
    username = email.split("@")[0].lower()  # derive a default username from email
    hashed, salt = _hash_password(password)
    try:
        with _get_conn() as conn:
            conn.execute(
                "INSERT INTO users (full_name, email, username, hashed_pw, pw_salt) VALUES (?, ?, ?, ?, ?)",
                (full_name.strip(), email.strip().lower(), username, hashed, salt),
            )
            conn.commit()
            row = conn.execute(
                "SELECT id, full_name, email, username, created_at FROM users WHERE email = ?",
                (email.strip().lower(),),
            ).fetchone()
            return dict(row)
    except sqlite3.IntegrityError as exc:
        raise UserExistsError(f"Email '{email}' is already registered.") from exc


def get_user_by_login(identifier: str) -> Optional[dict]:
    """Find by email OR username. Returns full row including hashed_pw/salt."""
    ident = identifier.strip().lower()
    with _get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM users WHERE email = ? OR username = ?",
            (ident, ident),
        ).fetchone()
    return dict(row) if row else None


def get_user_by_id(user_id: int) -> Optional[dict]:
    with _get_conn() as conn:
        row = conn.execute(
            "SELECT id, full_name, email, username, created_at FROM users WHERE id = ?",
            (user_id,),
        ).fetchone()
    return dict(row) if row else None
