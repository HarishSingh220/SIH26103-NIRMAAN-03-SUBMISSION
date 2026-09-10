"""
FastAPI router for NIRMAAN user authentication.

Endpoints:
  POST /api/auth/register  — Create a new account
  POST /api/auth/login     — Authenticate and receive a JWT
  GET  /api/auth/me        — Return current user info (requires Bearer token)
  POST /api/auth/logout    — Client-side; always returns 200 (token is stateless)

All responses use a consistent JSON envelope:
  { "success": true,  "data": { ... } }
  { "success": false, "error": { "code": "...", "message": "..." } }
"""
from __future__ import annotations

import logging
import re

from fastapi import APIRouter, Depends, Header, HTTPException, status
from pydantic import BaseModel, EmailStr, Field, field_validator

from .db import UserExistsError, create_user, get_user_by_id, get_user_by_login, init_db
from .tokens import create_access_token, decode_access_token

logger = logging.getLogger("paimana.api")

router = APIRouter()


# ──────────────────────────────────────────────────────────────────────────────
# Request / Response schemas
# ──────────────────────────────────────────────────────────────────────────────

class RegisterRequest(BaseModel):
    fullName: str = Field(..., min_length=1, max_length=120)
    email: str = Field(..., min_length=3, max_length=254)
    password: str = Field(..., min_length=6, max_length=128)
    confirmPassword: str = Field(..., min_length=6, max_length=128)

    @field_validator("email")
    @classmethod
    def validate_email(cls, v: str) -> str:
        if not re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", v):
            raise ValueError("Invalid email address.")
        return v.lower().strip()

    @field_validator("fullName")
    @classmethod
    def validate_name(cls, v: str) -> str:
        return v.strip()


class LoginRequest(BaseModel):
    username: str = Field(..., min_length=1, max_length=254, description="Email or username")
    password: str = Field(..., min_length=1, max_length=128)


def _user_public(user: dict) -> dict:
    return {
        "id": user["id"],
        "fullName": user["full_name"],
        "email": user["email"],
        "username": user["username"],
        "createdAt": user.get("created_at"),
        "role": "Project Officer",
    }


# ──────────────────────────────────────────────────────────────────────────────
# Auth dependency
# ──────────────────────────────────────────────────────────────────────────────

def _current_user_id(authorization: str = Header(None)) -> int:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"code": "UNAUTHORIZED", "message": "Authentication required."},
            headers={"WWW-Authenticate": "Bearer"},
        )
    token = authorization[len("Bearer "):]
    payload = decode_access_token(token)
    if payload is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"code": "INVALID_TOKEN", "message": "Token is invalid or has expired."},
            headers={"WWW-Authenticate": "Bearer"},
        )
    return int(payload["sub"])


# ──────────────────────────────────────────────────────────────────────────────
# Endpoints
# ──────────────────────────────────────────────────────────────────────────────

@router.on_event("startup")
def _startup():
    init_db()


@router.post("/register", status_code=status.HTTP_201_CREATED)
def register(body: RegisterRequest):
    """Create a new NIRMAAN account."""
    if body.password != body.confirmPassword:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={"code": "VALIDATION_ERROR", "message": "Passwords do not match."},
        )
    try:
        user = create_user(
            full_name=body.fullName,
            email=body.email,
            password=body.password,
        )
    except UserExistsError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={"code": "USER_EXISTS", "message": str(exc)},
        )
    except Exception:
        logger.exception("Unexpected error during registration")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"code": "SERVER_ERROR", "message": "Registration failed. Please try again."},
        )
    token = create_access_token(user["id"])
    return {"success": True, "data": {"token": token, "user": _user_public(user)}}


@router.post("/login")
def login(body: LoginRequest):
    """Authenticate and receive a JWT token."""
    user = get_user_by_login(body.username)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"code": "INVALID_CREDENTIALS", "message": "Invalid username or password."},
        )
    from .db import verify_password
    if not verify_password(body.password, user["hashed_pw"], user["pw_salt"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"code": "INVALID_CREDENTIALS", "message": "Invalid username or password."},
        )
    token = create_access_token(user["id"])
    return {"success": True, "data": {"token": token, "user": _user_public(user)}}


@router.get("/me")
def me(user_id: int = Depends(_current_user_id)):
    """Return the currently authenticated user's profile."""
    user = get_user_by_id(user_id)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "USER_NOT_FOUND", "message": "User account not found."},
        )
    return {"success": True, "data": {"user": _user_public(user)}}


@router.post("/logout")
def logout():
    """Stateless logout — client must discard the token locally."""
    return {"success": True, "data": {"message": "Logged out successfully."}}
