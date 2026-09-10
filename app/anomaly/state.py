"""Load/unload/access for the anomaly-detection model artifact.

Split out of the FastAPI app object so `app/main.py`'s lifespan can load and
unload both services' artifacts uniformly, and so `app/anomaly/router.py`
never touches a bare module-level global directly.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

import joblib
from fastapi import HTTPException

from . import config as cfg
from app.common.paths import PROJECT_ROOT

logger = logging.getLogger("paimana.api")

BASE_DIR = Path(__file__).resolve().parents[2]
_MODEL_DEFAULT = cfg.MODEL_PATH
_raw_model_path = os.getenv("ANOMALY_MODEL_ARTIFACT", os.getenv("MODEL_ARTIFACT"))
MODEL_PATH = str(Path(_raw_model_path) if _raw_model_path else Path(_MODEL_DEFAULT))
if not os.path.isabs(MODEL_PATH):
    MODEL_PATH = str(PROJECT_ROOT / MODEL_PATH)

_ARTIFACT: dict | None = None


def load() -> None:
    global _ARTIFACT
    if not os.path.exists(MODEL_PATH):
        logger.error("Anomaly model artifact not found at %s", MODEL_PATH)
        raise RuntimeError(
            f"Anomaly model artifact not found at {MODEL_PATH}. "
            "Build it offline with `python -m app.anomaly.train`."
        )
    try:
        _ARTIFACT = joblib.load(MODEL_PATH)
    except Exception:
        logger.exception("Failed to load anomaly model artifact from %s", MODEL_PATH)
        raise
    logger.info(
        "Loaded anomaly model artifact from %s (model_version=%s, fit_cutoff=%s)",
        MODEL_PATH,
        _ARTIFACT.get("model_version", cfg.MODEL_VERSION),
        _ARTIFACT.get("fit_cutoff"),
    )


def unload() -> None:
    global _ARTIFACT
    _ARTIFACT = None
    logger.info("Anomaly model artifact unloaded.")


def is_loaded() -> bool:
    return _ARTIFACT is not None


def get() -> dict:
    """Return the loaded artifact, or raise 503 if the service hasn't
    finished starting up (or startup failed)."""
    if _ARTIFACT is None:
        raise HTTPException(status_code=503, detail="Anomaly model is not loaded")
    return _ARTIFACT


def get_or_none() -> dict | None:
    """Same as `get()` but returns None instead of raising -- for read-only
    status checks (e.g. /health) that should report "not loaded" rather
    than fail with a 503."""
    return _ARTIFACT
