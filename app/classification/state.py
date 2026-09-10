"""Load/unload/access for the overrun-risk classification coordinator.

Split out of the FastAPI app object so `app/main.py`'s lifespan can load and
unload all three services' artifacts uniformly, and so
`app/classification/router.py` never touches the coordinator singleton
directly.
"""
from __future__ import annotations

import logging
import os

from fastapi import HTTPException

from . import config as cfg
from .coordinator import ModelCoordinator

logger = logging.getLogger("paimana.api")

_COORDINATOR: ModelCoordinator | None = None


def load() -> None:
    global _COORDINATOR
    for label, path in (("cost", cfg.COST_MODEL_PATH), ("time", cfg.TIME_MODEL_PATH)):
        if not os.path.exists(path):
            logger.error("Classification %s model artifact not found at %s", label, path)
            raise RuntimeError(
                f"Classification {label} model artifact not found at {path}. "
                "Retrain with `app/classification/training/train_"
                f"{label}{'_v4' if label == 'cost' else '_model'}.py` "
                "and place the artifact under models/classification/."
            )
    try:
        _COORDINATOR = ModelCoordinator(cfg.COST_MODEL_PATH, cfg.TIME_MODEL_PATH)
    except Exception:
        logger.exception("Failed to load classification model artifacts")
        raise
    logger.info(
        "Loaded classification models: cost=%s time=%s", cfg.COST_MODEL_PATH, cfg.TIME_MODEL_PATH
    )


def unload() -> None:
    global _COORDINATOR
    _COORDINATOR = None
    logger.info("Classification model artifacts unloaded.")


def is_loaded() -> bool:
    return _COORDINATOR is not None


def get() -> ModelCoordinator:
    if _COORDINATOR is None:
        raise HTTPException(status_code=503, detail="Classification models are not loaded")
    return _COORDINATOR
