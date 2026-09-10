"""Load/unload/access for the cost-overrun and time-overrun
`CombinedOverrunPipeline` artifacts.

Split out of the FastAPI app object so `app/main.py`'s lifespan can load and
unload both services' artifacts uniformly, and so `app/overrun/router.py`
never touches bare module-level globals directly.
"""
from __future__ import annotations

import logging
import os
import pickle

from fastapi import HTTPException

from . import config as cfg
from .core import CombinedOverrunPipeline  # noqa: F401  (import needed for unpickling)

logger = logging.getLogger("paimana.api")

_COST_PIPELINE: CombinedOverrunPipeline | None = None
_TIME_PIPELINE: CombinedOverrunPipeline | None = None


def load() -> None:
    global _COST_PIPELINE, _TIME_PIPELINE
    for label, path in (("cost", cfg.COST_MODEL_PATH), ("time", cfg.TIME_MODEL_PATH)):
        if not os.path.exists(path):
            logger.error("Overrun %s model artifact not found at %s", label, path)
            raise RuntimeError(
                f"Overrun {label}-overrun model artifact not found at {path}. "
                "Build it offline with `python -m app.overrun.train --target both`."
            )
    try:
        with open(cfg.COST_MODEL_PATH, "rb") as f:
            _COST_PIPELINE = pickle.load(f)
        with open(cfg.TIME_MODEL_PATH, "rb") as f:
            _TIME_PIPELINE = pickle.load(f)
    except Exception:
        logger.exception("Failed to load overrun model artifacts")
        raise
    logger.info(
        "Loaded overrun pipelines: cost=%s time=%s", cfg.COST_MODEL_PATH, cfg.TIME_MODEL_PATH
    )


def unload() -> None:
    global _COST_PIPELINE, _TIME_PIPELINE
    _COST_PIPELINE = None
    _TIME_PIPELINE = None
    logger.info("Overrun model artifacts unloaded.")


def is_loaded() -> bool:
    return _COST_PIPELINE is not None and _TIME_PIPELINE is not None


def get_cost() -> CombinedOverrunPipeline:
    if _COST_PIPELINE is None:
        raise HTTPException(status_code=503, detail="Cost-overrun model is not loaded")
    return _COST_PIPELINE


def get_time() -> CombinedOverrunPipeline:
    if _TIME_PIPELINE is None:
        raise HTTPException(status_code=503, detail="Time-overrun model is not loaded")
    return _TIME_PIPELINE
