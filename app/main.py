"""Combined inference API: PAIMANA overrun-risk classification + cost/time
overrun regression + hybrid anomaly detection, served from one FastAPI app.

- Overrun-risk classification lives under `/classification/*` (see
  app/classification/router.py). Predicts the *probability* a project will
  have a cost/time overrun.
- Cost/time overrun regression lives under `/overrun/*` (see
  app/overrun/router.py). Predicts overrun *magnitude* (%). Can be called
  standalone, or gated by a classifier's yes/no risk flags per target.
- Anomaly detection lives under `/anomaly/*` (see app/anomaly/router.py).
  Flags anomalous reporting patterns. Always runs standalone when called
  directly.
- `/gateway/predict` (see app/gateway/router.py) chains all three: it calls
  classification first, and only calls regression/anomaly for a project when
  the classifier's predicted probability is above 0.5. The three services'
  code is never merged together -- the gateway only imports and composes
  each one's already-existing public pieces.
- `/summary/*` (see app/summary/router.py) turns any of the above
  prediction responses into a short, human-readable summary using a free
  Google Gemini model (primary) with a Hugging Face-hosted Qwen model as
  fallback, orchestrated with LangChain + LangGraph.
  It calls no local model and loads nothing at startup; it's usable as long as
  GOOGLE_API_KEY and/or HUGGINGFACEHUB_API_TOKEN is set.

All three services train offline against their own dataset (never inside a
request handler) and are loaded once at startup, all writing under one
shared `models/` root (see each service's README setup step):

    python -m app.anomaly.train                          # -> models/anomaly_model.joblib
    python -m app.overrun.train --target both             # -> models/*.pkl
    (see app/classification/training/ for the classifier's offline
    train_cost_v4.py / train_time_model.py scripts)        # -> models/classification/*.joblib

Run locally:
    uvicorn app.main:app --host 0.0.0.0 --port 8000

Run via Docker: see Dockerfile.
"""
from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv

# Must run before any `app.*` module is imported below -- every config.py in
# this project reads its environment variables (GOOGLE_API_KEY,
# HUGGINGFACEHUB_API_TOKEN, MODEL_DIR, OVERRUN_RISK_PROBABILITY_THRESHOLD,
# ...) at IMPORT time via
# `os.environ.get(...)`, so `.env` has to be loaded into the process
# environment first. Looks for a `.env` file in the current working
# directory (where you run `uvicorn` from); does nothing if it's not
# there, so real environment variables / Docker `-e` flags still work
# unchanged and always take precedence over `.env` (override=False).
load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=False)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Scikit-learn version compatibility: register the bare '_loss' module name
# expected by artifacts trained on sklearn 1.8.x so they can be loaded under
# 1.9.x (where the same code lives at sklearn._loss).
import app.common.compat  # noqa: F401

from app.anomaly import router as anomaly_router
from app.anomaly import state as anomaly_state
from app.auth import router as auth_router
from app.auth.db import init_db as init_auth_db
from app.classification import router as classification_router
from app.classification import state as classification_state
from app.common.http import logger, register_error_handling
from app.gateway import router as gateway_router
from app.overrun import router as overrun_router
from app.overrun import state as overrun_state
from app.predict_risk import router as predict_risk_router
from app.summary import config as summary_cfg
from app.summary import router as summary_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialise the auth database (creates tables if they don't exist).
    init_auth_db()
    # All three artifacts are built offline (see module docstring). Fail
    # fast and loudly at startup if any is missing, rather than serving
    # 503s from every prediction endpoint until someone notices.
    anomaly_state.load()
    overrun_state.load()
    classification_state.load()
    yield
    anomaly_state.unload()
    overrun_state.unload()
    classification_state.unload()
    logger.info("All model artifacts unloaded; shutting down.")


app = FastAPI(
    title="NIRMAAN / PAIMANA Combined Prediction API",
    version="combined-v2",
    description=(
        "Production inference API combining three independently trained "
        "services: overrun-risk classification (`/classification/*`), "
        "cost/time overrun regression (`/overrun/*`), and hybrid anomaly "
        "detection (`/anomaly/*`) -- plus `/gateway/predict`, which chains "
        "them so regression and anomaly detection only run when the "
        "classifier's predicted overrun probability is above 0.5. Each "
        "service keeps its own dataset, training script, and model "
        "artifact(s); this app only loads all three and exposes them "
        "behind one process. "
        "Frontend-facing endpoints: POST /predict-risk (risk analysis), "
        "POST /api/auth/register, POST /api/auth/login, GET /api/auth/me."
    ),
    lifespan=lifespan,
)

# ─────────────────────────────────────────────────────────────────────────────
# CORS — allow the Vite dev server (and any configured FRONTEND_URL) to reach
# this API. The allowed origin is configurable via the FRONTEND_URL env var.
# ─────────────────────────────────────────────────────────────────────────────
_frontend_url = os.environ.get("FRONTEND_URL", "http://localhost:5173")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[_frontend_url],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

register_error_handling(app)

app.include_router(classification_router.router, prefix="/classification", tags=["overrun-risk-classification"])
app.include_router(overrun_router.router, prefix="/overrun", tags=["cost-time-overrun"])
app.include_router(anomaly_router.router, prefix="/anomaly", tags=["anomaly-detection"])
app.include_router(gateway_router.router, prefix="/gateway", tags=["gated-pipeline"])
app.include_router(summary_router.router, prefix="/summary", tags=["genai-summary"])
# Frontend-facing endpoints
app.include_router(predict_risk_router.router, prefix="/predict-risk", tags=["frontend-predict-risk"])
app.include_router(auth_router.router, prefix="/api/auth", tags=["auth"])


@app.get("/health", tags=["combined"])
def health():
    """Aggregate health check across all three prediction services. Each
    service also has its own `/classification/health` / `/overrun/health` /
    `/anomaly/health` for a per-service view. `/summary` is reported here
    for visibility but never flips the overall status to "degraded": it
    calls external Google Gemini / Hugging Face APIs rather than a locally
    loaded model, so its availability is independent of the other three."""
    classification_ok = classification_state.is_loaded()
    anomaly_ok = anomaly_state.is_loaded()
    overrun_ok = overrun_state.is_loaded()
    summary_configured = bool(summary_cfg.GOOGLE_API_KEY or summary_cfg.HUGGINGFACEHUB_API_TOKEN)
    return {
        "status": "ok" if (classification_ok and anomaly_ok and overrun_ok) else "degraded",
        "services": {
            "overrun_risk_classification": {"model_loaded": classification_ok},
            "cost_time_overrun": {"model_loaded": overrun_ok},
            "anomaly_detection": {"model_loaded": anomaly_ok},
            "genai_summary": {"configured": summary_configured},
        },
    }
