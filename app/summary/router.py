"""FastAPI routes for the LangGraph + Gemini/Qwen GenAI summary service."""
from __future__ import annotations

from fastapi import APIRouter, Body, HTTPException

from app.common.http import run_offloaded

from . import config as cfg
from .schemas import (
    SUMMARY_BATCH_EXAMPLE,
    SUMMARY_PREDICTION_EXAMPLE,
    SummaryBatchRequest,
    SummaryBatchResponse,
    SummaryRequest,
    SummaryResponse,
)
from .service import summarize_project

router = APIRouter()


@router.get("/health")
def health():
    primary_configured = bool(cfg.GOOGLE_API_KEY)
    fallback_configured = bool(cfg.HUGGINGFACEHUB_API_TOKEN)
    configured = primary_configured or fallback_configured
    notes = []
    if not primary_configured:
        notes.append("Set GOOGLE_API_KEY to enable the primary Gemini model.")
    if not fallback_configured:
        notes.append("Set HUGGINGFACEHUB_API_TOKEN to enable the Qwen fallback model.")
    return {
        "status": "ok" if configured else "unconfigured",
        "configured": configured,
        "provider": "google-gemini + huggingface",
        "primary_model": cfg.GEMINI_MODEL_ID,
        "fallback_models": [cfg.QWEN_MODEL_ID],
        "model_version": cfg.MODEL_VERSION,
        "note": None if (primary_configured and fallback_configured) else " ".join(notes),
    }


@router.get("/schema")
def get_schema():
    """Return the request examples and model configuration used by GenAI endpoints."""
    return {
        "model_version": cfg.MODEL_VERSION,
        "provider": "google-gemini + huggingface",
        "primary_model": cfg.GEMINI_MODEL_ID,
        "fallback_models": [cfg.QWEN_MODEL_ID],
        "endpoints": {
            "single": {
                "path": "/summary/generate",
                "method": "POST",
                "example_request": {"prediction": SUMMARY_PREDICTION_EXAMPLE, "project_name": "Doubling of XYZ Rail Line"},
            },
            "batch": {
                "path": "/summary/generate/batch",
                "method": "POST",
                "example_request": SUMMARY_BATCH_EXAMPLE,
                "max_projects": cfg.MAX_BATCH_PROJECTS,
            },
        },
    }


@router.post("/generate", response_model=SummaryResponse)
async def generate_summary(
    body: SummaryRequest = Body(..., openapi_examples={
        "prediction_result": {
            "summary": "Single project's prediction output",
            "value": {"prediction": SUMMARY_PREDICTION_EXAMPLE, "project_name": "Doubling of XYZ Rail Line"},
        }
    }),
):
    result = await run_offloaded(summarize_project, body.prediction, body.project_name)
    return SummaryResponse(**result)


@router.post("/generate/batch", response_model=SummaryBatchResponse)
async def generate_summary_batch(
    body: SummaryBatchRequest = Body(..., openapi_examples={
        "multiple_projects": {
            "summary": "Multiple prediction outputs",
            "value": SUMMARY_BATCH_EXAMPLE,
        }
    }),
):
    if len(body.predictions) > cfg.MAX_BATCH_PROJECTS:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Batch has {len(body.predictions)} projects, which exceeds the "
                f"{cfg.MAX_BATCH_PROJECTS}-project limit for /summary/generate/batch."
            ),
        )

    summaries = []
    for prediction in body.predictions:
        result = await run_offloaded(summarize_project, prediction, None)
        summaries.append(SummaryResponse(**result))

    used_models = {s.model for s in summaries if s.model != "none"}
    model_label = next(iter(used_models), cfg.GEMINI_MODEL_ID)
    return SummaryBatchResponse(model=model_label, summaries=summaries)
