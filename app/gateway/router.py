"""Backward-compatible alias for the classification pipeline.

Historically this module chained all three PAIMANA services together and
required the caller to describe a project TWICE -- once in the
classification service's own schema (metadata + reporting_periods) and once
more in the raw schema the regression/anomaly services use.

That two-schema requirement is gone: the raw schema (see
`app.common.raw_features`) now covers everything every service needs,
`app.classification.raw_adapter` derives the classification-specific shape
from it internally, and `app.classification.pipeline` runs the classifier
then automatically gates regression + anomaly off of it -- all from ONE
caller-supplied payload. See `app/classification/router.py`, which is now
the primary entry point for the whole pipeline.

`/gateway/predict` and `/gateway/predict/batch` are kept as thin aliases,
delegating to the exact same pipeline, purely so existing callers of this
path keep working. Same input contract as `/classification/*`: a one-row JSON array for
`/predict`, and an explicit `{"projects": [{"project_code": ..., "rows": [...]}, ...]}`
body for `/predict/batch`.
"""
from __future__ import annotations

import pandas as pd
from fastapi import APIRouter, Body, HTTPException, Query

from app.classification import config as classification_cfg
from app.classification.pipeline import run_pipeline
from app.common.http import run_offloaded
from app.common.raw_features import EXAMPLE_BATCH_REQUEST, EXAMPLE_ROW
from app.common.schemas import BatchPredictionRequest, SingleRowsField, flatten_batch_projects
from app.overrun import config as overrun_cfg
from app.overrun.router import _valid_preferred_model

router = APIRouter()


@router.post("/predict")
async def gated_predict(
    rows: SingleRowsField = Body(
        ...,
        description=(
            "A JSON array with EXACTLY ONE project row (see GET "
            "/classification/schema). Alias for POST /classification/predict "
            "-- kept for backward compatibility."
        ),
        openapi_examples={"single_row": {"summary": "One project row", "value": [EXAMPLE_ROW]}},
    ),
    preferred_model: str = Query(overrun_cfg.DEFAULT_PREFERRED_MODEL),
):
    """Run the classifier first; automatically run the cost/time-overrun
    regression and the anomaly detector for whichever target(s) the
    classifier flagged as overrun risk (predicted probability above the
    threshold from `GET /gateway/config`).

    This is a thin alias for `POST /classification/predict` -- see that
    endpoint (and `app.classification.pipeline`) for the full contract.
    Only raw, prediction-time fields are accepted; every engineered feature
    is computed internally from this single payload.
    """
    _valid_preferred_model(preferred_model)
    if len(rows) != 1:
        raise HTTPException(
            status_code=422,
            detail=(
                f"POST /gateway/predict takes exactly one row (got {len(rows)}). "
                "Use POST /gateway/predict/batch for multiple rows or projects."
            ),
        )
    raw = pd.DataFrame(rows)
    result = await run_offloaded(run_pipeline, raw, preferred_model)
    if not result["predictions"]:
        raise HTTPException(422, "No project could be scored.")
    return result["predictions"][0]


@router.post("/predict/batch")
async def gated_predict_batch(
    body: BatchPredictionRequest = Body(
        ...,
        description=(
            "Explicit multi-project batch payload. Alias for "
            "POST /classification/predict/batch."
        ),
        openapi_examples={"multiple_projects": {"summary": "Multiple projects with quarterly history", "value": EXAMPLE_BATCH_REQUEST}},
    ),
    preferred_model: str = Query(overrun_cfg.DEFAULT_PREFERRED_MODEL),
):
    """Alias for `POST /classification/predict/batch`."""
    _valid_preferred_model(preferred_model)
    raw = pd.DataFrame(flatten_batch_projects(body))
    return await run_offloaded(run_pipeline, raw, preferred_model)


@router.get("/config")
def gateway_config():
    return {
        "overrun_risk_probability_threshold": classification_cfg.OVERRUN_RISK_PROBABILITY_THRESHOLD,
        "note": (
            "cost/time-overrun regression and anomaly detection only run "
            "for a project when the classification service's predicted "
            "probability for that target is strictly greater than this "
            "threshold. Prefer POST /classification/predict, "
            "/classification/predict/batch, and /classification/predict/csv "
            "-- this router is now a thin backward-compatible alias for "
            "them."
        ),
    }
