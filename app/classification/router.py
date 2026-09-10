"""FastAPI router for the overrun-risk classification service.

Mounted under the `/classification` prefix by `app/main.py`. This is now the
SINGLE entry point for the whole PAIMANA pipeline: a caller supplies raw,
prediction-time project data here -- exactly once, in the same raw schema
used by the overrun-regression and anomaly-detection services (see
`app.common.raw_features` / `GET /classification/schema`) -- and this router:

  1. Derives every engineered feature from that raw data internally
     (`app.common.raw_features.derive_prediction_features`).
  2. Runs the overrun-risk classifier on it.
  3. Automatically runs the cost/time-overrun regression service
     (`app.overrun`) and the anomaly-detection service (`app.anomaly`) on
     the SAME already-derived data, gated on the classifier's own predicted
     probability (> `OVERRUN_RISK_PROBABILITY_THRESHOLD`, default 0.5).

The caller never supplies the project a second time and never precomputes
an engineered feature -- see `app.classification.pipeline` for the full
orchestration logic and `app.classification.raw_adapter` for how raw rows
are adapted into this service's own metadata + reporting_periods shape.

Endpoints:
- GET  /classification/health        - Health check
- GET  /classification/schema        - Raw input schema (required/optional
                                        columns + example row)
- POST /classification/predict       - Single project: a JSON array with
                                        EXACTLY ONE raw row
- POST /classification/predict/batch - Many raw rows across one or more
                                        projects, grouped by project_code
- POST /classification/predict/csv   - Same as /predict/batch, but the raw
                                        rows come from an uploaded CSV file

All three prediction endpoints return the same combined shape per project:
`classification` (cost/time overrun probability + label), `risk_gate` (the
threshold decision), `overrun` (latest regression result plus a
`quarterly` list -- one entry per supplied quarter, for time-trend graphs
-- or a `reason` block explaining why it was skipped), and `anomaly`
(latest score/risk_level plus its own `quarterly` list, or a `reason` block
explaining why it was skipped). `/predict/batch` and `/predict/csv` always
score every row they receive: a project may appear across many rows (its
reporting-period history) and there is no `include_history` toggle --
history is always used to build features AND always returned row-by-row.

Every `overrun.cost_overrun`/`overrun.time_overrun` block carries both the
predicted PERCENTAGE and the exact outcome it implies (see
`app.overrun.outcomes`): `predicted_final_cost_rs_cr`/
`predicted_cost_increase_rs_cr` for cost, `predicted_final_commissioning_date`/
`predicted_delay_months` for time.
"""
from __future__ import annotations


import pandas as pd
from fastapi import APIRouter, Body, File, HTTPException, Query, UploadFile, status
from starlette.concurrency import run_in_threadpool

from app.common.http import MAX_CSV_BYTES, parse_csv_bytes, read_upload_capped, run_offloaded
from app.common.raw_features import EXAMPLE_BATCH_REQUEST, EXAMPLE_ROW, RAW_OPTIONAL_COLS, RAW_REQUIRED_COLS
from app.common.schemas import BatchPredictionRequest, SingleRowsField, flatten_batch_projects
from app.overrun import config as overrun_cfg
from app.overrun.router import _valid_preferred_model

from . import config as cfg
from . import state
from .pipeline import run_pipeline

router = APIRouter()


# ==============================================================================
# ENDPOINTS
# ==============================================================================

@router.get("/health")
def health():
    """Health check. Returns OK if the classification models are loaded."""
    if not state.is_loaded():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Classification models are not loaded",
        )
    coordinator = state.get()
    model_info = coordinator.get_model_info()
    return {
        "status": "ok",
        "model_version": cfg.MODEL_VERSION,
        "risk_probability_threshold": cfg.OVERRUN_RISK_PROBABILITY_THRESHOLD,
        "models_loaded": {
            "cost": model_info["cost_model"]["target"],
            "time": model_info["time_model"]["target"],
        },
    }


@router.get("/schema")
def get_schema():
    """Raw, prediction-time input schema -- the SAME schema used by
    `/overrun/*` and `/anomaly/*`. Every engineered feature (metadata,
    reporting-period fields, etc.) is derived internally; callers must not
    supply them, and never need to call the other two services separately --
    `/predict`, `/predict/batch`, and `/predict/csv` here already cascade
    into gated regression + anomaly automatically.
    """
    return {
        "model_version": cfg.MODEL_VERSION,
        "risk_probability_threshold": cfg.OVERRUN_RISK_PROBABILITY_THRESHOLD,
        "input_schema": {
            "note": (
                "Every endpoint takes only raw, prediction-time fields -- "
                "the same schema the overrun-regression and "
                "anomaly-detection services take (see "
                "app.common.raw_features). Engineered features are "
                "computed internally; callers must not supply them. "
                "Send every available quarter for a project (not just the "
                "latest) so trajectory/temporal features reflect real "
                "history instead of being imputed from a single snapshot."
            ),
            "required": RAW_REQUIRED_COLS,
            "optional": RAW_OPTIONAL_COLS,
            "example": EXAMPLE_ROW,
            "batch_example": EXAMPLE_BATCH_REQUEST,
            "batch_shape": {
                "projects": [
                    {"project_code": "P001", "rows": ["raw quarter row", "raw quarter row"]},
                    {"project_code": "P002", "rows": ["raw quarter row"]},
                ]
            },
        },
        "cascade": {
            "note": (
                "Regression (/overrun) and anomaly detection (/anomaly) "
                "run automatically for a target once this classifier's "
                "predicted probability for that target exceeds the "
                "threshold above -- using the same data supplied here. "
                "No separate call or separate input is needed."
            ),
        },
    }


@router.post("/predict")
async def predict(
    rows: SingleRowsField = Body(
        ...,
        description=(
            "A JSON array with EXACTLY ONE project row (see GET "
            "/classification/schema for the raw fields / EXAMPLE_ROW). For "
            "more than one row -- multiple projects, or one project's "
            "quarterly history -- use /predict/batch instead."
        ),
        openapi_examples={"single_row": {"summary": "One project row", "value": [EXAMPLE_ROW]}},
    ),
    preferred_model: str = Query(
        overrun_cfg.DEFAULT_PREFERRED_MODEL,
        description=(
            "'model_b' or 'final_model' -- which regression ensemble is surfaced as "
            f"predicted_overrun_pct. Optional -- defaults to '{overrun_cfg.DEFAULT_PREFERRED_MODEL}'."
        ),
    ),
):
    """Classify ONE project's overrun risk, then automatically run the
    cost/time-overrun regression and anomaly-detection services on the same
    data for whichever target(s) the classifier flags as risk (probability
    above `OVERRUN_RISK_PROBABILITY_THRESHOLD`, see `GET /classification/schema`).

    Only raw, prediction-time fields are accepted -- every engineered
    feature is computed internally, and the data is supplied here ONCE for
    the whole pipeline.
    """
    _valid_preferred_model(preferred_model)
    if len(rows) != 1:
        raise HTTPException(
            status_code=422,
            detail=(
                f"POST /classification/predict takes exactly one row (got {len(rows)}). "
                "Use POST /classification/predict/batch for multiple rows or projects."
            ),
        )
    raw = pd.DataFrame(rows)
    result = await run_offloaded(run_pipeline, raw, preferred_model)
    if not result["predictions"]:
        raise HTTPException(422, "No project could be scored.")
    return result["predictions"][0]


@router.post("/predict/batch")
async def predict_batch(
    body: BatchPredictionRequest = Body(
        ...,
        description=(
            "Explicit multi-project batch payload. Each project contains its "
            "project_code and one or more quarterly raw rows."
        ),
        openapi_examples={"multiple_projects": {"summary": "Multiple projects with quarterly history", "value": EXAMPLE_BATCH_REQUEST}},
    ),
    preferred_model: str = Query(
        overrun_cfg.DEFAULT_PREFERRED_MODEL,
        description=(
            "'model_b' or 'final_model' -- which regression ensemble is surfaced as "
            f"predicted_overrun_pct. Optional -- defaults to '{overrun_cfg.DEFAULT_PREFERRED_MODEL}'."
        ),
    ),
):
    """Classify overrun risk for one or more projects from raw,
    prediction-time rows, then automatically run gated regression +
    anomaly detection for every project -- all from the SAME rows supplied
    here. Rows are grouped by `project_code`; a project's multiple
    quarters are treated as its history.

    Each project's result always includes the latest quarter's blocks plus
    a `quarterly` list inside `overrun`/`anomaly` -- one entry per row
    received for that project, tagged with its `reporting_period_date`, so
    a caller can plot a time-trend graph. There is no `include_history`
    flag; row-wise history is always used and always returned.
    """
    _valid_preferred_model(preferred_model)
    raw = pd.DataFrame(flatten_batch_projects(body))
    return await run_offloaded(run_pipeline, raw, preferred_model)


@router.post("/predict/csv")
async def predict_csv(
    file: UploadFile = File(
        ...,
        description=(
            "CSV of raw, prediction-time rows -- same columns as "
            "/predict/batch (see GET /classification/schema)."
        ),
    ),
    preferred_model: str = Query(
        overrun_cfg.DEFAULT_PREFERRED_MODEL,
        description=(
            "'model_b' or 'final_model' -- which regression ensemble is surfaced as "
            f"predicted_overrun_pct. Optional -- defaults to '{overrun_cfg.DEFAULT_PREFERRED_MODEL}'."
        ),
    ),
):
    """Classify overrun risk for projects from a CSV of raw,
    prediction-time rows, then automatically run gated regression +
    anomaly detection for every project -- all from the SAME uploaded
    file. Multiple rows per project_code are treated as that project's
    history. Same response shape as /predict/batch, including the
    always-on `quarterly` (row-wise) series inside `overrun`/`anomaly`.
    """
    _valid_preferred_model(preferred_model)
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=415, detail="Only CSV files are accepted")
    content = await read_upload_capped(file, MAX_CSV_BYTES)
    raw = await run_in_threadpool(parse_csv_bytes, content)
    if raw.empty:
        raise HTTPException(status_code=422, detail="Uploaded CSV has no data rows.")
    return await run_offloaded(run_pipeline, raw, preferred_model)
