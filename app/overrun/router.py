"""FastAPI router for the cost/time-overrun regression service.

Mounted under the `/overrun` prefix by `app/main.py`. Loads both
`CombinedOverrunPipeline` artifacts (built by `app.overrun.train`) and serves
them behind one API. Three ways to call it:

- `POST /overrun/predict`       -- exactly ONE project row (a JSON array
                                    with a single item); always predicts
                                    both targets, no classifier gating.
- `POST /overrun/predict/batch` -- a JSON array of raw-schema rows across
                                    one or more projects, grouped by
                                    project_code internally.
- `POST /overrun/predict/csv`   -- same as /predict/batch, but the rows come
                                    from an uploaded CSV file.

Every endpoint here takes only the same raw, prediction-time fields the
anomaly-detection service takes (see `app.common.raw_features` for the exact
raw column list and derivation rules, or `GET /overrun/schema`). Every
feature `CombinedOverrunPipeline.engineer_features()` needs on top of that
raw schema (`expenditure_to_cost_pct`, `project_age_at_report_months`,
`planned_duration_months`, `progress_ratio`, `landmark_index`,
`n_landmarks_total`, `horizon_months`, `reporting_period_date`, ...) is
computed internally by `derive_prediction_features()` -- the caller never
precomputes or supplies any of it.

The batch/csv endpoints mirror the anomaly-detection service's contract:
send every available quarter for a project (not just the latest) so the
temporal/velocity/rolling features reflect real history instead of being
imputed from a single snapshot.

Every `cost_overrun`/`time_overrun` block below carries both the predicted
PERCENTAGE and the concrete EXACT outcome it implies (see
`app.overrun.outcomes` for the exact formulas):

- `cost_overrun`: `predicted_overrun_pct` plus `predicted_final_cost_rs_cr`
  (projected total cost) and `predicted_cost_increase_rs_cr` (the Rs Cr
  increase over `original_cost_rs_cr`).
- `time_overrun`: `predicted_overrun_pct` plus `predicted_final_commissioning_date`
  (projected completion date) and `predicted_delay_months` (the delay over
  `original_commissioning_date`).
"""
from __future__ import annotations

from typing import Any

import pandas as pd
from fastapi import APIRouter, Body, File, HTTPException, Query, UploadFile
from starlette.concurrency import run_in_threadpool

from app.common.http import MAX_CSV_BYTES, parse_csv_bytes, read_upload_capped, run_offloaded
from app.common.raw_features import (
    EXAMPLE_BATCH_REQUEST,
    EXAMPLE_ROW,
    RAW_OPTIONAL_COLS,
    RAW_REQUIRED_COLS,
    derive_prediction_features,
)
from app.common.schemas import BatchPredictionRequest, SingleRowsField, flatten_batch_projects

from . import config as cfg
from . import state
from .scoring import predict_target_block, target_block_from_row

router = APIRouter()

_PREFERRED_MODEL_DESC = (
    "Which ensemble's prediction is surfaced as predicted_overrun_pct "
    f"(and therefore drives the exact cost/date outcome too): 'model_b' or "
    f"'final_model'. Optional -- defaults to '{cfg.DEFAULT_PREFERRED_MODEL}' "
    "when omitted, so most callers never need to set this."
)


def _valid_preferred_model(preferred_model: str) -> None:
    if preferred_model not in ("model_b", "final_model"):
        raise HTTPException(status_code=400, detail="preferred_model must be 'model_b' or 'final_model'.")


# =============================================================================
# GET /health, GET /schema
# =============================================================================
@router.get("/health")
def health():
    return {
        "status": "ok" if state.is_loaded() else "models not loaded",
        "model_loaded": state.is_loaded(),
        "model_version": cfg.MODEL_VERSION,
    }


@router.get("/schema")
def get_schema():
    cost_pipeline = state.get_cost()
    time_pipeline = state.get_time()
    return {
        "model_version": cfg.MODEL_VERSION,
        "input_schema": {
            "note": (
                "Every endpoint takes only raw, prediction-time fields -- "
                "the same schema the anomaly-detection service takes (see "
                "app.common.raw_features). Engineered features "
                "(reporting_period_date, expenditure_to_cost_pct, "
                "project_age_at_report_months, planned_duration_months, "
                "progress_ratio, landmark_index, n_landmarks_total, "
                "horizon_months, ...) are computed internally; callers must "
                "not supply them."
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
        "output": {
            "note": (
                "Each cost_overrun/time_overrun block carries both the "
                "predicted percentage AND the exact outcome it implies -- "
                "predicted_final_cost_rs_cr/predicted_cost_increase_rs_cr "
                "for cost, predicted_final_commissioning_date/"
                "predicted_delay_months for time. See app.overrun.outcomes "
                "for the exact formulas."
            ),
        },
        "preferred_model": {
            "options": ["model_b", "final_model"],
            "default": cfg.DEFAULT_PREFERRED_MODEL,
            "note": "Optional on every endpoint; omit it to use the default.",
        },
        "cost_model": {
            "models_available": ["model_b", "final_model"],
            "model_b_sector": cost_pipeline.sector_name,
        },
        "time_model": {
            "models_available": ["model_b", "final_model"],
            "model_b_sector": time_pipeline.sector_name,
        },
        "history_scoring": {
            "single_endpoint": "/overrun/predict",
            "batch_endpoint": "/overrun/predict/batch",
            "csv_endpoint": "/overrun/predict/csv",
        },
    }


# =============================================================================
# POST /predict -- ONE project, ONE row (bare list of exactly one item)
# =============================================================================
def _predict_single(rows: list[dict], preferred_model: str) -> dict[str, Any]:
    raw_df = pd.DataFrame(rows)
    project_code = str(rows[0].get(cfg.PROJECT_COL))
    cost_pipeline = state.get_cost()
    time_pipeline = state.get_time()

    # derive_prediction_features() raises ValueError on a missing raw
    # column or an empty frame; run_offloaded already maps ValueError/
    # KeyError/TypeError to a 422, so no extra try/except is needed here.
    expanded_df = derive_prediction_features(raw_df)
    cost_block = predict_target_block(cost_pipeline, expanded_df, preferred_model)
    time_block = predict_target_block(time_pipeline, expanded_df, preferred_model)

    return {
        "project_code": project_code,
        "cost_overrun": cost_block,
        "time_overrun": time_block,
        "rows_used": len(raw_df),
    }


@router.post("/predict")
async def predict(
    rows: SingleRowsField = Body(
        ...,
        description=(
            "A JSON array with EXACTLY ONE project row (see GET "
            "/overrun/schema for the raw fields / EXAMPLE_ROW). For more "
            "than one row -- multiple projects, or one project's quarterly "
            "history -- use /predict/batch instead."
        ),
        openapi_examples={"single_row": {"summary": "One project row", "value": [EXAMPLE_ROW]}},
    ),
    preferred_model: str = Query(cfg.DEFAULT_PREFERRED_MODEL, description=_PREFERRED_MODEL_DESC),
):
    """Cost/time-overrun prediction for ONE project from a single raw row
    (see app.common.raw_features / GET /overrun/schema). Every engineered
    feature is computed internally. Always predicts both targets -- there
    is no classifier-risk gating on this standalone endpoint (that's what
    `/classification/predict` does automatically, on top of this same
    regression).

    Each `cost_overrun`/`time_overrun` block includes both the predicted
    percentage and the exact outcome it implies -- see this router's
    module docstring or `GET /overrun/schema` for the field list.
    """
    _valid_preferred_model(preferred_model)
    if len(rows) != 1:
        raise HTTPException(
            status_code=422,
            detail=(
                f"POST /overrun/predict takes exactly one row (got {len(rows)}). "
                "Use POST /overrun/predict/batch for multiple rows or projects."
            ),
        )
    return await run_offloaded(_predict_single, rows, preferred_model)


# =============================================================================
# POST /predict/batch, POST /predict/csv -- bulk raw-schema rows, grouped by
# project_code internally (mirrors the anomaly-detection service's contract)
# =============================================================================
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
    preferred_model: str = Query(cfg.DEFAULT_PREFERRED_MODEL, description=_PREFERRED_MODEL_DESC),
):
    """Score cost + time overrun for multiple projects from the raw,
    prediction-time schema (the same one the anomaly-detection service
    takes -- see app.common.raw_features / GET /overrun/schema), each with
    zero or more quarterly history rows.

    Each project is grouped independently by `project_code`; histories are
    never mixed across projects. Send every available quarter for a project
    (not just the latest) so the temporal/velocity/rolling features reflect
    real history instead of being imputed from a single snapshot.

    Always returns each project's latest quarter as `latest_prediction`
    (cost_overrun + time_overrun blocks, each with `predicted_overrun_pct`,
    the exact cost/date outcome, `model_b_prediction`, and
    `final_model_prediction`), AND every scored quarter as
    `quarterly_predictions` (same shape, each tagged with its
    `reporting_period_date`) so a caller can plot a time-trend graph per
    project.
    """
    state.get_cost()
    state.get_time()
    _valid_preferred_model(preferred_model)
    raw = pd.DataFrame(flatten_batch_projects(body))
    return await run_offloaded(_score_batch_df, raw, preferred_model)


def _latest_rows(df: pd.DataFrame, project_col: str, date_col: str) -> pd.DataFrame:
    """One row per project_col value: the row with the latest date_col.

    Rows with a missing/unparseable date are sorted first
    (`na_position="first"`) so a project's genuine latest date is always
    the one `tail(1)` picks -- never shadowed by a NaT row landing last.
    """
    ordered = df.copy()
    ordered["_sort_date"] = pd.to_datetime(ordered[date_col], errors="coerce")
    ordered = ordered.sort_values([project_col, "_sort_date"], na_position="first")
    latest = ordered.groupby(project_col, sort=False, as_index=False).tail(1)
    return latest.drop(columns=["_sort_date"]).sort_values(project_col).reset_index(drop=True)


def _project_prediction(cost_row: dict, time_row: dict, preferred_model: str, include_period: bool = False) -> dict:
    out = {
        "project_code": str(cost_row.get(cfg.PROJECT_COL)),
        "cost_overrun": target_block_from_row(cost_row, cfg.COST_TARGET, preferred_model),
        "time_overrun": target_block_from_row(time_row, cfg.TIME_TARGET, preferred_model),
    }
    if include_period:
        out["reporting_period_date"] = cost_row.get(cfg.DATE_COL)
    return out


def _score_batch_df(raw: pd.DataFrame, preferred_model: str) -> dict:
    """Shared scoring path for both /predict/batch (JSON rows) and
    /predict/csv (CSV upload) -- both end up with a raw DataFrame here and
    are scored identically.

    derive_prediction_features() raises ValueError on a missing required raw
    column or an empty frame; run_offloaded already maps ValueError/KeyError/
    TypeError to a 422, so no extra try/except is needed here.
    """
    expanded = derive_prediction_features(raw)

    cost_pipeline = state.get_cost()
    time_pipeline = state.get_time()

    cost_frame = cost_pipeline.predict(expanded, return_frame=True)
    time_frame = time_pipeline.predict(expanded, return_frame=True)

    # Always key/display project_code stripped of incidental whitespace --
    # derive_prediction_features() already strips it, but each pipeline's
    # own engineer_features() re-derives project_code from its input frame
    # independently, so re-strip defensively here too.
    cost_frame[cfg.PROJECT_COL] = cost_frame[cfg.PROJECT_COL].astype(str).str.strip()
    time_frame[cfg.PROJECT_COL] = time_frame[cfg.PROJECT_COL].astype(str).str.strip()

    # Keyed by the stripped project_code, matching what the pipelines
    # produce in `cost_frame`/`time_frame` above -- an unstripped key (very
    # common straight out of a CSV export, e.g. "P001 ") would otherwise miss.
    project_counts = (
        raw.assign(**{cfg.PROJECT_COL: raw[cfg.PROJECT_COL].astype(str).str.strip()})
        .groupby(cfg.PROJECT_COL)
        .size()
        .to_dict()
    )

    latest_cost = _latest_rows(cost_frame, cfg.PROJECT_COL, cfg.DATE_COL)
    latest_time = _latest_rows(time_frame, cfg.PROJECT_COL, cfg.DATE_COL)
    latest_time_by_code = {str(r[cfg.PROJECT_COL]): r for r in latest_time.to_dict("records")}

    projects = []
    for cost_row in latest_cost.to_dict("records"):
        code = str(cost_row[cfg.PROJECT_COL])
        time_row = latest_time_by_code.get(code, {})
        cost_hist = cost_frame[cost_frame[cfg.PROJECT_COL] == code].sort_values(cfg.DATE_COL)
        time_hist = time_frame[time_frame[cfg.PROJECT_COL] == code].sort_values(cfg.DATE_COL)
        # Join by reporting date rather than positional zip: a malformed row,
        # duplicate date, or future model change must never pair a cost result
        # with a different quarter's time result.
        cost_hist = cost_hist.drop_duplicates(subset=[cfg.DATE_COL], keep="last")
        time_hist = time_hist.drop_duplicates(subset=[cfg.DATE_COL], keep="last")
        cost_by_date = {str(r[cfg.DATE_COL]): r for r in cost_hist.to_dict("records")}
        time_by_date = {str(r[cfg.DATE_COL]): r for r in time_hist.to_dict("records")}
        all_dates = sorted(set(cost_by_date) | set(time_by_date))
        quarterly_predictions = []
        for date_key in all_dates:
            c = cost_by_date.get(date_key)
            t = time_by_date.get(date_key)
            if c is None or t is None:
                # The two pipelines should normally emit identical rows. If
                # they diverge, preserve the valid side rather than silently
                # mispairing it with another date.
                partial = {"project_code": code, "reporting_period_date": date_key}
                if c is not None:
                    partial["cost_overrun"] = target_block_from_row(c, cfg.COST_TARGET, preferred_model)
                if t is not None:
                    partial["time_overrun"] = target_block_from_row(t, cfg.TIME_TARGET, preferred_model)
                quarterly_predictions.append(partial)
            else:
                quarterly_predictions.append(_project_prediction(c, t, preferred_model, include_period=True))
        item = {
            "project_code": code,
            "history_rows_received": int(project_counts.get(code, 0)),
            "latest_prediction": _project_prediction(cost_row, time_row, preferred_model),
            "quarterly_predictions": quarterly_predictions,
        }
        projects.append(item)

    return {
        "model_version": cfg.MODEL_VERSION,
        "preferred_model": preferred_model,
        "projects_received": len(project_counts),
        "rows_received": len(raw),
        "rows_scored": len(cost_frame),
        "projects": projects,
    }


@router.post("/predict/csv")
async def predict_csv(
    file: UploadFile = File(
        ...,
        description=(
            "CSV of raw, prediction-time rows -- same columns as "
            "/predict/batch (see GET /overrun/schema)."
        ),
    ),
    preferred_model: str = Query(cfg.DEFAULT_PREFERRED_MODEL, description=_PREFERRED_MODEL_DESC),
):
    """Score cost + time overrun for projects from a CSV of raw,
    prediction-time rows -- the same raw schema as /predict/batch and the
    anomaly-detection service (see app.common.raw_features / GET
    /overrun/schema). Multiple rows per project_code are treated as that
    project's history. Same response shape as /predict/batch: each
    project's latest quarter as `latest_prediction`, plus every scored
    quarter as `quarterly_predictions` for time-trend graphs.

    Example CSV header (one column per raw field, values omitted for
    brevity)::

        project_code,project_name,agency_name,sector,state,project_status,
        reporting_quarter,financial_year,original_cost_rs_cr,
        anticipated_cost_rs_cr,cumulative_expenditure_rs_cr,approval_date,
        original_commissioning_date,anticipated_commissioning_date
    """
    state.get_cost()
    state.get_time()
    _valid_preferred_model(preferred_model)
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=415, detail="Only CSV files are accepted")
    content = await read_upload_capped(file, MAX_CSV_BYTES)
    raw = await run_in_threadpool(parse_csv_bytes, content)
    if raw.empty:
        raise HTTPException(status_code=422, detail="Uploaded CSV has no data rows.")
    return await run_offloaded(_score_batch_df, raw, preferred_model)
