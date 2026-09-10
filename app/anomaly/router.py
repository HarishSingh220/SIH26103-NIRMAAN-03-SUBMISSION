"""FastAPI router for the PAIMANA hybrid anomaly-detection service.

Mounted under the `/anomaly` prefix by `app/main.py`. Every endpoint here
takes only raw, prediction-time fields -- see `app.anomaly.raw_features` for
the exact raw column list and derivation rules; every engineered feature is
computed internally, the caller never precomputes anything.
"""
from __future__ import annotations

import math

import pandas as pd
from fastapi import APIRouter, Body, File, HTTPException, UploadFile
from starlette.concurrency import run_in_threadpool

from app.common.http import MAX_CSV_BYTES, parse_csv_bytes, read_upload_capped, run_offloaded
from app.common.schemas import BatchPredictionRequest, flatten_batch_projects

from . import config as cfg
from . import raw_features
from . import state
from .core import score_all
from .raw_features import derive_prediction_features

router = APIRouter()


def _score_sync(raw: pd.DataFrame) -> pd.DataFrame:
    """Pure, synchronous scoring call -- no exception translation here.
    Always invoked through `run_offloaded`, the single place that maps
    exceptions to HTTP responses.
    """
    artifact = state.get()
    return score_all(raw, artifact["hybrid_reference"], artifact["track_b_reference"])


def _json_safe(df: pd.DataFrame):
    cols = [c for c in cfg.OUTPUT_COLS if c in df.columns]
    out = df[cols].copy()
    for c in out.columns:
        if pd.api.types.is_datetime64_any_dtype(out[c]):
            out[c] = out[c].dt.strftime("%Y-%m-%d")
    records = out.to_dict(orient="records")
    for record in records:
        for key, value in record.items():
            if isinstance(value, float) and math.isnan(value):
                record[key] = None
    return records


def _simple_prediction(record: dict, include_period: bool = False) -> dict:
    risk_level = record.get("risk_level")
    if risk_level == "Normal":
        reason = "The project is fine."
    else:
        reason = record.get("final_anomaly_reason") or "No further detail available."
    out = {
        "project_name": record.get("project_name"),
        "project_code": record.get("project_code"),
        "anomaly_score": record.get("final_anomaly_score"),
        "risk_level": risk_level,
        "anomaly_reason": reason,
    }
    if include_period:
        out["reporting_period_date"] = record.get("reporting_period_date")
    return out


@router.get("/health")
def health():
    artifact = state.get_or_none()  # read-only status check, not gated behind 503
    return {
        "status": "ok",
        "model_loaded": state.is_loaded(),
        "model_version": (artifact or {}).get("model_version", cfg.MODEL_VERSION),
    }


@router.get("/model-info")
def model_info():
    artifact = state.get()
    return {
        "model_version": artifact.get("model_version", cfg.MODEL_VERSION),
        "fit_cutoff": artifact.get("fit_cutoff"),
        "feature_families": artifact["hybrid_reference"].get("feature_families"),
        "risk_labels": ["Normal", "Review", "Critical"],
        "monitor_label": "Normal",
        "history_scoring": {
            "track_a_min_observations": 3,
            "track_b_max_observations": 2,
            "batch_endpoint": "/anomaly/predict/batch",
            "csv_endpoint": "/anomaly/predict/csv",
        },
        "input_schema": {
            "note": (
                "Every endpoint takes only raw, prediction-time fields -- "
                "see app.anomaly.raw_features.RAW_REQUIRED_COLS / "
                "RAW_OPTIONAL_COLS and the README. Engineered features are "
                "computed internally."
            ),
            "required": raw_features.RAW_REQUIRED_COLS,
            "optional": raw_features.RAW_OPTIONAL_COLS,
            "batch_example": raw_features.EXAMPLE_BATCH_REQUEST,
            "batch_shape": {
                "projects": [
                    {"project_code": "P001", "rows": ["raw quarter row", "raw quarter row"]},
                    {"project_code": "P002", "rows": ["raw quarter row"]},
                ]
            },
        },
    }


def _score_grouped_df(raw: pd.DataFrame) -> dict:
    """Shared scoring path for both /predict/batch (JSON rows) and
    /predict/csv (CSV upload) -- both end up with a raw DataFrame here and
    are scored/grouped identically. Rows are grouped by project_code;
    always returns each project's latest quarter as ``latest_prediction``
    plus every scored quarter as ``quarterly_predictions`` (for time-trend
    graphs).

    derive_prediction_features() raises ValueError on a missing raw
    column; run_offloaded already maps ValueError/KeyError/TypeError to a
    422, so no extra try/except is needed here.
    """
    artifact = state.get()
    expanded = derive_prediction_features(raw)
    result = _score_sync(expanded)
    if result.empty:
        raise HTTPException(status_code=422, detail="No rows were scored")

    latest = (
        result.sort_values(["project_code", "reporting_period_date", "_source_row_id"])
        .groupby("project_code", sort=False, as_index=False)
        .tail(1)
        .sort_values("project_code")
    )
    # Keyed by the stripped project_code, matching what score_all()/
    # coerce_dtypes() produce in `result`/`latest` -- an unstripped key (very
    # common straight out of a CSV export, e.g. "P001 ") would otherwise miss.
    project_counts = (
        raw.assign(project_code=raw["project_code"].astype(str).str.strip())
        .groupby("project_code")
        .size()
        .to_dict()
    )
    projects = []
    for _, row in latest.iterrows():
        code = str(row["project_code"])
        history_result = result[result["project_code"].astype(str) == code]
        projects.append({
            "project_code": code,
            "history_rows_received": int(project_counts[code]),
            "latest_prediction": _simple_prediction(_json_safe(pd.DataFrame([row]))[0]),
            "quarterly_predictions": [
                _simple_prediction(r, include_period=True)
                for r in _json_safe(history_result)
            ],
        })

    return {
        "model_version": artifact.get("model_version", cfg.MODEL_VERSION),
        "projects_received": len(project_counts),
        "rows_received": len(raw),
        "rows_scored": len(result),
        "projects": projects,
    }


@router.post("/predict/batch")
async def predict_batch(
    body: BatchPredictionRequest = Body(
        ...,
        description=(
            "Explicit multi-project batch payload. Each project contains its "
            "project_code and one or more quarterly raw rows."
        ),
        openapi_examples={"multiple_projects": {"summary": "Multiple projects with quarterly history", "value": raw_features.EXAMPLE_BATCH_REQUEST}},
    ),
):
    """Score multiple projects from the reduced raw-input schema, each with
    zero or more quarterly history rows (see README / app.anomaly.raw_features
    for the exact raw columns and derivation rules).

    Send every available quarter for a project, not just the newest one --
    the QoQ/CUSUM/trend features need the history to put a project on Track A
    instead of Track B.

    Always returns each project's latest quarter as ``latest_prediction``
    (project_name, project_code, anomaly_score, risk_level, anomaly_reason
    -- "The project is fine." when risk_level is "Normal"), AND every
    scored quarter as ``quarterly_predictions``, the same reduced shape for
    every scored quarter, each tagged with its ``reporting_period_date``,
    so a caller can plot a time-trend graph per project.
    """
    state.get()
    raw = pd.DataFrame(flatten_batch_projects(body))
    return await run_offloaded(_score_grouped_df, raw)


@router.post("/predict/csv")
async def predict_csv(file: UploadFile = File(...)):
    """Score projects from a CSV of raw-schema rows (see README /
    app.anomaly.raw_features for the exact columns). Multiple rows per
    project_code are treated as that project's history. Same response
    shape as /predict/batch: each project's latest quarter as
    ``latest_prediction``, plus every scored quarter as
    ``quarterly_predictions`` for time-trend graphs.
    """
    state.get()
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=415, detail="Only CSV files are accepted")
    # Reading the upload stays on the event loop (it's just I/O); parsing a
    # potentially large CSV and then scoring it are both CPU-bound, so both
    # go through the same threadpool offload as every other endpoint.
    content = await read_upload_capped(file, MAX_CSV_BYTES)
    raw = await run_in_threadpool(parse_csv_bytes, content)
    return await run_offloaded(_score_grouped_df, raw)
