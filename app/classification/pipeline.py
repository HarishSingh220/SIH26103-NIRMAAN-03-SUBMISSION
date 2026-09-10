"""Full prediction pipeline: classification -> gated regression + anomaly,
from a SINGLE caller-supplied raw payload.

The caller supplies raw, prediction-time project data exactly once, using
the same raw schema as the overrun-regression and anomaly-detection
services (see ``app.common.raw_features``). This module:

  1. Derives every engineered feature from that raw data ONE time
     (``app.common.raw_features.derive_prediction_features``).
  2. Adapts the result into the classification service's own
     metadata + reporting_periods shape and runs the classifier
     (``app.classification``).
  3. Gates the (already-built) regression service (``app.overrun``) and
     anomaly-detection service (``app.anomaly``) on the classifier's own
     predicted probability, reusing the SAME engineered data computed in
     step 1 -- neither downstream service is asked to recompute anything,
     and the caller never supplies the project a second time.

This supersedes the old two-schema contract in ``app/gateway/router.py``
(classification metadata/reporting_periods PLUS a separate raw row/history)
with a single raw schema and a single request.

Per the project's layering rule, this module composes each service's
already-existing public pieces (state accessors, ``core`` functions) rather
than merging any of their internals together.
"""
from __future__ import annotations

from typing import Any, Optional

import pandas as pd
from fastapi import HTTPException

from app.anomaly import state as anomaly_state
from app.anomaly.core import score_all as _anomaly_score_all
from app.anomaly.router import _simple_prediction as _anomaly_simple_prediction
from app.common.raw_features import derive_prediction_features
from app.overrun import config as overrun_cfg
from app.overrun import outcomes as overrun_outcomes
from app.overrun import state as overrun_state
from app.overrun.router import _valid_preferred_model
from app.overrun.scoring import latest_row_index as _overrun_latest_row_index
from app.overrun.scoring import target_block_from_row as _overrun_target_block_from_row

from . import config as cfg
from . import state as classification_state
from .raw_adapter import build_projects_from_engineered


# =============================================================================
# Risk gate (same 0.5-by-default cutoff previously enforced in the gateway)
# =============================================================================
def _risk_gate(classification_result: dict) -> dict[str, Any]:
    threshold = cfg.OVERRUN_RISK_PROBABILITY_THRESHOLD
    cost_probability = classification_result["cost_overrun"]["probability"]
    time_probability = classification_result["time_overrun"]["probability"]
    cost_risk = cost_probability > threshold
    time_risk = time_probability > threshold
    return {
        "threshold": threshold,
        "cost_overrun_risk": cost_risk,
        "time_overrun_risk": time_risk,
        "any_overrun_risk": cost_risk or time_risk,
    }


# =============================================================================
# Regression (overrun) leg -- reuses the already-derived engineered frame
# =============================================================================
def _overrun_skipped_block(expanded_df: pd.DataFrame, target_col: str, reason: str) -> dict[str, Any]:
    """Same shape as a real prediction block, with every predicted field
    `None` -- except the exact-outcome block's REFERENCE fields
    (`original_cost_rs_cr` / `original_commissioning_date` /
    `planned_duration_months`), which are still filled in from the
    project's latest known row even when regression itself was skipped, so
    a caller always has the baseline to compare against.
    """
    block = {
        "predicted_overrun_pct": None,
        "model_b_prediction": None,
        "final_model_prediction": None,
        "preferred_model": None,
        "reason": reason,
    }
    anchor = expanded_df.loc[_overrun_latest_row_index(expanded_df)]
    return overrun_outcomes.attach_exact_outcome(block, target_col, anchor)


def _overrun_target_series(
    pipeline: "CombinedOverrunPipeline", expanded_df: pd.DataFrame, target_col: str, preferred_model: str
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Score every supplied quarter for this target (not just the latest),
    sorted by reporting_period_date. Returns (latest_block, quarterly_rows)
    -- quarterly_rows is every row's block (percentages AND the exact
    cost/date outcome -- see app.overrun.outcomes) tagged with its date,
    for a caller to plot a time-trend graph.
    """
    scored = pipeline.predict(expanded_df, return_frame=True).sort_values(overrun_cfg.DATE_COL)
    quarterly = [
        {
            "reporting_period_date": row.get(overrun_cfg.DATE_COL),
            **_overrun_target_block_from_row(row, target_col, preferred_model),
        }
        for row in scored.to_dict("records")
    ]
    if not quarterly:
        return _overrun_skipped_block(expanded_df, target_col, "No rows were scored"), []
    latest = {k: v for k, v in quarterly[-1].items() if k != "reporting_period_date"}
    return latest, quarterly


def _run_overrun(expanded_df: pd.DataFrame, gate: dict[str, Any], preferred_model: str) -> dict[str, Any]:
    cost_pipeline = overrun_state.get_cost()
    time_pipeline = overrun_state.get_time()

    try:
        if gate["cost_overrun_risk"]:
            cost_block, cost_quarterly = _overrun_target_series(
                cost_pipeline, expanded_df, overrun_cfg.COST_TARGET, preferred_model
            )
        else:
            cost_block, cost_quarterly = (
                _overrun_skipped_block(
                    expanded_df,
                    overrun_cfg.COST_TARGET,
                    "Not flagged as cost-overrun risk by classifier (probability <= "
                    f"{gate['threshold']}) - regression skipped",
                ),
                [],
            )
        if gate["time_overrun_risk"]:
            time_block, time_quarterly = _overrun_target_series(
                time_pipeline, expanded_df, overrun_cfg.TIME_TARGET, preferred_model
            )
        else:
            time_block, time_quarterly = (
                _overrun_skipped_block(
                    expanded_df,
                    overrun_cfg.TIME_TARGET,
                    "Not flagged as time-overrun risk by classifier (probability <= "
                    f"{gate['threshold']}) - regression skipped",
                ),
                [],
            )
    except Exception as exc:
        raise HTTPException(422, f"Overrun regression failed: {exc}")

    # Merge cost/time quarterly series by date so each entry carries both
    # targets, mirroring the shape of the "latest" blocks above.
    by_date: dict[Any, dict[str, Any]] = {}
    for row in cost_quarterly:
        by_date.setdefault(row["reporting_period_date"], {})["cost_overrun"] = {
            k: v for k, v in row.items() if k != "reporting_period_date"
        }
    for row in time_quarterly:
        by_date.setdefault(row["reporting_period_date"], {})["time_overrun"] = {
            k: v for k, v in row.items() if k != "reporting_period_date"
        }
    quarterly = [
        {"reporting_period_date": date, **blocks}
        for date, blocks in sorted(by_date.items(), key=lambda kv: (kv[0] is None, kv[0]))
    ]

    return {"cost_overrun": cost_block, "time_overrun": time_block, "quarterly": quarterly}


# =============================================================================
# Anomaly leg -- reuses the same already-derived engineered frame
# =============================================================================
def _anomaly_skipped_block(reason: str) -> dict[str, Any]:
    return {
        "project_code": None,
        "anomaly_score": None,
        "risk_level": None,
        "anomaly_reason": reason,
        "quarterly": [],
    }


def _run_anomaly(expanded_df: pd.DataFrame) -> dict[str, Any]:
    artifact = anomaly_state.get()  # 503s early if the model isn't loaded
    try:
        result = _anomaly_score_all(
            expanded_df, artifact["hybrid_reference"], artifact["track_b_reference"]
        )
    except Exception as exc:
        raise HTTPException(422, f"Anomaly scoring failed: {exc}")

    if result.empty:
        return _anomaly_skipped_block("No rows were scored")

    ordered = result.sort_values("reporting_period_date")
    records = ordered.to_dict("records")
    latest_out = _anomaly_simple_prediction(records[-1])
    latest_out["quarterly"] = [_anomaly_simple_prediction(r, include_period=True) for r in records]
    return latest_out


# =============================================================================
# Per-project orchestration
# =============================================================================
def run_pipeline_for_project(
    project: dict[str, Any],
    project_expanded: pd.DataFrame,
    preferred_model: str,
    targets: Optional[list[str]] = None,
) -> dict[str, Any]:
    """Run classification -> gated regression/anomaly for ONE project, whose
    raw rows have already been derived into ``project_expanded`` (the slice
    of the shared engineered DataFrame belonging to just this project_code).
    """
    coordinator = classification_state.get()

    try:
        classification_result = coordinator.predict_project(
            metadata=project["metadata"],
            reporting_periods=project["reporting_periods"],
            targets=targets or ["cost", "time"],
        )
    except ValueError as exc:
        return {"project_code": project["metadata"].get("project_code"), "error": str(exc)}

    gate = _risk_gate(classification_result)
    overrun_result = _run_overrun(project_expanded, gate, preferred_model)
    anomaly_result = (
        _run_anomaly(project_expanded)
        if gate["any_overrun_risk"]
        else _anomaly_skipped_block(
            "Not flagged as overrun risk by classifier (probability <= "
            f"{gate['threshold']}) - anomaly detection skipped"
        )
    )

    return {
        "project_code": classification_result["project_code"],
        "rows_used": classification_result.get("rows_used"),
        "warnings": classification_result.get("warnings", []),
        "classification": {
            "cost_overrun": classification_result["cost_overrun"],
            "time_overrun": classification_result["time_overrun"],
        },
        "risk_gate": gate,
        # "overrun" and "anomaly" each carry the latest quarter's blocks at
        # the top level (unchanged shape) PLUS a "quarterly" list -- every
        # supplied quarter's block for a project, tagged with its
        # reporting_period_date, so a caller can plot a time-trend graph.
        # "quarterly" is empty when regression/anomaly was skipped by the
        # risk gate.
        "overrun": overrun_result,
        "anomaly": anomaly_result,
    }


# =============================================================================
# Top-level entry point -- single row, batch, or CSV all funnel through here
# =============================================================================
def run_pipeline(raw: pd.DataFrame, preferred_model: str = overrun_cfg.DEFAULT_PREFERRED_MODEL) -> dict[str, Any]:
    """Full entry point: raw-schema rows (one or more projects, one or more
    quarters each) -> classification + gated regression/anomaly results for
    every project, computed from ONE caller-supplied payload.

    Works identically for a single row, many rows for one project
    (history), or many rows across many projects (batch/CSV) -- the only
    difference is how many project dicts ``build_projects_from_engineered``
    produces.
    """
    _valid_preferred_model(preferred_model)

    expanded = derive_prediction_features(raw)
    projects = build_projects_from_engineered(expanded)
    if not projects:
        raise HTTPException(422, "No projects could be parsed from the supplied data.")

    expanded = expanded.copy()
    expanded["project_code"] = expanded["project_code"].astype(str).str.strip()

    predictions = []
    for project in projects:
        code = project["metadata"]["project_code"]
        project_expanded = expanded[expanded["project_code"] == code]
        try:
            predictions.append(run_pipeline_for_project(project, project_expanded, preferred_model))
        except HTTPException as exc:
            # A single project's regression/anomaly failure (e.g. a model
            # artifact issue surfaced only for that project's data) should
            # not take down the whole batch response -- mirrors
            # ModelCoordinator.predict_batch's own per-project resilience
            # for classification failures (ValueError, handled inside
            # run_pipeline_for_project). A 503 (model not loaded at all)
            # is systemic rather than per-project, so it still aborts the
            # whole request instead of being swallowed per project.
            if exc.status_code == 503:
                raise
            predictions.append({"project_code": code, "error": str(exc.detail)})

    return {
        "model_version": cfg.MODEL_VERSION,
        "preferred_model": preferred_model,
        "projects_received": len(projects),
        "rows_received": len(raw),
        "predictions": predictions,
    }
