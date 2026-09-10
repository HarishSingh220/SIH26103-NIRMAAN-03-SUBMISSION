"""Adapts the shared raw, prediction-time schema (see app.common.raw_features)
into the metadata + reporting_periods shape the classification service's own
preprocessing pipeline expects (see app.classification.preprocessing).

Why this exists
----------------
Historically the classification service spoke a completely different input
schema (``{"metadata": {...}, "reporting_periods": [...]}``) from the
overrun-regression and anomaly-detection services (the flat, raw,
prediction-time schema in ``app.common.raw_features``). That forced a caller
to describe the same project twice, in two shapes, to run the full pipeline
(see the old app/gateway/router.py).

This module lets the classification endpoint accept literally the same raw
columns as the other two services. A caller now supplies raw project data
ONCE; every engineered feature (project age, expenditure ratios, deltas,
...) is derived once (via ``app.common.raw_features.derive_prediction_features``)
and reused by all three services internally -- nothing is asked for twice.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from app.common.raw_features import derive_prediction_features

# app.classification.preprocessing.PROJECT_LEVEL_COLS -- project-level fields
# that are constant across a project's reporting periods.
METADATA_FIELDS = [
    "sector", "state", "agency_name", "n_landmarks_total", "original_cost_rs_cr",
]

# app.classification.preprocessing.LANDMARK_LEVEL_COLS (minus landmark_index/
# quarter/financial_year, added explicitly below) -- fields that vary per
# reporting period. Every one of these is already produced by
# derive_prediction_features(), so no extra input is needed from the caller.
REPORTING_PERIOD_FIELDS = [
    "landmark_index", "quarter", "financial_year", "horizon_months",
    "cumulative_expenditure_rs_cr", "expenditure_to_cost_pct",
    "project_age_at_report_months", "planned_duration_months",
    "progress_ratio", "anticipated_cost_rs_cr",
    "anticipated_cost_change_pct", "anticipated_date_change_months",
]


def _clean_value(v: Any) -> Any:
    """NaN/NaT -> None so downstream validation (e.g. Optional[float] fields)
    never chokes on a numpy NaN, which is not JSON/pydantic-friendly."""
    if v is None:
        return None
    if isinstance(v, (float, np.floating)) and np.isnan(v):
        return None
    try:
        if pd.isna(v):
            return None
    except (TypeError, ValueError):
        pass
    return v


def build_projects_from_engineered(engineered: pd.DataFrame) -> list[dict[str, Any]]:
    """``engineered`` is the output of
    ``app.common.raw_features.derive_prediction_features()`` -- one row per
    project-quarter, possibly several projects and/or several quarters per
    project mixed together in any order.

    Groups by ``project_code``, sorts each project's own rows by
    ``reporting_period_date``, and returns one classification-ready project
    dict per project_code::

        {"metadata": {...}, "reporting_periods": [{...}, ...]}

    ready to hand straight to
    ``ModelCoordinator.predict_project``/``predict_batch`` -- no additional
    caller input required.
    """
    if engineered.empty:
        return []

    df = engineered.copy()
    df["project_code"] = df["project_code"].astype(str).str.strip()
    # Invalid reporting dates must never outrank valid dates when selecting the latest row.
    df = df.assign(_valid_reporting_date=df["reporting_period_date"].notna())
    df = df.sort_values(
        ["project_code", "_valid_reporting_date", "reporting_period_date"],
        ascending=[True, False, True],
    )

    projects: list[dict[str, Any]] = []
    for project_code, group in df.groupby("project_code", sort=False):
        group = group.reset_index(drop=True)
        latest = group.loc[group["_valid_reporting_date"].idxmax()]

        metadata: dict[str, Any] = {"project_code": project_code}
        for field in METADATA_FIELDS:
            metadata[field] = _clean_value(latest.get(field))

        reporting_periods = []
        for _, row in group.iterrows():
            period = {field: _clean_value(row.get(field)) for field in REPORTING_PERIOD_FIELDS}
            reporting_periods.append(period)

        projects.append({"metadata": metadata, "reporting_periods": reporting_periods})

    return projects


def raw_df_to_projects(raw: pd.DataFrame) -> tuple[list[dict[str, Any]], pd.DataFrame]:
    """Raw-schema DataFrame (``RAW_REQUIRED_COLS``/``RAW_OPTIONAL_COLS``) ->
    ``(classification-ready project dicts, engineered DataFrame)``.

    The engineered DataFrame is returned alongside the project dicts so the
    caller (see ``app.classification.pipeline``) can hand that SAME
    already-derived data straight to the overrun-regression and
    anomaly-detection services without recomputing anything or asking the
    original caller for it a second time.
    """
    engineered = derive_prediction_features(raw)
    projects = build_projects_from_engineered(engineered)
    engineered = engineered.drop(columns=["_valid_reporting_date"], errors="ignore")
    return projects, engineered
