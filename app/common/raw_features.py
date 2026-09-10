"""Shared raw prediction-time schema + feature derivation.

Both the anomaly-detection service and the cost/time-overrun regression
service are trained on PAIMANA project-reporting data and, at prediction
time, need the same handful of engineered features -- reporting period
date, project age, expenditure ratio, contemporaneous cost/time overruns,
and so on -- computed from the same small set of raw fields a caller
actually has on hand when a quarterly report comes in. This module is the
one place that raw schema and derivation logic live, so both services
accept literally the same request-body shape and can never drift apart on
how a feature is computed.

Per the top-level README's architecture note, neither service's code
imports from the other; both import this module instead
(`app.anomaly.raw_features` and `app.overrun.router` each import from here).

`derive_prediction_features()` raises `ValueError` for a missing required
raw column -- callers running it inside `run_offloaded()` (see
`app.common.http`) get that translated into a 422 automatically. Anything
else (an unparseable date, a non-numeric cost figure, an unknown quarter
label, ...) degrades to `NaN` in the relevant engineered column rather than
raising, so one malformed row never crashes an entire batch; per-row data
quality is the concerned model's problem to flag, not this function's to
gate on.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Required vs optional -- audited against every downstream consumer
# (app.common.raw_features, app.anomaly.core, app.overrun, app.classification
# .preprocessing) rather than assumed:
#
# REQUIRED = feeds a numeric engineered feature directly (age, duration,
# expenditure ratio, contemporaneous/anticipated deltas) via _months_between/
# _pct_change above, or is the row/entity identifier + time key every
# service groups on (project_code, reporting_quarter, financial_year).
# Missing any of these would silently zero out a real feature rather than
# just lose a categorical signal, so these still hard-fail with a clear 422
# instead of quietly degrading a prediction.
#
# OPTIONAL = never appears in any numeric formula in this file, and every
# downstream reader already treats a missing/NaN value as a normal,
# handled case rather than a crash:
#   - project_name: carried through purely for display in responses
#     (`app.anomaly.router` does `record.get("project_name")`); the only
#     place it feeds a *feature* is app.anomaly.core's entity-change flags,
#     which already guard every comparison with `.notna()` and even treat
#     a missing name as its own "missing critical data" anomaly signal
#     (`track_b_missing_critical_flag`) -- i.e. omitting it is a supported,
#     meaningful input, not an edge case.
#   - project_status: audited across the whole codebase and found to have
#     no computational use anywhere -- it's accepted and string-cleaned
#     but never read by any feature, model, or grouping key.
# revised_cost_rs_cr / revised_commissioning_date were already optional
# (see the "revised-if-present, else-original" fallback below).
# ---------------------------------------------------------------------------
RAW_REQUIRED_COLS = [
    "project_code", "agency_name", "sector", "state",
    "reporting_quarter", "financial_year",
    "original_cost_rs_cr", "anticipated_cost_rs_cr",
    "cumulative_expenditure_rs_cr", "approval_date",
    "original_commissioning_date", "anticipated_commissioning_date",
]
RAW_OPTIONAL_COLS = [
    "project_name", "project_status",
    "revised_cost_rs_cr", "revised_commissioning_date",
]

# Columns derive_prediction_features() is responsible for adding on top of
# the raw ones above. Used only as a defensive self-check at the end of the
# function (a bug here should fail loudly, not silently ship a missing
# column downstream) -- not an authoritative list of what any one caller's
# model actually needs.
ENGINEERED_COLS = [
    "quarter", "reporting_period_date",
    "project_age_at_report_months", "planned_duration_months",
    "progress_ratio", "expenditure_to_cost_pct",
    "contemporaneous_cost_overrun_pct", "contemporaneous_time_overrun_pct",
    "anticipated_cost_change_pct", "anticipated_date_change_months",
    "landmark_index", "n_landmarks_total", "horizon_months",
]

# One realistic, fully-populated example row -- reused for every endpoint's
# OpenAPI example (single predict, batch predict, schema docs) so they can
# never drift out of sync with each other or with RAW_REQUIRED_COLS/
# RAW_OPTIONAL_COLS above.
EXAMPLE_ROW = {
    "project_code": "P001",
    "project_name": "Doubling of XYZ Rail Line",
    "agency_name": "Ministry of Railways",
    "sector": "RAILWAYS",
    "state": "MAHARASHTRA",
    "project_status": "Ongoing",
    "reporting_quarter": "Q1",
    "financial_year": "2023-24",
    "original_cost_rs_cr": 200.0,
    "anticipated_cost_rs_cr": 210.0,
    "cumulative_expenditure_rs_cr": 20.0,
    "approval_date": "2020-04-01",
    "original_commissioning_date": "2024-03-31",
    "anticipated_commissioning_date": "2024-06-30",
}


# Explicit multi-project batch example. Each project has two reporting rows so
# Swagger/OpenAPI makes it obvious that the batch schema supports multiple
# projects and quarterly history. The batch endpoints do not reuse the single-
# row example above.
EXAMPLE_BATCH_REQUEST = {
    "projects": [
        {
            "project_code": "P001",
            "rows": [
                EXAMPLE_ROW,
                {
                    **EXAMPLE_ROW,
                    "project_code": "P001",
                    "reporting_quarter": "Q2",
                    "financial_year": "2023-24",
                    "cumulative_expenditure_rs_cr": 42.0,
                    "anticipated_cost_rs_cr": 214.0,
                },
            ],
        },
        {
            "project_code": "P002",
            "rows": [
                {
                    **EXAMPLE_ROW,
                    "project_code": "P002",
                    "project_name": "New National Highway Package",
                    "sector": "ROAD TRANSPORT AND HIGHWAYS",
                    "state": "KARNATAKA",
                    "original_cost_rs_cr": 320.0,
                    "anticipated_cost_rs_cr": 335.0,
                    "cumulative_expenditure_rs_cr": 30.0,
                },
                {
                    "project_code": "P002",
                    "project_name": "New National Highway Package",
                    "sector": "ROAD TRANSPORT AND HIGHWAYS",
                    "state": "KARNATAKA",
                    "reporting_quarter": "Q2",
                    "financial_year": "2023-24",
                    "original_cost_rs_cr": 320.0,
                    "anticipated_cost_rs_cr": 340.0,
                    "cumulative_expenditure_rs_cr": 67.0,
                    "approval_date": "2020-04-01",
                    "original_commissioning_date": "2024-03-31",
                    "anticipated_commissioning_date": "2024-09-30",
                },
            ],
        },
    ]
}

_DATE_RAW_COLS = [
    "approval_date", "original_commissioning_date",
    "revised_commissioning_date", "anticipated_commissioning_date",
]
_NUMERIC_RAW_COLS = [
    "original_cost_rs_cr", "revised_cost_rs_cr", "anticipated_cost_rs_cr",
    "cumulative_expenditure_rs_cr",
]

# Indian financial year: Apr-Jun / Jul-Sep / Oct-Dec / Jan-Mar. Q4 falls in
# the FY's second calendar year (FY "2023-24" Q4 ends 2024-03-31).
_QUARTER_START_MONTH_DAY = {
    "Q1": (4, 1),
    "Q2": (7, 1),
    "Q3": (10, 1),
    "Q4": (1, 1),
}

_AVG_MONTH_DAYS = 30.4375  # 365.25 / 12


def _fy_start_year(financial_year) -> float:
    """'2023-24' -> 2023. NaN (not a raised error) on anything that doesn't
    parse -- a single bad row should surface later as a normal
    validation error downstream, not crash feature derivation for the batch."""
    try:
        return float(int(str(financial_year).strip().split("-")[0]))
    except (ValueError, AttributeError, IndexError):
        return np.nan


def _quarter_start_date(financial_year, quarter):
    start_year = _fy_start_year(financial_year)
    q = str(quarter).strip().upper()
    if pd.isna(start_year) or q not in _QUARTER_START_MONTH_DAY:
        return pd.NaT
    month, day = _QUARTER_START_MONTH_DAY[q]
    year = int(start_year) + (1 if q == "Q4" else 0)
    try:
        return pd.Timestamp(year=year, month=month, day=day)
    except ValueError:
        return pd.NaT


def _months_between(start: pd.Series, end: pd.Series) -> pd.Series:
    """Elapsed months as a float, using a fixed-length average month.
    NaT on either side propagates to NaN (never raises)."""
    delta_days = (end - start).dt.days
    return (delta_days / _AVG_MONTH_DAYS).astype(float)


def _pct_change(new: pd.Series, base: pd.Series) -> pd.Series:
    out = (new - base) / base * 100
    return out.replace([np.inf, -np.inf], np.nan)


def derive_prediction_features(raw: pd.DataFrame) -> pd.DataFrame:
    """Raw, prediction-time schema (RAW_REQUIRED_COLS + RAW_OPTIONAL_COLS)
    -> the same frame with every engineered feature (ENGINEERED_COLS)
    appended. Row order is preserved; row count never changes.
    """
    if not isinstance(raw, pd.DataFrame):
        raise TypeError(f"raw must be a pandas DataFrame, got {type(raw).__name__}")
    if raw.empty:
        raise ValueError("Input has no rows to score.")

    missing = [c for c in RAW_REQUIRED_COLS if c not in raw.columns]
    if missing:
        raise ValueError(f"Missing required raw column(s): {', '.join(missing)}")

    x = raw.copy()
    for c in RAW_OPTIONAL_COLS:
        if c not in x.columns:
            x[c] = np.nan

    for c in ["project_code", "project_name", "agency_name", "sector", "state",
              "project_status", "financial_year"]:
        x[c] = x[c].astype("string").str.strip()

    for c in _DATE_RAW_COLS:
        x[c] = pd.to_datetime(x[c], errors="coerce", format="mixed")
    for c in _NUMERIC_RAW_COLS:
        x[c] = pd.to_numeric(x[c], errors="coerce")

    # ---- reporting_period_date / quarter --------------------------------
    x["quarter"] = x["reporting_quarter"].astype("string").str.strip().str.upper()
    x["reporting_period_date"] = [
        _quarter_start_date(fy, q) for fy, q in zip(x["financial_year"], x["quarter"])
    ]

    # ---- revised-if-present, else-original baselines --------------------
    effective_cost = x["revised_cost_rs_cr"].where(
        x["revised_cost_rs_cr"].notna(), x["original_cost_rs_cr"]
    )
    effective_completion = x["revised_commissioning_date"].where(
        x["revised_commissioning_date"].notna(), x["original_commissioning_date"]
    )

    # ---- engineered numeric features -------------------------------------
    x["project_age_at_report_months"] = _months_between(
        x["approval_date"], x["reporting_period_date"]
    )
    x["planned_duration_months"] = _months_between(
        x["approval_date"], effective_completion
    )

    x["progress_ratio"] = (
        x["project_age_at_report_months"] / x["planned_duration_months"]
    ).replace([np.inf, -np.inf], np.nan)

    x["expenditure_to_cost_pct"] = (
        (x["cumulative_expenditure_rs_cr"] / effective_cost) * 100
    ).replace([np.inf, -np.inf], np.nan)

    x["contemporaneous_cost_overrun_pct"] = _pct_change(
        x["anticipated_cost_rs_cr"], x["original_cost_rs_cr"]
    )
    x["contemporaneous_time_overrun_pct"] = (
        _months_between(x["original_commissioning_date"], x["anticipated_commissioning_date"])
        / _months_between(x["approval_date"], x["original_commissioning_date"])
        * 100
    ).replace([np.inf, -np.inf], np.nan)
    x["anticipated_cost_change_pct"] = _pct_change(x["anticipated_cost_rs_cr"], effective_cost)
    x["anticipated_date_change_months"] = _months_between(
        effective_completion, x["anticipated_commissioning_date"]
    )

    # ---- reporting sequence / cohort-size features -----------------------
    tmp = x.reset_index().sort_values(["project_code", "reporting_period_date", "index"])
    tmp["landmark_index"] = tmp.groupby("project_code").cumcount() + 1
    x["landmark_index"] = tmp.set_index("index")["landmark_index"].reindex(x.index)
    x["n_landmarks_total"] = x.groupby("project_code")["project_code"].transform("size")
    x["horizon_months"] = _months_between(x["reporting_period_date"], effective_completion)

    missing_engineered = [c for c in ENGINEERED_COLS if c not in x.columns]
    if missing_engineered:  # pragma: no cover - defensive; indicates a bug here
        raise AssertionError(
            f"derive_prediction_features() forgot to populate: {missing_engineered}"
        )

    return x
