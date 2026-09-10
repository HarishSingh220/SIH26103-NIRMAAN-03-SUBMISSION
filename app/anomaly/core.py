"""PAIMANA production anomaly scoring core.

The detector math is refactored from the validated final notebook without
changing the scoring formulas. Reference statistics/models are fitted offline
and reused at inference time.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from . import config as cfg

# 10.1 Leakage-safe reference fitting + peer and trajectory feature families

# NOTE: HYBRID_IF_N_ESTIMATORS / HYBRID_IF_N_JOBS / TRACK_B_* thresholds all
# live in app/config.py, the single source of truth for validated production
# configuration (see README). This module used to re-assign those same
# attributes on `cfg` at import time -- harmless while the values agreed, but
# a future edit to config.py would have been silently overwritten by whatever
# was hardcoded here. Removed; everything is read from `cfg` directly.

HYBRID_FIT_CUTOFF = cfg.FY_HOLDOUT_START


# ---------------------------------------------------------------------------
# Input validation & dtype coercion
#
# score_all() depends on both of these. They were referenced but never
# defined anywhere in this module -- every call to score_all() (i.e. every
# /predict* request) raised NameError. Implemented below.
# ---------------------------------------------------------------------------

_DATE_COLS = ["reporting_period_date", "anticipated_commissioning_date"]
_STRING_COLS = [
    "project_code", "project_name", "sector", "state", "agency_name",
    "financial_year", "quarter",
]
_NUMERIC_COLS = [
    c for c in cfg.REQUIRED_COLS if c not in _DATE_COLS + _STRING_COLS
]


# ---------------------------------------------------------------------------
# Human-readable reason text
#
# The detectors below originally emitted machine-oriented tokens straight
# into the *_reason output fields, e.g. "robust_z_max=6.23",
# "cusum_score=14.20", "duplicate_project_quarter",
# "anticipated_cost_below_original_cost". That's fine for a developer
# grepping logs, but these reason fields are the part of the response a
# reviewer actually reads to decide whether to act on a flagged project --
# and cryptic snake_case codes with no explanation of what they mean or why
# the number matters don't serve that audience. Everything below turns those
# tokens into plain-English sentences a non-technical reviewer can act on
# without a data dictionary, while keeping the underlying numbers (z-scores,
# thresholds, percentages) so the explanation stays specific and auditable
# rather than vague.
# ---------------------------------------------------------------------------

_METRIC_LABELS = {
    "expenditure_to_cost_pct": "the expenditure-to-cost ratio",
    "progress_ratio": "the physical progress ratio",
    "contemporaneous_cost_overrun_pct": "the cost overrun percentage",
    "contemporaneous_time_overrun_pct": "the time overrun percentage",
    "anticipated_cost_change_pct": "the anticipated cost change",
    "anticipated_date_change_months": "the anticipated commissioning-date change",
}

_TRACK_B_FEATURE_LABELS = {
    "original_cost_rs_cr": "the original project cost",
    "cumulative_expenditure_rs_cr": "cumulative expenditure",
    "anticipated_cost_rs_cr": "the anticipated final cost",
    "expenditure_to_cost_pct": "the expenditure-to-cost ratio",
    "project_age_at_report_months": "the project's age",
    "planned_duration_months": "the planned project duration",
    "progress_ratio": "the physical progress ratio",
}


def _pretty_metric(feat, labels=_METRIC_LABELS):
    """Human-readable name for a raw feature column, with a safe fallback."""
    return labels.get(feat, feat.replace("_", " "))


def _describe_z_signal(label):
    """Turn a signal label like "base::progress_ratio" into a plain-English
    description of what moved and how, e.g. "the physical progress ratio,
    from one quarter to the next".
    """
    kind, _, feat = label.partition("::")
    metric = _pretty_metric(feat)
    if kind == "peer":
        return f"{metric}, compared with similar projects in its peer group"
    if kind == "accel":
        return f"the pace of change in {metric.replace('the ', '', 1)} (speeding up or reversing faster than usual)"
    if kind == "trend":
        return f"the 3-quarter trend in {metric.replace('the ', '', 1)}"
    # "base" (or any future family): plain quarter-over-quarter change.
    return f"{metric}, from one quarter to the next"



def validate_input(raw: pd.DataFrame) -> None:
    """Fail fast with a clear, catchable error before any scoring runs.

    Raises ValueError/TypeError on problems the API layer turns into a 422.
    Duplicate (project_code, reporting_period_date) rows are intentionally
    NOT rejected here -- detecting them is one of the Branch B/Track B data
    quality signals, so they must survive to reach that logic.
    """
    if not isinstance(raw, pd.DataFrame):
        raise TypeError("Input must be tabular (a list of row objects / a DataFrame).")
    if raw.empty:
        raise ValueError("Input contains no rows.")

    missing = [c for c in cfg.REQUIRED_COLS if c not in raw.columns]
    if missing:
        raise ValueError(f"Missing required column(s): {', '.join(missing)}")

    if raw["project_code"].isna().any() or raw["project_code"].astype(str).str.strip().eq("").any():
        raise ValueError("project_code must be present and non-empty for every row.")

    parsed_dates = pd.to_datetime(raw["reporting_period_date"], errors="coerce")
    if parsed_dates.isna().any():
        bad_codes = (
            raw.loc[parsed_dates.isna(), "project_code"].astype(str).unique().tolist()
        )
        raise ValueError(
            "reporting_period_date could not be parsed (expected YYYY-MM-DD) "
            f"for project_code(s): {bad_codes[:10]}"
        )


def coerce_dtypes(raw: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of ``raw`` with production dtypes applied.

    - date columns -> datetime64 (required for .diff()/.dt accessors downstream)
    - identifier/categorical columns -> stripped strings
    - everything else in REQUIRED_COLS -> numeric (invalid values become NaN,
      which the existing null-handling in the feature/DQ logic already covers)
    """
    x = raw.copy()

    for c in _STRING_COLS:
        if c in x.columns:
            x[c] = x[c].astype("string").str.strip()

    for c in _DATE_COLS:
        if c in x.columns:
            x[c] = pd.to_datetime(x[c], errors="coerce")

    for c in _NUMERIC_COLS:
        if c in x.columns:
            x[c] = pd.to_numeric(x[c], errors="coerce")

    return x

def _robust_location_scale(s):
    s = pd.Series(s).replace([np.inf, -np.inf], np.nan).dropna()
    if s.empty:
        return np.nan, np.nan
    med = s.median()
    mad = (s - med).abs().median()
    if pd.notna(mad) and mad > 0:
        return med, mad
    std = s.std()
    if pd.notna(std) and std > 0:
        return med, std
    return med, 1.0

def _prepare_hybrid_frame(raw, causal=False):
    x = raw.copy()
    if "_source_row_id" not in x.columns:
        x["_source_row_id"] = np.arange(len(x), dtype=np.int64)
    x["_source_row_id"] = x["_source_row_id"].astype(np.int64)
    x = x.sort_values(
        ["project_code", "reporting_period_date", "_source_row_id"]
    ).reset_index(drop=True)

    x["n_quarters_on_record"] = (
        x.groupby("project_code")["reporting_period_date"].transform("count")
    )

    dur = x["planned_duration_months"].mask(
        x["planned_duration_months"].eq(0)
    )
    x["planned_duration_months_model"] = dur.groupby(
        x["project_code"]
    ).transform(lambda s: s.ffill() if causal else s.ffill().bfill())

    progress = x["progress_ratio"].copy()
    recoverable = (
        (progress.isna() | np.isinf(progress))
        & x["planned_duration_months_model"].notna()
    )
    progress.loc[recoverable] = (
        x.loc[recoverable, "project_age_at_report_months"]
        / x.loc[recoverable, "planned_duration_months_model"]
    )
    x["progress_ratio_model"] = progress.replace(
        [np.inf, -np.inf], np.nan
    )

    for feat in cfg.delta_features:
        src = "progress_ratio_model" if feat == "progress_ratio" else feat
        dcol = f"{feat}__qoq_delta"
        x[dcol] = x.groupby("project_code")[src].diff()
        x[dcol] = x[dcol].replace([np.inf, -np.inf], np.nan)

    for feat in cfg.TRAJECTORY_BASE_FEATURES:
        dcol = f"{feat}__qoq_delta"
        acol = f"{feat}__acceleration"
        tcol = f"{feat}__trend_3q"
        g = x.groupby("project_code")[dcol]
        d1 = g.shift(1)
        d2 = g.shift(2)
        x[acol] = x[dcol] - d1
        valid_count = (
            x[dcol].notna().astype(int)
            + d1.notna().astype(int)
            + d2.notna().astype(int)
        )
        rolling_sum = (
            x[dcol].fillna(0)
            + d1.fillna(0)
            + d2.fillna(0)
        )
        x[tcol] = (
            rolling_sum / valid_count.replace(0, np.nan)
        ).where(valid_count >= 2, np.nan)
        x[acol] = x[acol].replace([np.inf, -np.inf], np.nan)
        x[tcol] = x[tcol].replace([np.inf, -np.inf], np.nan)

    prev_duration = x.groupby("project_code")[
        "planned_duration_months_model"
    ].shift(1)
    x["duration_extension_months"] = (
        x["planned_duration_months_model"] - prev_duration
    )
    return x

def _variant_signal_specs(feature_families):
    specs = []
    if "base" in feature_families:
        specs += [
            (f"{feat}__qoq_delta", f"base::{feat}")
            for feat in cfg.delta_features
        ]
    if "peer" in feature_families:
        specs += [
            (f"{feat}__peer_z", f"peer::{feat}")
            for feat in cfg.PEER_FEATURES
        ]
    if "trajectory" in feature_families:
        for feat in cfg.TRAJECTORY_BASE_FEATURES:
            specs += [
                (f"{feat}__acceleration", f"accel::{feat}"),
                (f"{feat}__trend_3q", f"trend::{feat}"),
            ]
    return specs


def _fit_project_size_cutpoints(x, fit_mask):
    vals = pd.to_numeric(x.loc[fit_mask, "original_cost_rs_cr"], errors="coerce").dropna()
    q = np.unique(vals.quantile([0.20, 0.40, 0.60, 0.80]).to_numpy())
    return q.tolist()


def _apply_project_size_bucket(x, cutpoints):
    labels = ["XS", "S", "M", "L", "XL"]
    q = np.asarray(cutpoints, dtype=float)
    if len(q) < 4:
        x["project_size_bucket"] = "M"
        return x
    cost = pd.to_numeric(x["original_cost_rs_cr"], errors="coerce")
    x["project_size_bucket"] = cost.map(
        lambda v: np.nan if pd.isna(v) else labels[min(np.searchsorted(q, v, side="right"), 4)]
    )
    return x


def _peer_keys(frame, cols):
    return frame[cols].astype("string").fillna("__MISSING__").agg("||".join, axis=1)

def fit_hybrid_reference(
    raw, fit_cutoff=HYBRID_FIT_CUTOFF, feature_families=None,
    if_n_estimators=None, if_contamination=None, fit_isolation_forest=True
):
    if feature_families is None:
        feature_families = ["base"]

    x = _prepare_hybrid_frame(raw, causal=False)
    fit_mask = x["financial_year"] < fit_cutoff
    size_cutpoints = _fit_project_size_cutpoints(x, fit_mask)
    x = _apply_project_size_bucket(x, size_cutpoints)

    source_features = []
    if "base" in feature_families:
        source_features += [f"{f}__qoq_delta" for f in cfg.delta_features]
    if "trajectory" in feature_families:
        for f in cfg.TRAJECTORY_BASE_FEATURES:
            source_features += [
                f"{f}__acceleration", f"{f}__trend_3q"
            ]

    z_params = {}
    for source_col in source_features:
        global_med, global_scale = _robust_location_scale(
            x.loc[fit_mask, source_col]
        )
        z_params[source_col] = {
            "__global__": (global_med, global_scale)
        }
        for sector, gidx in x.groupby("sector", dropna=False).groups.items():
            train_idx = gidx[fit_mask.loc[gidx].to_numpy()]
            med, scale = _robust_location_scale(
                x.loc[train_idx, source_col]
            )
            if pd.isna(med):
                med, scale = global_med, global_scale
            z_params[source_col][sector] = (med, scale)

    for source_col, params in z_params.items():
        med_map = {
            s: v[0] for s, v in params.items() if s != "__global__"
        }
        scale_map = {
            s: v[1] for s, v in params.items() if s != "__global__"
        }
        med = x["sector"].map(med_map).fillna(params["__global__"][0])
        scale = x["sector"].map(scale_map).fillna(params["__global__"][1])
        x[f"{source_col}__z"] = (
            0.6745 * (x[source_col] - med) / scale
        ).replace([np.inf, -np.inf], np.nan)

    peer_params = {}
    if "peer" in feature_families:
        for feat in cfg.PEER_FEATURES:
            peer_params[feat] = {
                "__global__": _robust_location_scale(
                    x.loc[fit_mask, feat]
                )
            }
            key = _peer_keys(x.loc[fit_mask], cfg.HYBRID_PEER_GROUP_COLS)
            tmp = x.loc[fit_mask].copy()
            tmp["_peer_group_key"] = key
            for group_key, g in tmp.groupby("_peer_group_key", dropna=False):
                n = g[feat].notna().sum()
                if n >= cfg.HYBRID_MIN_PEER_GROUP_SIZE:
                    med, scale = _robust_location_scale(g[feat])
                    if pd.notna(med):
                        peer_params[feat][group_key] = (med, scale)

        # Sector fallback is retained explicitly for sparse state/size groups.
        for feat in cfg.PEER_FEATURES:
            sector_map = {}
            sector_scale = {}
            tmp = x.loc[fit_mask].copy()
            for sector, g in tmp.groupby("sector", dropna=False):
                med, scale = _robust_location_scale(g[feat])
                if pd.notna(med):
                    sector_map[sector] = med
                    sector_scale[sector] = scale
            peer_params[feat]["__sector__"] = (sector_map, sector_scale)

        for feat in cfg.PEER_FEATURES:
            params = peer_params[feat]
            group_key = _peer_keys(x, cfg.HYBRID_PEER_GROUP_COLS)
            gmed = {k: v[0] for k, v in params.items() if k not in ["__global__", "__sector__"]}
            gscale = {k: v[1] for k, v in params.items() if k not in ["__global__", "__sector__"]}
            med = group_key.map(gmed)
            scale = group_key.map(gscale)
            if cfg.HYBRID_PEER_FALLBACK_TO_SECTOR:
                smed, sscale = params["__sector__"]
                med = med.fillna(x["sector"].map(smed))
                scale = scale.fillna(x["sector"].map(sscale))
            med = med.fillna(params["__global__"][0])
            scale = scale.fillna(params["__global__"][1])
            x[f"{feat}__peer_z"] = (0.6745 * (x[feat] - med) / scale).replace([np.inf, -np.inf], np.nan)

    signal_specs = _variant_signal_specs(feature_families)
    zcols = []
    signal_labels = []
    for source_col, label in signal_specs:
        zcol = (
            source_col
            if source_col.endswith("__peer_z")
            else f"{source_col}__z"
        )
        zcols.append(zcol)
        signal_labels.append(label)

    model_mask = fit_mask & (x["n_quarters_on_record"] >= 3)
    if_models = {}
    if fit_isolation_forest:
        sector_groups = x[x["n_quarters_on_record"] >= 3].groupby(
            "sector", dropna=False
        )
    else:
        sector_groups = []
    for sector, g in sector_groups:
        train = g.loc[model_mask.loc[g.index]]
        if len(train) < cfg.HYBRID_IF_MIN_ROWS or not zcols:
            continue
        imputer = SimpleImputer(strategy="median")
        X_train = imputer.fit_transform(train[zcols])
        model = IsolationForest(
            n_estimators=(cfg.HYBRID_IF_N_ESTIMATORS if if_n_estimators is None else if_n_estimators),
            contamination=(cfg.HYBRID_IF_CONTAMINATION if if_contamination is None else if_contamination),
            random_state=42,
            n_jobs=cfg.HYBRID_IF_N_JOBS
        )
        model.fit(X_train)
        train_scores = -model.decision_function(X_train)
        if_models[sector] = (
            imputer, model, np.sort(train_scores)
        )

    return {
        "fit_cutoff": fit_cutoff,
        "feature_families": list(feature_families),
        "z_params": z_params,
        "peer_params": peer_params,
        "zcols": zcols,
        "signal_labels": signal_labels,
        "if_models": if_models,
        "project_size_cutpoints": size_cutpoints,
    }


# 10.2 Hybrid scoring: base + peer + trajectory + risk ranking

def _add_risk_ranking(frame):
    """Add deterministic score ranking plus score-based risk levels.

    Ranking is retained for diagnostics, but it NO LONGER determines risk.
    Risk labels depend only on the actual anomaly score, so the distribution
    can change when detector outputs or thresholds change.
    """
    out = frame.sort_values(
        ["final_anomaly_score", "project_code",
         "reporting_period_date", "_source_row_id"],
        ascending=[False, True, True, True]
    ).copy()
    n = len(out)
    out["risk_rank"] = np.arange(1, n + 1, dtype=np.int64)
    out["risk_percentile"] = (
        1 - (out["risk_rank"] - 1) / max(n - 1, 1)
    ).clip(0, 1)

    # IMPORTANT: risk_level is score-based, not top-k based.
    out["risk_level"] = np.select(
        [
            out["final_anomaly_score"] >= cfg.RISK_CRITICAL_THRESHOLD,
            out["final_anomaly_score"] >= cfg.RISK_REVIEW_THRESHOLD,
            out["final_anomaly_score"] >= cfg.RISK_MONITOR_THRESHOLD,
        ],
        ["Critical", "Review", "Normal"],
        default="Normal"
    )

    out["review_queue_flag"] = (
        out["final_anomaly_score"] >= cfg.RISK_REVIEW_THRESHOLD
    ).astype(int)
    out["critical_queue_flag"] = (
        out["final_anomaly_score"] >= cfg.RISK_CRITICAL_THRESHOLD
    ).astype(int)
    return out

def _cusum_sequence(values):
    s_pos, s_neg = 0.0, 0.0
    flags, scores = [], []
    for value in values:
        if pd.isna(value):
            flags.append(0)
            scores.append(np.nan)
            continue
        s_pos = max(0.0, s_pos + value - cfg.HYBRID_CUSUM_K)
        s_neg = min(0.0, s_neg + value + cfg.HYBRID_CUSUM_K)
        score = max(s_pos, -s_neg)
        flag = int(
            s_pos > cfg.HYBRID_CUSUM_H or -s_neg > cfg.HYBRID_CUSUM_H
        )
        flags.append(flag)
        scores.append(score)
        if flag:
            s_pos, s_neg = 0.0, 0.0
    return np.asarray(flags), np.asarray(scores)

def run_hybrid_framework(
    raw,
    reference=None,
    retrospective=True,
    causal=True,
    feature_families=None,
    enable_cusum=True,
    enable_if=True,
    numerical_weight=None,
    dq_weight=None,
    numerical_flag_mode="corroborated"
):
    if reference is None:
        raise ValueError("hybrid_reference is required for inference")
    if feature_families is None:
        feature_families = reference.get(
            "feature_families", ["base"]
        )

    x = _prepare_hybrid_frame(raw, causal=causal)
    if reference.get("project_size_cutpoints") is not None:
        x = _apply_project_size_bucket(x, reference["project_size_cutpoints"])

    # Historical z-score references.
    for source_col, params in reference["z_params"].items():
        med_map = {
            s: v[0] for s, v in params.items()
            if s != "__global__"
        }
        scale_map = {
            s: v[1] for s, v in params.items()
            if s != "__global__"
        }
        med = x["sector"].map(med_map).fillna(
            params["__global__"][0]
        )
        scale = x["sector"].map(scale_map).fillna(
            params["__global__"][1]
        )
        x[f"{source_col}__z"] = (
            0.6745 * (x[source_col] - med) / scale
        ).replace([np.inf, -np.inf], np.nan)

    # Peer features: current-level project values against historical
    # sector-peer median/MAD. This is independent of the longitudinal
    # QoQ reference and therefore adds a complementary signal.
    if "peer" in feature_families:
        for feat in cfg.PEER_FEATURES:
            params = reference["peer_params"][feat]
            med_map = {
                s: v[0] for s, v in params.items()
                if s != "__global__"
            }
            scale_map = {
                s: v[1] for s, v in params.items()
                if s != "__global__"
            }
            group_key = _peer_keys(x, cfg.HYBRID_PEER_GROUP_COLS)
            med_map = {s: v[0] for s, v in params.items() if s not in ["__global__", "__sector__"]}
            scale_map = {s: v[1] for s, v in params.items() if s not in ["__global__", "__sector__"]}
            med = group_key.map(med_map)
            scale = group_key.map(scale_map)
            if cfg.HYBRID_PEER_FALLBACK_TO_SECTOR and "__sector__" in params:
                smed, sscale = params["__sector__"]
                med = med.fillna(x["sector"].map(smed))
                scale = scale.fillna(x["sector"].map(sscale))
            med = med.fillna(params["__global__"][0])
            scale = scale.fillna(params["__global__"][1])
            x[f"{feat}__peer_z"] = (0.6745 * (x[feat] - med) / scale).replace([np.inf, -np.inf], np.nan)

    ta = x[x["n_quarters_on_record"] >= 3].copy()
    ta = ta.sort_values(
        ["project_code", "reporting_period_date", "_source_row_id"]
    )

    signal_specs = _variant_signal_specs(feature_families)
    zcols = []
    signal_labels = []
    for source_col, label in signal_specs:
        zcol = (
            source_col if source_col.endswith("__peer_z")
            else f"{source_col}__z"
        )
        zcols.append(zcol)
        signal_labels.append(label)

    # Numerical branch.
    ta["zscore_max_abs"] = (
        ta[zcols].abs().max(axis=1) if zcols else 0.0
    )
    ta["robust_z_flag"] = (
        ta["zscore_max_abs"] > cfg.HYBRID_Z_THRESHOLD
    ).astype(int)
    ta["robust_z_anomaly_score"] = (
        ta["zscore_max_abs"] / cfg.HYBRID_Z_THRESHOLD
    ).clip(0, 1).fillna(0)

    # Which tracked metric actually drove the max z-score, so the reason
    # text can name it (e.g. "expenditure-to-cost ratio") instead of just
    # reporting an unlabeled number. fillna(-1) avoids idxmax raising on a
    # row where every signal is NaN -- that row's zscore_max_abs is 0/NaN
    # and robust_z_flag is False regardless, so its (meaningless) driver
    # label is never surfaced in the reason text.
    if zcols:
        _driver_col = ta[zcols].abs().fillna(-1.0).idxmax(axis=1)
        ta["_zscore_driver_label"] = _driver_col.map(
            dict(zip(zcols, signal_labels))
        )
    else:
        ta["_zscore_driver_label"] = None

    if enable_cusum:
        cusum_flag_cols, cusum_score_cols = [], []
        for source_col, label in signal_specs:
            zcol = (
                source_col if source_col.endswith("__peer_z")
                else f"{source_col}__z"
            )
            fcol = f"{label}__cusum_flag"
            scol = f"{label}__cusum_score"
            flags = np.zeros(len(ta), dtype=np.int8)
            scores = np.full(len(ta), np.nan)
            positions = {idx: i for i, idx in enumerate(ta.index)}
            for _, group in ta.groupby("project_code", sort=False):
                f, ss = _cusum_sequence(group[zcol].to_numpy())
                inds = np.fromiter(
                    (positions[idx] for idx in group.index),
                    dtype=int, count=len(group)
                )
                flags[inds] = f
                scores[inds] = ss
            ta[fcol] = flags
            ta[scol] = scores
            cusum_flag_cols.append(fcol)
            cusum_score_cols.append(scol)

        ta["cusum_flag"] = (
            ta[cusum_flag_cols].sum(axis=1) > 0
        ).astype(int)
        ta["cusum_max_score"] = ta[cusum_score_cols].max(axis=1)
        ta["cusum_anomaly_score"] = (
            ta["cusum_max_score"] / cfg.HYBRID_CUSUM_H
        ).clip(0, 1).fillna(0)
    else:
        ta["cusum_flag"] = 0
        ta["cusum_max_score"] = 0.0
        ta["cusum_anomaly_score"] = 0.0

    # Isolation Forest.
    ta["iso_forest_score"] = np.nan
    ta["iso_forest_flag"] = 0
    ta["iso_forest_anomaly_score"] = 0.0
    # Whether this row's sector actually had a fitted IF model at train time.
    # A sector with no model isn't "confirmed normal" by IF -- it was simply
    # never scored by it (too few training rows, or unseen at fit time).
    ta["iso_forest_model_available"] = False

    if enable_if:
        for sector, group in ta.groupby("sector", dropna=False):
            if sector not in reference["if_models"]:
                continue
            imputer, model, train_scores = reference["if_models"][sector]
            X = imputer.transform(group[zcols])
            scores = -model.decision_function(X)
            flags = (model.predict(X) == -1).astype(int)
            tails = (
                np.searchsorted(train_scores, scores, side="right") + 0.5
            ) / (len(train_scores) + 1)
            ta.loc[group.index, "iso_forest_score"] = scores
            ta.loc[group.index, "iso_forest_flag"] = flags
            ta.loc[group.index, "iso_forest_anomaly_score"] = tails
            ta.loc[group.index, "iso_forest_model_available"] = True

    # Fusion weights for the three numerical detectors. IF normally carries
    # half the numerical weight. Previously, any row whose sector had no
    # fitted IF model (too few training rows, or a sector unseen at fit
    # time) silently got iso_forest_anomaly_score=0 plugged into that 0.50
    # slot -- which doesn't mean "IF found nothing," it means "IF was never
    # run," and it quietly halved the effective numerical signal for every
    # row in that sector. Redistribute IF's weight onto the two remaining
    # detectors (keeping their 1:1 ratio) whenever IF didn't run, so
    # unmonitored sectors aren't scored as if they were confirmed normal.
    _w_z, _w_cusum, _w_if = 0.25, 0.25, 0.50
    _if_available = ta["iso_forest_model_available"]
    _w_z_row = np.where(_if_available, _w_z, _w_z + _w_if / 2)
    _w_cusum_row = np.where(_if_available, _w_cusum, _w_cusum + _w_if / 2)
    _w_if_row = np.where(_if_available, _w_if, 0.0)

    ta["numerical_anomaly_score"] = (
        _w_z_row * ta["robust_z_anomaly_score"]
        + _w_cusum_row * ta["cusum_anomaly_score"].fillna(0)
        + _w_if_row * ta["iso_forest_anomaly_score"].fillna(0)
    ).fillna(0).clip(0, 1)

    # Numerical flag corroboration fix.
    # Previously a single noisy detector could flag a row on its own via
    # either the "extreme Z" escape (1.5x threshold) or the "extreme IF
    # score" escape (>=0.995). Isolation Forest never exceeded F1~0.06 at
    # any tuned setting (see 5B), so letting it independently flag rows
    # added false positives without adding real detections. "legacy_or"
    # keeps the original behavior for comparison; "corroborated" (the new
    # default) requires >=2 of 3 detectors to agree, or a much more extreme
    # lone Z-score (2.5x) -- the lone-IF escape is removed entirely.
    # Corroboration means at least 2 of the 3 detectors agree.
    # The previous implementation multiplied detector flags by weights
    # and then compared the weighted sum to 2. Because the weights sum to 1,
    # that condition could NEVER be true.
    _numerical_corroboration = (
        ta[["robust_z_flag", "cusum_flag", "iso_forest_flag"]].sum(axis=1) >= 2
    )
    if numerical_flag_mode == "legacy_or":
        ta["numerical_anomaly_flag"] = (
            _numerical_corroboration
            | (ta["zscore_max_abs"] > 1.5 * cfg.HYBRID_Z_THRESHOLD)
            | (ta["iso_forest_anomaly_score"] >= 0.995)
        ).astype(int)
    else:
        ta["numerical_anomaly_flag"] = (
            _numerical_corroboration
            | (ta["zscore_max_abs"] > 2.5 * cfg.HYBRID_Z_THRESHOLD)
        ).astype(int)

    def _numerical_reason(row):
        reasons = []
        if row["robust_z_flag"]:
            driver = row.get("_zscore_driver_label")
            what = (
                _describe_z_signal(driver) if isinstance(driver, str)
                else "one of this project's tracked metrics"
            )
            reasons.append(
                f"An unusually large shift was detected in {what} "
                f"(z-score {row['zscore_max_abs']:.1f}, beyond the normal "
                f"range of ±{cfg.HYBRID_Z_THRESHOLD:.1f})"
            )
        if row["cusum_flag"]:
            reasons.append(
                "A sustained drift has been building up across recent "
                f"quarters rather than a single one-off jump (CUSUM score "
                f"{row['cusum_max_score']:.1f}, past the alert level of "
                f"{cfg.HYBRID_CUSUM_H:.1f})"
            )
        if row["iso_forest_flag"]:
            reasons.append(
                "This quarter's overall combination of metrics looks "
                "statistically unusual compared with similar projects "
                "(flagged by the isolation forest model)"
            )
        if not row["iso_forest_model_available"]:
            reasons.append(
                "The isolation forest check could not be run for this "
                "project's sector (not enough historical projects to "
                "train it)"
            )
        return "; ".join(reasons) if reasons else (
            "No unusual numerical patterns were detected this quarter"
        )

    ta["numerical_anomaly_reason"] = ta.apply(
        _numerical_reason, axis=1
    )
    ta = ta.drop(columns=["_zscore_driver_label"])

    # Branch B: data quality / temporal / logical consistency.
    entity_cols = ["project_name", "agency_name", "state", "sector"]
    for col in entity_cols:
        previous = ta.groupby("project_code")[col].shift(1)
        next_value = ta.groupby("project_code")[col].shift(-1)
        ta[f"{col}_changed_from_previous_flag"] = (
            ta[col].notna() & previous.notna() & (ta[col] != previous)
        ).astype(int)
        if retrospective:
            ta[f"{col}_sandwich_flag"] = (
                ta[col].notna() & previous.notna() & next_value.notna()
                & (ta[col] != previous) & (ta[col] != next_value)
            ).astype(int)
        else:
            ta[f"{col}_sandwich_flag"] = 0

    ta["duplicate_project_quarter_flag"] = ta.duplicated(
        ["project_code", "reporting_period_date"], keep=False
    ).astype(int)
    ta["report_gap_days"] = (
        ta.groupby("project_code")["reporting_period_date"]
        .diff().dt.days
    )
    ta["near_duplicate_flag"] = (
        ta["report_gap_days"] < 30
    ).fillna(False).astype(int)
    ta["unexpected_reporting_gap_flag"] = (
        ta["report_gap_days"] > 120
    ).fillna(False).astype(int)
    ta["rule_expenditure_decrease_flag"] = (
        ta.groupby("project_code")["cumulative_expenditure_rs_cr"]
        .diff() < 0
    ).fillna(False).astype(int)
    ta["rule_large_date_revision_flag"] = (
        ta["anticipated_date_change_months"].abs() > 24
    ).fillna(False).astype(int)
    ta["rule_large_cost_revision_flag"] = (
        ta["anticipated_cost_change_pct"].abs() > 200
    ).fillna(False).astype(int)
    ta["rule_duration_extension_flag"] = (
        ta["duration_extension_months"] > 0
    ).fillna(False).astype(int)
    ta["rule_contradictory_numeric_flag"] = (
        (ta["project_age_at_report_months"] < 0)
        | (ta["planned_duration_months_model"] < 0)
        | (ta["progress_ratio_model"] < 0)
        | (ta["cumulative_expenditure_rs_cr"] < 0)
        | (ta["original_cost_rs_cr"] < 0)
        | (ta["anticipated_cost_rs_cr"] < 0)
    ).fillna(False).astype(int)

    # BUG FIX: entity consistency was completely inert in production.
    # ``retrospective`` is always False at inference (score_all() always
    # calls run_hybrid_framework(..., retrospective=False)), which zeroes
    # out every ``*_sandwich_flag`` above -- correctly, since "changed then
    # reverted" needs a *next* quarter's value that a live causal request
    # cannot see yet. But entity_consistency_score was built ONLY from those
    # sandwich flags, so it was hardcoded to 0.0 for every single production
    # prediction: a project could have its agency, sector, state, or name
    # silently changed in the exact quarter it happened and this detector
    # would never fire, regardless of anomaly_type/final_anomaly_reason.
    # `*_changed_from_previous_flag` (computed above, right alongside the
    # sandwich flags) is the causal-safe half of this check -- it only looks
    # backward, so it is always available, including in live requests -- but
    # it was computed and then never read anywhere. Track B already does the
    # analogous thing correctly (track_b_*_changed_flag feeds
    # track_b_entity_change_flag unconditionally); Track A did not. Fold the
    # causal signal in via max() so retrospective/offline audits keep
    # exactly their previous (stronger, confirmed-by-reversion) score, and
    # live requests get a real, non-zero signal the moment a change occurs
    # instead of staying silent until -- at best, never -- a future quarter
    # confirms it reverted.
    ta["entity_consistency_score"] = pd.concat([
        (
            1.00 * ta["project_name_sandwich_flag"]
            + 0.40 * ta["sector_sandwich_flag"]
            + 0.20 * ta["state_sandwich_flag"]
            + 0.10 * ta["agency_name_sandwich_flag"]
        ),
        (
            0.55 * ta["project_name_changed_from_previous_flag"]
            + 0.30 * ta["sector_changed_from_previous_flag"]
            + 0.20 * ta["state_changed_from_previous_flag"]
            + 0.10 * ta["agency_name_changed_from_previous_flag"]
        ),
    ], axis=1).max(axis=1).clip(0, 1)
    ta["temporal_consistency_score"] = (
        1.00 * ta["duplicate_project_quarter_flag"]
        + 0.80 * ta["near_duplicate_flag"]
        + 0.55 * ta["unexpected_reporting_gap_flag"]
    ).clip(0, 1)
    ta["cross_field_logical_score"] = (
        1.00 * ta["rule_expenditure_decrease_flag"]
        + 0.50 * ta["rule_large_date_revision_flag"]
        + 0.50 * ta["rule_large_cost_revision_flag"]
        + 0.35 * ta["rule_duration_extension_flag"]
        + 1.00 * ta["rule_contradictory_numeric_flag"]
    ).clip(0, 1)
    ta["data_quality_score"] = pd.concat(
        [
            ta["entity_consistency_score"],
            ta["temporal_consistency_score"],
            ta["cross_field_logical_score"]
        ], axis=1
    ).max(axis=1).clip(0, 1)
    ta["data_quality_flag"] = (
        ta["data_quality_score"] >= 0.50
    ).astype(int)

    def _dq_reason(row):
        reasons = []
        # Report the stronger, confirmed-by-reversion reason when it fired
        # (retrospective/offline audits only); otherwise fall back to the
        # causal-safe "changed since last quarter" reason, which is the
        # only one that can ever fire on a live/production request. See the
        # entity_consistency_score fix above for why both exist.
        if row["project_name_sandwich_flag"]:
            reasons.append(
                "The project name changed this quarter and then reverted "
                "back to the original name the following quarter -- often "
                "a sign of a data-entry mistake rather than a real rename"
            )
        elif row["project_name_changed_from_previous_flag"]:
            reasons.append("The project name changed from the previous quarter")
        if row["sector_sandwich_flag"]:
            reasons.append(
                "The sector was changed this quarter and then reverted "
                "back the following quarter"
            )
        elif row["sector_changed_from_previous_flag"]:
            reasons.append("The sector changed from the previous quarter")
        if row["state_sandwich_flag"]:
            reasons.append(
                "The state was changed this quarter and then reverted "
                "back the following quarter"
            )
        elif row["state_changed_from_previous_flag"]:
            reasons.append("The state changed from the previous quarter")
        if row["agency_name_sandwich_flag"]:
            reasons.append(
                "The implementing agency was changed this quarter and "
                "then reverted back the following quarter"
            )
        elif row["agency_name_changed_from_previous_flag"]:
            reasons.append("The implementing agency changed from the previous quarter")
        if row["duplicate_project_quarter_flag"]:
            reasons.append(
                "This project reported the same quarter more than once "
                "(a duplicate entry for the same reporting period)"
            )
        if row["near_duplicate_flag"]:
            reasons.append(
                "This report was filed less than 30 days after the "
                "previous one, unusually close together"
            )
        if row["unexpected_reporting_gap_flag"]:
            reasons.append(
                f"More than 120 days passed since this project's previous "
                f"report ({row['report_gap_days']:.0f}-day gap), longer "
                f"than expected between quarterly updates"
            )
        if row["rule_expenditure_decrease_flag"]:
            reasons.append(
                "Cumulative expenditure decreased from the previous "
                "quarter, which should not normally happen"
            )
        if row["rule_large_date_revision_flag"]:
            reasons.append(
                "The anticipated commissioning date was revised by more "
                f"than 24 months in a single quarter "
                f"({row['anticipated_date_change_months']:.1f} months)"
            )
        if row["rule_large_cost_revision_flag"]:
            reasons.append(
                "The anticipated final cost was revised by more than "
                f"200% in a single quarter "
                f"({row['anticipated_cost_change_pct']:.0f}%)"
            )
        if row["rule_duration_extension_flag"]:
            reasons.append(
                "The planned project duration was extended compared with "
                "the previous quarter"
            )
        if row["rule_contradictory_numeric_flag"]:
            reasons.append(
                "Some reported figures are logically inconsistent (a "
                "negative cost, expenditure, age, duration, or progress "
                "value)"
            )
        return "; ".join(reasons) if reasons else (
            "No data-quality issues were detected this quarter"
        )

    ta["data_quality_reason"] = ta.apply(_dq_reason, axis=1)

    # Hybrid risk score. Numerical evidence receives twice the fusion weight
    # of the DQ branch by default. Weights remain overridable for sensitivity
    # analysis, but the selected research configuration is 2:1 numerical:DQ.
    _w_num = cfg.HYBRID_NUMERICAL_WEIGHT if numerical_weight is None else numerical_weight
    _w_dq = cfg.HYBRID_DQ_WEIGHT if dq_weight is None else dq_weight
    ta["final_anomaly_score"] = (
        _w_num * ta["numerical_anomaly_score"]
        + _w_dq * ta["data_quality_score"]
    ).clip(0, 1)

    # Track B is reported independently as well as being included in the
    # 2:1 hybrid score. This is important because a DQ-only anomaly can have
    # a maximum hybrid contribution of 1/3 and therefore should not disappear
    # merely because the research objective prioritizes numerical anomalies.
    ta["track_b_anomaly_score"] = ta["data_quality_score"]
    ta["track_b_anomaly_flag"] = ta["data_quality_flag"]
    ta["track_b_risk_level"] = np.select(
        [
            ta["track_b_anomaly_score"] >= cfg.TRACK_B_CRITICAL_THRESHOLD,
            ta["track_b_anomaly_score"] >= cfg.TRACK_B_REVIEW_THRESHOLD,
            ta["track_b_anomaly_score"] >= cfg.TRACK_B_MONITOR_THRESHOLD,
        ],
        ["Critical", "Review", "Normal"],
        default="Normal"
    )

    # Final hybrid decision.
    ta["final_anomaly_flag"] = (
        ta["final_anomaly_score"] >= cfg.HYBRID_REVIEW_THRESHOLD
    ).astype(int)

    # Risk level is score-based, so it can change with the anomaly score.
    # Scores below the review threshold are reported as Normal.
    ta["final_priority"] = np.select(
        [
            ta["final_anomaly_score"] >= cfg.HYBRID_CRITICAL_THRESHOLD,
            ta["final_anomaly_score"] >= cfg.HYBRID_REVIEW_THRESHOLD,
            ta["final_anomaly_score"] >= cfg.RISK_MONITOR_THRESHOLD,
        ],
        ["Critical", "Review", "Normal"],
        default="Normal"
    )

    def _anomaly_type(row):
        types = []
        if row["numerical_anomaly_flag"]:
            types.append("NUMERICAL_BEHAVIOR")
        if any(row[c] for c in [
            "project_name_sandwich_flag", "agency_name_sandwich_flag",
            "state_sandwich_flag", "sector_sandwich_flag",
            # Causal-safe half of the same check -- see the
            # entity_consistency_score fix above. Without these, a live
            # request could have entity_consistency_score > 0 (driving
            # data_quality_score/final_anomaly_score) while anomaly_type
            # still reported something else entirely, or "NONE".
            "project_name_changed_from_previous_flag",
            "agency_name_changed_from_previous_flag",
            "state_changed_from_previous_flag",
            "sector_changed_from_previous_flag",
        ]):
            types.append("ENTITY_CONSISTENCY")
        if any(row[c] for c in [
            "duplicate_project_quarter_flag", "near_duplicate_flag",
            "unexpected_reporting_gap_flag"
        ]):
            types.append("TEMPORAL_CONSISTENCY")
        if any(row[c] for c in [
            "rule_expenditure_decrease_flag", "rule_large_date_revision_flag",
            "rule_large_cost_revision_flag", "rule_duration_extension_flag",
            "rule_contradictory_numeric_flag"
        ]):
            types.append("CROSS_FIELD_LOGICAL")
        if len(types) > 1:
            return "MULTIPLE"
        if types:
            return types[0]
        if row["data_quality_flag"]:
            return "DATA_QUALITY"
        return "NONE"

    ta["anomaly_type"] = ta.apply(_anomaly_type, axis=1)

    def _final_reason(row):
        parts = []
        if row["numerical_anomaly_flag"]:
            parts.append(
                "Numerical behavior: " + row["numerical_anomaly_reason"] + "."
            )
        # Gating on data_quality_score > 0 (any sub-signal fired) rather
        # than on data_quality_flag (the aggregate >=0.50 threshold) keeps
        # this text consistent with anomaly_type, which is also driven by
        # the individual sub-flags rather than the aggregate threshold.
        # Without this, a project could come back as anomaly_type =
        # "ENTITY_CONSISTENCY" / "CROSS_FIELD_LOGICAL" (a single weaker
        # signal fired, e.g. a lone metadata change or a lone duration
        # extension) while final_anomaly_reason said "No anomalies
        # detected" -- correct about the *score* not crossing the review
        # threshold, but a confusing, self-contradicting explanation.
        # final_anomaly_flag/final_priority/risk_level are unaffected by
        # this -- they still use data_quality_score against the validated
        # thresholds exactly as before; only the text explanation changes.
        if row["data_quality_score"] > 0:
            parts.append(
                "Data quality: " + row["data_quality_reason"] + "."
            )
        return " ".join(parts) if parts else (
            "No anomalies detected -- this quarter's report is "
            "consistent with the project's history."
        )

    ta["final_anomaly_reason"] = ta.apply(_final_reason, axis=1)
    ta["feature_families"] = "+".join(feature_families)

    return _add_risk_ranking(ta)


# ============================================================
# TRACK B — short-history peer plausibility + Branch B quality checks
# ============================================================
# Track B has <3 observations and is therefore NOT forced through
# longitudinal Z/CUSUM/Isolation-Forest scoring. It uses the dedicated
# peer-reference + Branch B approach.
#
# Thresholds/features for Track B live in app/config.py (cfg.TRACK_B_*) --
# removed the duplicate reassignment that used to live here, since it just
# re-wrote cfg with the same values and risked silently overriding any
# future edit made in config.py instead.


def _fit_track_b_peer_reference(raw, cutoff=cfg.FY_HOLDOUT_START):
    x = raw.copy()
    fit_mask = x["financial_year"] < cutoff
    fit = x.loc[fit_mask].copy()
    cuts = _fit_project_size_cutpoints(x, fit_mask)
    fit = _apply_project_size_bucket(fit, cuts)
    params = {"__global__": {}, "__sector__": {}, "size_cutpoints": cuts}

    for feat in cfg.TRACK_B_PEER_FEATURES:
        params["__global__"][feat] = _robust_location_scale(fit[feat])
        for sector, g in fit.groupby("sector", dropna=False):
            if sector not in params["__sector__"]:
                params["__sector__"][sector] = {}
            params["__sector__"][sector][feat] = _robust_location_scale(g[feat])
    return params


def _track_b_peer_score(x, reference):
    out = x.copy()
    cuts = reference.get("size_cutpoints")
    if cuts is not None:
        out = _apply_project_size_bucket(out, cuts)

    z_cols = []
    for feat in cfg.TRACK_B_PEER_FEATURES:
        med_global, scale_global = reference["__global__"][feat]
        sector_vals = reference.get("__sector__", {})
        sector_meds = {k: v[feat][0] for k, v in sector_vals.items() if feat in v}
        sector_scales = {k: v[feat][1] for k, v in sector_vals.items() if feat in v}
        med = out["sector"].map(sector_meds).fillna(med_global)
        scale = out["sector"].map(sector_scales).fillna(scale_global)
        z = (out[feat] - med).abs() * 0.6745 / scale
        z = z.replace([np.inf, -np.inf], np.nan)
        zcol = f"track_b_{feat}__peer_z"
        out[zcol] = z
        z_cols.append(zcol)

    out["track_b_peer_z_max"] = out[z_cols].max(axis=1)
    out["track_b_peer_anomaly_score"] = (
        out["track_b_peer_z_max"] / cfg.TRACK_B_PEER_Z_THRESHOLD
    ).clip(0, 1).fillna(0)
    out["track_b_peer_flag"] = (
        out["track_b_peer_z_max"] > cfg.TRACK_B_PEER_Z_THRESHOLD
    ).astype(int)

    def _peer_reason(row):
        if pd.isna(row["track_b_peer_z_max"]):
            return (
                "Not enough peer projects with comparable data were "
                "available to judge whether this project's figures are "
                "typical"
            )
        drivers = []
        for feat in cfg.TRACK_B_PEER_FEATURES:
            z = row[f"track_b_{feat}__peer_z"]
            if pd.notna(z) and z >= cfg.TRACK_B_PEER_Z_THRESHOLD:
                drivers.append(_pretty_metric(feat, _TRACK_B_FEATURE_LABELS))
        if drivers:
            return (
                "This project looks unusual compared with similar "
                "projects (same sector/state/size band), particularly in "
                + ", ".join(drivers)
            )
        return "This project's figures are in line with similar projects"

    out["track_b_peer_reason"] = out.apply(_peer_reason, axis=1)
    return out


def run_track_b_branch(raw, peer_reference):
    """Score short-history projects using peer plausibility + Branch B rules.

    ``peer_reference`` (from ``_fit_track_b_peer_reference``) is required.
    BUG FIX: this used to default to ``peer_reference=None`` and, when the
    caller omitted it, fell back to a module-level name
    ``track_b_peer_reference`` that was never defined anywhere in this file
    or imported from elsewhere. That fallback branch could never have
    succeeded -- it always raised ``NameError: name 'track_b_peer_reference'
    is not defined`` at call time. ``score_all()`` (the only in-repo caller)
    always passes ``peer_reference`` explicitly, so production requests
    never hit this path, but any other caller -- a notebook, a script, a
    future endpoint -- that used the documented default would get a
    confusing NameError instead of the real problem ("you forgot to pass a
    fitted reference"). Making the argument required fails fast with a
    normal, clear ``TypeError: run_track_b_branch() missing 1 required
    positional argument`` instead.
    """
    x = raw[raw["n_quarters_on_record"] < 3].copy()
    if "_source_row_id" not in x.columns:
        x["_source_row_id"] = np.arange(len(x), dtype=np.int64)
    x["anticipated_commissioning_date"] = pd.to_datetime(
        x["anticipated_commissioning_date"], errors="coerce"
    )
    x = x.sort_values(["project_code", "reporting_period_date", "_source_row_id"])
    x = _track_b_peer_score(x, peer_reference)

    # -------- BRANCH B: entity / temporal / logical quality --------
    x["track_b_duplicate_project_quarter_flag"] = x.duplicated(
        ["project_code", "reporting_period_date"], keep=False
    ).astype(int)

    prev_date = x.groupby("project_code")["reporting_period_date"].shift(1)
    gap_days = (x["reporting_period_date"] - prev_date).dt.days
    x["track_b_reporting_gap_flag"] = (gap_days > 120).fillna(False).astype(int)

    x["track_b_missing_critical_flag"] = (
        x[["project_name", "sector", "state", "agency_name", "original_cost_rs_cr"]]
        .isna().any(axis=1)
    ).astype(int)

    x["track_b_duration_invalid_flag"] = (
        x["planned_duration_months"].notna() & (x["planned_duration_months"] <= 0)
    ).astype(int)

    x["track_b_cum_gt_original_flag"] = (
        x["cumulative_expenditure_rs_cr"].notna()
        & x["original_cost_rs_cr"].notna()
        & (x["cumulative_expenditure_rs_cr"] > x["original_cost_rs_cr"])
    ).astype(int)

    x["track_b_cum_gt_anticipated_flag"] = (
        x["cumulative_expenditure_rs_cr"].notna()
        & x["anticipated_cost_rs_cr"].notna()
        & (x["cumulative_expenditure_rs_cr"] > x["anticipated_cost_rs_cr"])
    ).astype(int)

    x["track_b_anticipated_lt_original_flag"] = (
        x["anticipated_cost_rs_cr"].notna()
        & x["original_cost_rs_cr"].notna()
        & (x["anticipated_cost_rs_cr"] < x["original_cost_rs_cr"])
    ).astype(int)

    x["track_b_anticipated_date_overdue_flag"] = (
        x["anticipated_commissioning_date"].notna()
        & (x["anticipated_commissioning_date"] < x["reporting_period_date"])
    ).astype(int)

    entity_change_cols = []
    for col in ["project_name", "agency_name", "state", "sector"]:
        prev = x.groupby("project_code")[col].shift(1)
        c = f"track_b_{col}_changed_flag"
        x[c] = (prev.notna() & x[col].notna() & prev.ne(x[col])).astype(int)
        entity_change_cols.append(c)

    x["track_b_entity_change_flag"] = (x[entity_change_cols].sum(axis=1) > 0).astype(int)

    x["track_b_data_quality_score"] = pd.concat([
        x["track_b_duplicate_project_quarter_flag"].astype(float),
        x["track_b_reporting_gap_flag"].astype(float) * 0.50,
        x["track_b_missing_critical_flag"].astype(float) * 0.45,
        x["track_b_duration_invalid_flag"].astype(float) * 0.90,
        x["track_b_cum_gt_original_flag"].astype(float) * 0.80,
        x["track_b_cum_gt_anticipated_flag"].astype(float),
        x["track_b_anticipated_lt_original_flag"].astype(float) * 0.60,
        x["track_b_anticipated_date_overdue_flag"].astype(float) * 0.60,
        x["track_b_entity_change_flag"].astype(float) * 0.25,
    ], axis=1).max(axis=1).clip(0, 1)

    x["track_b_data_quality_flag"] = (x["track_b_data_quality_score"] >= 0.50).astype(int)

    def _branch_b_reason(row):
        reasons = []
        if row["track_b_duplicate_project_quarter_flag"]:
            reasons.append(
                "This project reported the same quarter more than once "
                "(a duplicate entry for the same reporting period)"
            )
        if row["track_b_reporting_gap_flag"]:
            reasons.append(
                "More than 120 days passed since this project's previous "
                "report, longer than expected between quarterly updates"
            )
        if row["track_b_missing_critical_flag"]:
            reasons.append(
                "Some required project details are missing (project "
                "name, sector, state, agency, or original cost)"
            )
        if row["track_b_duration_invalid_flag"]:
            reasons.append(
                "The planned project duration is zero or negative, "
                "which is not a valid value"
            )
        if row["track_b_cum_gt_original_flag"]:
            reasons.append(
                "Cumulative expenditure has already exceeded the "
                "originally approved project cost"
            )
        if row["track_b_cum_gt_anticipated_flag"]:
            reasons.append(
                "Cumulative expenditure has already exceeded even the "
                "most recently anticipated final cost"
            )
        if row["track_b_anticipated_lt_original_flag"]:
            reasons.append(
                "The anticipated final cost is lower than the original "
                "approved cost, which is unusual"
            )
        if row["track_b_anticipated_date_overdue_flag"]:
            reasons.append(
                "The anticipated commissioning date has already passed "
                "as of this report"
            )
        if row["track_b_entity_change_flag"]:
            reasons.append(
                "Project details (name, agency, state, or sector) "
                "changed even though this project has very little "
                "reporting history, which makes it hard to tell if this "
                "is a correction or a data problem"
            )
        return "; ".join(reasons) if reasons else (
            "No data-quality issues were detected this quarter"
        )

    x["track_b_data_quality_reason"] = x.apply(_branch_b_reason, axis=1)
    x["track_b_final_anomaly_score"] = (
        0.60 * x["track_b_peer_anomaly_score"]
        + 0.40 * x["track_b_data_quality_score"]
    ).clip(0, 1)

    x["track_b_final_anomaly_flag"] = (
        (x["track_b_final_anomaly_score"] >= cfg.TRACK_B_REVIEW_THRESHOLD)
        | ((x["track_b_peer_flag"] == 1) & (x["track_b_data_quality_flag"] == 1))
    ).astype(int)

    x["track_b_final_priority"] = np.select(
        [
            x["track_b_final_anomaly_score"] >= cfg.TRACK_B_CRITICAL_THRESHOLD,
            x["track_b_final_anomaly_score"] >= cfg.TRACK_B_REVIEW_THRESHOLD,
            x["track_b_final_anomaly_score"] >= cfg.TRACK_B_MONITOR_THRESHOLD,
        ],
        ["Critical", "Review", "Normal"],
        default="Normal"
    )

    x["track_b_anomaly_type"] = np.select(
        [
            (x["track_b_peer_flag"] == 1) & (x["track_b_data_quality_flag"] == 1),
            x["track_b_peer_flag"] == 1,
            x["track_b_data_quality_flag"] == 1,
        ],
        ["PEER_PLAUSIBILITY_AND_DATA_QUALITY", "PEER_PLAUSIBILITY", "DATA_QUALITY"],
        default="NONE"
    )

    x["track_b_final_anomaly_reason"] = (
        "Peer comparison: " + x["track_b_peer_reason"] + ". "
        + "Data quality: " + x["track_b_data_quality_reason"] + "."
    )
    x["track"] = "B"
    x["anomaly_confidence"] = "LOWER_THAN_TRACK_A"
    x["track_b_anomaly_score"] = x["track_b_final_anomaly_score"]
    x["track_b_anomaly_flag"] = x["track_b_final_anomaly_flag"]
    x["track_b_risk_level"] = x["track_b_final_priority"]

    # BUG FIX: Track A populates the shared/top-level output fields
    # (risk_level, final_anomaly_score, final_anomaly_flag, final_priority,
    # anomaly_type, final_anomaly_reason) via _add_risk_ranking() and the
    # hybrid scoring in run_hybrid_framework(). Track B never set those same
    # top-level field names -- only their track_b_-prefixed equivalents --
    # so any request that scored at least one short-history project came
    # back with risk_level=None for those rows (and, for a request made up
    # entirely of short-history projects, the column was missing from the
    # result altogether: `result["risk_level"]` raised KeyError). This
    # broke the "risk_level" contract documented in config.OUTPUT_COLS and
    # the README's "risk labels: Normal/Review/Critical" section for every
    # Track B row. Mirror the Track B equivalents onto the shared field
    # names so callers get a consistent schema regardless of track.
    x["risk_level"] = x["track_b_final_priority"]
    x["final_anomaly_score"] = x["track_b_final_anomaly_score"]
    x["final_anomaly_flag"] = x["track_b_final_anomaly_flag"]
    x["final_priority"] = x["track_b_final_priority"]
    x["anomaly_type"] = x["track_b_anomaly_type"]
    x["final_anomaly_reason"] = x["track_b_final_anomaly_reason"]
    # No numerical-detector equivalent exists on Track B (too little history
    # for Z/CUSUM/Isolation Forest); fill with explicit not-applicable
    # defaults instead of leaving them silently absent/NaN.
    x["numerical_anomaly_score"] = np.nan
    x["robust_z_flag"] = 0
    x["cusum_flag"] = 0
    x["iso_forest_flag"] = 0
    x["zscore_max_abs"] = np.nan
    x["numerical_anomaly_reason"] = (
        "Not applicable -- this project has fewer than 3 reported "
        "quarters, so longitudinal numerical checks (z-score, CUSUM, "
        "isolation forest) could not be run. See the peer-comparison and "
        "data-quality reasons instead."
    )
    x["data_quality_score"] = x["track_b_data_quality_score"]
    x["data_quality_reason"] = x["track_b_data_quality_reason"]
    return x

def score_all(raw: pd.DataFrame, hybrid_reference: dict, track_b_reference: dict) -> pd.DataFrame:
    """Run the full production pipeline on project histories.

    ``raw`` may contain multiple quarterly rows for one project or many projects.
    Rows are grouped by ``project_code`` and ordered by ``reporting_period_date``.
    Projects with >=3 observations are scored by longitudinal Track A; projects
    with <3 observations are scored by Track B. One scored row is returned for
    every input row.
    """
    validate_input(raw)
    x = coerce_dtypes(raw)
    x["_source_row_id"] = np.arange(len(x), dtype=np.int64)

    # Prepare once so the project-history count is calculated from the complete
    # request, not separately for each row. This is what allows an API request
    # containing Q1/Q2/Q3/Q4 for one project to enter Track A.
    prepped = _prepare_hybrid_frame(x, causal=True)
    track_a_input = prepped[prepped["n_quarters_on_record"] >= 3].copy()
    track_b_input = prepped[prepped["n_quarters_on_record"] < 3].copy()

    results = []

    if not track_a_input.empty:
        track_a_results = run_hybrid_framework(
            track_a_input,
            hybrid_reference,
            retrospective=False,
            causal=True,
            feature_families=hybrid_reference.get(
                "feature_families", ["base", "peer", "trajectory"]
            ),
            numerical_weight=cfg.HYBRID_NUMERICAL_WEIGHT,
            dq_weight=cfg.HYBRID_DQ_WEIGHT,
            numerical_flag_mode="corroborated",
        )
        track_a_results["track"] = "A"
        track_a_results["anomaly_confidence"] = "STANDARD_TRACK_A"
        for c, default in {
            "track_b_final_anomaly_score": np.nan,
            "track_b_final_anomaly_flag": 0,
            "track_b_final_priority": "NOT_APPLICABLE",
            "track_b_anomaly_type": "NOT_APPLICABLE",
            "track_b_final_anomaly_reason": (
                "Not scored on Track B -- this project has 3 or more "
                "reported quarters, so it was scored on Track A "
                "(longitudinal) instead."
            ),
        }.items():
            track_a_results[c] = default
        results.append(track_a_results)

    if not track_b_input.empty:
        track_b_results = run_track_b_branch(track_b_input, track_b_reference)
        results.append(track_b_results)

    if not results:
        return pd.DataFrame()

    all_results = pd.concat(results, ignore_index=True, sort=False)
    return all_results.sort_values(
        ["project_code", "reporting_period_date", "_source_row_id"]
    ).reset_index(drop=True)