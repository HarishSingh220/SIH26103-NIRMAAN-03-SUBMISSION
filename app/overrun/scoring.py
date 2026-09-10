"""Composes ONE target's (cost or time) full prediction block -- predicted
percentages from `CombinedOverrunPipeline`, plus the exact-amount/exact-date
outcome from `app.overrun.outcomes` -- from a project's already-derived
engineered rows.

This is the single place that decides WHICH of a project's rows is
"current" for both (a) reading off the ensembles' predicted percentage and
(b) anchoring the exact-outcome calculation. Both `app/overrun/router.py`'s
single/history endpoint and `app.classification.pipeline`'s automatic
regression cascade call into this module, so they can never drift apart
on that choice -- a real bug in an earlier version of this endpoint took
"the row the caller happened to place last in their JSON array" instead of
"the row with the truly-latest `reporting_period_date`", which silently
scored the wrong quarter whenever a caller's `history` list wasn't already
sorted chronologically. This module fixes that in one place instead of
patching each caller separately.
"""
from __future__ import annotations

from typing import Any, Optional, Union

import pandas as pd

from . import config as cfg
from . import outcomes
from .core import CombinedOverrunPipeline


def latest_row_index(expanded_df: pd.DataFrame, date_col: str = cfg.DATE_COL):
    """Row LABEL (not position -- use `expanded_df.index.get_loc(...)` to
    convert if you need a position into a same-order numpy array) of
    `expanded_df`'s row with the latest `date_col` value.

    Falls back to the last row in supplied order only if NOTHING parses as
    a date (should not happen once `derive_prediction_features()` has run,
    but keeps this deterministic rather than raising on a fully-malformed
    input).
    """
    dates = pd.to_datetime(expanded_df[date_col], errors="coerce")
    if dates.notna().any():
        return dates.idxmax()
    return expanded_df.index[-1]


def predict_target_block(
    pipeline: CombinedOverrunPipeline,
    expanded_df: pd.DataFrame,
    preferred_model: str,
    risk: Optional[bool] = None,
) -> dict[str, Any]:
    """Full response block for ONE target (cost or time), given a
    project's already-derived engineered rows (`app.common.raw_features.
    derive_prediction_features()` output for just that project_code):
    `predicted_overrun_pct` (+ both raw ensemble predictions) AND the
    concrete exact-amount/exact-date outcome (see `app.overrun.outcomes`),
    both computed off the row with the TRULY latest `reporting_period_date`
    -- not just whichever row happened to be last in the caller's array.

    `risk=False` skips the (expensive) regression call entirely and
    returns a block with every prediction field set to `None` plus a
    `reason` -- used when an upstream classifier has already cleared this
    project/target. `risk=None` (default) or `True` both run it: `None`
    keeps this pipeline usable stand-alone, without a classifier gating it.
    """
    latest_idx = latest_row_index(expanded_df)
    anchor = expanded_df.loc[latest_idx]

    if risk is False:
        block: dict[str, Any] = {
            "predicted_overrun_pct": None,
            "model_b_prediction": None,
            "final_model_prediction": None,
            "preferred_model": None,
            "reason": "Not flagged as overrun risk by classifier - regression skipped",
        }
    else:
        result = pipeline.predict(expanded_df, return_frame=False)
        pos = expanded_df.index.get_loc(latest_idx)
        model_b_pred = float(result["model_b"][pos])
        final_model_pred = float(result["final_model"][pos])
        primary = model_b_pred if preferred_model == "model_b" else final_model_pred
        block = {
            "predicted_overrun_pct": round(primary, 4),
            "model_b_prediction": round(model_b_pred, 4),
            "final_model_prediction": round(final_model_pred, 4),
            "preferred_model": preferred_model,
        }

    return outcomes.attach_exact_outcome(block, pipeline.target_col, anchor)


def target_block_from_row(row: Union[dict, pd.Series], target_col: str, preferred_model: str) -> dict[str, Any]:
    """Same output shape as `predict_target_block`, but for the bulk
    /predict/batch and /predict/csv path -- there, `row` already IS the
    project's correctly-identified latest row (see `_latest_rows()` in
    `app/overrun/router.py`), with `CombinedOverrunPipeline.predict(...,
    return_frame=True)`'s two prediction columns already computed and
    attached to it, so no second model call or row-selection is needed
    here -- just read the columns off and attach the exact outcome.
    """
    model_b_pred = float(row[f"predicted_{target_col}_model_b"])
    final_pred = float(row[f"predicted_{target_col}_final_model"])
    primary = model_b_pred if preferred_model == "model_b" else final_pred
    block = {
        "predicted_overrun_pct": round(primary, 4),
        "model_b_prediction": round(model_b_pred, 4),
        "final_model_prediction": round(final_pred, 4),
        "preferred_model": preferred_model,
    }
    return outcomes.attach_exact_outcome(block, target_col, row)
