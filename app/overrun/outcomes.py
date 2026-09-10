"""Turns a predicted overrun PERCENTAGE into the concrete numbers a caller
actually wants: the projected final cost (Rs Cr) for the cost-overrun
target, and the projected final commissioning date for the time-overrun
target.

Both `final_cost_overrun_pct` and `final_time_overrun_pct` are defined
against the project's ORIGINAL baseline, never a later revision -- the
same convention `app.common.raw_features` uses for the contemporaneous
cost/time features this pipeline is trained alongside::

    final_cost_overrun_pct = (final_cost_rs_cr - original_cost_rs_cr)
                              / original_cost_rs_cr * 100

    final_time_overrun_pct = delay_months
                              / months(approval_date -> original_commissioning_date)
                              * 100

(verified directly against `data/paiman_projects_landmark_dataset.csv`:
e.g. project 020100040 has `original_cost_rs_cr=13171`,
`final_cost_overrun_pct=70.5413`, and its cumulative expenditure converges
to ~22,460 Rs Cr -- exactly `13171 * (1 + 0.705413)`).

Both formulas are inverted here the same way to recover the exact
amount/date from a predicted percentage.
"""
from __future__ import annotations

from typing import Any, Optional, Union

import numpy as np
import pandas as pd

from . import config as cfg

_AVG_MONTH_DAYS = 30.4375  # 365.25 / 12 -- matches app.common.raw_features


def _clean_float(v: Any) -> Optional[float]:
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return v if np.isfinite(v) else None


def _clean_date(v: Any) -> Optional[pd.Timestamp]:
    ts = pd.to_datetime(v, errors="coerce")
    return None if pd.isna(ts) else ts


def _iso_date(ts: Optional[pd.Timestamp]) -> Optional[str]:
    return None if ts is None else ts.date().isoformat()


def planned_duration_months_baseline(approval_date: Any, original_commissioning_date: Any) -> Optional[float]:
    """The project's ORIGINAL planned duration (approval -> original
    commissioning date), in months -- the same denominator
    `contemporaneous_time_overrun_pct` uses in `app.common.raw_features`.

    Deliberately re-derived from the two raw dates here rather than reused
    from an already-engineered `planned_duration_months` column: that
    column swaps in a REVISED commissioning date when one is present,
    which would corrupt this baseline (the training target was always
    defined against the ORIGINAL date, revision or not).
    """
    approval = _clean_date(approval_date)
    original_commissioning = _clean_date(original_commissioning_date)
    if approval is None or original_commissioning is None:
        return None
    return (original_commissioning - approval).days / _AVG_MONTH_DAYS


def exact_cost_outcome(original_cost_rs_cr: Any, predicted_overrun_pct: Optional[float]) -> dict[str, Optional[float]]:
    """`{"original_cost_rs_cr", "predicted_final_cost_rs_cr", "predicted_cost_increase_rs_cr"}`,
    all None if either input is missing/unparseable (e.g. the block being
    filled in is a "skipped - not flagged as risk" block)."""
    original_cost = _clean_float(original_cost_rs_cr)
    pct = _clean_float(predicted_overrun_pct)
    if original_cost is None or pct is None:
        return {
            "original_cost_rs_cr": original_cost,
            "predicted_final_cost_rs_cr": None,
            "predicted_cost_increase_rs_cr": None,
        }
    final_cost = original_cost * (1 + pct / 100)
    return {
        "original_cost_rs_cr": round(original_cost, 4),
        "predicted_final_cost_rs_cr": round(final_cost, 4),
        "predicted_cost_increase_rs_cr": round(final_cost - original_cost, 4),
    }


def exact_time_outcome(
    approval_date: Any, original_commissioning_date: Any, predicted_overrun_pct: Optional[float]
) -> dict[str, Optional[Any]]:
    """`{"original_commissioning_date", "planned_duration_months",
    "predicted_delay_months", "predicted_final_commissioning_date"}`, all
    None (beyond whatever baseline could be computed) if any input is
    missing/unparseable."""
    original_commissioning = _clean_date(original_commissioning_date)
    baseline_months = planned_duration_months_baseline(approval_date, original_commissioning_date)
    pct = _clean_float(predicted_overrun_pct)

    if original_commissioning is None or baseline_months is None or pct is None:
        return {
            "original_commissioning_date": _iso_date(original_commissioning),
            "planned_duration_months": round(baseline_months, 4) if baseline_months is not None else None,
            "predicted_delay_months": None,
            "predicted_final_commissioning_date": None,
        }

    delay_months = (pct / 100) * baseline_months
    final_date = original_commissioning + pd.Timedelta(days=delay_months * _AVG_MONTH_DAYS)
    return {
        "original_commissioning_date": _iso_date(original_commissioning),
        "planned_duration_months": round(baseline_months, 4),
        "predicted_delay_months": round(delay_months, 4),
        "predicted_final_commissioning_date": _iso_date(final_date),
    }


def attach_exact_outcome(
    block: dict[str, Any], target_col: str, anchor: Union[pd.Series, dict],
) -> dict[str, Any]:
    """Adds the exact-amount/exact-date fields to an existing
    cost_overrun/time_overrun response `block` (mutated in place, and also
    returned for convenience).

    `anchor` supplies the raw values the percentage is inverted against --
    `original_cost_rs_cr` for the cost target, `approval_date` +
    `original_commissioning_date` for the time target -- and should be the
    project's TRUTHFULLY LATEST reporting-period row (a pandas Series or a
    plain dict both work; only `.get(...)` is used). `block["predicted_overrun_pct"]`
    supplies the percentage to invert.

    Safe to call on a "skipped" block (`predicted_overrun_pct` is `None`):
    every exact field simply comes back `None` too, same as the
    percentages already do.
    """
    pct = block.get("predicted_overrun_pct")
    if target_col == cfg.COST_TARGET:
        block.update(exact_cost_outcome(anchor.get("original_cost_rs_cr"), pct))
    elif target_col == cfg.TIME_TARGET:
        block.update(
            exact_time_outcome(anchor.get("approval_date"), anchor.get("original_commissioning_date"), pct)
        )
    else:
        raise ValueError(f"Unknown target_col: {target_col!r}")
    return block
