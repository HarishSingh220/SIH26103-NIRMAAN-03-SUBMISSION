"""Raw prediction-time schema + feature derivation for the anomaly-detection
service.

The actual schema/derivation logic now lives in `app.common.raw_features`
(shared with the cost/time-overrun service so both accept literally the
same request-body shape -- see that module's docstring). This module
re-exports it under its original names so existing imports elsewhere in
this package (`from . import raw_features`, `from .raw_features import
derive_prediction_features`) keep working unchanged, and additionally
checks the result against this service's own `cfg.REQUIRED_COLS` before
returning -- a defensive, service-specific guarantee the shared module
itself doesn't make.
"""
from __future__ import annotations

import pandas as pd

from app.common.raw_features import (
    EXAMPLE_BATCH_REQUEST,
    EXAMPLE_ROW,
    RAW_OPTIONAL_COLS,
    RAW_REQUIRED_COLS,
    derive_prediction_features as _derive_prediction_features,
)

from . import config as cfg

__all__ = [
    "RAW_REQUIRED_COLS",
    "RAW_OPTIONAL_COLS",
    "EXAMPLE_ROW",
    "EXAMPLE_BATCH_REQUEST",
    "derive_prediction_features",
]


def derive_prediction_features(raw: pd.DataFrame) -> pd.DataFrame:
    x = _derive_prediction_features(raw)
    missing_engineered = [c for c in cfg.REQUIRED_COLS if c not in x.columns]
    if missing_engineered:  # pragma: no cover - defensive; indicates a bug here
        raise AssertionError(
            f"derive_prediction_features() forgot to populate: {missing_engineered}"
        )
    return x
