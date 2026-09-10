"""
Central configuration for the overrun-risk classification service.

Mirrors the pattern used by `app.overrun.config` / `app.anomaly.config`: every
environment-specific value (model paths, the risk-gating threshold) lives
here so nothing else in this package hardcodes a path or a magic number.
"""
import os
from pathlib import Path

from app.common.paths import PROJECT_ROOT, CLASSIFICATION_MODEL_DIR


def _resolve_path(value: str | None, default: Path) -> str:
    path = Path(value) if value else default
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return str(path)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
MODEL_DIR = _resolve_path(os.environ.get("CLASSIFICATION_MODEL_DIR"), CLASSIFICATION_MODEL_DIR)

COST_MODEL_PATH = _resolve_path(
    os.environ.get("CLASSIFICATION_COST_MODEL_PATH"),
    Path(MODEL_DIR) / "paiman_cost_overrun_ensemble.joblib",
)
TIME_MODEL_PATH = _resolve_path(
    os.environ.get("CLASSIFICATION_TIME_MODEL_PATH"),
    Path(MODEL_DIR) / "paiman_time_overrun_ensemble.joblib",
)

# ---------------------------------------------------------------------------
# Risk gating
# ---------------------------------------------------------------------------
# The classifier's own trained decision threshold (used for its own
# "Overrun Risk" / "No Overrun" label) is stored inside each model artifact
# and can differ from 0.5. The gateway that decides whether to run the
# downstream regression/anomaly services, however, always gates on the raw
# probability against this fixed cutoff, independent of the model's own
# tuned threshold.
OVERRUN_RISK_PROBABILITY_THRESHOLD = float(
    os.environ.get("OVERRUN_RISK_PROBABILITY_THRESHOLD", "0.5")
)

MODEL_VERSION = "classification-v1"
