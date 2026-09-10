"""
Central configuration for the cost/time-overrun regression service.

Everything environment-specific (paths, thresholds) lives here so `train.py` and
`main.py`/`router.py` never hardcode a path themselves.
"""
import os
from pathlib import Path

from app.common.paths import (
    PROJECT_ROOT,
    DATA_DIR as SHARED_DATA_DIR,
    MODEL_DIR as SHARED_MODEL_DIR,
)


def _resolve_path(value: str | None, default: Path) -> str:
    path = Path(value) if value else default
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return str(path)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_DIR = _resolve_path(os.environ.get("DATA_DIR"), SHARED_DATA_DIR)
MODEL_DIR = _resolve_path(os.environ.get("MODEL_DIR"), SHARED_MODEL_DIR)

RAW_DATA_PATH = _resolve_path(
    os.environ.get("OVERRUN_RAW_DATA_PATH"), Path(DATA_DIR) / "paiman_projects_landmark_dataset.csv"
)

COST_MODEL_PATH = _resolve_path(None, Path(MODEL_DIR) / "final_cost_overrun_pct_combined_model.pkl")
TIME_MODEL_PATH = _resolve_path(None, Path(MODEL_DIR) / "final_time_overrun_pct_combined_model.pkl")
COST_SCHEMA_PATH = _resolve_path(None, Path(MODEL_DIR) / "cost_overrun_feature_schema.json")
TIME_SCHEMA_PATH = _resolve_path(None, Path(MODEL_DIR) / "time_overrun_feature_schema.json")

# ---------------------------------------------------------------------------
# Column names (shared by both targets' feature engineering)
# ---------------------------------------------------------------------------
PROJECT_COL = "project_code"
DATE_COL = "reporting_period_date"
GLOBAL_TIME_COL = "global_time_index"
PROJECT_TIME_COL = "project_reporting_index"

COST_TARGET = "final_cost_overrun_pct"
TIME_TARGET = "final_time_overrun_pct"
LOG_TARGET = "log_target"

# Leakage guards: for each target, the *other* outcome family plus this target's own flag
LEAKAGE_COLUMNS_BY_TARGET = {
    COST_TARGET: ["final_time_overrun_pct", "final_time_overrun_flag", "final_cost_overrun_flag"],
    TIME_TARGET: ["final_cost_overrun_pct", "final_cost_overrun_flag", "final_time_overrun_flag"],
}

RAW_NUMERIC_COLS = [
    "original_cost_rs_cr", "cumulative_expenditure_rs_cr", "expenditure_to_cost_pct",
    "project_age_at_report_months", "planned_duration_months", "progress_ratio",
    "landmark_index", "n_landmarks_total", "horizon_months",
]
RAW_TEXT_COLS = ["project_name", "sector", "state", "agency_name", "quarter", "financial_year"]
MISSINGNESS_SOURCE_COLS = [
    "state", "agency_name", "cumulative_expenditure_rs_cr",
    "expenditure_to_cost_pct", "project_age_at_report_months", "planned_duration_months",
]
CATEGORICAL_FEATURES = ["sector", "state", "agency_name"]

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
RANDOM_STATE = 42
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
N_OOF_FOLDS = 5
EXTREME_THRESHOLD = 500  # % overrun defining the "extreme tail" used for Model B's sector detection

XGB_PARAMS = dict(
    n_estimators=1200, learning_rate=0.03, max_depth=6, min_child_weight=5,
    subsample=0.85, colsample_bytree=0.85, reg_alpha=0.1, reg_lambda=2.0,
    objective="reg:squarederror", n_jobs=-1,
)
LGBM_PARAMS = dict(
    n_estimators=1200, learning_rate=0.03, num_leaves=31, max_depth=-1,
    min_child_samples=30, subsample=0.85, colsample_bytree=0.85,
    reg_alpha=0.1, reg_lambda=2.0, objective="regression", n_jobs=-1, verbosity=-1,
)
CATBOOST_PARAMS = dict(
    iterations=1200, learning_rate=0.03, depth=7, loss_function="RMSE",
    verbose=False, thread_count=-1,
)

# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------
API_HOST = os.environ.get("API_HOST", "0.0.0.0")
API_PORT = int(os.environ.get("API_PORT", "8000"))
DEFAULT_PREFERRED_MODEL = os.environ.get("DEFAULT_PREFERRED_MODEL", "model_b")
if DEFAULT_PREFERRED_MODEL not in {"model_b", "final_model"}:
    DEFAULT_PREFERRED_MODEL = "model_b"
MODEL_VERSION = "overrun-combined-v1"
