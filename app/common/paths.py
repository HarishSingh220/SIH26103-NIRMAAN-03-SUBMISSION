"""Project-wide filesystem paths used by training and inference code."""
from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"
CLASSIFICATION_MODEL_DIR = MODEL_DIR / "classification"

LANDMARK_DATASET_PATH = DATA_DIR / "paiman_projects_landmark_dataset.csv"
ANOMALY_DATASET_PATH = DATA_DIR / "paiman_projects_for_anomaly_detection.csv"
ANOMALY_MODEL_PATH = MODEL_DIR / "anomaly_model.joblib"

COST_OVERRUN_MODEL_PATH = MODEL_DIR / "final_cost_overrun_pct_combined_model.pkl"
TIME_OVERRUN_MODEL_PATH = MODEL_DIR / "final_time_overrun_pct_combined_model.pkl"
COST_SCHEMA_PATH = MODEL_DIR / "cost_overrun_feature_schema.json"
TIME_SCHEMA_PATH = MODEL_DIR / "time_overrun_feature_schema.json"

CLASSIFICATION_COST_MODEL_PATH = CLASSIFICATION_MODEL_DIR / "paiman_cost_overrun_ensemble.joblib"
CLASSIFICATION_TIME_MODEL_PATH = CLASSIFICATION_MODEL_DIR / "paiman_time_overrun_ensemble.joblib"
