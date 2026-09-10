"""Production configuration for the PAIMANA hybrid anomaly detector.

These values are the validated final-model configuration. Changing them
requires re-running the notebook benchmark/audit before deployment.
"""
from app.common.paths import ANOMALY_MODEL_PATH, ANOMALY_DATASET_PATH

FY_HOLDOUT_START = "2023-24"

DATA_PATH = ANOMALY_DATASET_PATH
MODEL_PATH = ANOMALY_MODEL_PATH

delta_features = [
    "expenditure_to_cost_pct","progress_ratio",
    "contemporaneous_cost_overrun_pct","contemporaneous_time_overrun_pct",
    "anticipated_cost_change_pct","anticipated_date_change_months",
]
PEER_FEATURES = delta_features.copy()
TRAJECTORY_BASE_FEATURES = delta_features.copy()

HYBRID_Z_THRESHOLD = 5.5
HYBRID_PEER_GROUP_COLS = ["sector","state","project_size_bucket"]
HYBRID_MIN_PEER_GROUP_SIZE = 100
HYBRID_PEER_FALLBACK_TO_SECTOR = True
HYBRID_CUSUM_K = 0.15
HYBRID_CUSUM_H = 12.0
HYBRID_IF_MIN_ROWS = 50
HYBRID_IF_CONTAMINATION = 0.10
HYBRID_IF_N_ESTIMATORS = 500
HYBRID_IF_N_JOBS = 1

HYBRID_NUMERICAL_WEIGHT = 0.60
HYBRID_DQ_WEIGHT = 0.40
NUMERICAL_RECALL_FLOOR = 0.55
NUMERICAL_RECALL_OBJECTIVE_WEIGHT = 1 / 3
DQ_RECALL_OBJECTIVE_WEIGHT = 2 / 3
HYBRID_REVIEW_THRESHOLD = 0.60
HYBRID_CRITICAL_THRESHOLD = 0.80

RISK_MONITOR_THRESHOLD = 0.25
RISK_REVIEW_THRESHOLD = 0.60
RISK_CRITICAL_THRESHOLD = 0.80

TRACK_B_PEER_Z_THRESHOLD = 4.5
TRACK_B_MONITOR_THRESHOLD = 0.25
TRACK_B_REVIEW_THRESHOLD = 0.50
TRACK_B_CRITICAL_THRESHOLD = 0.80
TRACK_B_PEER_FEATURES = [
    "original_cost_rs_cr","cumulative_expenditure_rs_cr","anticipated_cost_rs_cr",
    "expenditure_to_cost_pct","project_age_at_report_months",
    "planned_duration_months","progress_ratio",
]
REQUIRED_COLS = [
    "project_code","project_name","sector","state","agency_name",
    "financial_year","quarter","reporting_period_date",
    "original_cost_rs_cr","cumulative_expenditure_rs_cr",
    "anticipated_cost_rs_cr","anticipated_commissioning_date",
    "planned_duration_months","project_age_at_report_months","progress_ratio",
    "expenditure_to_cost_pct","contemporaneous_cost_overrun_pct",
    "contemporaneous_time_overrun_pct","anticipated_cost_change_pct",
    "anticipated_date_change_months","landmark_index","n_landmarks_total",
    "horizon_months",
]
OUTPUT_COLS = [
    "project_code","project_name","sector","state","agency_name",
    "financial_year","quarter","reporting_period_date","track",
    "n_quarters_on_record",
    "numerical_anomaly_score","robust_z_flag","cusum_flag","iso_forest_flag",
    "zscore_max_abs","data_quality_score","final_anomaly_score",
    "final_anomaly_flag","final_priority","risk_level",
    "track_b_anomaly_score","track_b_anomaly_flag","track_b_risk_level",
    "anomaly_type","numerical_anomaly_reason","data_quality_reason",
    "final_anomaly_reason","track_b_final_anomaly_reason",
]
MODEL_VERSION="paimana-hybrid-final-v1"
