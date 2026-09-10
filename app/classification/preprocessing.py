"""
Preprocessing module - mirrors the training pipeline EXACTLY.
This ensures inference-time behavior matches training-time behavior.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any


# ==============================================================================
# CONSTANTS (must match training pipeline)
# ==============================================================================

MAX_LANDMARK_INDEX = 3

INDICATOR_COLS = ['planned_duration_months', 'project_age_at_report_months']

TRAJECTORY_NUMERIC_COLS = [
    'expenditure_to_cost_pct',
    'progress_ratio',
    'anticipated_cost_change_pct',
    'anticipated_date_change_months'
]

# Project-level fields (constant across all landmarks for a project)
PROJECT_LEVEL_COLS = ['sector', 'state', 'agency_name', 'n_landmarks_total', 'original_cost_rs_cr']

# Landmark-level fields (vary per reporting period)
LANDMARK_LEVEL_COLS = [
    'landmark_index', 'quarter', 'horizon_months',
    'cumulative_expenditure_rs_cr', 'expenditure_to_cost_pct',
    'project_age_at_report_months', 'planned_duration_months',
    'progress_ratio', 'anticipated_cost_rs_cr',
    'anticipated_cost_change_pct', 'anticipated_date_change_months'
]


# ==============================================================================
# MAIN PREPROCESSING FUNCTION
# ==============================================================================

def preprocess_project(
    metadata: Dict[str, Any],
    reporting_periods: List[Dict[str, Any]],
    artifact: Dict[str, Any]
) -> pd.DataFrame:
    """
    Main entry point: Convert user input to model-ready 28-feature DataFrame.

    This function EXACTLY mirrors the training pipeline steps 2-8:
    1. Build DataFrame from reporting periods
    2. Filter to landmark_index <= 3
    3. Add project-level metadata
    4. Add missingness indicators
    5. Compute trajectory deltas
    6. Apply label encoding, clipping, imputation (using artifact's fitted params)

    Args:
        metadata: ProjectMetadata dict (sector, state, agency_name, etc.)
        reporting_periods: List of ReportingPeriod dicts
        artifact: Loaded .joblib artifact containing fitted preprocessors

    Returns:
        pd.DataFrame with shape (n_filtered_rows, n_features) ready for model.predict()
    """
    # Step 1: Build DataFrame from reporting periods
    df = pd.DataFrame(reporting_periods)

    # Step 2: Filter to early landmarks only (landmark_index <= MAX_LANDMARK_INDEX)
    df = filter_early_landmarks(df)

    if len(df) == 0:
        raise ValueError(
            f"No reporting periods with landmark_index <= {MAX_LANDMARK_INDEX}. "
            f"Model only uses data from early milestones."
        )

    # Step 3: Add project-level metadata to each row
    df = add_project_metadata(df, metadata)

    # Step 4: Add missingness indicators (before imputation)
    df = add_missingness_indicators(df)

    # Step 5: Compute trajectory features (deltas across landmarks)
    df = compute_trajectory_features(df)

    # Step 6: Apply preprocessing (label encoding, clipping, imputation)
    # Uses the artifact's fitted params to ensure consistency with training
    df = apply_preprocessor(df, artifact)

    # Step 7: Select only the feature columns the model was trained on
    feature_cols = artifact['feature_columns']
    df = df[feature_cols]

    return df


def filter_early_landmarks(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only rows where landmark_index <= MAX_LANDMARK_INDEX."""
    return df[df['landmark_index'] <= MAX_LANDMARK_INDEX].copy()


def add_project_metadata(df: pd.DataFrame, metadata: Dict[str, Any]) -> pd.DataFrame:
    """
    Add project-level fields to each row.
    Also adds rows_per_project (count of reporting periods).
    """
    for col in PROJECT_LEVEL_COLS:
        df[col] = metadata.get(col)

    # Add rows_per_project for sample weight calculation
    df['rows_per_project'] = len(df)

    return df


def add_missingness_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add _was_missing columns for high-missing-rate columns.
    Matches training pipeline Step 5.
    """
    for col in INDICATOR_COLS:
        if col in df.columns:
            df[f"{col}_was_missing"] = df[col].isna().astype(int)

    return df


def compute_trajectory_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute delta features (last - first) for trajectory columns.
    Matches training pipeline Step 6 (improvement #4).

    IMPORTANT: This assumes data is sorted by landmark_index.
    The "first" value is the earliest landmark, "last" is the latest.
    """
    # Sort by landmark_index to ensure correct ordering
    df = df.sort_values('landmark_index').reset_index(drop=True)

    # Check if single-row project (all rows have same rows_per_project == 1)
    is_single_row = len(df) == 1 or (df['rows_per_project'] == 1).all()

    for col in TRAJECTORY_NUMERIC_COLS:
        if col not in df.columns:
            continue

        # For single-row projects: delta = 0 (no trajectory to measure)
        if is_single_row:
            delta = 0.0
        else:
            # Get first and last valid values
            first_val = df[col].iloc[0]  # Earliest landmark
            last_val = df[col].iloc[-1]  # Latest landmark (within early window)

            # Compute delta (last - first)
            # Handle None/NaN values
            if first_val is None or last_val is None or pd.isna(first_val) or pd.isna(last_val):
                delta = None  # Will be handled below
            else:
                delta = last_val - first_val

        # Add delta column (same value for all rows of this project)
        delta_col = f'{col}_delta'
        was_missing_col = f'{col}_delta_missing'

        if delta is None or pd.isna(delta):
            df[delta_col] = 0.0
            df[was_missing_col] = 1
        else:
            df[delta_col] = delta
            df[was_missing_col] = 0

        # Single-row projects: was_missing = 0
        if is_single_row:
            df[was_missing_col] = 0

    return df


def apply_preprocessor(df: pd.DataFrame, artifact: Dict[str, Any]) -> pd.DataFrame:
    """
    Apply the same preprocessing as training pipeline:
    1. Label encode categorical columns
    2. Clip anticipated_cost_change_pct to 1st/99th percentile
    3. Median-impute missing numeric values

    Uses the artifact's fitted parameters to ensure consistency.
    """
    df = df.copy()
    label_encoders = artifact['label_encoders']
    clip_bounds = artifact['clip_bounds']
    numeric_medians = artifact['numeric_medians']

    # ----- Step 0: Convert object columns to numeric where possible -----
    # This handles None values that make columns 'object' dtype
    for col in df.columns:
        if df[col].dtype == 'object' and col not in ['sector', 'state', 'agency_name', 'quarter', 'financial_year']:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # ----- Step 1: Label Encode Categoricals -----
    categorical_cols = ['sector', 'state', 'agency_name', 'quarter', 'financial_year']

    for col in categorical_cols:
        if col not in label_encoders:
            continue

        encoder = label_encoders[col]
        values = df[col].fillna('MISSING').astype(str)

        # Handle unseen categories
        unseen_mask = ~values.isin(encoder.classes_)
        if unseen_mask.any():
            # Map unseen to the most common category recorded during training.
            categorical_modes = artifact.get("categorical_modes", {})
            most_common = categorical_modes.get(col)
            if most_common is None:
                raise ValueError(
                    f"Model artifact is missing categorical_modes[{col!r}]; "
                    "retrain/export the classification artifact with the fixed training pipeline."
                )
            values = values.replace(dict(zip(values[unseen_mask], [most_common] * unseen_mask.sum())))

        df[col] = encoder.transform(values)

    # ----- Step 2: Clip anticipated_cost_change_pct -----
    accp_col = 'anticipated_cost_change_pct'
    if accp_col in df.columns and accp_col in clip_bounds:
        bounds = clip_bounds[accp_col]
        p1, p99 = bounds['p1'], bounds['p99']
        median_val = bounds['median']

        # Replace inf/-inf with NaN first
        df[accp_col] = df[accp_col].replace([np.inf, -np.inf], np.nan)

        # Clip to percentile range
        df[accp_col] = df[accp_col].clip(lower=p1, upper=p99)

        # Median impute remaining NaN
        df[accp_col] = df[accp_col].fillna(median_val)

    # ----- Step 3: Median-impute all other numeric columns -----
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # rows_per_project is metadata, not a model feature.
    numeric_cols = [c for c in numeric_cols if c != 'rows_per_project']

    for col in numeric_cols:
        # Replace inf/-inf with NaN
        df[col] = df[col].replace([np.inf, -np.inf], np.nan)

        # Use median from artifact if available, else from data
        if col in numeric_medians:
            median_val = numeric_medians[col]
        else:
            median_val = df[col].median()

        df[col] = df[col].fillna(median_val)

    return df


# ==============================================================================
# UTILITY FUNCTIONS
# ==============================================================================

def validate_reporting_periods(reporting_periods: List[Dict]) -> List[str]:
    """
    Validate reporting periods and return list of warnings.
    """
    warnings = []

    # Check for landmark_index duplicates
    indices = [rp.get('landmark_index') for rp in reporting_periods]
    if len(indices) != len(set(indices)):
        warnings.append("Duplicate landmark_index values found. Only first occurrence will be used.")

    # Check for gaps in landmark_index
    sorted_indices = sorted([i for i in indices if i is not None])
    if sorted_indices:
        expected = set(range(1, max(sorted_indices) + 1))
        actual = set(sorted_indices)
        missing = expected - actual
        if missing:
            warnings.append(f"Missing landmark indices: {sorted(missing)}. Deltas may be incomplete.")

    return warnings


def compute_overrun_reason(
    target: str,
    metadata: Dict[str, Any],
    reporting_periods: List[Dict[str, Any]],
    probability: float,
    prediction: str,
    threshold: float
) -> str:
    """
    Compute a human-readable reason for the overrun prediction.

    Based on the input features that are most important for each model,
    derived from the trained model's feature importances.

    Args:
        target: 'cost' or 'time'
        metadata: Project metadata dict
        reporting_periods: List of reporting period dicts
        probability: Model predicted probability
        prediction: 'Overrun Risk' or 'No Overrun'
        threshold: Model decision threshold

    Returns:
        Human-readable reason string, or empty string if prediction is 'No Overrun'
    """
    if prediction != "Overrun Risk":
        return ""

    reasons = []

    if target == "cost":
        # Feature importance order (from model training):
        # 1. anticipated_cost_change_pct_delta_first_to_last (0.095)
        # 2. anticipated_cost_change_pct (0.061)
        # 3. expenditure_to_cost_pct (0.056)
        # 4. n_landmarks_total (0.061)
        # 5. sector (0.054)
        # 6. project_age_at_report_months (0.047)

        # Check anticipated_cost_change_pct across reporting periods
        for rp in reporting_periods:
            accp = rp.get("anticipated_cost_change_pct")
            if accp is not None:
                if accp > 15:
                    reasons.append(
                        f"significant anticipated cost increase of {accp:.1f}%"
                    )
                elif accp > 5:
                    reasons.append(
                        f"moderate anticipated cost increase of {accp:.1f}%"
                    )

        # Check expenditure_to_cost_pct
        for rp in reporting_periods:
            etc = rp.get("expenditure_to_cost_pct")
            if etc is not None and etc > 0:
                if etc > 60:
                    reasons.append(
                        f"high expenditure-to-cost ratio ({etc:.1f}% spent)"
                    )
                elif etc > 35:
                    reasons.append(
                        f"elevated expenditure-to-cost ratio ({etc:.1f}% spent)"
                    )

        # Check overall project age vs planned duration
        proj_age = metadata.get("project_age_at_report_months", 0)
        planned_dur = metadata.get("planned_duration_months", 0)
        if planned_dur > 0 and proj_age / planned_dur > 0.8:
            ratio = proj_age / planned_dur
            reasons.append(
                f"project age ({proj_age:.0f} months) is {ratio:.0%} of planned duration ({planned_dur:.0f} months)"
            )

        if not reasons:
            reasons.append(
                "model identifies risk factors from early project indicators"
            )

    elif target == "time":
        # Feature importance order (from model training):
        # 1. rows_per_project (0.152) - note: metadata, not a direct reason
        # 2. n_landmarks_total (0.095)
        # 3. financial_year (0.081)
        # 4. progress_ratio (0.078) - KEY: how far along the project is
        # 5. anticipated_date_change_months (0.052) - KEY: anticipated delay
        # 6. horizon_months (0.049)
        # 7. planned_duration_months (0.032)
        # 8. state (0.028)
        # 9. agency_name (0.025)

        # Check anticipated_date_change_months - most important time indicator
        for rp in reporting_periods:
            adc = rp.get("anticipated_date_change_months")
            if adc is not None:
                if adc > 6:
                    reasons.append(
                        f"significant anticipated delay of {adc:.1f} months"
                    )
                elif adc > 3:
                    reasons.append(
                        f"moderate anticipated delay of {adc:.1f} months"
                    )

        # Check progress_ratio - how much of planned duration is complete
        for rp in reporting_periods:
            pr = rp.get("progress_ratio")
            if pr is not None and pr > 0:
                if pr > 1.0:
                    reasons.append(
                        f"project behind schedule (progress ratio {pr:.2f})"
                    )
                elif pr > 0.7:
                    reasons.append(
                        f"progress ratio {pr:.2f} indicates approaching deadline"
                    )

        # Check planned_duration_months vs project age
        proj_age = metadata.get("project_age_at_report_months", 0)
        planned_dur = metadata.get("planned_duration_months", 0)
        if planned_dur > 0 and proj_age / planned_dur > 0.7:
            ratio = proj_age / planned_dur
            reasons.append(
                f"project age ({proj_age:.0f} months) is {ratio:.0%} of planned duration ({planned_dur:.0f} months)"
            )

        if not reasons:
            reasons.append(
                "model identifies risk factors from early project tracking data"
            )

    # Return top 3 reasons, joined by "; "
    return "; ".join(reasons[:3])


def get_feature_importance_summary(artifact: Dict[str, Any]) -> pd.DataFrame:
    """
    Get feature importance from the XGBoost model.
    Returns DataFrame sorted by importance.
    """
    importances = pd.DataFrame({
        'feature': artifact['feature_columns'],
        'importance': artifact['models']['xgb'].feature_importances_
    }).sort_values('importance', ascending=False)

    return importances
