"""
SHAP-based Explanation Module for Overrun Predictions.

Uses SHAP (SHapley Additive exPlanations) to provide model-agnostic
feature attributions for each prediction, making the model's decisions
interpretable and actionable.

Key concepts:
- SHAP values represent each feature's contribution to the prediction
- Positive SHAP values push prediction toward "Overrun Risk"
- Negative SHAP values push prediction toward "No Overrun"
- The base value is the expected output (average prediction)
"""

import logging
from typing import Dict, List, Any, Optional

import numpy as np
import pandas as pd
import shap

logger = logging.getLogger(__name__)


# ==============================================================================
# FEATURE DESCRIPTIONS
# ==============================================================================

# Human-readable descriptions for features (used in explanations)
FEATURE_DESCRIPTIONS: Dict[str, Dict[str, Any]] = {
    # Cost model features
    'anticipated_cost_change_pct': {
        'name': 'Anticipated Cost Change',
        'description': 'Projected percentage change in total cost',
        'type': 'numeric',
        'unit': '%',
        'higher_is_risk': True,
    },
    'anticipated_cost_change_pct_delta': {
        'name': 'Cost Change Trend',
        'description': 'Change in anticipated cost between first and last report',
        'type': 'numeric',
        'unit': 'percentage points',
        'higher_is_risk': True,
    },
    'expenditure_to_cost_pct': {
        'name': 'Expenditure Ratio',
        'description': 'Percentage of budget already spent',
        'type': 'numeric',
        'unit': '%',
        'higher_is_risk': True,
    },
    'expenditure_to_cost_pct_delta': {
        'name': 'Spending Trend',
        'description': 'Change in spending rate between reports',
        'type': 'numeric',
        'unit': 'percentage points',
        'higher_is_risk': True,
    },
    'n_landmarks_total': {
        'name': 'Total Milestones',
        'description': 'Total number of project milestones',
        'type': 'numeric',
        'unit': 'milestones',
        'higher_is_risk': True,
    },
    'project_age_at_report_months': {
        'name': 'Project Age',
        'description': 'Time elapsed since project start',
        'type': 'numeric',
        'unit': 'months',
        'higher_is_risk': True,
    },
    'planned_duration_months': {
        'name': 'Planned Duration',
        'description': 'Originally planned project duration',
        'type': 'numeric',
        'unit': 'months',
        'higher_is_risk': False,  # Longer planned = less risky per month
    },
    'progress_ratio': {
        'name': 'Progress Ratio',
        'description': 'Actual progress vs planned progress',
        'type': 'numeric',
        'unit': '',
        'higher_is_risk': True,  # High ratio = behind schedule
    },
    'progress_ratio_delta': {
        'name': 'Progress Trend',
        'description': 'Change in progress ratio between reports',
        'type': 'numeric',
        'unit': '',
        'higher_is_risk': True,
    },
    'anticipated_date_change_months': {
        'name': 'Anticipated Delay',
        'description': 'Projected delay in completion date',
        'type': 'numeric',
        'unit': 'months',
        'higher_is_risk': True,
    },
    'anticipated_date_change_months_delta': {
        'name': 'Delay Trend',
        'description': 'Change in anticipated delay between reports',
        'type': 'numeric',
        'unit': 'months',
        'higher_is_risk': True,
    },
    'original_cost_rs_cr': {
        'name': 'Original Cost',
        'description': 'Original project budget',
        'type': 'numeric',
        'unit': 'Rs Cr',
        'higher_is_risk': None,  # Ambiguous - large projects have different dynamics
    },
    'cumulative_expenditure_rs_cr': {
        'name': 'Cumulative Expenditure',
        'description': 'Total amount spent so far',
        'type': 'numeric',
        'unit': 'Rs Cr',
        'higher_is_risk': None,
    },
    'horizon_months': {
        'name': 'Horizon',
        'description': 'Months remaining until completion',
        'type': 'numeric',
        'unit': 'months',
        'higher_is_risk': False,  # Less time left = more concerning
    },
    'rows_per_project': {
        'name': 'Reporting Frequency',
        'description': 'Number of reporting periods available',
        'type': 'numeric',
        'unit': 'reports',
        'higher_is_risk': None,
    },
    'sector': {
        'name': 'Sector',
        'description': 'Project sector/industry',
        'type': 'categorical',
        'unit': '',
        'higher_is_risk': None,
    },
    'state': {
        'name': 'State',
        'description': 'State where project is located',
        'type': 'categorical',
        'unit': '',
        'higher_is_risk': None,
    },
    'agency_name': {
        'name': 'Agency',
        'description': 'Implementing agency',
        'type': 'categorical',
        'unit': '',
        'higher_is_risk': None,
    },
    'financial_year': {
        'name': 'Financial Year',
        'description': 'Year of reporting period',
        'type': 'categorical',
        'unit': '',
        'higher_is_risk': None,
    },
    'quarter': {
        'name': 'Quarter',
        'description': 'Quarter of the financial year',
        'type': 'categorical',
        'unit': '',
        'higher_is_risk': None,
    },
    'landmark_index': {
        'name': 'Milestone Index',
        'description': 'Current milestone number',
        'type': 'numeric',
        'unit': '',
        'higher_is_risk': True,
    },
    # Missingness indicators
    'planned_duration_months_was_missing': {
        'name': 'Planned Duration Missing',
        'description': 'Whether planned duration was recorded',
        'type': 'binary',
        'unit': '',
        'higher_is_risk': True,
    },
    'project_age_at_report_months_was_missing': {
        'name': 'Project Age Missing',
        'description': 'Whether project age was recorded',
        'type': 'binary',
        'unit': '',
        'higher_is_risk': True,
    },
    'expenditure_to_cost_pct_delta_missing': {
        'name': 'Spending Trend Missing',
        'description': 'Whether spending trend data was available',
        'type': 'binary',
        'unit': '',
        'higher_is_risk': True,
    },
    'progress_ratio_delta_missing': {
        'name': 'Progress Trend Missing',
        'description': 'Whether progress trend data was available',
        'type': 'binary',
        'unit': '',
        'higher_is_risk': True,
    },
    'anticipated_cost_change_pct_delta_missing': {
        'name': 'Cost Trend Missing',
        'description': 'Whether cost trend data was available',
        'type': 'binary',
        'unit': '',
        'higher_is_risk': True,
    },
    'anticipated_date_change_months_delta_missing': {
        'name': 'Delay Trend Missing',
        'description': 'Whether delay trend data was available',
        'type': 'binary',
        'unit': '',
        'higher_is_risk': True,
    },
}


def get_feature_description(feature_name: str) -> Dict[str, Any]:
    """Get human-readable description for a feature."""
    if feature_name in FEATURE_DESCRIPTIONS:
        return FEATURE_DESCRIPTIONS[feature_name]

    # Default description for unknown features
    return {
        'name': feature_name.replace('_', ' ').title(),
        'description': feature_name,
        'type': 'unknown',
        'unit': '',
        'higher_is_risk': None,
    }


# ==============================================================================
# SHAP EXPLAINER CLASS
# ==============================================================================

class SHAPExplainer:
    """
    SHAP-based explanation generator for ensemble models.

    Handles both XGBoost and HistGradientBoosting models by computing
    SHAP values for each model and combining them using ensemble weights.
    """

    def __init__(self, artifact: Dict[str, Any], target: str = 'unknown'):
        """
        Initialize SHAP explainer with model artifact.

        Args:
            artifact: Model artifact containing models and metadata
            target: 'cost' or 'time' for labeling
        """
        self.artifact = artifact
        self.models = artifact['models']
        self.weights = artifact['ensemble_weights']
        self.feature_columns = artifact['feature_columns']
        self.target = target
        self.label_encoders = artifact.get('label_encoders', {})

        # Initialize SHAP explainers for each model
        self._init_explainers()

    def _init_explainers(self):
        """Initialize SHAP TreeExplainer for each model."""
        self.explainers = {}

        # XGBoost - has native SHAP support
        if 'xgb' in self.models:
            try:
                self.explainers['xgb'] = shap.TreeExplainer(self.models['xgb'])
                logger.info(f"Initialized XGBoost SHAP explainer")
            except Exception as e:
                logger.warning(f"Failed to init XGBoost explainer: {e}")

        # HistGradientBoosting - also supported by TreeExplainer
        if 'hgb' in self.models:
            try:
                self.explainers['hgb'] = shap.TreeExplainer(self.models['hgb'])
                logger.info(f"Initialized HistGradientBoosting SHAP explainer")
            except Exception as e:
                logger.warning(f"Failed to init HGB explainer: {e}")

    def explain(
        self,
        features_df: pd.DataFrame,
        n_top_features: int = 5,
        aggregation: str = 'mean'
    ) -> Dict[str, Any]:
        """
        Generate SHAP-based explanation for predictions.

        Args:
            features_df: DataFrame with feature values (shape: n_samples, n_features)
            n_top_features: Number of top contributing features to return
            aggregation: 'mean' or 'max' - how to aggregate SHAP values across rows

        Returns:
            Dict with:
                - shap_values: Array of SHAP values (n_samples, n_features)
                - base_value: Expected prediction value
                - top_features: List of top contributing features with explanations
                - feature_values: Actual values of top features
        """
        if features_df.empty:
            return self._empty_explanation()

        # Ensure columns match expected order
        features_df = features_df[self.feature_columns]

        # Compute SHAP values for each model
        shap_values_per_model = {}
        base_values_per_model = {}

        for model_name, explainer in self.explainers.items():
            try:
                result = explainer(features_df)
                shap_values_per_model[model_name] = result.values
                base_values_per_model[model_name] = result.base_values[0] if hasattr(result, 'base_values') else 0.0
            except Exception as e:
                logger.warning(f"SHAP explanation failed for {model_name}: {e}")
                shap_values_per_model[model_name] = None
                base_values_per_model[model_name] = None

        # Combine SHAP values using ensemble weights
        combined_shap = None
        combined_base = 0.0
        total_weight = 0.0

        for model_name, shap_vals in shap_values_per_model.items():
            if shap_vals is not None:
                weight = self.weights.get(model_name, 1.0)
                if combined_shap is None:
                    combined_shap = weight * shap_vals
                else:
                    combined_shap += weight * shap_vals

                if base_values_per_model[model_name] is not None:
                    combined_base += weight * base_values_per_model[model_name]
                total_weight += weight

        if combined_shap is not None and total_weight > 0:
            combined_shap /= total_weight
            combined_base /= total_weight

        # Aggregate across rows if multiple reporting periods
        if aggregation == 'mean':
            agg_shap = combined_shap.mean(axis=0) if combined_shap is not None else None
        elif aggregation == 'max':
            agg_shap = np.abs(combined_shap).max(axis=0) if combined_shap is not None else None
        else:
            agg_shap = combined_shap.mean(axis=0) if combined_shap is not None else None

        # Decode categorical features for readable output
        decoded_features = self._decode_features(features_df)

        # Get top contributing features
        top_features = self._get_top_features(agg_shap, decoded_features, n_top_features)

        return {
            'shap_values': combined_shap,
            'base_value': combined_base,
            'top_features': top_features,
            'feature_values': decoded_features.iloc[0].to_dict() if len(decoded_features) > 0 else {},
        }

    def _decode_features(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """
        Decode categorical features back to original labels.
        """
        decoded = features_df.copy()

        for col, encoder in self.label_encoders.items():
            if col in decoded.columns:
                # Get the encoded value
                encoded_val = decoded[col].iloc[0] if len(decoded) > 0 else None

                # Decode back to original label
                if encoded_val is not None and not np.isnan(encoded_val):
                    try:
                        idx = int(encoded_val)
                        if idx < len(encoder.classes_):
                            decoded[col] = encoder.classes_[idx]
                    except (ValueError, TypeError):
                        pass  # Keep original value

        return decoded

    def _get_top_features(
        self,
        shap_values: np.ndarray,
        features_df: pd.DataFrame,
        n_top: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Get top contributing features based on absolute SHAP values.

        Args:
            shap_values: Aggregated SHAP values (n_features,)
            features_df: DataFrame with feature values
            n_top: Number of top features to return

        Returns:
            List of feature explanations
        """
        if shap_values is None:
            return []

        # Get absolute SHAP values and sort
        abs_shap = np.abs(shap_values)
        sorted_indices = np.argsort(abs_shap)[::-1]  # Descending order

        explanations = []
        feature_values = features_df.iloc[0] if len(features_df) > 0 else None

        for idx in sorted_indices[:n_top]:
            feature_name = self.feature_columns[idx]
            shap_val = shap_values[idx]
            abs_val = abs_shap[idx]

            if feature_values is not None:
                feature_val = feature_values[feature_name]
            else:
                feature_val = None

            explanation = self._create_feature_explanation(
                feature_name, shap_val, abs_val, feature_val
            )

            if explanation:
                explanations.append(explanation)

        return explanations

    def _create_feature_explanation(
        self,
        feature_name: str,
        shap_value: float,
        abs_shap: float,
        feature_value: Any
    ) -> Optional[Dict[str, Any]]:
        """
        Create a human-readable explanation for a single feature.

        Args:
            feature_name: Name of the feature
            shap_value: SHAP value for this feature
            abs_shap: Absolute SHAP value
            feature_value: Actual value of the feature

        Returns:
            Dict with explanation details
        """
        desc = get_feature_description(feature_name)

        # Format the value
        formatted_value = self._format_value(feature_value, desc['unit'], desc['type'])

        # Determine direction and create explanation text
        if shap_value > 0:
            direction = 'increases'
            direction_text = 'pushing toward'
            risk_indicator = '⚠️'
        else:
            direction = 'decreases'
            direction_text = 'pushing away from'
            risk_indicator = '✓'

        # Create explanation text
        if desc['type'] == 'categorical':
            explanation_text = (
                f"{desc['name']} is '{formatted_value}' "
                f"({direction_text} overrun risk)"
            )
        elif desc['type'] == 'binary':
            if feature_value == 1:
                explanation_text = (
                    f"{desc['name']}: data was unavailable "
                    f"({direction_text} overrun risk)"
                )
            else:
                explanation_text = (
                    f"{desc['name']}: data was available "
                    f"({direction_text} overrun risk)"
                )
        else:
            if desc['higher_is_risk'] is True:
                explanation_text = (
                    f"{desc['name']} = {formatted_value} "
                    f"({direction} overrun risk by {abs_shap:.3f})"
                )
            elif desc['higher_is_risk'] is False:
                explanation_text = (
                    f"{desc['name']} = {formatted_value} "
                    f"({direction} overrun risk by {abs_shap:.3f})"
                )
            else:
                explanation_text = (
                    f"{desc['name']} = {formatted_value} "
                    f"(contribution: {shap_value:+.3f})"
                )

        return {
            'feature': feature_name,
            'display_name': desc['name'],
            'value': feature_value,
            'formatted_value': formatted_value,
            'shap_value': float(shap_value),
            'abs_shap_value': float(abs_shap),
            'explanation': explanation_text,
            'direction': direction,
            'type': desc['type'],
        }

    def _format_value(
        self,
        value: Any,
        unit: str,
        feature_type: str
    ) -> str:
        """Format a feature value for display."""
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return 'N/A'

        if feature_type == 'categorical':
            return str(value)
        elif feature_type == 'binary':
            return 'Yes' if value == 1 else 'No'
        else:
            # Numeric
            if isinstance(value, (int, np.integer)):
                if unit == 'Rs Cr':
                    return f"Rs {value:.1f} Cr"
                elif unit:
                    return f"{value} {unit}"
                else:
                    return str(value)
            else:
                # Float
                try:
                    val = float(value)
                    if abs(val) >= 100:
                        formatted = f"{val:.1f}"
                    elif abs(val) >= 1:
                        formatted = f"{val:.2f}"
                    else:
                        formatted = f"{val:.3f}"

                    if unit:
                        return f"{formatted} {unit}"
                    return formatted
                except (ValueError, TypeError):
                    return str(value)

    def _empty_explanation(self) -> Dict[str, Any]:
        """Return empty explanation structure."""
        return {
            'shap_values': None,
            'base_value': None,
            'top_features': [],
            'feature_values': {},
        }


def compute_shap_reasoning(
    target: str,
    metadata: Dict[str, Any],
    reporting_periods: List[Dict[str, Any]],
    probability: float,
    prediction: str,
    threshold: float,
    artifact: Dict[str, Any],
    features_df: pd.DataFrame,
    n_top_features: int = 5
) -> str:
    """
    Compute SHAP-based reasoning for an overrun prediction.

    This is the main entry point called from the coordinator.

    Args:
        target: 'cost' or 'time'
        metadata: Project metadata dict
        reporting_periods: List of reporting period dicts
        probability: Model predicted probability
        prediction: 'Overrun Risk' or 'No Overrun'
        threshold: Model decision threshold
        artifact: Model artifact containing trained models
        features_df: Preprocessed features DataFrame
        n_top_features: Number of top features to include in explanation

    Returns:
        Human-readable explanation string
    """
    if prediction != "Overrun Risk":
        return ""

    try:
        # Create explainer
        explainer = SHAPExplainer(artifact, target=target)

        # Get SHAP explanations
        explanation = explainer.explain(
            features_df,
            n_top_features=n_top_features,
            aggregation='mean'
        )

        if not explanation['top_features']:
            return "Model identifies project as high risk based on multiple factors"

        # Build human-readable explanation
        parts = []

        # Add a summary line
        prob_exceed = (probability - threshold) / threshold * 100
        if prob_exceed > 50:
            confidence = "significantly"
        elif prob_exceed > 20:
            confidence = "moderately"
        else:
            confidence = "slightly"

        parts.append(
            f"Project is {confidence} above the {threshold:.1%} risk threshold "
            f"(predicted probability: {probability:.1%}). "
            f"Key risk factors:"
        )

        # Add top contributing features
        for i, feat in enumerate(explanation['top_features'], 1):
            parts.append(f"  {i}. {feat['explanation']}")

        return " ".join(parts)

    except Exception as e:
        logger.warning(f"SHAP reasoning failed: {e}")
        # Fallback
        return (
            "Model identifies project as high risk. "
            "Detailed explanation unavailable - please review project metrics."
        )
