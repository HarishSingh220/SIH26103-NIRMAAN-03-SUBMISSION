"""
Inference module - loads models and generates predictions.
"""

import joblib
import pandas as pd
from typing import Dict, Any

from . import config as cfg

# ==============================================================================
# CONSTANTS
# ==============================================================================

DEFAULT_COST_MODEL_PATH = cfg.COST_MODEL_PATH
DEFAULT_TIME_MODEL_PATH = cfg.TIME_MODEL_PATH


# ==============================================================================
# OVERRUN PREDICTOR CLASS
# ==============================================================================

class OverrunPredictor:
    """
    Loads a single model artifact and provides prediction.

    The artifact contains:
    - models['xgb']: Trained XGBoost classifier
    - models['hgb']: Trained HistGradientBoosting classifier
    - threshold: Optimized decision threshold
    - ensemble_weights: Weights for combining model predictions
    """

    def __init__(self, artifact_path: str):
        """
        Load the model artifact.

        Args:
            artifact_path: Path to the .joblib file
        """
        self.artifact_path = artifact_path
        self.artifact = joblib.load(artifact_path)

        # Cache model references
        self.models = self.artifact['models']
        self.threshold = self.artifact['threshold']
        self.weights = self.artifact['ensemble_weights']
        self.target = self.artifact['target']

    def predict(self, features_df: pd.DataFrame, aggregation: str = 'mean') -> Dict[str, Any]:
        """
        Get project-level prediction by aggregating across rows.

        Works for both single-row and multi-row inputs:
        - 1 row: returns that row's probability
        - n rows: averages all probabilities

        Args:
            features_df: DataFrame with shape (n_samples, n_features)
            aggregation: 'mean' (default) or 'max'

        Returns:
            Dict with:
                - probability: Aggregated probability
                - prediction: 0 or 1
                - threshold_used: The decision threshold
                - n_rows: Number of rows aggregated
        """
        # Get ensemble probabilities (one per row)
        xgb_probs = self.models['xgb'].predict_proba(features_df)[:, 1]
        hgb_probs = self.models['hgb'].predict_proba(features_df)[:, 1]

        probs = (
            self.weights['xgb'] * xgb_probs +
            self.weights['hgb'] * hgb_probs
        )

        # Aggregate (works for 1 row or many)
        if aggregation == 'mean':
            agg_prob = probs.mean()
        elif aggregation == 'max':
            agg_prob = probs.max()
        else:
            raise ValueError(f"Unknown aggregation: {aggregation}")

        prediction = 1 if agg_prob >= self.threshold else 0

        return {
            'probability': float(agg_prob),
            'prediction': int(prediction),
            'threshold_used': self.threshold,
            'n_rows': len(features_df)
        }


# ==============================================================================
# CONVENIENCE FUNCTIONS
# ==============================================================================

def load_cost_predictor(artifact_path: str = DEFAULT_COST_MODEL_PATH) -> OverrunPredictor:
    """Load the cost overrun predictor."""
    return OverrunPredictor(artifact_path)


def load_time_predictor(artifact_path: str = DEFAULT_TIME_MODEL_PATH) -> OverrunPredictor:
    """Load the time overrun predictor."""
    return OverrunPredictor(artifact_path)
