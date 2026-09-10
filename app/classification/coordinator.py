"""
Model Coordinator - orchestrates preprocessing and inference.
Holds loaded models in memory and handles the full prediction pipeline.
"""

from typing import Dict, List, Any, Optional
import logging

from . import config as cfg
from . import preprocessing
from .inference import load_cost_predictor, load_time_predictor
from .explainers import compute_shap_reasoning


# ==============================================================================
# LOGGING
# ==============================================================================

logger = logging.getLogger(__name__)


# ==============================================================================
# COORDINATOR CLASS
# ==============================================================================

class ModelCoordinator:
    """
    Singleton-ish class that holds both model predictors and orchestrates
    the full prediction pipeline: preprocessing → inference → aggregation.

    Usage:
        coordinator = ModelCoordinator()
        result = coordinator.predict_project(metadata, reporting_periods)
    """

    def __init__(
        self,
        cost_model_path: str = cfg.COST_MODEL_PATH,
        time_model_path: str = cfg.TIME_MODEL_PATH
    ):
        """
        Initialize and load both models.

        Args:
            cost_model_path: Path to cost overrun model artifact
            time_model_path: Path to time overrun model artifact
        """
        logger.info("Loading models...")
        self.cost_predictor = load_cost_predictor(cost_model_path)
        self.time_predictor = load_time_predictor(time_model_path)
        logger.info("Models loaded successfully")

        # Cache artifact references for preprocessing
        self.cost_artifact = self.cost_predictor.artifact
        self.time_artifact = self.time_predictor.artifact

    def predict_project(
        self,
        metadata: Dict[str, Any],
        reporting_periods: List[Dict[str, Any]],
        targets: List[str] = ['cost', 'time']
    ) -> Dict[str, Any]:
        """
        Predict overrun for a single project.

        Args:
            metadata: Project metadata dict (project_code, sector, state, etc.)
            reporting_periods: List of reporting period dicts
            targets: List of targets to predict ('cost', 'time', or both)

        Returns:
            Dict with predictions for each target
        """
        # Validate and get warnings
        warnings = preprocessing.validate_reporting_periods(reporting_periods)

        # Preprocess for cost model (uses cost artifact for fitted preprocessors)
        try:
            cost_features = preprocessing.preprocess_project(
                metadata, reporting_periods, self.cost_artifact
            )
        except ValueError as e:
            raise ValueError(f"Failed to preprocess for cost prediction: {e}")

        # Preprocess for time model (uses time artifact for fitted preprocessors)
        try:
            time_features = preprocessing.preprocess_project(
                metadata, reporting_periods, self.time_artifact
            )
        except ValueError as e:
            raise ValueError(f"Failed to preprocess for time prediction: {e}")

        result = {
            'project_code': metadata['project_code'],
            'rows_used': len(cost_features),  # Same for both
            'warnings': warnings
        }

        # Cost overrun prediction
        if 'cost' in targets:
            cost_pred = self.cost_predictor.predict(cost_features, aggregation='mean')
            cost_prediction_label = 'Overrun Risk' if cost_pred['prediction'] == 1 else 'No Overrun'

            # Use SHAP-based reasoning for "Overrun Risk" predictions
            if cost_prediction_label == 'Overrun Risk':
                cost_reason = compute_shap_reasoning(
                    target='cost',
                    metadata=metadata,
                    reporting_periods=reporting_periods,
                    probability=cost_pred['probability'],
                    prediction=cost_prediction_label,
                    threshold=cost_pred['threshold_used'],
                    artifact=self.cost_artifact,
                    features_df=cost_features,
                    n_top_features=5
                )
            else:
                cost_reason = ""

            result['cost_overrun'] = {
                'probability': round(cost_pred['probability'], 4),
                'prediction': cost_prediction_label,
                'reason': cost_reason
            }

        # Time overrun prediction
        if 'time' in targets:
            time_pred = self.time_predictor.predict(time_features, aggregation='mean')
            time_prediction_label = 'Overrun Risk' if time_pred['prediction'] == 1 else 'No Overrun'

            # Use SHAP-based reasoning for "Overrun Risk" predictions
            if time_prediction_label == 'Overrun Risk':
                time_reason = compute_shap_reasoning(
                    target='time',
                    metadata=metadata,
                    reporting_periods=reporting_periods,
                    probability=time_pred['probability'],
                    prediction=time_prediction_label,
                    threshold=time_pred['threshold_used'],
                    artifact=self.time_artifact,
                    features_df=time_features,
                    n_top_features=5
                )
            else:
                time_reason = ""

            result['time_overrun'] = {
                'probability': round(time_pred['probability'], 4),
                'prediction': time_prediction_label,
                'reason': time_reason
            }

        return result

    def predict_batch(
        self,
        projects: List[Dict[str, Any]],
        targets: List[str] = ['cost', 'time']
    ) -> List[Dict[str, Any]]:
        """
        Predict overrun for multiple projects.

        Args:
            projects: List of project dicts, each containing:
                - metadata: ProjectMetadata dict
                - reporting_periods: List of ReportingPeriod dicts
            targets: List of targets to predict

        Returns:
            List of prediction results
        """
        results = []

        for project_req in projects:
            try:
                result = self.predict_project(
                    metadata=project_req['metadata'],
                    reporting_periods=project_req['reporting_periods'],
                    targets=targets
                )
                results.append(result)
            except ValueError as e:
                # Log error but continue with other projects
                logger.warning(
                    f"Failed to predict project {project_req.get('metadata', {}).get('project_code', 'UNKNOWN')}: {e}"
                )
                results.append({
                    'project_code': project_req.get('metadata', {}).get('project_code', 'UNKNOWN'),
                    'error': str(e)
                })

        return results

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about both deployed models.

        Returns:
            Dict with info for both cost and time models
        """
        return {
            'cost_model': {
                'target': self.cost_predictor.target,
                'max_landmark_index': self.cost_artifact['max_landmark_index'],
                'threshold': self.cost_artifact['threshold'],
                'ensemble_weights': self.cost_artifact['ensemble_weights'],
                'training_metrics': self.cost_artifact['train_metrics'],
                'train_metrics': self.cost_artifact['train_metrics'],
                'n_features': len(self.cost_artifact['feature_columns'])
            },
            'time_model': {
                'target': self.time_predictor.target,
                'max_landmark_index': self.time_artifact['max_landmark_index'],
                'threshold': self.time_artifact['threshold'],
                'ensemble_weights': self.time_artifact['ensemble_weights'],
                'training_metrics': self.time_artifact['train_metrics'],
                'train_metrics': self.time_artifact['train_metrics'],
                'n_features': len(self.time_artifact['feature_columns'])
            }
        }


# ==============================================================================
# SINGLETON INSTANCE
# ==============================================================================

# Lazy-loaded coordinator instance
_coordinator: Optional[ModelCoordinator] = None


def get_coordinator(
    cost_model_path: str = cfg.COST_MODEL_PATH,
    time_model_path: str = cfg.TIME_MODEL_PATH
) -> ModelCoordinator:
    """
    Get or create the global coordinator instance.
    Models are loaded once and reused across requests.
    """
    global _coordinator

    if _coordinator is None:
        _coordinator = ModelCoordinator(cost_model_path, time_model_path)

    return _coordinator


def reset_coordinator():
    """Reset the coordinator (for testing or reloading models)."""
    global _coordinator
    _coordinator = None
