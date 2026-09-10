"""
Pulls a flat, LLM-friendly set of facts out of a prediction-endpoint output
dict, tolerant of the couple of shapes those endpoints actually return.

Handles:
  - The combined pipeline shape returned by `POST /classification/predict`
    and `POST /gateway/predict` (`app.classification.pipeline`):
    {"project_code", "classification": {"cost_overrun", "time_overrun"},
     "overrun": {...}, "anomaly": {...}, "warnings": [...]}
  - The flatter shape from `app.classification.schemas.ProjectPredictionResponse`
    (top-level "cost_overrun" / "time_overrun", no "anomaly" block).
  - A per-project error dict (`{"project_code", "error"}`), passed through
    unsummarized rather than sent to the LLM.

Never raises on a missing/odd field -- every accessor degrades to `None`
so a partial or future-shaped payload still produces a best-effort summary
instead of a 500.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class ProjectFacts:
    project_code: Optional[str] = None
    error: Optional[str] = None
    warnings: list[str] = field(default_factory=list)

    cost_overrun_prediction: Optional[str] = None
    cost_overrun_probability: Optional[float] = None
    cost_overrun_reason: Optional[str] = None
    cost_overrun_pct: Optional[float] = None
    predicted_cost_increase_rs_cr: Optional[float] = None
    predicted_final_cost_rs_cr: Optional[float] = None

    time_overrun_prediction: Optional[str] = None
    time_overrun_probability: Optional[float] = None
    time_overrun_reason: Optional[str] = None
    predicted_delay_months: Optional[float] = None
    predicted_final_commissioning_date: Optional[str] = None

    anomaly_risk_level: Optional[str] = None
    anomaly_reason: Optional[str] = None
    anomaly_score: Optional[float] = None

    def has_any_signal(self) -> bool:
        return any(
            v is not None
            for v in (
                self.cost_overrun_prediction,
                self.time_overrun_prediction,
                self.anomaly_risk_level,
            )
        )


def _get(d: Any, *keys: str) -> Any:
    """Nested-dict getter that returns None instead of raising on any
    missing key or non-dict intermediate value."""
    cur = d
    for key in keys:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    return cur


def extract_project_facts(prediction: dict[str, Any]) -> ProjectFacts:
    if not isinstance(prediction, dict):
        return ProjectFacts(error="Prediction payload was not a JSON object.")

    project_code = prediction.get("project_code")

    if "error" in prediction:
        return ProjectFacts(project_code=project_code, error=str(prediction["error"]))

    # "classification" block (combined pipeline shape) takes precedence;
    # fall back to top-level cost_overrun/time_overrun (flat shape).
    classification_block = prediction.get("classification")
    if not isinstance(classification_block, dict):
        classification_block = prediction

    overrun_block = prediction.get("overrun") if isinstance(prediction.get("overrun"), dict) else {}
    anomaly_block = prediction.get("anomaly") if isinstance(prediction.get("anomaly"), dict) else {}

    facts = ProjectFacts(
        project_code=project_code,
        warnings=list(prediction.get("warnings") or []),
        cost_overrun_prediction=_get(classification_block, "cost_overrun", "prediction"),
        cost_overrun_probability=_get(classification_block, "cost_overrun", "probability"),
        cost_overrun_reason=_get(classification_block, "cost_overrun", "reason"),
        cost_overrun_pct=_get(overrun_block, "cost_overrun", "predicted_overrun_pct"),
        predicted_cost_increase_rs_cr=_get(overrun_block, "cost_overrun", "predicted_cost_increase_rs_cr"),
        predicted_final_cost_rs_cr=_get(overrun_block, "cost_overrun", "predicted_final_cost_rs_cr"),
        time_overrun_prediction=_get(classification_block, "time_overrun", "prediction"),
        time_overrun_probability=_get(classification_block, "time_overrun", "probability"),
        time_overrun_reason=_get(classification_block, "time_overrun", "reason"),
        predicted_delay_months=_get(overrun_block, "time_overrun", "predicted_delay_months"),
        predicted_final_commissioning_date=_get(overrun_block, "time_overrun", "predicted_final_commissioning_date"),
        anomaly_risk_level=anomaly_block.get("risk_level"),
        anomaly_reason=anomaly_block.get("anomaly_reason"),
        anomaly_score=anomaly_block.get("anomaly_score"),
    )
    return facts
