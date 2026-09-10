"""
Pydantic schemas for FastAPI request/response validation.
"""

from typing import List, Optional, Literal, Union
from pydantic import BaseModel, Field


# ==============================================================================
# REQUEST SCHEMAS
# ==============================================================================

class ReportingPeriod(BaseModel):
    """
    Per-landmark data. Sent once per reporting period (landmark).
    Only landmark_index <= 3 is used by the model.
    """
    landmark_index: int = Field(..., ge=1, description="Milestone number (1, 2, 3, ...)")
    quarter: str = Field(..., pattern=r"^Q[1-4]$", description="Quarter: Q1, Q2, Q3, or Q4")
    financial_year: str = Field(..., description="Financial year (e.g., '2017-18')")
    horizon_months: int = Field(..., ge=0, description="Months to next landmark")

    # Optional per-period numeric fields (can be None if not yet reported)
    cumulative_expenditure_rs_cr: Optional[float] = Field(None, ge=0, description="Cumulative expenditure in Rs. Crores")
    expenditure_to_cost_pct: Optional[float] = Field(None, ge=0, description="(Expenditure / Original Cost) × 100")
    project_age_at_report_months: Optional[float] = Field(None, ge=0, description="Months since project start")
    planned_duration_months: Optional[float] = Field(None, ge=0, description="Total planned duration in months")
    progress_ratio: Optional[float] = Field(None, ge=0, description="Project Age / Planned Duration")
    anticipated_cost_rs_cr: Optional[float] = Field(None, ge=0, description="Latest anticipated cost in Rs. Crores")
    anticipated_cost_change_pct: Optional[float] = Field(None, description="Anticipated cost change percentage")
    anticipated_date_change_months: Optional[float] = Field(None, description="Anticipated delay in months")


class ProjectMetadata(BaseModel):
    """
    Project-level metadata. Sent once per project (constant across all landmarks).
    """
    project_code: str = Field(..., min_length=1, max_length=50, description="Unique project identifier")
    sector: str = Field(..., min_length=1, description="Sector (e.g., ATOMIC ENERGY, ROAD TRANSPORT AND HIGHWAYS)")
    state: str = Field(..., min_length=1, description="State name (e.g., TAMIL NADU)")
    agency_name: str = Field(..., min_length=1, description="Executing agency name")
    n_landmarks_total: int = Field(..., ge=1, description="Total number of planned milestones")
    original_cost_rs_cr: float = Field(..., gt=0, description="Originally sanctioned cost in Rs. Crores")


class ProjectInput(BaseModel):
    """
    Single project input: metadata + reporting periods.
    """
    metadata: ProjectMetadata
    reporting_periods: List[ReportingPeriod] = Field(..., min_length=1, description="At least 1 reporting period")


class BatchPredictionRequest(BaseModel):
    """
    Batch prediction request. Body shape: {"projects": [<project>, ...]}.
    """
    projects: Union[List[ProjectInput], ProjectInput] = Field(
        ...,
        description="List of projects to predict, or a single project object"
    )


# ==============================================================================
# FLEXIBLE REQUEST UNION
# ==============================================================================

# The endpoint accepts four body shapes:
#   1. {"projects": [...]}  — wrapped list
#   2. {"projects": {...}}  — wrapped single project
#   3. [...]                — bare list
#   4. {...}                — bare single project
#
# FastAPI generates a request body schema from the parameter annotation.
# We use a single Pydantic model that allows both `projects` key and direct
# project fields. The bare-list case is handled by accepting `Body(...)` in
# api.py and normalizing it before validation.


# ==============================================================================
# RESPONSE SCHEMAS
# ==============================================================================

class OverrunResult(BaseModel):
    """Result for either cost or time overrun prediction."""
    probability: float = Field(..., ge=0, le=1, description="Probability of overrun (0.0 to 1.0)")
    prediction: Literal["No Overrun", "Overrun Risk"] = Field(..., description="Binary prediction")
    reason: Optional[str] = Field(None, description="Human-readable reason for overrun prediction (only present when prediction is 'Overrun Risk')")


class ProjectPredictionResponse(BaseModel):
    """Prediction result for a single project."""
    project_code: str
    cost_overrun: OverrunResult
    time_overrun: OverrunResult


class BatchPredictionResponse(BaseModel):
    """Response for batch prediction."""
    predictions: List[ProjectPredictionResponse]
