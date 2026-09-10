"""Shared request schemas for single-row and multi-project prediction endpoints."""
from __future__ import annotations

from typing import Annotated, Any

from pydantic import AfterValidator, BaseModel, Field, model_validator

_ROW_SCALAR_TYPES = (str, int, float, bool, type(None))


def validate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Validate that every prediction row is a non-empty flat scalar mapping."""
    for i, row in enumerate(rows):
        if not row:
            raise ValueError(f"Row {i} is empty; every row must have at least one field.")
        for key, value in row.items():
            if not isinstance(value, _ROW_SCALAR_TYPES):
                raise ValueError(
                    f"Row {i}, field {key!r}: expected a string, number, boolean, or "
                    f"null, got {type(value).__name__}. Nested objects/arrays are not supported."
                )
    return rows


SingleRowsField = Annotated[
    list[dict[str, Any]],
    Field(
        min_length=1,
        max_length=1,
        description="Exactly one raw project row. Use the batch schema for multiple projects or quarters.",
    ),
    AfterValidator(validate_rows),
]


BatchRows = Annotated[
    list[dict[str, Any]],
    Field(
        min_length=1,
        max_length=10000,
        description="One or more reporting rows belonging to the project identified by project_code.",
    ),
    AfterValidator(validate_rows),
]


class BatchProjectInput(BaseModel):
    """One project plus all reporting-quarter rows that belong to it."""

    project_code: str = Field(..., min_length=1, max_length=100, description="Stable project identifier.")
    rows: BatchRows

    @model_validator(mode="after")
    def validate_project_code(self) -> "BatchProjectInput":
        for index, row in enumerate(self.rows):
            row_code = row.get("project_code")
            if row_code is not None and str(row_code).strip() != self.project_code.strip():
                raise ValueError(
                    f"Project {self.project_code!r}, row {index}: project_code must match the "
                    "batch project's project_code when supplied."
                )
        return self


class BatchPredictionRequest(BaseModel):
    """Explicit multi-project batch request.

    Example body:
        {"projects": [{"project_code": "P001", "rows": [{...}, {...}]}, ...]}
    """

    projects: list[BatchProjectInput] = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="Multiple projects; each project may contain one or more quarterly history rows.",
    )

    @model_validator(mode="after")
    def validate_total_rows(self) -> "BatchPredictionRequest":
        total_rows = sum(len(project.rows) for project in self.projects)
        if total_rows > 10000:
            raise ValueError(
                f"Batch contains {total_rows} rows; the maximum supported total is 10000 rows."
            )
        return self


def flatten_batch_projects(body: BatchPredictionRequest) -> list[dict[str, Any]]:
    """Flatten the explicit batch schema into the existing model row format."""
    flattened: list[dict[str, Any]] = []
    for project in body.projects:
        code = project.project_code.strip()
        for row in project.rows:
            item = dict(row)
            item["project_code"] = code
            flattened.append(item)
    return flattened

