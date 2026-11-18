"""Evaluation artifact and model-performance response models."""

from typing import Any, Literal

from pydantic import BaseModel, Field

DatasetCategory = Literal["train", "student_test"]


class PerformanceModel(BaseModel):
	model_name: str
	accuracy: float | None = Field(default=None, ge=0, le=1)
	precision: float | None = Field(default=None, ge=0, le=1)
	recall: float | None = Field(default=None, ge=0, le=1)
	f1_score: float | None = Field(default=None, ge=0, le=1)
	confusion_matrix: list[list[int]] | None = None
	cross_validation: dict[str, float | None] = Field(default_factory=dict)
	hyperparameter_results: list[dict[str, Any]] = Field(default_factory=list)
	route_metrics: dict[str, float | int | None] = Field(default_factory=dict)


class PerformanceArtifact(BaseModel):
	dataset_category: DatasetCategory
	student_test_evaluation_permitted: bool = False
	models: list[PerformanceModel]


class PerformanceResponse(BaseModel):
	status: Literal["available", "empty", "blocked"]
	dataset_category: DatasetCategory | None = None
	models: list[PerformanceModel] = Field(default_factory=list)
	message: str