"""Prediction API request and response models."""

from typing import Literal

from pydantic import BaseModel, Field

from backend.schemas.performance import DatasetCategory


class PredictionRequest(BaseModel):
	dataset_category: DatasetCategory
	model_id: str | None = None


class PredictedRoute(BaseModel):
	arc_ids: list[int]
	coordinates: list[tuple[float, float]]
	arc_count: int = Field(ge=0)
	node_count: int = Field(ge=0)
	connected: bool
	valid: bool
	distance_m: float | None = Field(default=None, ge=0)


class PredictionResponse(BaseModel):
	status: Literal["completed"]
	trip_id: str
	dataset_category: DatasetCategory
	model_id: str | None = None
	gps_observation_count: int = Field(ge=0)
	processing_time_s: float = Field(ge=0)
	route: PredictedRoute