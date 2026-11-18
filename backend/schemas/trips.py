"""Pydantic schemas for dataset trip endpoints."""

from typing import Literal

from pydantic import BaseModel

DatasetCategory = Literal["train", "student_test"]


class RequiredFiles(BaseModel):
	track: bool
	nodes: bool
	arcs: bool


class TripSummary(BaseModel):
	trip_id: str
	dataset_category: DatasetCategory
	files: RequiredFiles
	available: bool


class TripListResponse(BaseModel):
	training_trips: list[TripSummary]
	test_trips: list[TripSummary]


class GPSObservation(BaseModel):
	coordinates: tuple[float, float]
	third_column: float


class RoadNode(BaseModel):
	node_index: int
	coordinates: tuple[float, float]


class RoadArc(BaseModel):
	arc_index: int
	source_node: int
	target_node: int


class RoadNetwork(BaseModel):
	nodes: list[RoadNode]
	arcs: list[RoadArc]
	truncated: bool
	limit: int


class TripDetailResponse(TripSummary):
	gps_observation_count: int
	road_node_count: int
	road_arc_count: int
	gps_observations: list[GPSObservation]
	road_network: RoadNetwork | None