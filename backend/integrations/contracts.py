"""Standardized data contracts between backend adapters."""

from collections.abc import Mapping, Sequence
from typing import Any, Protocol

Coordinate = tuple[float, float]
GPSPoints = Sequence[Coordinate]
RoadGraph = Mapping[str, Any]
CandidateSet = Mapping[str, Any]
FeatureSet = Mapping[str, Any]
ProbabilitySet = Mapping[str, Any]


class Member1Implementation(Protocol):
	def load_trip(self, trip_id: str, dataset_category: str) -> Mapping[str, Any]: ...

	def generate_candidates(
		self,
		gps_points: GPSPoints,
		road_graph: RoadGraph,
	) -> CandidateSet: ...

	def extract_features(self, candidates: CandidateSet) -> FeatureSet: ...


class Member2Implementation(Protocol):
	def load_best_model(self, model_id: str | None = None) -> Any: ...

	def predict_candidates(self, model: Any, features: FeatureSet) -> ProbabilitySet: ...


class Member3Implementation(Protocol):
	def reconstruct_route(
		self,
		gps_points: GPSPoints,
		candidates: CandidateSet,
		probabilities: ProbabilitySet,
		road_graph: RoadGraph,
	) -> Mapping[str, Any]: ...

	def validate_route(
		self,
		predicted_arc_ids: Sequence[int],
		road_graph: RoadGraph,
	) -> Mapping[str, Any]: ...