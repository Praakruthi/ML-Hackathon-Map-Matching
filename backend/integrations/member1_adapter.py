"""Replaceable adapter boundary for Member 1 data preparation."""

from collections.abc import Mapping
from typing import Any

from backend.integrations.base import BaseAdapter
from backend.integrations.contracts import (
	CandidateSet,
	FeatureSet,
	GPSPoints,
	Member1Implementation,
	RoadGraph,
)


class Member1Adapter(BaseAdapter):
	name = "Member 1 - Data Engineering"
	package = "src.member1"

	def __init__(self, implementation: Member1Implementation | None = None) -> None:
		super().__init__(implementation)

	def load_trip(self, trip_id: str, dataset_category: str) -> Mapping[str, Any]:
		return self.implementation().load_trip(trip_id, dataset_category)

	def generate_candidates(
		self,
		gps_points: GPSPoints,
		road_graph: RoadGraph,
	) -> CandidateSet:
		return self.implementation().generate_candidates(gps_points, road_graph)

	def extract_features(self, candidates: CandidateSet) -> FeatureSet:
		return self.implementation().extract_features(candidates)