"""Replaceable adapter boundary for Member 3 route reconstruction."""

from collections.abc import Mapping, Sequence
from typing import Any

from backend.integrations.base import BaseAdapter
from backend.integrations.contracts import (
	CandidateSet,
	GPSPoints,
	Member3Implementation,
	ProbabilitySet,
	RoadGraph,
)


class Member3Adapter(BaseAdapter):
	name = "Member 3 - HMM and Route Reconstruction"
	package = "src.decoder.hmm"

	def __init__(self, implementation: Member3Implementation | None = None) -> None:
		super().__init__(implementation)

	def reconstruct_route(
		self,
		gps_points: GPSPoints,
		candidates: CandidateSet,
		probabilities: ProbabilitySet,
		road_graph: RoadGraph,
	) -> Mapping[str, Any]:
		return self.implementation().reconstruct_route(
			gps_points,
			candidates,
			probabilities,
			road_graph,
		)

	def validate_route(
		self,
		predicted_arc_ids: Sequence[int],
		road_graph: RoadGraph,
	) -> Mapping[str, Any]:
		return self.implementation().validate_route(predicted_arc_ids, road_graph)