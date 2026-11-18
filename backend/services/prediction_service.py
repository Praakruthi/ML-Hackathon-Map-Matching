"""Adapter-driven map-matching orchestration with unavailable-safe behavior."""

import logging
import time
from collections.abc import Mapping, Sequence
from typing import Any

from backend.integrations.base import ModuleUnavailableError
from backend.integrations.member1_adapter import Member1Adapter
from backend.integrations.member2_adapter import Member2Adapter
from backend.integrations.member3_adapter import Member3Adapter
from backend.integrations.route_serializer import serialize_route
from backend.services.module_service import ModuleRegistry

logger = logging.getLogger(__name__)


class PredictionUnavailableError(Exception):
	"""Raised when one or more standardized member adapters are not connected."""

	def __init__(self, modules: list[dict[str, object]]) -> None:
		self.modules = modules
		super().__init__("Prediction is unavailable until all member adapters are connected.")


class PredictionService:
	def __init__(
		self,
		member1: Member1Adapter | None = None,
		member2: Member2Adapter | None = None,
		member3: Member3Adapter | None = None,
	) -> None:
		self.member1 = member1 or Member1Adapter()
		self.member2 = member2 or Member2Adapter()
		self.member3 = member3 or Member3Adapter()
		self.module_registry = ModuleRegistry((self.member1, self.member2, self.member3))

	def predict_trip(
		self,
		trip_id: str,
		dataset_category: str,
		model_id: str | None = None,
	) -> dict[str, Any]:
		readiness = self.module_registry.get_status()
		if not readiness["pipeline_ready"]:
			raise PredictionUnavailableError(readiness["modules"])

		started = time.perf_counter()
		trip = self.member1.load_trip(trip_id, dataset_category)
		gps_points = trip.get("gps_points")
		road_graph = trip.get("road_graph")
		if not isinstance(gps_points, Sequence) or not isinstance(road_graph, Mapping):
			raise ValueError("Member 1 adapter returned an invalid trip payload.")
		candidates = self.member1.generate_candidates(gps_points, road_graph)
		features = self.member1.extract_features(candidates)
		model = self.member2.load_best_model(model_id)
		probabilities = self.member2.predict_candidates(model, features)
		route = self.member3.reconstruct_route(
			gps_points,
			candidates,
			probabilities,
			road_graph,
		)
		arc_ids = route.get("arc_ids")
		if not isinstance(arc_ids, Sequence) or isinstance(arc_ids, (str, bytes)):
			raise ValueError("Member 3 adapter returned no ordered arc_ids sequence.")
		serialized_route = serialize_route(
			arc_ids,
			road_graph.get("nodes", []),
			road_graph.get("arcs", []),
		)
		validation = self.member3.validate_route(arc_ids, road_graph)
		serialized_route["valid"] = bool(validation.get("valid", False))
		serialized_route["connected"] = bool(
			validation.get("connected", serialized_route["connected"])
		) and serialized_route["connected"]
		serialized_route["distance_m"] = route.get("distance_m")
		return {
			"status": "completed",
			"trip_id": trip_id,
			"dataset_category": dataset_category,
			"model_id": model_id,
			"gps_observation_count": len(gps_points),
			"processing_time_s": time.perf_counter() - started,
			"route": serialized_route,
		}


prediction_service = PredictionService()