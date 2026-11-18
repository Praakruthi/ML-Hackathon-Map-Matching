"""Convert ordered road arc IDs into validated geographic route coordinates."""

from collections.abc import Mapping, Sequence
import math
from typing import Any


class RouteSerializationError(ValueError):
	"""Raised when predicted arc IDs cannot be mapped to road coordinates."""


def serialize_route(
	predicted_arc_ids: Sequence[int],
	road_nodes: Sequence[Sequence[float]],
	road_arcs: Sequence[Sequence[int] | Mapping[str, Any]],
) -> dict[str, Any]:
	"""Return ordered arc IDs and a [longitude, latitude] coordinate sequence."""
	coordinates: list[tuple[float, float]] = []
	connected = True
	previous_target: int | None = None
	for arc_id in predicted_arc_ids:
		if not isinstance(arc_id, int) or isinstance(arc_id, bool):
			raise RouteSerializationError("Predicted arc IDs must be integers.")
		if not 0 <= arc_id < len(road_arcs):
			raise RouteSerializationError(f"Unknown predicted arc ID {arc_id}.")
		arc = road_arcs[arc_id]
		if isinstance(arc, Mapping):
			source = arc.get("source_node", arc.get("source"))
			target = arc.get("target_node", arc.get("target"))
		else:
			if len(arc) != 2:
				raise RouteSerializationError(f"Arc {arc_id} must have two node indices.")
			source, target = arc
		if not isinstance(source, int) or not isinstance(target, int):
			raise RouteSerializationError(f"Arc {arc_id} has invalid node indices.")
		if not 0 <= source < len(road_nodes) or not 0 <= target < len(road_nodes):
			raise RouteSerializationError(f"Arc {arc_id} references an unknown node.")
		if previous_target is not None and source != previous_target:
			connected = False
		for node_index in (source, target):
			node = road_nodes[node_index]
			if len(node) != 2:
				raise RouteSerializationError(f"Node {node_index} must have two coordinates.")
			point = (float(node[0]), float(node[1]))
			if not all(math.isfinite(value) for value in point):
				raise RouteSerializationError(f"Node {node_index} has non-finite coordinates.")
			if not -180.0 <= point[0] <= 180.0 or not -90.0 <= point[1] <= 90.0:
				raise RouteSerializationError(f"Node {node_index} has invalid coordinates.")
			if not coordinates or coordinates[-1] != point:
				coordinates.append(point)
		previous_target = target
	return {
		"arc_ids": list(predicted_arc_ids),
		"coordinates": coordinates,
		"arc_count": len(predicted_arc_ids),
		"node_count": len(coordinates),
		"connected": connected,
	}