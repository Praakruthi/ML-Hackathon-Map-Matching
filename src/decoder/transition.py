"""Directed road-network distances and transition scores for PathSnap.

The network is trip-scoped: arc IDs are keys within one ``.arcs`` file and
must not be reused across network instances.  Coordinates are supplied as
``(longitude, latitude)`` pairs, matching the raw PathSnap files.
"""

from __future__ import annotations

from dataclasses import dataclass
import heapq
import math
from collections.abc import Mapping, Sequence
from numbers import Real
from typing import Any, TypeAlias

import networkx as nx

from .emission import Coordinate
from .viterbi import ViterbiCandidate, ViterbiObservation


_EARTH_RADIUS_M = 6_371_008.8
ArcRow: TypeAlias = Sequence[int] | Mapping[str, int]


@dataclass(frozen=True)
class Arc:
    """A directed road-network arc with its original per-trip identifier."""

    arc_id: int
    source: int
    target: int
    length_m: float


@dataclass(frozen=True)
class CandidatePosition:
    """A projected position along one directed arc."""

    arc_id: int
    distance_from_start_m: float
    arc_length_m: float

    def __post_init__(self) -> None:
        _validate_non_negative_finite(self.arc_length_m, "arc_length_m")
        _validate_non_negative_finite(
            self.distance_from_start_m, "distance_from_start_m"
        )
        if self.distance_from_start_m > self.arc_length_m + 1e-9:
            raise ValueError("distance_from_start_m cannot exceed arc_length_m")

    @property
    def distance_to_end_m(self) -> float:
        """Distance remaining before reaching the arc's destination node."""

        return max(0.0, self.arc_length_m - self.distance_from_start_m)

    @classmethod
    def from_fraction(
        cls, arc_id: int, arc_length_m: float, fraction: float
    ) -> "CandidatePosition":
        if not math.isfinite(fraction) or not 0.0 <= fraction <= 1.0:
            raise ValueError("fraction must be finite and within [0, 1]")
        return cls(arc_id, float(arc_length_m) * fraction, float(arc_length_m))


@dataclass(frozen=True)
class RoutePath:
    """A feasible directed path between two candidate positions."""

    distance_m: float
    arc_ids: tuple[int, ...]


class RoadNetwork:
    """Trip-specific directed road graph with cached node-to-node paths."""

    def __init__(
        self,
        nodes: Sequence[Coordinate],
        arcs: Sequence[Arc],
        *,
        trip_id: str | None = None,
    ) -> None:
        self.trip_id = trip_id
        self.nodes = tuple(_validate_coordinate(node, f"node {i}") for i, node in enumerate(nodes))
        self.arcs = {arc.arc_id: arc for arc in arcs}
        if len(self.arcs) != len(arcs):
            raise ValueError("arc IDs must be unique within one trip network")
        self.graph = nx.MultiDiGraph()
        self.graph.add_nodes_from(range(len(self.nodes)))
        for arc in arcs:
            if not 0 <= arc.source < len(self.nodes) or not 0 <= arc.target < len(self.nodes):
                raise ValueError(
                    f"arc {arc.arc_id} references a node outside the supplied node list"
                )
            self.graph.add_edge(
                arc.source,
                arc.target,
                key=arc.arc_id,
                arc_id=arc.arc_id,
                length_m=arc.length_m,
            )
        self._node_path_cache: dict[tuple[int, int], RoutePath | None] = {}

    @classmethod
    def from_data(
        cls,
        nodes: Sequence[Coordinate],
        arcs: Sequence[ArcRow],
        *,
        trip_id: str | None = None,
    ) -> "RoadNetwork":
        """Build a network from raw ``.nodes`` coordinates and ``.arcs`` rows."""

        validated_nodes = tuple(
            _validate_coordinate(node, f"node {i}") for i, node in enumerate(nodes)
        )
        parsed_arcs: list[Arc] = []
        for arc_id, row in enumerate(arcs):
            source, target = _parse_arc_row(row, arc_id)
            if not 0 <= source < len(validated_nodes) or not 0 <= target < len(validated_nodes):
                raise ValueError(
                    f"arc {arc_id} references a node outside the supplied node list"
                )
            parsed_arcs.append(
                Arc(
                    arc_id=arc_id,
                    source=source,
                    target=target,
                    length_m=geodesic_distance_m(
                        validated_nodes[source], validated_nodes[target]
                    ),
                )
            )
        return cls(validated_nodes, parsed_arcs, trip_id=trip_id)

    def candidate_position(
        self, arc_id: int, distance_from_start_m: float
    ) -> CandidatePosition:
        """Create a validated position using the network's stored arc length."""

        arc = self.arc(arc_id)
        return CandidatePosition(arc_id, distance_from_start_m, arc.length_m)

    def arc(self, arc_id: int) -> Arc:
        try:
            return self.arcs[arc_id]
        except KeyError as error:
            raise KeyError(f"Unknown arc_id {arc_id!r} for trip {self.trip_id!r}") from error

    def shortest_node_path(self, source: int, target: int) -> RoutePath | None:
        """Return the shortest directed node path, including its arc IDs."""

        if not 0 <= source < len(self.nodes) or not 0 <= target < len(self.nodes):
            raise ValueError("source and target node IDs must refer to supplied nodes")
        key = (source, target)
        if key in self._node_path_cache:
            return self._node_path_cache[key]

        distances: dict[int, float] = {source: 0.0}
        paths: dict[int, tuple[int, ...]] = {source: ()}
        queue: list[tuple[float, int]] = [(0.0, source)]
        result: RoutePath | None = None
        while queue:
            distance, node = heapq.heappop(queue)
            if distance > distances.get(node, math.inf):
                continue
            if node == target:
                result = RoutePath(distance, paths[node])
                break
            for _, next_node, arc_id, data in self.graph.out_edges(node, keys=True, data=True):
                next_distance = distance + float(data["length_m"])
                next_path = paths[node] + (int(arc_id),)
                known_distance = distances.get(next_node, math.inf)
                # The path tuple makes equal-length choices reproducible.
                if next_distance < known_distance - 1e-12 or (
                    math.isclose(next_distance, known_distance, abs_tol=1e-12)
                    and next_path < paths.get(next_node, next_path)
                ):
                    distances[next_node] = next_distance
                    paths[next_node] = next_path
                    heapq.heappush(queue, (next_distance, next_node))
        self._node_path_cache[key] = result
        return result

    def path_between_positions(
        self, start: CandidatePosition, destination: CandidatePosition
    ) -> RoutePath | None:
        """Return the shortest directed path between two projected positions."""

        _validate_position_for_network(start, self)
        _validate_position_for_network(destination, self)
        start_arc = self.arc(start.arc_id)
        destination_arc = self.arc(destination.arc_id)

        if start.arc_id == destination.arc_id:
            if destination.distance_from_start_m >= start.distance_from_start_m:
                return RoutePath(
                    destination.distance_from_start_m - start.distance_from_start_m,
                    (start.arc_id,),
                )
            node_path = self.shortest_node_path(start_arc.target, start_arc.source)
            if node_path is None:
                return None
            return RoutePath(
                start.distance_to_end_m
                + node_path.distance_m
                + destination.distance_from_start_m,
                (start.arc_id,) + node_path.arc_ids + (destination.arc_id,),
            )

        node_path = self.shortest_node_path(start_arc.target, destination_arc.source)
        if node_path is None:
            return None
        return RoutePath(
            start.distance_to_end_m
            + node_path.distance_m
            + destination.distance_from_start_m,
            (start.arc_id,) + node_path.arc_ids + (destination.arc_id,),
        )


def build_road_network(
    nodes: Sequence[Coordinate],
    arcs: Sequence[ArcRow],
    *,
    trip_id: str | None = None,
) -> RoadNetwork:
    """Convenience wrapper for constructing a trip-scoped road network."""

    return RoadNetwork.from_data(nodes, arcs, trip_id=trip_id)


def geodesic_distance_m(first: Coordinate, second: Coordinate) -> float:
    """Return great-circle distance between two lon/lat points in metres."""

    first = _validate_coordinate(first, "first coordinate")
    second = _validate_coordinate(second, "second coordinate")
    longitude_1, latitude_1 = map(math.radians, first)
    longitude_2, latitude_2 = map(math.radians, second)
    delta_latitude = latitude_2 - latitude_1
    delta_longitude = longitude_2 - longitude_1
    haversine = math.sin(delta_latitude / 2) ** 2 + math.cos(latitude_1) * math.cos(latitude_2) * math.sin(delta_longitude / 2) ** 2
    return 2.0 * _EARTH_RADIUS_M * math.asin(min(1.0, math.sqrt(haversine)))


class RoadNetworkTransitionScorer:
    """Callable transition scorer compatible with :func:`decode_viterbi`.

    ``candidate_positions`` is keyed by observation ID and then arc ID.  This
    keeps projected offsets out of Member 2's probability payload.  GPS
    coordinates are read from observation metadata using ``longitude`` and
    ``latitude`` keys.  ``timestamp_unit_seconds`` must be supplied explicitly
    when timestamps are not already in seconds.

    Scores are unnormalised log compatibility values.  A missing path returns
    ``-math.inf``.  A configured speed limit also returns ``-math.inf`` when a
    route is physically too slow to explain in the observed time interval.
    ``last_rejection_reason`` records whether the most recent rejection was
    caused by reachability or by time/speed validation.
    """

    def __init__(
        self,
        network: RoadNetwork,
        candidate_positions: Mapping[int, Mapping[int, CandidatePosition]],
        *,
        beta_m: float = 50.0,
        max_speed_mps: float | None = None,
        timestamp_unit_seconds: float = 1.0,
    ) -> None:
        _validate_positive_finite(beta_m, "beta_m")
        if max_speed_mps is not None:
            _validate_positive_finite(max_speed_mps, "max_speed_mps")
        _validate_positive_finite(timestamp_unit_seconds, "timestamp_unit_seconds")
        self.network = network
        self.candidate_positions = candidate_positions
        self.beta_m = float(beta_m)
        self.max_speed_mps = None if max_speed_mps is None else float(max_speed_mps)
        self.timestamp_unit_seconds = float(timestamp_unit_seconds)
        self.last_rejection_reason: str | None = None
        self._position_path_cache: dict[tuple[int, float, int, float], RoutePath | None] = {}

    def route_distance(
        self, start: CandidatePosition, destination: CandidatePosition
    ) -> RoutePath | None:
        """Return a cached feasible path between two candidate positions."""

        key = (
            start.arc_id,
            round(start.distance_from_start_m, 9),
            destination.arc_id,
            round(destination.distance_from_start_m, 9),
        )
        if key not in self._position_path_cache:
            self._position_path_cache[key] = self.network.path_between_positions(
                start, destination
            )
        return self._position_path_cache[key]

    def __call__(
        self,
        previous_candidate: ViterbiCandidate,
        current_candidate: ViterbiCandidate,
        previous_observation: ViterbiObservation,
        current_observation: ViterbiObservation,
    ) -> float:
        self.last_rejection_reason = None
        start = self._position_for(previous_observation, previous_candidate)
        destination = self._position_for(current_observation, current_candidate)
        route = self.route_distance(start, destination)
        if route is None:
            self.last_rejection_reason = "unreachable"
            return -math.inf

        if self.max_speed_mps is not None:
            elapsed = _elapsed_seconds(
                previous_observation.timestamp,
                current_observation.timestamp,
                self.timestamp_unit_seconds,
            )
            if elapsed is not None:
                if elapsed <= 0:
                    self.last_rejection_reason = "invalid_time"
                    return -math.inf
                if route.distance_m / elapsed > self.max_speed_mps:
                    self.last_rejection_reason = "speed_limit"
                    return -math.inf

        previous_point = _observation_coordinate(previous_observation)
        current_point = _observation_coordinate(current_observation)
        gps_distance = geodesic_distance_m(previous_point, current_point)
        return -abs(route.distance_m - gps_distance) / self.beta_m

    def _position_for(
        self, observation: ViterbiObservation, candidate: ViterbiCandidate
    ) -> CandidatePosition:
        metadata = candidate.metadata or {}
        position = metadata.get("position")
        if isinstance(position, CandidatePosition):
            return position
        try:
            return self.candidate_positions[observation.observation_id][candidate.arc_id]
        except KeyError as error:
            raise KeyError(
                f"No projected position for observation {observation.observation_id} "
                f"and arc {candidate.arc_id}"
            ) from error


def _parse_arc_row(row: ArcRow, arc_id: int) -> tuple[int, int]:
    if isinstance(row, Mapping):
        source = row.get("source", row.get("from_node_id"))
        target = row.get("target", row.get("to_node_id"))
    elif isinstance(row, Sequence) and len(row) == 2:
        source, target = row
    else:
        raise ValueError(f"arc row {arc_id} must contain exactly two node IDs")
    if not isinstance(source, int) or isinstance(source, bool) or not isinstance(target, int) or isinstance(target, bool):
        raise ValueError(f"arc row {arc_id} must contain integer node IDs")
    return source, target


def _validate_position_for_network(position: CandidatePosition, network: RoadNetwork) -> None:
    arc = network.arc(position.arc_id)
    if not math.isclose(position.arc_length_m, arc.length_m, rel_tol=1e-9, abs_tol=1e-6):
        raise ValueError("candidate position uses a different arc length than the network")


def _observation_coordinate(observation: ViterbiObservation) -> Coordinate:
    metadata: Mapping[str, Any] = observation.metadata or {}
    if "longitude" not in metadata or "latitude" not in metadata:
        raise ValueError(
            f"observation {observation.observation_id} must provide longitude and latitude metadata"
        )
    return _validate_coordinate(
        (metadata["longitude"], metadata["latitude"]), "observation coordinate"
    )


def _elapsed_seconds(
    previous_timestamp: float | None,
    current_timestamp: float | None,
    timestamp_unit_seconds: float,
) -> float | None:
    if previous_timestamp is None or current_timestamp is None:
        return None
    if not isinstance(previous_timestamp, Real) or not isinstance(current_timestamp, Real):
        raise ValueError("timestamps must be numeric when supplied")
    if not math.isfinite(float(previous_timestamp)) or not math.isfinite(float(current_timestamp)):
        raise ValueError("timestamps must be finite when supplied")
    return (float(current_timestamp) - float(previous_timestamp)) * timestamp_unit_seconds


def _validate_coordinate(coordinate: Coordinate, name: str) -> Coordinate:
    if not isinstance(coordinate, Sequence) or len(coordinate) != 2:
        raise ValueError(f"{name} must be a (longitude, latitude) pair")
    longitude, latitude = coordinate
    if not isinstance(longitude, Real) or not isinstance(latitude, Real):
        raise ValueError(f"{name} must contain numeric coordinates")
    longitude = float(longitude)
    latitude = float(latitude)
    if not math.isfinite(longitude) or not math.isfinite(latitude) or not -180 <= longitude <= 180 or not -90 <= latitude <= 90:
        raise ValueError(f"{name} contains invalid longitude/latitude values")
    return longitude, latitude


def _validate_non_negative_finite(value: float, name: str) -> None:
    if not isinstance(value, Real) or isinstance(value, bool) or not math.isfinite(float(value)) or float(value) < 0:
        raise ValueError(f"{name} must be finite and non-negative")


def _validate_positive_finite(value: float, name: str) -> None:
    if not isinstance(value, Real) or isinstance(value, bool) or not math.isfinite(float(value)) or float(value) <= 0:
        raise ValueError(f"{name} must be finite and positive")
