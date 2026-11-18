"""Complete directed route reconstruction from an HMM candidate sequence.

The HMM selects one candidate arc per observation.  This module fills the
directed road arcs between those selected positions and preserves partial
traversal information for the first and last arcs of each transition.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from collections.abc import Mapping
from typing import Any

from .hmm import HMMResult
from .transition import CandidatePosition, RoadNetwork, RoutePath


class RouteReconstructionError(ValueError):
    """Raised when a decoded candidate sequence cannot form a valid route."""


@dataclass(frozen=True)
class ArcTraversal:
    """One forward traversal interval along a directed arc."""

    arc_id: int
    distance_from_start_m: float
    distance_to_end_m: float

    @property
    def distance_m(self) -> float:
        return self.distance_to_end_m - self.distance_from_start_m


@dataclass(frozen=True)
class ReconstructedRoute:
    """A complete route, distinct from the HMM's selected candidate sequence."""

    trip_id: str | None
    selected_arc_sequence: tuple[int, ...]
    route_arc_sequence: tuple[int, ...]
    total_distance_m: float
    selected_observation_count: int
    reconstructed_arc_count: int
    traversals: tuple[ArcTraversal, ...]


def reconstruct_route(
    hmm_result: HMMResult,
    road_network: RoadNetwork,
    candidate_positions: Mapping[int, Mapping[int, CandidatePosition]],
) -> ReconstructedRoute:
    """Expand an HMM result into a connected, ordered directed arc route.

    Candidate positions must be the same trip-specific positions used during
    HMM transition scoring.  The function does not rerun Viterbi, consult
    reference routes, or mutate the HMM result or position mapping.
    """

    _validate_trip_identity(hmm_result, road_network)
    selected_arcs = tuple(getattr(hmm_result, "arc_sequence", ()))
    observation_ids = tuple(getattr(hmm_result, "observation_ids", ()))
    if len(selected_arcs) != len(observation_ids):
        raise RouteReconstructionError(
            "HMM arc_sequence and observation_ids must have the same length"
        )
    if not selected_arcs:
        raise RouteReconstructionError("HMM result contains no selected observations")

    positions = tuple(
        _lookup_position(observation_id, arc_id, candidate_positions, road_network)
        for observation_id, arc_id in zip(observation_ids, selected_arcs)
    )

    if len(positions) == 1:
        traversals = (
            ArcTraversal(
                positions[0].arc_id,
                positions[0].distance_from_start_m,
                positions[0].distance_from_start_m,
            ),
        )
    else:
        traversals_list: list[ArcTraversal] = []
        for index, (start, destination) in enumerate(zip(positions, positions[1:])):
            path = road_network.path_between_positions(start, destination)
            if path is None:
                raise RouteReconstructionError(
                    "No directed path between observation "
                    f"{observation_ids[index]} arc {start.arc_id} and observation "
                    f"{observation_ids[index + 1]} arc {destination.arc_id}"
                )
            segment_traversals = _traversals_for_path(
                start, destination, path, road_network
            )
            _append_with_boundary_merge(traversals_list, segment_traversals)
        traversals = tuple(traversals_list)

    _validate_connected_order(traversals, road_network)
    total_distance = sum(traversal.distance_m for traversal in traversals)
    if not math.isfinite(total_distance) or total_distance < 0:
        raise RouteReconstructionError("reconstructed distance must be finite and non-negative")

    return ReconstructedRoute(
        trip_id=getattr(hmm_result, "trip_id", None),
        selected_arc_sequence=selected_arcs,
        route_arc_sequence=tuple(traversal.arc_id for traversal in traversals),
        total_distance_m=total_distance,
        selected_observation_count=len(selected_arcs),
        reconstructed_arc_count=len(traversals),
        traversals=traversals,
    )


def _traversals_for_path(
    start: CandidatePosition,
    destination: CandidatePosition,
    path: RoutePath,
    network: RoadNetwork,
) -> tuple[ArcTraversal, ...]:
    if not path.arc_ids:
        raise RouteReconstructionError("a non-empty candidate transition returned an empty path")
    if path.arc_ids[0] != start.arc_id or path.arc_ids[-1] != destination.arc_id:
        raise RouteReconstructionError("RoutePath endpoints do not match candidate positions")

    if start.arc_id == destination.arc_id and len(path.arc_ids) == 1:
        if destination.distance_from_start_m < start.distance_from_start_m:
            raise RouteReconstructionError("backward same-arc movement returned a direct path")
        return (
            ArcTraversal(
                start.arc_id,
                start.distance_from_start_m,
                destination.distance_from_start_m,
            ),
        )

    traversals: list[ArcTraversal] = [
        ArcTraversal(start.arc_id, start.distance_from_start_m, start.arc_length_m)
    ]
    for arc_id in path.arc_ids[1:-1]:
        arc_length = network.arc(arc_id).length_m
        traversals.append(ArcTraversal(arc_id, 0.0, arc_length))
    traversals.append(
        ArcTraversal(destination.arc_id, 0.0, destination.distance_from_start_m)
    )
    expected_distance = sum(traversal.distance_m for traversal in traversals)
    if not math.isclose(expected_distance, path.distance_m, rel_tol=1e-9, abs_tol=1e-6):
        raise RouteReconstructionError("RoutePath distance disagrees with traversal intervals")
    return tuple(traversals)


def _append_with_boundary_merge(
    output: list[ArcTraversal], incoming: tuple[ArcTraversal, ...]
) -> None:
    for traversal in incoming:
        if output:
            previous = output[-1]
            if (
                previous.arc_id == traversal.arc_id
                and math.isclose(
                    previous.distance_to_end_m,
                    traversal.distance_from_start_m,
                    rel_tol=1e-9,
                    abs_tol=1e-6,
                )
            ):
                output[-1] = ArcTraversal(
                    previous.arc_id,
                    previous.distance_from_start_m,
                    traversal.distance_to_end_m,
                )
                continue
        output.append(traversal)


def _validate_connected_order(
    traversals: tuple[ArcTraversal, ...], network: RoadNetwork
) -> None:
    for previous, current in zip(traversals, traversals[1:]):
        previous_arc = network.arc(previous.arc_id)
        current_arc = network.arc(current.arc_id)
        if previous_arc.target != current_arc.source:
            raise RouteReconstructionError(
                f"arcs {previous.arc_id} and {current.arc_id} are not directionally connected"
            )
        if previous.distance_m < -1e-9 or current.distance_m < -1e-9:
            raise RouteReconstructionError("traversal interval has negative distance")


def _lookup_position(
    observation_id: int,
    arc_id: int,
    candidate_positions: Mapping[int, Mapping[int, CandidatePosition]],
    network: RoadNetwork,
) -> CandidatePosition:
    try:
        position = candidate_positions[observation_id][arc_id]
    except KeyError as error:
        raise RouteReconstructionError(
            f"missing candidate position for observation {observation_id}, arc {arc_id}"
        ) from error
    if not isinstance(position, CandidatePosition):
        raise RouteReconstructionError("candidate position must be a CandidatePosition")
    try:
        arc = network.arc(arc_id)
    except KeyError as error:
        raise RouteReconstructionError(str(error)) from error
    if position.arc_id != arc_id:
        raise RouteReconstructionError(
            f"candidate position arc_id {position.arc_id} does not match selected arc {arc_id}"
        )
    if not math.isclose(position.arc_length_m, arc.length_m, rel_tol=1e-9, abs_tol=1e-6):
        raise RouteReconstructionError(
            f"candidate position for arc {arc_id} belongs to a different network"
        )
    return position


def _validate_trip_identity(hmm_result: HMMResult, network: RoadNetwork) -> None:
    trip_id = getattr(hmm_result, "trip_id", None)
    if trip_id is not None and network.trip_id is not None and trip_id != network.trip_id:
        raise RouteReconstructionError(
            f"HMM trip_id {trip_id!r} does not match network trip_id {network.trip_id!r}"
        )
