"""Emission scores for GPS observations and candidate road segments.

The dataset stores coordinates as ``longitude, latitude`` pairs.  Distances
are calculated in a local metre-based projection before the Gaussian score is
formed; longitude and latitude degrees are never treated as Cartesian units.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
from numbers import Real
from typing import Any, TypeAlias

from .viterbi import ViterbiCandidate, ViterbiObservation


Coordinate: TypeAlias = tuple[float, float]
Geometry: TypeAlias = Sequence[Coordinate]

_EARTH_RADIUS_M = 6_371_008.8


def point_to_segment_distance_m(
    point: Coordinate,
    segment_start: Coordinate,
    segment_end: Coordinate,
) -> float:
    """Return the shortest distance from a lon/lat point to a segment in metres.

    A local equirectangular projection is accurate for the short road segments
    used by map matching and avoids introducing a third-party GIS dependency.
    """

    point = _validate_coordinate(point, "point")
    segment_start = _validate_coordinate(segment_start, "segment_start")
    segment_end = _validate_coordinate(segment_end, "segment_end")

    reference_latitude = math.radians(point[1])
    projected_point = _project(point, reference_latitude)
    projected_start = _project(segment_start, reference_latitude)
    projected_end = _project(segment_end, reference_latitude)

    dx = projected_end[0] - projected_start[0]
    dy = projected_end[1] - projected_start[1]
    segment_length_squared = dx * dx + dy * dy
    if segment_length_squared == 0:
        return math.hypot(
            projected_point[0] - projected_start[0],
            projected_point[1] - projected_start[1],
        )

    projection = (
        (projected_point[0] - projected_start[0]) * dx
        + (projected_point[1] - projected_start[1]) * dy
    ) / segment_length_squared
    projection = min(1.0, max(0.0, projection))
    closest_x = projected_start[0] + projection * dx
    closest_y = projected_start[1] + projection * dy
    return math.hypot(projected_point[0] - closest_x, projected_point[1] - closest_y)


def point_to_geometry_distance_m(point: Coordinate, geometry: Geometry) -> float:
    """Return the shortest distance from a point to a segment or polyline."""

    if not isinstance(geometry, Sequence) or len(geometry) < 2:
        raise ValueError("geometry must contain at least two coordinates")
    distances = [
        point_to_segment_distance_m(point, start, end)
        for start, end in zip(geometry, geometry[1:])
    ]
    distance = min(distances)
    if not math.isfinite(distance) or distance < 0:
        raise ValueError("calculated distance must be finite and non-negative")
    return distance


def gaussian_log_emission(
    observation: Coordinate,
    road_geometry: Geometry,
    sigma_m: float,
) -> float:
    """Return the Gaussian log emission score for a GPS point and road geometry."""

    _validate_sigma(sigma_m)
    distance_m = point_to_geometry_distance_m(observation, road_geometry)
    return -math.log(math.sqrt(2.0 * math.pi) * sigma_m) - (
        distance_m * distance_m
    ) / (2.0 * sigma_m * sigma_m)


def ml_log_emission(
    candidate_or_probability: ViterbiCandidate | Real,
    epsilon: float = 1e-12,
) -> float:
    """Convert an ML positive-class probability into a log compatibility score.

    The returned value is a compatibility score for Viterbi decoding, not a
    claim that the classifier probability is a calibrated HMM likelihood.  The
    input probability is read only; it is not normalised or written back.
    """

    _validate_epsilon(epsilon)
    probability = (
        candidate_or_probability.ml_probability
        if isinstance(candidate_or_probability, ViterbiCandidate)
        else candidate_or_probability
    )
    if not isinstance(probability, Real) or isinstance(probability, bool):
        raise ValueError("ML probability must be a real number")
    probability = float(probability)
    if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
        raise ValueError(
            f"ML probability must be finite and within [0, 1], got {probability!r}"
        )
    return math.log(max(probability, epsilon))


def gaussian_log_emission_for_context(
    candidate: ViterbiCandidate,
    observation: ViterbiObservation,
    geometry_by_arc_id: Mapping[int, Geometry],
    sigma_m: float,
) -> float:
    """Score a Viterbi candidate using GPS coordinates stored in observation metadata.

    The context adapter expects ``longitude`` and ``latitude`` keys in the
    observation metadata.  It is intentionally separate from the core geometry
    function so future preprocessing can choose its own observation container.
    """

    if candidate.arc_id not in geometry_by_arc_id:
        raise KeyError(f"No geometry supplied for arc_id {candidate.arc_id}")
    point = _coordinate_from_metadata(observation.metadata)
    return gaussian_log_emission(point, geometry_by_arc_id[candidate.arc_id], sigma_m)


def _project(coordinate: Coordinate, reference_latitude: float) -> tuple[float, float]:
    longitude, latitude = map(math.radians, coordinate)
    return (
        _EARTH_RADIUS_M * longitude * math.cos(reference_latitude),
        _EARTH_RADIUS_M * latitude,
    )


def _validate_coordinate(coordinate: Coordinate, name: str) -> Coordinate:
    if not isinstance(coordinate, Sequence) or len(coordinate) != 2:
        raise ValueError(f"{name} must be a (longitude, latitude) pair")
    longitude, latitude = coordinate
    if not isinstance(longitude, Real) or not isinstance(latitude, Real):
        raise ValueError(f"{name} must contain numeric coordinates")
    longitude = float(longitude)
    latitude = float(latitude)
    if (
        not math.isfinite(longitude)
        or not math.isfinite(latitude)
        or not -180.0 <= longitude <= 180.0
        or not -90.0 <= latitude <= 90.0
    ):
        raise ValueError(f"{name} contains invalid longitude/latitude values")
    return longitude, latitude


def _validate_sigma(sigma_m: float) -> None:
    if not isinstance(sigma_m, Real) or isinstance(sigma_m, bool):
        raise ValueError("sigma_m must be a positive finite number")
    if not math.isfinite(float(sigma_m)) or float(sigma_m) <= 0:
        raise ValueError("sigma_m must be a positive finite number")


def _validate_epsilon(epsilon: float) -> None:
    if not isinstance(epsilon, Real) or isinstance(epsilon, bool):
        raise ValueError("epsilon must be a positive finite number no greater than 1")
    if not math.isfinite(float(epsilon)) or not 0 < float(epsilon) <= 1:
        raise ValueError("epsilon must be a positive finite number no greater than 1")


def _coordinate_from_metadata(metadata: Mapping[str, Any] | None) -> Coordinate:
    if metadata is None:
        raise ValueError("observation metadata must contain longitude and latitude")
    if "longitude" not in metadata or "latitude" not in metadata:
        raise ValueError("observation metadata must contain longitude and latitude")
    return _validate_coordinate(
        (metadata["longitude"], metadata["latitude"]), "observation"
    )
