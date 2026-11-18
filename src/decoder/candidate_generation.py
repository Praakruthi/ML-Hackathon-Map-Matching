"""Geometry helpers for expanding PathSnap candidate coverage.

The original preprocessing notebook uses a midpoint tree to choose the first
15 arcs.  That is fast, but a long arc can pass close to a GPS point while its
midpoint is comparatively far away.  This module keeps the midpoint tree as a
broad spatial index and ranks the retrieved arcs by their true point-to-
segment distance.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from .emission import Coordinate
from .transition import Arc

try:
    from scipy.spatial import cKDTree
except ImportError:  # pragma: no cover - the preprocessing environment has scipy
    cKDTree = None


EARTH_RADIUS_M = 6_371_000.0


def local_xy(
    coordinates: Sequence[Coordinate], origin: Coordinate
) -> list[tuple[float, float]]:
    """Project lon/lat coordinates into local metres around ``origin``."""

    lon0, lat0 = origin
    scale = math.cos(math.radians(lat0))
    return [
        (
            EARTH_RADIUS_M * math.radians(lon - lon0) * scale,
            EARTH_RADIUS_M * math.radians(lat - lat0),
        )
        for lon, lat in coordinates
    ]


def point_to_segment_distance_m(
    point: tuple[float, float],
    start: tuple[float, float],
    end: tuple[float, float],
) -> tuple[float, float]:
    """Return minimum distance and clamped fraction along a metric segment."""

    dx = end[0] - start[0]
    dy = end[1] - start[1]
    denominator = dx * dx + dy * dy
    if denominator == 0.0:
        return math.hypot(point[0] - start[0], point[1] - start[1]), 0.0
    fraction = (
        (point[0] - start[0]) * dx + (point[1] - start[1]) * dy
    ) / denominator
    fraction = min(1.0, max(0.0, fraction))
    nearest = (start[0] + fraction * dx, start[1] + fraction * dy)
    return math.hypot(point[0] - nearest[0], point[1] - nearest[1]), fraction


class TrueDistanceCandidateIndex:
    """Broad midpoint lookup followed by exact segment-distance ranking."""

    def __init__(self, nodes: Sequence[Coordinate], arcs: Sequence[Arc]):
        self._origin = _origin(nodes)
        self._nodes_xy = local_xy(nodes, self._origin)
        self._arcs = tuple(arcs)
        self._arc_xy = tuple(
            (self._nodes_xy[arc.source], self._nodes_xy[arc.target])
            for arc in self._arcs
        )
        self._midpoints = tuple(
            (
                (start[0] + end[0]) / 2.0,
                (start[1] + end[1]) / 2.0,
            )
            for start, end in self._arc_xy
        )
        self._tree = cKDTree(self._midpoints) if cKDTree is not None else None

    def rank(
        self,
        point: Coordinate,
        *,
        pool_size: int = 64,
        max_distance_m: float | None = None,
    ) -> list[tuple[int, float, float]]:
        """Return ``(arc_id, distance_m, fraction)`` sorted by true distance."""

        if pool_size <= 0:
            raise ValueError("pool_size must be positive")
        point_xy = local_xy([point], self._origin)[0]
        count = min(pool_size, len(self._arcs))
        if self._tree is None:
            local_indices = range(len(self._arcs))
        else:
            _, nearest = self._tree.query(point_xy, k=count)
            local_indices = [int(index) for index in _as_list(nearest)]

        ranked: list[tuple[int, float, float]] = []
        for index in local_indices:
            distance, fraction = point_to_segment_distance_m(
                point_xy, *self._arc_xy[index]
            )
            if max_distance_m is None or distance <= max_distance_m:
                ranked.append((self._arcs[index].arc_id, distance, fraction))
        ranked.sort(key=lambda item: (item[1], item[0]))
        return ranked


def _origin(coordinates: Sequence[Coordinate] | Coordinate) -> Coordinate:
    if len(coordinates) == 2 and isinstance(coordinates[0], (int, float)):
        return coordinates  # type: ignore[return-value]
    values = list(coordinates)  # type: ignore[arg-type]
    return (
        sum(float(item[0]) for item in values) / len(values),
        sum(float(item[1]) for item in values) / len(values),
    )


def _as_list(value) -> list:
    if hasattr(value, "tolist"):
        value = value.tolist()
    return value if isinstance(value, list) else [value]
