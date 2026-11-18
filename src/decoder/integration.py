"""Adapters for the supplied PathSnap preprocessing and prediction artifacts.

This module consumes predictions from one frozen selected model. It derives
candidate positions from GPS observations and arc geometry using the same local
equirectangular projection used by the preprocessing notebook. Reference route
files are intentionally not read here.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from zipfile import ZipFile
from collections.abc import Mapping, Sequence
from typing import Any

from .transition import CandidatePosition, RoadNetwork, build_road_network


@dataclass(frozen=True)
class LoadedTripInputs:
    """All non-ground-truth inputs needed by Member 3 for one trip."""

    trip_id: str
    predictions: Mapping[str, Any]
    road_network: RoadNetwork
    observation_data: Mapping[int, Mapping[str, float]]
    candidate_positions: Mapping[int, Mapping[int, CandidatePosition]]
    selected_model: str | None

    @property
    def observation_count(self) -> int:
        return len(self.predictions["observations"])

    @property
    def candidate_count(self) -> int:
        return sum(len(item["candidates"]) for item in self.predictions["observations"])


def load_prediction_payload(
    source: str | Path,
    *,
    trip_id: str,
    member_name: str = "pathsnap_hmm_input.json",
) -> tuple[dict[str, Any], str | None]:
    """Load one trip payload from JSON or the model-output ZIP.

    The JSON may contain one mapping or a list of trip mappings. Optional model
    identity is read from ``model_name``/``model_id`` but is not required by the
    HMM.
    """

    source = Path(source)
    metrics_raw: bytes | None = None
    if source.suffix.lower() == ".zip":
        with ZipFile(source) as archive:
            try:
                raw = archive.read(member_name)
            except KeyError as error:
                raise FileNotFoundError(f"{member_name} not found in {source}") from error
            metrics_raw = archive.read("pathsnap_final_metrics.json") if "pathsnap_final_metrics.json" in archive.namelist() else None
        payload = json.loads(raw.decode("utf-8"))
    else:
        payload = json.loads(source.read_text(encoding="utf-8"))

    payloads = payload if isinstance(payload, list) else [payload]
    matches = [item for item in payloads if isinstance(item, Mapping) and str(item.get("trip_id")) == str(trip_id)]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one prediction payload for trip {trip_id}, found {len(matches)}")
    selected = dict(matches[0])
    model_name = selected.get("model_name") or selected.get("model_id")
    if model_name is None and metrics_raw is not None:
        model_name = json.loads(metrics_raw.decode("utf-8")).get("selected_model")
    return selected, str(model_name) if model_name is not None else None


def load_trip_inputs_from_zip(
    dataset_zip: str | Path,
    prediction_source: str | Path,
    *,
    split: str,
    trip_id: str,
    selected_model: str | None = None,
) -> LoadedTripInputs:
    """Load one trip's GPS/network files and prediction payload from ZIPs."""

    predictions, payload_model = load_prediction_payload(prediction_source, trip_id=trip_id)
    with ZipFile(dataset_zip) as archive:
        nodes = _read_coordinates(
            archive.read(_find_trip_member(archive, split, trip_id, "nodes"))
        )
        arcs = _read_integer_pairs(
            archive.read(_find_trip_member(archive, split, trip_id, "arcs"))
        )
        track = _read_track(
            archive.read(_find_trip_member(archive, split, trip_id, "track"))
        )

    if len(track) < len(predictions.get("observations", ())):
        raise ValueError(f"track has {len(track)} rows but predictions have {len(predictions.get('observations', ()))} observations")
    network = build_road_network(nodes, arcs, trip_id=trip_id)
    observation_data: dict[int, Mapping[str, float]] = {}
    positions: dict[int, dict[int, CandidatePosition]] = {}
    for index, raw_observation in enumerate(predictions.get("observations", ())):
        if not isinstance(raw_observation, Mapping):
            raise ValueError(f"prediction observation {index} is not a mapping")
        observation_id = int(raw_observation.get("observation_id", index))
        longitude, latitude, timestamp = track[index]
        observation_data[observation_id] = {
            "longitude": longitude,
            "latitude": latitude,
            "timestamp": timestamp,
        }
        positions[observation_id] = {}
        for candidate in raw_observation.get("candidates", ()):
            arc_id = int(candidate["arc_id"])
            arc = network.arc(arc_id)
            fraction = _projection_fraction(
                (longitude, latitude),
                network.nodes[arc.source],
                network.nodes[arc.target],
            )
            positions[observation_id][arc_id] = CandidatePosition.from_fraction(
                arc_id, arc.length_m, fraction
            )

    return LoadedTripInputs(
        trip_id=trip_id,
        predictions=predictions,
        road_network=network,
        observation_data=observation_data,
        candidate_positions=positions,
        selected_model=selected_model or payload_model,
    )


def _find_trip_member(archive: ZipFile, split: str, trip_id: str, extension: str) -> str:
    suffix = f"/{split}/{trip_id}/{trip_id}.{extension}"
    candidates = [name for name in archive.namelist() if name.endswith(suffix)]
    if not candidates:
        candidates = [name for name in archive.namelist() if name.endswith(f"/{trip_id}/{trip_id}.{extension}")]
    if len(candidates) != 1:
        raise FileNotFoundError(f"could not uniquely find {split}/{trip_id}/{trip_id}.{extension}")
    return candidates[0]


def _read_coordinates(raw: bytes) -> list[tuple[float, float]]:
    return [tuple(map(float, line.split())) for line in raw.decode("utf-8").splitlines() if line.strip()]


def _read_integer_pairs(raw: bytes) -> list[tuple[int, int]]:
    return [tuple(map(int, line.split())) for line in raw.decode("utf-8").splitlines() if line.strip()]


def _read_track(raw: bytes) -> list[tuple[float, float, float]]:
    return [tuple(map(float, line.split())) for line in raw.decode("utf-8").splitlines() if line.strip()]


def _projection_fraction(
    point: tuple[float, float], start: tuple[float, float], end: tuple[float, float]
) -> float:
    """Project a GPS point onto an arc and return its clamped along-arc fraction."""

    lon0, lat0 = point
    scale = 6_371_000.0
    cos_lat = math.cos(math.radians(lat0))
    point_xy = (0.0, 0.0)
    start_xy = (
        scale * math.radians(start[0] - lon0) * cos_lat,
        scale * math.radians(start[1] - lat0),
    )
    end_xy = (
        scale * math.radians(end[0] - lon0) * cos_lat,
        scale * math.radians(end[1] - lat0),
    )
    dx = end_xy[0] - start_xy[0]
    dy = end_xy[1] - start_xy[1]
    denominator = dx * dx + dy * dy
    if denominator == 0:
        return 0.0
    fraction = ((point_xy[0] - start_xy[0]) * dx + (point_xy[1] - start_xy[1]) * dy) / denominator
    return min(1.0, max(0.0, fraction))
