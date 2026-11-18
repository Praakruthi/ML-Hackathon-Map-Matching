"""HMM orchestration for candidate-road map matching.

This module connects the existing emission, transition, and Viterbi modules. It
returns one selected candidate arc per GPS observation; it does not expand that
sequence into the complete travelled route.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from collections.abc import Mapping, Sequence
from numbers import Real
from typing import Any

from .emission import gaussian_log_emission, ml_log_emission
from .transition import CandidatePosition, RoadNetwork, RoadNetworkTransitionScorer
from .viterbi import ViterbiCandidate, ViterbiObservation, decode_viterbi


@dataclass(frozen=True)
class HMMConfig:
    """Configuration for one trip-matching run.

    ``sigma_m`` and ``beta_m`` are metres. ``timestamp_unit_seconds`` converts
    input timestamp differences to seconds. The initial values are defaults,
    not tuned model parameters.
    """

    emission_mode: str = "ml"
    emission_weight: float = 1.0
    transition_weight: float = 1.0
    sigma_m: float = 20.0
    beta_m: float = 50.0
    epsilon: float = 1e-12
    max_speed_mps: float | None = None
    timestamp_unit_seconds: float = 1.0

    def __post_init__(self) -> None:
        if self.emission_mode not in {"ml", "distance"}:
            raise ValueError("emission_mode must be 'ml' or 'distance'")
        _validate_non_negative_finite(self.emission_weight, "emission_weight")
        _validate_non_negative_finite(self.transition_weight, "transition_weight")
        _validate_positive_finite(self.sigma_m, "sigma_m")
        _validate_positive_finite(self.beta_m, "beta_m")
        _validate_positive_finite(self.epsilon, "epsilon")
        if self.epsilon > 1.0:
            raise ValueError("epsilon must not exceed 1")
        _validate_positive_finite(self.timestamp_unit_seconds, "timestamp_unit_seconds")
        if self.max_speed_mps is not None:
            _validate_positive_finite(self.max_speed_mps, "max_speed_mps")


@dataclass(frozen=True)
class HMMResult:
    """Decoded candidate sequence, distinct from the complete reconstructed route."""

    trip_id: str | None
    observation_ids: tuple[int, ...]
    arc_sequence: tuple[int, ...]
    final_score: float
    emission_mode: str
    configuration: Mapping[str, Any]


def match_trip(
    predictions: Mapping[str, Any],
    road_network: RoadNetwork,
    observation_data: Mapping[int, Mapping[str, Any]] | Sequence[Mapping[str, Any]],
    candidate_positions: Mapping[int, Mapping[int, CandidatePosition]],
    config: HMMConfig | None = None,
) -> HMMResult:
    """Decode one trip using Member 2 predictions and Member 1 metadata.

    ``observation_data`` must provide ``longitude`` and ``latitude`` for every
    predicted observation and may provide ``timestamp``. ``candidate_positions``
    must be keyed by observation ID and then arc ID, with values created for the
    supplied ``road_network``. The prediction payload is never modified.
    """

    if not isinstance(predictions, Mapping):
        raise TypeError("predictions must be the Member 2 mapping payload")
    config = config or HMMConfig()
    trip_id = predictions.get("trip_id")
    if trip_id is not None and road_network.trip_id is not None and trip_id != road_network.trip_id:
        raise ValueError(
            f"prediction trip_id {trip_id!r} does not match network trip_id {road_network.trip_id!r}"
        )

    member_observations = predictions.get("observations")
    if not isinstance(member_observations, Sequence) or isinstance(member_observations, (str, bytes)):
        raise ValueError("predictions must contain an 'observations' sequence")
    metadata_by_id = _normalise_observation_data(observation_data)
    observations = _build_observations(member_observations, metadata_by_id, road_network)
    if not observations:
        raise ValueError("at least one observation is required")
    _validate_candidate_positions(observations, candidate_positions, road_network)

    transition_scorer = RoadNetworkTransitionScorer(
        road_network,
        candidate_positions,
        beta_m=config.beta_m,
        max_speed_mps=config.max_speed_mps,
        timestamp_unit_seconds=config.timestamp_unit_seconds,
    )
    emission_scorer = _make_emission_scorer(config, road_network)

    def weighted_emission(candidate: ViterbiCandidate, observation: ViterbiObservation) -> float:
        return config.emission_weight * emission_scorer(candidate, observation)

    def weighted_transition(
        previous_candidate: ViterbiCandidate,
        current_candidate: ViterbiCandidate,
        previous_observation: ViterbiObservation,
        current_observation: ViterbiObservation,
    ) -> float:
        score = transition_scorer(
            previous_candidate,
            current_candidate,
            previous_observation,
            current_observation,
        )
        if score == -math.inf:
            return -math.inf
        return config.transition_weight * score

    decoded = decode_viterbi(
        observations,
        weighted_transition,
        epsilon=config.epsilon,
        emission_score=weighted_emission,
    )
    return HMMResult(
        trip_id=trip_id if trip_id is not None else road_network.trip_id,
        observation_ids=tuple(observation.observation_id for observation in observations),
        arc_sequence=decoded.arc_sequence,
        final_score=decoded.final_score,
        emission_mode=config.emission_mode,
        configuration=asdict(config),
    )


def _make_emission_scorer(config: HMMConfig, network: RoadNetwork):
    if config.emission_mode == "ml":
        return lambda candidate, observation: ml_log_emission(candidate, config.epsilon)

    def distance_score(candidate: ViterbiCandidate, observation: ViterbiObservation) -> float:
        arc = network.arc(candidate.arc_id)
        geometry = (network.nodes[arc.source], network.nodes[arc.target])
        metadata = observation.metadata or {}
        return gaussian_log_emission(
            (metadata["longitude"], metadata["latitude"]), geometry, config.sigma_m
        )

    return distance_score


def _build_observations(
    raw_observations: Sequence[Mapping[str, Any]],
    metadata_by_id: Mapping[int, Mapping[str, Any]],
    network: RoadNetwork,
) -> list[ViterbiObservation]:
    observations: list[ViterbiObservation] = []
    seen_ids: set[int] = set()
    for index, raw_observation in enumerate(raw_observations):
        if not isinstance(raw_observation, Mapping):
            raise ValueError(f"observation {index} must be a mapping")
        observation_id = raw_observation.get("observation_id", index)
        if not isinstance(observation_id, int) or isinstance(observation_id, bool):
            raise ValueError("observation_id must be an integer")
        if observation_id in seen_ids:
            raise ValueError(f"duplicate observation_id {observation_id}")
        seen_ids.add(observation_id)
        if observation_id not in metadata_by_id:
            raise ValueError(f"missing Member 1 metadata for observation {observation_id}")
        member_metadata = dict(metadata_by_id[observation_id])
        for key in ("longitude", "latitude"):
            if key not in member_metadata:
                raise ValueError(
                    f"observation {observation_id} metadata must contain {key}"
                )
        if "timestamp" not in member_metadata and "timestamp" in raw_observation:
            member_metadata["timestamp"] = raw_observation["timestamp"]
        candidates_raw = raw_observation.get("candidates")
        if not isinstance(candidates_raw, Sequence) or isinstance(candidates_raw, (str, bytes)) or not candidates_raw:
            raise ValueError(f"observation {observation_id} must contain candidates")
        candidates: list[ViterbiCandidate] = []
        for candidate_raw in candidates_raw:
            if not isinstance(candidate_raw, Mapping) or "arc_id" not in candidate_raw:
                raise ValueError(f"observation {observation_id} contains an invalid candidate")
            arc_id = candidate_raw["arc_id"]
            if not isinstance(arc_id, int) or isinstance(arc_id, bool):
                raise ValueError("candidate arc_id must be an integer")
            network.arc(arc_id)
            if "ml_probability" not in candidate_raw:
                raise ValueError("candidate must contain ml_probability")
            probability = candidate_raw["ml_probability"]
            if not isinstance(probability, Real) or isinstance(probability, bool):
                raise ValueError("ml_probability must be numeric")
            candidates.append(
                ViterbiCandidate(
                    arc_id=arc_id,
                    ml_probability=float(probability),
                    metadata=dict(candidate_raw),
                )
            )
        timestamp = member_metadata.get("timestamp")
        observations.append(
            ViterbiObservation(
                observation_id=observation_id,
                timestamp=timestamp,
                candidates=tuple(candidates),
                metadata=member_metadata,
            )
        )
    return observations


def _normalise_observation_data(
    observation_data: Mapping[int, Mapping[str, Any]] | Sequence[Mapping[str, Any]],
) -> dict[int, Mapping[str, Any]]:
    if isinstance(observation_data, Mapping):
        result = dict(observation_data)
    else:
        result = {}
        for index, item in enumerate(observation_data):
            if not isinstance(item, Mapping):
                raise ValueError(f"observation metadata {index} must be a mapping")
            observation_id = item.get("observation_id", index)
            result[observation_id] = item
    for observation_id, metadata in result.items():
        if not isinstance(observation_id, int) or isinstance(observation_id, bool):
            raise ValueError("observation metadata keys must be integer observation IDs")
        if not isinstance(metadata, Mapping):
            raise ValueError(f"metadata for observation {observation_id} must be a mapping")
    return result


def _validate_candidate_positions(
    observations: Sequence[ViterbiObservation],
    candidate_positions: Mapping[int, Mapping[int, CandidatePosition]],
    network: RoadNetwork,
) -> None:
    for observation in observations:
        positions = candidate_positions.get(observation.observation_id)
        if positions is None:
            raise ValueError(
                f"missing candidate positions for observation {observation.observation_id}"
            )
        for candidate in observation.candidates:
            position = positions.get(candidate.arc_id)
            if position is None:
                raise ValueError(
                    f"missing candidate position for observation {observation.observation_id}, "
                    f"arc {candidate.arc_id}"
                )
            if not isinstance(position, CandidatePosition):
                raise TypeError("candidate_positions values must be CandidatePosition instances")
            network.arc(candidate.arc_id)
            if position.arc_id != candidate.arc_id:
                raise ValueError("candidate position arc_id does not match candidate arc_id")
            if not math.isclose(
                position.arc_length_m,
                network.arc(candidate.arc_id).length_m,
                rel_tol=1e-9,
                abs_tol=1e-6,
            ):
                raise ValueError("candidate position does not belong to the supplied network")


def _validate_non_negative_finite(value: float, name: str) -> None:
    if not isinstance(value, Real) or isinstance(value, bool) or not math.isfinite(float(value)) or float(value) < 0:
        raise ValueError(f"{name} must be finite and non-negative")


def _validate_positive_finite(value: float, name: str) -> None:
    if not isinstance(value, Real) or isinstance(value, bool) or not math.isfinite(float(value)) or float(value) <= 0:
        raise ValueError(f"{name} must be finite and positive")
