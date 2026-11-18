"""A classifier-independent Viterbi decoder for candidate road segments.

The decoder deliberately knows nothing about how candidates were generated or
how a transition score is calculated.  Those details belong to the data and
road-network components that will be added later.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Callable, Iterator, Mapping, Sequence


class InvalidProbabilityError(ValueError):
    """Raised when a candidate compatibility probability is outside [0, 1]."""


class EmptyCandidateSetError(ValueError):
    """Raised when an observation has no candidates to decode."""


@dataclass(frozen=True)
class ViterbiCandidate:
    """One candidate state at a GPS observation."""

    arc_id: int
    ml_probability: float
    metadata: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class ViterbiObservation:
    """The context passed to a transition scorer for one observation."""

    observation_id: int
    candidates: tuple[ViterbiCandidate, ...]
    timestamp: float | None = None
    metadata: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class ViterbiResult:
    """The best decoded path and its accumulated log score."""

    arc_sequence: tuple[int, ...]
    final_score: float

    @property
    def path(self) -> tuple[int, ...]:
        """Alias that makes the result convenient to use in later modules."""

        return self.arc_sequence

    @property
    def score(self) -> float:
        """Alias for the final accumulated log score."""

        return self.final_score

    def __iter__(self) -> Iterator[object]:
        """Allow ``path, score = decode_viterbi(...)`` when desired."""

        yield self.arc_sequence
        yield self.final_score


TransitionScorer = Callable[
    [ViterbiCandidate, ViterbiCandidate, ViterbiObservation, ViterbiObservation],
    float | None,
]
EmissionScorer = Callable[[ViterbiCandidate, ViterbiObservation], float]


def decode_viterbi(
    predictions: Mapping[str, Any] | Sequence[ViterbiObservation | Mapping[str, Any] | Sequence[Any]],
    transition_score: TransitionScorer,
    *,
    epsilon: float = 1e-12,
    emission_score: EmissionScorer | None = None,
) -> ViterbiResult:
    """Decode the highest-scoring candidate sequence in log space.

    ``predictions`` may be the agreed Member 2 payload (a mapping containing
    ``observations``) or a sequence of observation objects.  Each observation
    is expected to contain ``candidates``.  A candidate mapping must contain
    ``arc_id`` and ``ml_probability``; ``probability`` and
    ``compatibility_score`` are accepted as neutral aliases for integration
    with earlier preprocessing code.

    The transition scorer returns a log transition score.  Returning ``None``
    or ``-math.inf`` marks a transition as impossible.  It receives both
    candidate states and both observation contexts, so timestamps and future
    road-network metadata can be used without changing this API.

    Ties are resolved by keeping the first candidate encountered.  This makes
    the result reproducible and preserves the input ordering.
    """

    _validate_epsilon(epsilon)
    observations = _normalise_observations(predictions)

    def score_emission(candidate: ViterbiCandidate, observation: ViterbiObservation) -> float:
        score = (
            _log_compatibility(candidate.ml_probability, epsilon)
            if emission_score is None
            else emission_score(candidate, observation)
        )
        return _validate_emission_score(score)

    first = observations[0]
    scores = [
        score_emission(candidate, first)
        for candidate in first.candidates
    ]
    backpointers: list[list[int]] = []

    for current_observation in observations[1:]:
        previous_observation = observations[len(backpointers)]
        next_scores = [-math.inf] * len(current_observation.candidates)
        previous_indices = [-1] * len(current_observation.candidates)

        for current_index, current_candidate in enumerate(current_observation.candidates):
            emission = score_emission(current_candidate, current_observation)
            best_score = -math.inf
            best_previous_index = -1

            for previous_index, previous_candidate in enumerate(previous_observation.candidates):
                if scores[previous_index] == -math.inf:
                    continue
                transition = transition_score(
                    previous_candidate,
                    current_candidate,
                    previous_observation,
                    current_observation,
                )
                transition = _validate_transition_score(transition)
                if transition == -math.inf:
                    continue

                candidate_score = scores[previous_index] + transition + emission
                # Strict comparison intentionally keeps the earliest path on ties.
                if candidate_score > best_score:
                    best_score = candidate_score
                    best_previous_index = previous_index

            next_scores[current_index] = best_score
            previous_indices[current_index] = best_previous_index

        scores = next_scores
        backpointers.append(previous_indices)

    best_index = max(range(len(scores)), key=lambda index: (scores[index], -index))
    final_score = scores[best_index]
    if final_score == -math.inf:
        raise ValueError("No valid path exists across the supplied observations")

    path_indices = [best_index]
    for previous_indices in reversed(backpointers):
        best_index = previous_indices[best_index]
        path_indices.append(best_index)
    path_indices.reverse()

    return ViterbiResult(
        arc_sequence=tuple(
            observation.candidates[index].arc_id
            for observation, index in zip(observations, path_indices)
        ),
        final_score=final_score,
    )


def _normalise_observations(
    predictions: Mapping[str, Any] | Sequence[ViterbiObservation | Mapping[str, Any] | Sequence[Any]],
) -> list[ViterbiObservation]:
    if isinstance(predictions, Mapping):
        raw_observations = predictions.get("observations")
        if raw_observations is None:
            raise ValueError("Prediction payload must contain an 'observations' field")
    else:
        raw_observations = predictions

    observations: list[ViterbiObservation] = []
    for index, raw_observation in enumerate(raw_observations):
        if isinstance(raw_observation, ViterbiObservation):
            observation = raw_observation
        elif isinstance(raw_observation, Mapping):
            raw_candidates = raw_observation.get("candidates")
            if raw_candidates is None:
                raise ValueError(f"Observation {index} must contain a 'candidates' field")
            observation = ViterbiObservation(
                observation_id=int(raw_observation.get("observation_id", index)),
                timestamp=raw_observation.get("timestamp"),
                metadata=raw_observation,
                candidates=tuple(_normalise_candidate(candidate) for candidate in raw_candidates),
            )
        else:
            observation = ViterbiObservation(
                observation_id=index,
                candidates=tuple(_normalise_candidate(candidate) for candidate in raw_observation),
            )
        if not observation.candidates:
            raise EmptyCandidateSetError(f"Observation {observation.observation_id} has no candidates")
        observations.append(observation)

    if not observations:
        raise ValueError("At least one observation is required")
    return observations


def _normalise_candidate(candidate: ViterbiCandidate | Mapping[str, Any]) -> ViterbiCandidate:
    if isinstance(candidate, ViterbiCandidate):
        return candidate
    if not isinstance(candidate, Mapping) or "arc_id" not in candidate:
        raise ValueError("Each candidate must contain an 'arc_id' field")

    probability_key = next(
        (key for key in ("ml_probability", "compatibility_score", "probability") if key in candidate),
        None,
    )
    if probability_key is None:
        raise ValueError("Each candidate must contain a compatibility probability")
    return ViterbiCandidate(
        arc_id=int(candidate["arc_id"]),
        ml_probability=float(candidate[probability_key]),
        metadata=candidate,
    )


def _validate_epsilon(epsilon: float) -> None:
    if not math.isfinite(epsilon) or not 0 < epsilon <= 1:
        raise ValueError("epsilon must be finite and satisfy 0 < epsilon <= 1")


def _log_compatibility(probability: float, epsilon: float) -> float:
    if not math.isfinite(probability) or probability < 0 or probability > 1:
        raise InvalidProbabilityError(
            f"Candidate probability must be finite and within [0, 1], got {probability!r}"
        )
    return math.log(max(probability, epsilon))


def _validate_transition_score(score: float | None) -> float:
    if score is None:
        return -math.inf
    if math.isnan(score) or score == math.inf:
        raise ValueError("Transition scores must be finite or -math.inf")
    return score


def _validate_emission_score(score: float) -> float:
    if not isinstance(score, (int, float)) or math.isnan(score) or score == math.inf:
        raise ValueError("Emission scores must be finite or -math.inf")
    return float(score)
