"""Evaluation utilities for completed PathSnap map-matching runs.

Reference routes enter this module only after prediction and reconstruction are
complete.  No evaluator ground truth is used by candidate selection, HMM
decoding, or route reconstruction.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, replace
import csv
import math
from pathlib import Path
import statistics
import time
from collections.abc import Mapping, Sequence
from numbers import Real
from typing import Any

from .hmm import HMMConfig, HMMResult, match_trip
from .route_reconstruction import RouteReconstructionError, ReconstructedRoute, reconstruct_route
from .transition import CandidatePosition, RoadNetwork
from .viterbi import ViterbiObservation
from .emission import ml_log_emission


@dataclass(frozen=True)
class ClassificationMetrics:
    """Precision/recall/F1 counts for one sequence-comparison convention."""

    true_positive: int
    false_positive: int
    false_negative: int
    precision: float
    recall: float
    f1: float


@dataclass(frozen=True)
class RouteLengthMetrics:
    """Length comparison in metres; relative error is undefined for zero reference length."""

    predicted_length_m: float
    reference_length_m: float
    absolute_error_m: float
    relative_error: float | None


@dataclass(frozen=True)
class BaselineSelection:
    """Independent ML-only candidate selection before route reconstruction."""

    trip_id: str | None
    observation_ids: tuple[int, ...]
    arc_sequence: tuple[int, ...]
    selection_time_s: float


@dataclass(frozen=True)
class EvaluationMethodResult:
    """Per-method result, including failures that have no route metrics."""

    trip_id: str | None
    method: str
    observation_count: int
    selected_arc_sequence: tuple[int, ...]
    reconstructed_arc_sequence: tuple[int, ...]
    reconstruction_succeeded: bool
    failure_reason: str | None
    unique_metrics: ClassificationMetrics | None
    occurrence_metrics: ClassificationMetrics | None
    lcs_metrics: ClassificationMetrics | None
    length_metrics: RouteLengthMetrics | None
    selection_time_s: float
    decoding_time_s: float
    route_reconstruction_time_s: float
    total_time_s: float


@dataclass(frozen=True)
class TripEvaluation:
    """All comparison methods evaluated for one trip."""

    trip_id: str | None
    reference_arc_sequence: tuple[int, ...]
    methods: tuple[EvaluationMethodResult, ...]


@dataclass(frozen=True)
class AggregateMethodResult:
    """Aggregate statistics with explicit valid and failed denominators."""

    method: str
    trip_count: int
    successful_trip_count: int
    failed_trip_count: int
    reconstruction_success_rate: float
    valid_unique_f1_count: int
    mean_unique_f1: float | None
    median_unique_f1: float | None
    valid_occurrence_f1_count: int
    mean_occurrence_f1: float | None
    median_occurrence_f1: float | None
    valid_lcs_f1_count: int
    mean_lcs_f1: float | None
    median_lcs_f1: float | None
    valid_length_error_count: int
    mean_absolute_length_error_m: float | None
    median_absolute_length_error_m: float | None
    mean_total_time_s: float | None
    median_total_time_s: float | None


def unique_arc_metrics(
    predicted_arc_sequence: Sequence[int], reference_arc_sequence: Sequence[int]
) -> ClassificationMetrics:
    """Compare unique arc-ID sets, ignoring order and repeated traversals."""

    predicted = set(predicted_arc_sequence)
    reference = set(reference_arc_sequence)
    return _classification_metrics(len(predicted & reference), len(predicted - reference), len(reference - predicted))


def occurrence_arc_metrics(
    predicted_arc_sequence: Sequence[int], reference_arc_sequence: Sequence[int]
) -> ClassificationMetrics:
    """Compare arc-ID multisets, preserving repeated occurrences."""

    predicted = Counter(predicted_arc_sequence)
    reference = Counter(reference_arc_sequence)
    true_positive = sum((predicted & reference).values())
    false_positive = sum((predicted - reference).values())
    false_negative = sum((reference - predicted).values())
    return _classification_metrics(true_positive, false_positive, false_negative)


def lcs_arc_metrics(
    predicted_arc_sequence: Sequence[int], reference_arc_sequence: Sequence[int]
) -> ClassificationMetrics:
    """Compare traversal order using LCS precision/recall/F1.

    Repeated arc IDs remain repeated. The implementation uses one DP row, so it
    uses O(min(n, m)) memory and O(n*m) time. It measures arc-order agreement,
    not exact geometric trajectory similarity.
    """

    first = tuple(predicted_arc_sequence)
    second = tuple(reference_arc_sequence)
    if len(second) > len(first):
        first, second = second, first
    previous = [0] * (len(second) + 1)
    for item in first:
        current = [0]
        for index, other in enumerate(second, start=1):
            if item == other:
                current.append(previous[index - 1] + 1)
            else:
                current.append(max(previous[index], current[-1]))
        previous = current
    return _classification_metrics(previous[-1], len(predicted_arc_sequence) - previous[-1], len(reference_arc_sequence) - previous[-1])


def route_length_metrics(predicted_length_m: float, reference_length_m: float) -> RouteLengthMetrics:
    """Compare predicted and reference route lengths in metres."""

    _validate_distance(predicted_length_m, "predicted_length_m")
    _validate_distance(reference_length_m, "reference_length_m")
    absolute_error = abs(float(predicted_length_m) - float(reference_length_m))
    relative_error = None if reference_length_m == 0 else absolute_error / float(reference_length_m)
    return RouteLengthMetrics(float(predicted_length_m), float(reference_length_m), absolute_error, relative_error)


def reference_route_length_m(route_arc_sequence: Sequence[int], network: RoadNetwork) -> float:
    """Sum full trip-specific arc lengths, preserving repeated route occurrences."""

    total = 0.0
    for arc_id in route_arc_sequence:
        total += network.arc(arc_id).length_m
    return total


def load_reference_route(path: str | Path, network: RoadNetwork) -> tuple[int, ...]:
    """Load and validate a whitespace-separated one-arc-ID-per-line route file."""

    route_path = Path(path)
    if not route_path.is_file():
        raise FileNotFoundError(f"reference route file not found: {route_path}")
    arc_ids: list[int] = []
    for line_number, line in enumerate(route_path.read_text(encoding="utf-8").splitlines(), start=1):
        value = line.strip()
        if not value:
            continue
        parts = value.split()
        if len(parts) != 1:
            raise ValueError(f"malformed reference route line {line_number}")
        try:
            arc_id = int(parts[0])
        except ValueError as error:
            raise ValueError(f"invalid arc ID on reference route line {line_number}") from error
        try:
            network.arc(arc_id)
        except KeyError as error:
            raise ValueError(f"reference route contains unknown arc_id {arc_id}") from error
        arc_ids.append(arc_id)
    return tuple(arc_ids)


def select_ml_baseline(predictions: Mapping[str, Any]) -> BaselineSelection:
    """Select each observation's highest ML probability independently.

    Ties keep the first candidate in input order. This function never consults
    transitions and never repairs disconnected selections.
    """

    if not isinstance(predictions, Mapping) or not isinstance(predictions.get("observations"), Sequence):
        raise ValueError("predictions must contain an observations sequence")
    start = time.perf_counter()
    observation_ids: list[int] = []
    selected: list[int] = []
    for index, observation in enumerate(predictions["observations"]):
        if not isinstance(observation, Mapping):
            raise ValueError(f"observation {index} must be a mapping")
        observation_id = observation.get("observation_id", index)
        candidates = observation.get("candidates")
        if not isinstance(candidates, Sequence) or not candidates:
            raise ValueError(f"observation {observation_id} has no candidates")
        best_arc: int | None = None
        best_probability = -math.inf
        for candidate in candidates:
            if not isinstance(candidate, Mapping) or "arc_id" not in candidate or "ml_probability" not in candidate:
                raise ValueError("each candidate must contain arc_id and ml_probability")
            probability = float(candidate["ml_probability"])
            ml_log_emission(probability)
            if probability > best_probability:
                best_probability = probability
                best_arc = int(candidate["arc_id"])
        observation_ids.append(int(observation_id))
        selected.append(best_arc)  # type: ignore[arg-type]
    return BaselineSelection(predictions.get("trip_id"), tuple(observation_ids), tuple(selected), time.perf_counter() - start)


def evaluate_trip(
    predictions: Mapping[str, Any],
    road_network: RoadNetwork,
    observation_data: Mapping[int, Mapping[str, Any]] | Sequence[Mapping[str, Any]],
    candidate_positions: Mapping[int, Mapping[int, CandidatePosition]],
    reference_arc_sequence: Sequence[int],
    *,
    config: HMMConfig | None = None,
) -> TripEvaluation:
    """Evaluate ML-only, distance-HMM, and ML-HMM methods for one trip.

    The reference sequence is consumed only after method inputs are prepared;
    callers should load it after prediction and reconstruction data is ready.
    """

    config = config or HMMConfig()
    reference = tuple(reference_arc_sequence)
    for arc_id in reference:
        road_network.arc(arc_id)
    methods: list[EvaluationMethodResult] = []

    baseline = select_ml_baseline(predictions)
    baseline_result = HMMResult(
        trip_id=baseline.trip_id,
        observation_ids=baseline.observation_ids,
        arc_sequence=baseline.arc_sequence,
        final_score=0.0,
        emission_mode="ml_only",
        configuration={"method": "ml_only"},
    )
    methods.append(_evaluate_decoded_method("ml_only", baseline_result, baseline.selection_time_s, road_network, candidate_positions, reference))

    methods.append(_run_hmm_method("distance_hmm", predictions, road_network, observation_data, candidate_positions, reference, replace(config, emission_mode="distance")))
    methods.append(_run_hmm_method("ml_hmm", predictions, road_network, observation_data, candidate_positions, reference, replace(config, emission_mode="ml")))
    return TripEvaluation(predictions.get("trip_id", road_network.trip_id), reference, tuple(methods))


def aggregate_evaluations(evaluations: Sequence[TripEvaluation]) -> tuple[AggregateMethodResult, ...]:
    """Aggregate per-method results without hiding failed reconstructions."""

    methods = sorted({method.method for evaluation in evaluations for method in evaluation.methods})
    aggregates: list[AggregateMethodResult] = []
    for method_name in methods:
        rows = [method for evaluation in evaluations for method in evaluation.methods if method.method == method_name]
        successful = [row for row in rows if row.reconstruction_succeeded]
        unique = [row.unique_metrics.f1 for row in successful if row.unique_metrics is not None]
        occurrence = [row.occurrence_metrics.f1 for row in successful if row.occurrence_metrics is not None]
        lcs = [row.lcs_metrics.f1 for row in successful if row.lcs_metrics is not None]
        errors = [row.length_metrics.absolute_error_m for row in successful if row.length_metrics is not None]
        times = [row.total_time_s for row in rows]
        aggregates.append(AggregateMethodResult(
            method=method_name,
            trip_count=len(rows),
            successful_trip_count=len(successful),
            failed_trip_count=len(rows) - len(successful),
            reconstruction_success_rate=len(successful) / len(rows) if rows else 0.0,
            valid_unique_f1_count=len(unique),
            mean_unique_f1=_mean_or_none(unique),
            median_unique_f1=_median_or_none(unique),
            valid_occurrence_f1_count=len(occurrence),
            mean_occurrence_f1=_mean_or_none(occurrence),
            median_occurrence_f1=_median_or_none(occurrence),
            valid_lcs_f1_count=len(lcs),
            mean_lcs_f1=_mean_or_none(lcs),
            median_lcs_f1=_median_or_none(lcs),
            valid_length_error_count=len(errors),
            mean_absolute_length_error_m=_mean_or_none(errors),
            median_absolute_length_error_m=_median_or_none(errors),
            mean_total_time_s=_mean_or_none(times),
            median_total_time_s=_median_or_none(times),
        ))
    return tuple(aggregates)


def export_trip_results_csv(evaluations: Sequence[TripEvaluation], path: str | Path) -> None:
    """Write deterministic per-method CSV rows without modifying source data."""

    fields = [
        "trip_id", "method", "observation_count", "selected_candidate_count", "reconstructed_arc_count",
        "reconstruction_succeeded", "failure_reason", "unique_precision", "unique_recall", "unique_f1",
        "occurrence_precision", "occurrence_recall", "occurrence_f1", "lcs_precision", "lcs_recall", "lcs_f1",
        "predicted_length_m", "reference_length_m", "absolute_length_error_m", "relative_length_error",
        "selection_time_s", "decoding_time_s", "route_reconstruction_time_s", "total_time_s",
    ]
    rows = []
    for evaluation in sorted(evaluations, key=lambda item: str(item.trip_id)):
        for method in sorted(evaluation.methods, key=lambda item: item.method):
            row = {
                "trip_id": evaluation.trip_id, "method": method.method,
                "observation_count": method.observation_count,
                "selected_candidate_count": len(method.selected_arc_sequence),
                "reconstructed_arc_count": len(method.reconstructed_arc_sequence),
                "reconstruction_succeeded": method.reconstruction_succeeded,
                "failure_reason": method.failure_reason,
                "selection_time_s": method.selection_time_s, "decoding_time_s": method.decoding_time_s,
                "route_reconstruction_time_s": method.route_reconstruction_time_s, "total_time_s": method.total_time_s,
            }
            row.update(_metric_row("unique", method.unique_metrics))
            row.update(_metric_row("occurrence", method.occurrence_metrics))
            row.update(_metric_row("lcs", method.lcs_metrics))
            row.update(_length_row(method.length_metrics))
            rows.append(row)
    _write_csv(path, fields, rows)


def export_aggregate_results_csv(aggregates: Sequence[AggregateMethodResult], path: str | Path) -> None:
    """Write deterministic aggregate method-comparison CSV rows."""

    fields = list(AggregateMethodResult.__dataclass_fields__)
    _write_csv(path, fields, [dict((field, getattr(row, field)) for field in fields) for row in sorted(aggregates, key=lambda item: item.method)])


def _run_hmm_method(method: str, predictions, network, observation_data, positions, reference, config):
    start = time.perf_counter()
    try:
        hmm_result = match_trip(predictions, network, observation_data, positions, config)
    except (ValueError, KeyError, RouteReconstructionError) as error:
        elapsed = time.perf_counter() - start
        return EvaluationMethodResult(predictions.get("trip_id", network.trip_id), method, len(predictions.get("observations", ())), (), (), False, str(error), None, None, None, None, 0.0, elapsed, 0.0, elapsed)
    decoding = time.perf_counter() - start
    return _evaluate_decoded_method(method, hmm_result, decoding, network, positions, reference, decoding_is_total=True)


def _evaluate_decoded_method(method, hmm_result, decoding_time, network, positions, reference, decoding_is_total=False):
    reconstruction_start = time.perf_counter()
    try:
        route = reconstruct_route(hmm_result, network, positions)
    except (ValueError, KeyError, RouteReconstructionError) as error:
        reconstruction_time = time.perf_counter() - reconstruction_start
        return EvaluationMethodResult(hmm_result.trip_id, method, len(hmm_result.observation_ids), tuple(hmm_result.arc_sequence), (), False, str(error), None, None, None, None, decoding_time if not decoding_is_total else 0.0, decoding_time if decoding_is_total else 0.0, reconstruction_time, decoding_time + reconstruction_time)
    reconstruction_time = time.perf_counter() - reconstruction_start
    length = route_length_metrics(route.total_distance_m, reference_route_length_m(reference, network))
    return EvaluationMethodResult(
        hmm_result.trip_id, method, len(hmm_result.observation_ids), tuple(hmm_result.arc_sequence), route.route_arc_sequence, True, None,
        unique_arc_metrics(route.route_arc_sequence, reference), occurrence_arc_metrics(route.route_arc_sequence, reference), lcs_arc_metrics(route.route_arc_sequence, reference), length,
        decoding_time if not decoding_is_total else 0.0, decoding_time if decoding_is_total else 0.0, reconstruction_time, decoding_time + reconstruction_time,
    )


def _classification_metrics(true_positive, false_positive, false_negative):
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return ClassificationMetrics(true_positive, false_positive, false_negative, precision, recall, f1)


def _validate_distance(value, name):
    if not isinstance(value, Real) or isinstance(value, bool) or not math.isfinite(float(value)) or float(value) < 0:
        raise ValueError(f"{name} must be finite and non-negative")


def _mean_or_none(values):
    return statistics.mean(values) if values else None


def _median_or_none(values):
    return statistics.median(values) if values else None


def _metric_row(prefix, metrics):
    return {f"{prefix}_{field}": getattr(metrics, field) if metrics else None for field in ("precision", "recall", "f1")}


def _length_row(metrics):
    return {
        "predicted_length_m": metrics.predicted_length_m if metrics else None,
        "reference_length_m": metrics.reference_length_m if metrics else None,
        "absolute_length_error_m": metrics.absolute_error_m if metrics else None,
        "relative_length_error": metrics.relative_error if metrics else None,
    }


def _write_csv(path, fields, rows):
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
