import csv
import math
import unittest
from pathlib import Path

from src.decoder.evaluation import (
    AggregateMethodResult,
    evaluate_trip,
    export_aggregate_results_csv,
    export_trip_results_csv,
    lcs_arc_metrics,
    load_reference_route,
    occurrence_arc_metrics,
    reference_route_length_m,
    route_length_metrics,
    select_ml_baseline,
    unique_arc_metrics,
)
from src.decoder.hmm import HMMConfig
from src.decoder.transition import build_road_network


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.network = build_road_network(
            [(0.0, 0.0), (0.001, 0.0), (0.002, 0.0)],
            [(0, 1), (1, 2)],
            trip_id="trip-1",
        )
        self.predictions = {
            "trip_id": "trip-1",
            "observations": [
                {"observation_id": 0, "candidates": [{"arc_id": 0, "ml_probability": 0.9}, {"arc_id": 1, "ml_probability": 0.1}]},
                {"observation_id": 1, "candidates": [{"arc_id": 1, "ml_probability": 0.9}, {"arc_id": 0, "ml_probability": 0.1}]},
            ],
        }
        self.observations = [
            {"observation_id": 0, "longitude": 0.0001, "latitude": 0.0, "timestamp": 0.0},
            {"observation_id": 1, "longitude": 0.0011, "latitude": 0.0, "timestamp": 1.0},
        ]
        self.positions = {
            0: {0: self.network.candidate_position(0, 10.0), 1: self.network.candidate_position(1, 1.0)},
            1: {1: self.network.candidate_position(1, 10.0), 0: self.network.candidate_position(0, 10.0)},
        }

    def test_unique_occurrence_and_lcs_metrics(self):
        self.assertEqual(unique_arc_metrics((1, 2, 2), (2, 3)).true_positive, 1)
        occurrence = occurrence_arc_metrics((1, 2, 2), (2, 2, 3))
        self.assertEqual((occurrence.true_positive, occurrence.false_positive, occurrence.false_negative), (2, 1, 1))
        lcs = lcs_arc_metrics((1, 3, 2, 4, 5), (1, 2, 3, 4, 5))
        self.assertEqual(lcs.true_positive, 4)
        self.assertLess(lcs.f1, 1.0)

    def test_empty_metrics_are_safe(self):
        for function in (unique_arc_metrics, occurrence_arc_metrics, lcs_arc_metrics):
            metrics = function((), ())
            self.assertEqual((metrics.true_positive, metrics.false_positive, metrics.false_negative), (0, 0, 0))
            self.assertEqual(metrics.f1, 0.0)
        self.assertEqual(route_length_metrics(0.0, 0.0).relative_error, None)

    def test_route_length_and_invalid_distances(self):
        metrics = route_length_metrics(120.0, 100.0)
        self.assertEqual(metrics.absolute_error_m, 20.0)
        self.assertEqual(metrics.relative_error, 0.2)
        with self.assertRaises(ValueError):
            route_length_metrics(-1.0, 1.0)
        self.assertAlmostEqual(reference_route_length_m((0, 0, 1), self.network), 2 * self.network.arc(0).length_m + self.network.arc(1).length_m)

    def test_reference_route_loader_preserves_repetitions_and_validates(self):
        path = Path("tests") / "_evaluation_trip.route"
        try:
            path.write_text("0\n1\n0\n", encoding="utf-8")
            self.assertEqual(load_reference_route(path, self.network), (0, 1, 0))
            path.write_text("9\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                load_reference_route(path, self.network)
            path.write_text("0 1\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                load_reference_route(path, self.network)
        finally:
            if path.exists():
                path.unlink()

    def test_ml_baseline_is_deterministic_and_does_not_mutate(self):
        original = repr(self.predictions)
        result = select_ml_baseline(self.predictions)
        self.assertEqual(result.arc_sequence, (0, 1))
        self.assertEqual(result.observation_ids, (0, 1))
        self.assertEqual(repr(self.predictions), original)
        tied = {"observations": [{"candidates": [{"arc_id": 1, "ml_probability": 0.5}, {"arc_id": 0, "ml_probability": 0.5}]}]}
        self.assertEqual(select_ml_baseline(tied).arc_sequence, (1,))

    def test_end_to_end_methods_and_result_structure(self):
        evaluation = evaluate_trip(
            self.predictions, self.network, self.observations, self.positions, (0, 1),
        )
        self.assertEqual(evaluation.trip_id, "trip-1")
        self.assertEqual({row.method for row in evaluation.methods}, {"ml_only", "distance_hmm", "ml_hmm"})
        self.assertTrue(all(row.reconstruction_succeeded for row in evaluation.methods))
        self.assertTrue(all(row.unique_metrics is not None for row in evaluation.methods))

    def test_failed_baseline_is_reported_without_repair(self):
        predictions = {
            "trip_id": "trip-1",
            "observations": [
                {"observation_id": 0, "candidates": [{"arc_id": 1, "ml_probability": 1.0}]},
                {"observation_id": 1, "candidates": [{"arc_id": 0, "ml_probability": 1.0}]},
            ],
        }
        positions = {
            0: {1: self.network.candidate_position(1, 10.0)},
            1: {0: self.network.candidate_position(0, 10.0)},
        }
        evaluation = evaluate_trip(predictions, self.network, self.observations, positions, (0, 1))
        baseline = next(row for row in evaluation.methods if row.method == "ml_only")
        self.assertEqual(baseline.selected_arc_sequence, (1, 0))
        self.assertFalse(baseline.reconstruction_succeeded)
        self.assertIsNone(baseline.unique_metrics)

    def test_aggregate_and_csv_export(self):
        evaluation = evaluate_trip(self.predictions, self.network, self.observations, self.positions, (0, 1))
        from src.decoder.evaluation import aggregate_evaluations
        aggregates = aggregate_evaluations((evaluation,))
        self.assertEqual(len(aggregates), 3)
        self.assertEqual(aggregates[0].trip_count, 1)
        trip_csv = Path("tests") / "_evaluation_trip.csv"
        aggregate_csv = Path("tests") / "_evaluation_aggregate.csv"
        try:
            export_trip_results_csv((evaluation,), trip_csv)
            export_aggregate_results_csv(aggregates, aggregate_csv)
            with trip_csv.open(newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 3)
            self.assertIn("unique_f1", rows[0])
            with aggregate_csv.open(newline="", encoding="utf-8") as stream:
                self.assertEqual(len(list(csv.DictReader(stream))), 3)
        finally:
            for path in (trip_csv, aggregate_csv):
                if path.exists():
                    path.unlink()


if __name__ == "__main__":
    unittest.main()
