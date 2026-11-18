import math
import unittest
from types import SimpleNamespace

from src.decoder.hmm import HMMConfig, match_trip
from src.decoder.route_reconstruction import (
    RouteReconstructionError,
    reconstruct_route,
)
from src.decoder.transition import CandidatePosition, build_road_network


class RouteReconstructionTests(unittest.TestCase):
    def setUp(self):
        self.network = build_road_network(
            [(0.0, 0.0), (0.001, 0.0), (0.002, 0.0), (0.003, 0.0), (0.001, 0.001)],
            [(0, 1), (1, 2), (2, 3), (1, 4), (4, 1), (4, 0)],
            trip_id="trip-1",
        )

    def result(self, arcs, observations=None, trip_id="trip-1"):
        observations = observations or tuple(range(len(arcs)))
        return SimpleNamespace(
            trip_id=trip_id,
            arc_sequence=tuple(arcs),
            observation_ids=tuple(observations),
        )

    def positions(self, entries):
        return {
            observation_id: {
                arc_id: self.network.candidate_position(arc_id, offset)
                for arc_id, offset in candidates
            }
            for observation_id, candidates in entries.items()
        }

    def test_direct_and_intermediate_paths(self):
        positions = self.positions({
            0: [(0, 10.0)],
            1: [(2, 20.0)],
        })
        route = reconstruct_route(self.result((0, 2)), self.network, positions)
        self.assertEqual(route.route_arc_sequence, (0, 1, 2))
        self.assertEqual(route.selected_arc_sequence, (0, 2))
        self.assertGreater(route.total_distance_m, 0.0)

    def test_same_arc_forward_merges_boundary_occurrences(self):
        positions = self.positions({0: [(0, 10.0)], 1: [(0, 30.0)], 2: [(1, 10.0)]})
        route = reconstruct_route(self.result((0, 0, 1)), self.network, positions)
        self.assertEqual(route.route_arc_sequence, (0, 1))
        self.assertAlmostEqual(
            route.traversals[0].distance_m,
            self.network.arc(0).length_m - 10.0,
            delta=1e-6,
        )

    def test_same_arc_backward_without_cycle_is_rejected(self):
        acyclic = build_road_network(
            [(0.0, 0.0), (0.001, 0.0)], [(0, 1)], trip_id="trip-1"
        )
        positions = {
            0: {0: acyclic.candidate_position(0, 80.0)},
            1: {0: acyclic.candidate_position(0, 20.0)},
        }
        with self.assertRaises(RouteReconstructionError):
            reconstruct_route(self.result((0, 0)), acyclic, positions)

    def test_same_arc_backward_cycle_preserves_repeated_arc(self):
        positions = self.positions({0: [(0, 80.0)], 1: [(0, 20.0)]})
        route = reconstruct_route(self.result((0, 0)), self.network, positions)
        self.assertEqual(route.route_arc_sequence, (0, 3, 5, 0))
        self.assertGreaterEqual(route.reconstructed_arc_count, 4)

    def test_partial_first_and_last_arc_distances(self):
        positions = self.positions({0: [(0, 20.0)], 1: [(2, 30.0)]})
        route = reconstruct_route(self.result((0, 2)), self.network, positions)
        self.assertAlmostEqual(route.traversals[0].distance_from_start_m, 20.0)
        self.assertAlmostEqual(route.traversals[-1].distance_to_end_m, 30.0)
        expected = self.network.arc(0).length_m - 20.0 + self.network.arc(1).length_m + 30.0
        self.assertAlmostEqual(route.total_distance_m, expected, delta=1e-6)

    def test_parallel_arcs_keep_the_selected_arc_id(self):
        network = build_road_network([(0.0, 0.0), (0.001, 0.0), (0.002, 0.0)], [(0, 1), (0, 1), (1, 2)], trip_id="parallel")
        positions = {
            0: {1: network.candidate_position(1, 1.0)},
            1: {2: network.candidate_position(2, 1.0)},
        }
        route = reconstruct_route(self.result((1, 2), (0, 1), "parallel"), network, positions)
        self.assertEqual(route.route_arc_sequence, (1, 2))

    def test_single_observation_returns_zero_distance_interval(self):
        positions = self.positions({0: [(1, 5.0)]})
        route = reconstruct_route(self.result((1,)), self.network, positions)
        self.assertEqual(route.route_arc_sequence, (1,))
        self.assertEqual(route.total_distance_m, 0.0)

    def test_disconnected_unknown_and_missing_positions_are_rejected(self):
        disconnected = build_road_network([(0.0, 0.0), (0.001, 0.0), (0.0, 1.0), (0.001, 1.0)], [(0, 1), (2, 3)], trip_id="trip-1")
        positions = {
            0: {0: disconnected.candidate_position(0, 1.0)},
            1: {1: CandidatePosition(1, 1.0, disconnected.arc(1).length_m)},
        }
        with self.assertRaises(RouteReconstructionError):
            reconstruct_route(self.result((0, 1)), disconnected, positions)
        with self.assertRaises(RouteReconstructionError):
            reconstruct_route(self.result((99,)), self.network, {})
        with self.assertRaises(RouteReconstructionError):
            reconstruct_route(self.result((0, 1)), self.network, {0: positions[0]})

    def test_alignment_and_trip_identity_are_validated(self):
        positions = self.positions({0: [(0, 1.0)], 1: [(1, 1.0)]})
        with self.assertRaises(RouteReconstructionError):
            reconstruct_route(self.result((0, 1), (0,)), self.network, positions)
        with self.assertRaises(RouteReconstructionError):
            reconstruct_route(self.result((0,), (0,), "other"), self.network, {0: positions[0]})

    def test_deterministic_path_and_no_global_deduplication(self):
        positions = self.positions({0: [(0, 80.0)], 1: [(0, 20.0)]})
        route_one = reconstruct_route(self.result((0, 0)), self.network, positions)
        route_two = reconstruct_route(self.result((0, 0)), self.network, positions)
        self.assertEqual(route_one.route_arc_sequence, route_two.route_arc_sequence)
        self.assertEqual(route_one.route_arc_sequence.count(0), 2)

    def test_no_input_mutation_and_hmm_integration(self):
        predictions = {
            "trip_id": "trip-1",
            "observations": [
                {"observation_id": 0, "candidates": [{"arc_id": 0, "ml_probability": 1.0}]},
                {"observation_id": 1, "candidates": [{"arc_id": 2, "ml_probability": 1.0}]},
            ],
        }
        observations = [
            {"observation_id": 0, "longitude": 0.0001, "latitude": 0.0, "timestamp": 0.0},
            {"observation_id": 1, "longitude": 0.0021, "latitude": 0.0, "timestamp": 1.0},
        ]
        positions = self.positions({0: [(0, 10.0)], 1: [(2, 10.0)]})
        original = repr(predictions)
        from src.decoder.hmm import match_trip
        hmm_result = match_trip(predictions, self.network, observations, positions, HMMConfig())
        route = reconstruct_route(hmm_result, self.network, positions)
        self.assertEqual(route.route_arc_sequence, (0, 1, 2))
        self.assertEqual(repr(predictions), original)

    def test_invalid_position_network_is_rejected(self):
        positions = {0: {0: CandidatePosition(0, 1.0, 999.0)}}
        with self.assertRaises(RouteReconstructionError):
            reconstruct_route(self.result((0,)), self.network, positions)


if __name__ == "__main__":
    unittest.main()
