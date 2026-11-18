import math
import unittest

from src.decoder.transition import (
    CandidatePosition,
    RoadNetworkTransitionScorer,
    build_road_network,
    geodesic_distance_m,
)
from src.decoder.viterbi import ViterbiCandidate, ViterbiObservation, decode_viterbi


class TransitionTests(unittest.TestCase):
    def setUp(self):
        # Each coordinate step is approximately 111 metres at the equator.
        self.nodes = [(0.0, 0.0), (0.001, 0.0), (0.002, 0.0), (0.003, 0.0)]

    def test_graph_construction_and_arc_lengths(self):
        network = build_road_network(self.nodes, [(0, 1), (1, 2)], trip_id="trip-a")
        self.assertEqual(network.trip_id, "trip-a")
        self.assertEqual(network.arc(0).source, 0)
        self.assertAlmostEqual(network.arc(0).length_m, 111.195, delta=0.2)

    def test_directed_connectivity_and_parallel_arcs(self):
        network = build_road_network(self.nodes[:3], [(0, 1), (0, 1), (1, 2)])
        self.assertEqual(network.graph.number_of_edges(0, 1), 2)
        self.assertIsNotNone(network.shortest_node_path(0, 2))
        self.assertIsNone(network.shortest_node_path(2, 0))
        self.assertEqual(network.shortest_node_path(0, 2).arc_ids, (0, 2))

    def test_same_arc_forward_movement(self):
        network = build_road_network(self.nodes[:2], [(0, 1)])
        start = network.candidate_position(0, 20.0)
        end = network.candidate_position(0, 70.0)
        path = network.path_between_positions(start, end)
        self.assertAlmostEqual(path.distance_m, 50.0)
        self.assertEqual(path.arc_ids, (0,))

    def test_same_arc_backward_requires_cycle(self):
        no_cycle = build_road_network(self.nodes[:2], [(0, 1)])
        self.assertIsNone(
            no_cycle.path_between_positions(
                no_cycle.candidate_position(0, 70.0),
                no_cycle.candidate_position(0, 20.0),
            )
        )
        cycle = build_road_network(self.nodes[:3], [(0, 1), (1, 2), (2, 0)])
        path = cycle.path_between_positions(
            cycle.candidate_position(0, 70.0), cycle.candidate_position(0, 20.0)
        )
        self.assertEqual(path.arc_ids, (0, 1, 2, 0))
        self.assertGreater(path.distance_m, 0.0)

    def test_connected_and_intermediate_arcs(self):
        network = build_road_network(self.nodes, [(0, 1), (1, 2), (2, 3)])
        path = network.path_between_positions(
            network.candidate_position(0, 50.0),
            network.candidate_position(2, 20.0),
        )
        self.assertEqual(path.arc_ids, (0, 1, 2))
        self.assertAlmostEqual(
            path.distance_m,
            network.arc(0).length_m - 50.0 + network.arc(1).length_m + 20.0,
            delta=0.01,
        )

    def test_disconnected_and_zero_length_arcs(self):
        network = build_road_network([(0.0, 0.0), (0.0, 0.0), (1.0, 1.0)], [(0, 1), (2, 2)])
        self.assertEqual(network.arc(0).length_m, 0.0)
        self.assertIsNone(
            network.path_between_positions(
                network.candidate_position(0, 0.0), network.candidate_position(1, 0.0)
            )
        )

    def test_position_validation_and_endpoints(self):
        network = build_road_network(self.nodes[:2], [(0, 1)])
        length = network.arc(0).length_m
        self.assertEqual(network.candidate_position(0, 0).distance_to_end_m, length)
        self.assertEqual(network.candidate_position(0, length).distance_to_end_m, 0.0)
        with self.assertRaises(ValueError):
            network.candidate_position(0, length + 1.0)
        with self.assertRaises(ValueError):
            CandidatePosition.from_fraction(0, length, 1.1)

    def _scorer(self, *, beta=10.0, speed=None, timestamps=(0.0, 1.0)):
        network = build_road_network(self.nodes[:3], [(0, 1), (1, 2)])
        positions = {
            0: {0: network.candidate_position(0, 10.0)},
            1: {1: network.candidate_position(1, 10.0)},
        }
        scorer = RoadNetworkTransitionScorer(
            network, positions, beta_m=beta, max_speed_mps=speed
        )
        previous = ViterbiObservation(
            0, (ViterbiCandidate(0, 1.0),), timestamps[0], {"longitude": 0.0001, "latitude": 0.0}
        )
        current = ViterbiObservation(
            1, (ViterbiCandidate(1, 1.0),), timestamps[1], {"longitude": 0.0011, "latitude": 0.0}
        )
        return network, positions, scorer, previous, current

    def test_gps_displacement_and_log_score(self):
        network, _, scorer, previous, current = self._scorer(beta=10.0)
        score = scorer(previous.candidates[0], current.candidates[0], previous, current)
        route_distance = network.arc(0).length_m - 10.0 + 10.0
        gps_distance = geodesic_distance_m((0.0001, 0.0), (0.0011, 0.0))
        self.assertAlmostEqual(score, -abs(route_distance - gps_distance) / 10.0, places=7)

    def test_invalid_beta_and_timestamps(self):
        network = build_road_network(self.nodes[:2], [(0, 1)])
        with self.assertRaises(ValueError):
            RoadNetworkTransitionScorer(network, {}, beta_m=0)
        with self.assertRaises(ValueError):
            RoadNetworkTransitionScorer(network, {}, timestamp_unit_seconds=0)

    def test_missing_timestamp_skips_speed_check(self):
        _, _, scorer, previous, current = self._scorer(speed=0.001)
        previous = ViterbiObservation(previous.observation_id, previous.candidates, None, previous.metadata)
        current = ViterbiObservation(current.observation_id, current.candidates, None, current.metadata)
        self.assertTrue(math.isfinite(scorer(previous.candidates[0], current.candidates[0], previous, current)))

    def test_zero_or_negative_elapsed_time_rejects_speed_checked_transition(self):
        _, _, scorer, previous, current = self._scorer(speed=1000.0, timestamps=(1.0, 1.0))
        self.assertEqual(scorer(previous.candidates[0], current.candidates[0], previous, current), -math.inf)
        self.assertEqual(scorer.last_rejection_reason, "invalid_time")

    def test_maximum_speed_check(self):
        _, _, scorer, previous, current = self._scorer(speed=1.0, timestamps=(0.0, 1.0))
        self.assertEqual(scorer(previous.candidates[0], current.candidates[0], previous, current), -math.inf)
        self.assertEqual(scorer.last_rejection_reason, "speed_limit")

    def test_shortest_path_and_position_results_are_cached(self):
        network, _, scorer, previous, current = self._scorer()
        first = scorer.route_distance(
            network.candidate_position(0, 10.0), network.candidate_position(1, 10.0)
        )
        second = scorer.route_distance(
            network.candidate_position(0, 10.0), network.candidate_position(1, 10.0)
        )
        self.assertIs(first, second)
        self.assertEqual(len(network._node_path_cache), 1)

    def test_viterbi_callback_compatibility(self):
        _, _, scorer, previous, current = self._scorer()
        result = decode_viterbi(
            [previous, current], scorer
        )
        self.assertEqual(result.path, (0, 1))

    def test_trip_scoped_arc_identity(self):
        first = build_road_network(self.nodes[:2], [(0, 1)], trip_id="a")
        second = build_road_network(self.nodes[:2], [(0, 1)], trip_id="b")
        self.assertEqual(first.arc(0).arc_id, second.arc(0).arc_id)
        self.assertNotEqual(first.trip_id, second.trip_id)

    def test_malformed_graph_rows_are_rejected(self):
        with self.assertRaises(ValueError):
            build_road_network(self.nodes[:2], [(0, 2)])
        with self.assertRaises(ValueError):
            build_road_network(self.nodes[:2], [(0,)])


if __name__ == "__main__":
    unittest.main()
