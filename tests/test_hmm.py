import math
import unittest

from src.decoder.hmm import HMMConfig, match_trip
from src.decoder.transition import CandidatePosition, build_road_network


class HMMIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.network = build_road_network(
            [(0.0, 0.0), (0.001, 0.0), (0.002, 0.0), (0.001, 0.001)],
            [(0, 1), (1, 2), (1, 3), (3, 1)],
            trip_id="trip-1",
        )
        self.observation_data = [
            {"observation_id": 0, "longitude": 0.0001, "latitude": 0.0, "timestamp": 0.0},
            {"observation_id": 1, "longitude": 0.0011, "latitude": 0.0, "timestamp": 1.0},
        ]
        self.positions = {
            0: {
                0: self.network.candidate_position(0, 10.0),
                2: self.network.candidate_position(2, 1.0),
            },
            1: {
                1: self.network.candidate_position(1, 10.0),
                3: self.network.candidate_position(3, 1.0),
            },
        }

    def _payload(self, first_candidates=None, second_candidates=None):
        return {
            "trip_id": "trip-1",
            "observations": [
                {"observation_id": 0, "candidates": first_candidates or [
                    {"arc_id": 0, "ml_probability": 0.9},
                    {"arc_id": 2, "ml_probability": 0.1},
                ]},
                {"observation_id": 1, "candidates": second_candidates or [
                    {"arc_id": 1, "ml_probability": 0.9},
                    {"arc_id": 3, "ml_probability": 0.1},
                ]},
            ],
        }

    def test_ml_end_to_end_and_input_is_not_mutated(self):
        payload = self._payload()
        original = repr(payload)
        result = match_trip(payload, self.network, self.observation_data, self.positions)
        self.assertEqual(result.arc_sequence, (0, 1))
        self.assertEqual(result.observation_ids, (0, 1))
        self.assertEqual(result.emission_mode, "ml")
        self.assertEqual(repr(payload), original)
        self.assertTrue(math.isfinite(result.final_score))

    def test_distance_mode_uses_geometry_not_ml_probability(self):
        payload = self._payload(
            [{"arc_id": 0, "ml_probability": 0.01}, {"arc_id": 2, "ml_probability": 0.99}],
            [{"arc_id": 1, "ml_probability": 0.01}, {"arc_id": 3, "ml_probability": 0.99}],
        )
        result = match_trip(
            payload,
            self.network,
            self.observation_data,
            self.positions,
            HMMConfig(emission_mode="distance", sigma_m=10.0),
        )
        self.assertEqual(result.arc_sequence, (0, 1))

    def test_emission_and_transition_weights_are_configurable(self):
        result = match_trip(
            self._payload(), self.network, self.observation_data, self.positions,
            HMMConfig(emission_weight=0.0, transition_weight=0.0),
        )
        self.assertEqual(result.arc_sequence, (0, 1))
        self.assertEqual(result.configuration["emission_weight"], 0.0)

    def test_connectivity_rejects_impossible_candidate(self):
        payload = self._payload(
            [{"arc_id": 0, "ml_probability": 1.0}],
            [{"arc_id": 3, "ml_probability": 1.0}],
        )
        # Arc 0 ends at node 1 and arc 3 starts at node 3; the graph does have
        # a return edge, so this remains feasible through arc 2. Remove it to
        # make the disconnected case explicit.
        disconnected = build_road_network(
            self.network.nodes, [(0, 1), (1, 2), (3, 1)], trip_id="trip-1"
        )
        positions = {
            0: {0: disconnected.candidate_position(0, 10.0)},
            1: {2: disconnected.candidate_position(2, 1.0)},
        }
        impossible_payload = self._payload(
            [{"arc_id": 0, "ml_probability": 1.0}],
            [{"arc_id": 2, "ml_probability": 1.0}],
        )
        with self.assertRaises(ValueError):
            match_trip(impossible_payload, disconnected, self.observation_data, positions)

    def test_single_observation_and_variable_candidates(self):
        payload = {"trip_id": "trip-1", "observations": [{
            "observation_id": 0,
            "candidates": [{"arc_id": 0, "ml_probability": 0.2}],
        }]}
        result = match_trip(payload, self.network, self.observation_data[:1], {0: {0: self.positions[0][0]}})
        self.assertEqual(result.arc_sequence, (0,))

    def test_missing_metadata_unknown_arc_and_trip_mismatch(self):
        with self.assertRaisesRegex(ValueError, "longitude"):
            match_trip(self._payload(), self.network, [{"observation_id": 0}], self.positions)
        with self.assertRaises(KeyError):
            match_trip(self._payload([{"arc_id": 99, "ml_probability": 1.0}]), self.network, self.observation_data, self.positions)
        with self.assertRaisesRegex(ValueError, "does not match"):
            match_trip({**self._payload(), "trip_id": "other"}, self.network, self.observation_data, self.positions)

    def test_missing_positions_and_invalid_config(self):
        with self.assertRaisesRegex(ValueError, "missing candidate position"):
            match_trip(self._payload(), self.network, self.observation_data, {0: {}, 1: {}})
        with self.assertRaises(ValueError):
            HMMConfig(emission_mode="unknown")
        with self.assertRaises(ValueError):
            HMMConfig(emission_weight=-1.0)

    def test_small_probabilities_and_deterministic_ties(self):
        payload = self._payload(
            [{"arc_id": 0, "ml_probability": 0.0}, {"arc_id": 2, "ml_probability": 0.0}],
            [{"arc_id": 1, "ml_probability": 0.0}, {"arc_id": 3, "ml_probability": 0.0}],
        )
        result = match_trip(
            payload, self.network, self.observation_data, self.positions,
            HMMConfig(epsilon=1e-300, transition_weight=0.0),
        )
        self.assertEqual(result.arc_sequence, (0, 1))
        self.assertTrue(math.isfinite(result.final_score))

    def test_callback_contract_preserves_observation_order_and_arc_ids(self):
        payload = self._payload(
            [{"arc_id": 2, "ml_probability": 0.6}, {"arc_id": 0, "ml_probability": 0.4}],
            [{"arc_id": 3, "ml_probability": 0.6}, {"arc_id": 1, "ml_probability": 0.4}],
        )
        result = match_trip(payload, self.network, self.observation_data, self.positions)
        self.assertEqual(result.observation_ids, (0, 1))
        self.assertTrue(all(arc_id in self.network.arcs for arc_id in result.arc_sequence))


if __name__ == "__main__":
    unittest.main()
