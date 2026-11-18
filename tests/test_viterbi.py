import math
import unittest

from src.decoder.viterbi import (
    EmptyCandidateSetError,
    InvalidProbabilityError,
    decode_viterbi,
)


def always_connected(previous, current, previous_observation, current_observation):
    return 0.0


class ViterbiDecoderTests(unittest.TestCase):
    def test_simple_path_and_log_score(self):
        payload = {"observations": [
            {"observation_id": 0, "candidates": [{"arc_id": 10, "ml_probability": 0.8}]},
            {"observation_id": 1, "candidates": [{"arc_id": 11, "ml_probability": 0.5}]},
        ]}
        result = decode_viterbi(payload, always_connected)
        self.assertEqual(result.arc_sequence, (10, 11))
        self.assertAlmostEqual(result.final_score, math.log(0.8) + math.log(0.5))

    def test_sequence_score_can_beat_independent_maximum(self):
        payload = [[
            {"arc_id": 1, "ml_probability": 0.9},
            {"arc_id": 2, "ml_probability": 0.8},
        ], [
            {"arc_id": 3, "ml_probability": 0.9},
            {"arc_id": 4, "ml_probability": 0.8},
        ]]
        def transitions(previous, current, *_):
            return 0.0 if (previous.arc_id, current.arc_id) == (2, 4) else math.log(0.01)
        result = decode_viterbi(payload, transitions)
        self.assertEqual(result.path, (2, 4))

    def test_different_candidate_counts_and_prediction_contract(self):
        payload = {"trip_id": "00000001", "observations": [
            {"observation_id": 4, "timestamp": 1.0, "candidates": [{"arc_id": 2641, "ml_probability": 0.7}]},
            {"observation_id": 5, "timestamp": 2.0, "candidates": [
                {"arc_id": 64973, "ml_probability": 0.6},
                {"arc_id": 133948, "ml_probability": 0.4},
            ]},
        ]}
        seen = []
        def scorer(previous, current, previous_observation, current_observation):
            seen.append((previous_observation.timestamp, current_observation.timestamp))
            return 0.0
        result = decode_viterbi(payload, scorer)
        self.assertEqual(result.arc_sequence, (2641, 64973))
        self.assertEqual(seen, [(1.0, 2.0), (1.0, 2.0)])

    def test_single_observation(self):
        result = decode_viterbi([[{"arc_id": 7, "ml_probability": 0.2}, {"arc_id": 8, "ml_probability": 0.3}]], always_connected)
        self.assertEqual(result.arc_sequence, (8,))

    def test_impossible_transitions_raise(self):
        def impossible(*_):
            return None
        with self.assertRaisesRegex(ValueError, "No valid path"):
            decode_viterbi([[{"arc_id": 1, "ml_probability": 1.0}], [{"arc_id": 2, "ml_probability": 1.0}]], impossible)

    def test_zero_probability_uses_epsilon(self):
        result = decode_viterbi([[{"arc_id": 1, "ml_probability": 0.0}]], always_connected, epsilon=1e-6)
        self.assertAlmostEqual(result.final_score, math.log(1e-6))

    def test_invalid_probabilities_and_empty_candidates(self):
        for probability in (-0.1, 1.1, math.nan, math.inf):
            with self.subTest(probability=probability), self.assertRaises(InvalidProbabilityError):
                decode_viterbi([[{"arc_id": 1, "ml_probability": probability}]], always_connected)
        with self.assertRaises(EmptyCandidateSetError):
            decode_viterbi([[]], always_connected)

    def test_ties_keep_input_order(self):
        payload = [[
            {"arc_id": 20, "ml_probability": 0.5},
            {"arc_id": 10, "ml_probability": 0.5},
        ], [
            {"arc_id": 30, "ml_probability": 0.5},
            {"arc_id": 40, "ml_probability": 0.5},
        ]]
        self.assertEqual(decode_viterbi(payload, always_connected).arc_sequence, (20, 30))

    def test_negative_infinity_transition_is_impossible(self):
        def scorer(previous, current, *_):
            return -math.inf if previous.arc_id == 1 else 0.0
        payload = [[{"arc_id": 1, "ml_probability": 1.0}, {"arc_id": 2, "ml_probability": 0.5}], [{"arc_id": 3, "ml_probability": 1.0}]]
        self.assertEqual(decode_viterbi(payload, scorer).path, (2, 3))

    def test_custom_emission_callback_is_used(self):
        payload = [[
            {"arc_id": 1, "ml_probability": 0.1},
            {"arc_id": 2, "ml_probability": 0.9},
        ]]
        def custom_emission(candidate, observation):
            return 0.0 if candidate.arc_id == 1 else -10.0
        result = decode_viterbi(
            payload, always_connected, emission_score=custom_emission
        )
        self.assertEqual(result.path, (1,))
        self.assertEqual(result.score, 0.0)


if __name__ == "__main__":
    unittest.main()
