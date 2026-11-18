import math
import unittest

from src.decoder.emission import (
    gaussian_log_emission,
    gaussian_log_emission_for_context,
    ml_log_emission,
    point_to_geometry_distance_m,
    point_to_segment_distance_m,
)
from src.decoder.viterbi import ViterbiCandidate, ViterbiObservation


class EmissionTests(unittest.TestCase):
    def test_point_on_segment_has_zero_distance(self):
        segment = ((0.0, 0.0), (0.001, 0.0))
        self.assertAlmostEqual(
            point_to_segment_distance_m((0.0005, 0.0), *segment), 0.0, places=8
        )

    def test_perpendicular_distance_is_measured_in_metres(self):
        segment = ((0.0, 0.0), (0.001, 0.0))
        distance = point_to_segment_distance_m((0.0005, 0.001), *segment)
        self.assertAlmostEqual(distance, 111.195, delta=0.2)

    def test_closest_endpoint_is_used(self):
        segment = ((0.0, 0.0), (0.001, 0.0))
        distance = point_to_segment_distance_m((0.002, 0.0), *segment)
        self.assertAlmostEqual(distance, 111.195, delta=0.2)

    def test_gaussian_log_emission_matches_formula(self):
        sigma = 10.0
        score = gaussian_log_emission((0.0005, 0.0), ((0.0, 0.0), (0.001, 0.0)), sigma)
        expected = -math.log(math.sqrt(2.0 * math.pi) * sigma)
        self.assertAlmostEqual(score, expected)

    def test_sigma_changes_distance_penalty(self):
        point = (0.0005, 0.001)
        geometry = ((0.0, 0.0), (0.001, 0.0))
        narrow = gaussian_log_emission(point, geometry, 10.0)
        wide = gaussian_log_emission(point, geometry, 100.0)
        self.assertGreater(wide, narrow)

    def test_invalid_sigma_and_coordinates_are_rejected(self):
        for sigma in (0.0, -1.0, math.nan, math.inf):
            with self.subTest(sigma=sigma), self.assertRaises(ValueError):
                gaussian_log_emission((0.0, 0.0), ((0.0, 0.0), (0.001, 0.0)), sigma)
        with self.assertRaises(ValueError):
            gaussian_log_emission((181.0, 0.0), ((0.0, 0.0), (0.001, 0.0)), 10.0)
        with self.assertRaises(ValueError):
            gaussian_log_emission((0.0, 0.0), ((0.0, 0.0),), 10.0)

    def test_polyline_geometry_uses_closest_piece(self):
        geometry = ((0.0, 0.0), (0.001, 0.0), (0.001, 0.001))
        self.assertAlmostEqual(
            point_to_geometry_distance_m((0.001, 0.0005), geometry), 0.0, places=8
        )

    def test_ml_zero_probability_uses_epsilon(self):
        self.assertEqual(ml_log_emission(0.0, epsilon=1e-6), math.log(1e-6))

    def test_ml_probability_one(self):
        self.assertEqual(ml_log_emission(1.0), 0.0)

    def test_invalid_ml_probabilities_are_rejected(self):
        for probability in (-0.1, 1.1, math.nan, math.inf):
            with self.subTest(probability=probability), self.assertRaises(ValueError):
                ml_log_emission(probability)

    def test_viterbi_data_class_compatibility(self):
        candidate = ViterbiCandidate(arc_id=2641, ml_probability=0.7)
        observation = ViterbiObservation(
            observation_id=0,
            timestamp=1.0,
            candidates=(candidate,),
            metadata={"longitude": 0.0005, "latitude": 0.0},
        )
        score = gaussian_log_emission_for_context(
            candidate, observation, {2641: ((0.0, 0.0), (0.001, 0.0))}, 10.0
        )
        self.assertAlmostEqual(score, -math.log(math.sqrt(2.0 * math.pi) * 10.0))
        self.assertAlmostEqual(ml_log_emission(candidate), math.log(0.7))

    def test_large_distance_remains_finite(self):
        score = gaussian_log_emission(
            (179.0, 80.0), ((-179.0, -80.0), (-178.0, -80.0)), 5.0
        )
        self.assertTrue(math.isfinite(score))
        self.assertLess(score, 0.0)


if __name__ == "__main__":
    unittest.main()
