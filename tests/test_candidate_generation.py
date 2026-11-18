import unittest

from src.decoder.candidate_generation import TrueDistanceCandidateIndex
from src.decoder.transition import Arc


class CandidateGenerationTests(unittest.TestCase):
    def test_ranks_long_segment_by_true_distance_not_midpoint(self):
        nodes = [(0.0, 0.0), (0.01, 0.0), (0.005, 0.0003), (0.006, 0.0003)]
        arcs = [Arc(0, 0, 1, 1111.0), Arc(1, 2, 3, 111.0)]
        index = TrueDistanceCandidateIndex(nodes, arcs)
        ranked = index.rank((0.005, 0.0001), pool_size=2)
        self.assertEqual(ranked[0][0], 0)
        self.assertLess(ranked[0][1], ranked[1][1])

    def test_preserves_directed_arc_ids(self):
        nodes = [(0.0, 0.0), (0.001, 0.0)]
        arcs = [Arc(42, 1, 0, 111.0)]
        ranked = TrueDistanceCandidateIndex(nodes, arcs).rank((0.0005, 0.0), pool_size=1)
        self.assertEqual(ranked[0][0], 42)


if __name__ == "__main__":
    unittest.main()
