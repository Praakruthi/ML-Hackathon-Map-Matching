import json
from pathlib import Path
import unittest
from zipfile import ZipFile

from src.decoder.integration import load_trip_inputs_from_zip


class IntegrationAdapterTests(unittest.TestCase):
    dataset_zip = Path("tests") / "_integration_dataset.zip"
    prediction_zip = Path("tests") / "_integration_predictions.zip"

    @classmethod
    def setUpClass(cls):
        with ZipFile(cls.dataset_zip, "w") as archive:
            prefix = "Student folder/student_test/00000000/00000000"
            archive.writestr(prefix + ".nodes", "0 0\n0.001 0\n0.002 0\n")
            archive.writestr(prefix + ".arcs", "0 1\n1 2\n")
            archive.writestr(prefix + ".track", "0.0001 0 1\n0.0011 0 2\n")
        payload = [{
            "trip_id": "00000000",
            "observations": [
                {"observation_id": 0, "candidates": [{"arc_id": 0, "ml_probability": 0.8}]},
                {"observation_id": 1, "candidates": [{"arc_id": 1, "ml_probability": 0.7}]},
            ],
        }]
        with ZipFile(cls.prediction_zip, "w") as archive:
            archive.writestr("pathsnap_hmm_input.json", json.dumps(payload))
            archive.writestr("pathsnap_final_metrics.json", json.dumps({"selected_model": "XGBoost"}))

    @classmethod
    def tearDownClass(cls):
        for path in (cls.dataset_zip, cls.prediction_zip):
            if path.exists():
                path.unlink()

    def test_zip_adapter_reads_contents_and_derives_positions(self):
        loaded = load_trip_inputs_from_zip(
            self.dataset_zip,
            self.prediction_zip,
            split="student_test",
            trip_id="00000000",
        )
        self.assertEqual(loaded.selected_model, "XGBoost")
        self.assertEqual(loaded.observation_count, 2)
        self.assertEqual(loaded.candidate_count, 2)
        self.assertEqual(len(loaded.road_network.arcs), 2)
        self.assertEqual(loaded.observation_data[1]["timestamp"], 2.0)
        self.assertGreaterEqual(loaded.candidate_positions[0][0].distance_from_start_m, 0.0)
        self.assertLessEqual(
            loaded.candidate_positions[0][0].distance_from_start_m,
            loaded.road_network.arc(0).length_m,
        )


if __name__ == "__main__":
    unittest.main()
