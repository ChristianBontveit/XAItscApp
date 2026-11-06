import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.tsc_model import (
    build_prediction_rows,
    load_prediction_artifact,
    save_prediction_artifact,
)


class TestPredictionCache(unittest.TestCase):
    def test_build_prediction_rows_uses_supplied_predictions(self):
        rows = build_prediction_rows(
            np.array([[1.0, 2.0], [3.0, 4.0]]),
            np.array([1, 0]),
            np.array([0, 0]),
        )
        self.assertEqual(rows[0]["predicted_label"], 0)
        self.assertFalse(rows[0]["is_correct"])
        self.assertEqual(rows[1]["predicted_label"], 0)
        self.assertTrue(rows[1]["is_correct"])

    def test_artifact_round_trip_and_stale_detection(self):
        class FakeModel:
            def predict(self, samples):
                return np.array([1, 0])

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model_path = root / "model.pkl"
            validation_path = root / "validation.npy"
            artifact_path = root / "predictions.npz"
            model_path.write_bytes(b"model")
            validation_path.write_bytes(b"validation")

            save_prediction_artifact(
                FakeModel(),
                np.array([[1.0], [2.0]]),
                np.array([1, 0]),
                model_path,
                validation_path,
                artifact_path,
            )
            np.testing.assert_array_equal(
                load_prediction_artifact(
                    artifact_path, model_path, validation_path, sample_count=2
                ),
                [1, 0],
            )

            model_path.write_bytes(b"changed model")
            with self.assertRaisesRegex(ValueError, "stale or incompatible"):
                load_prediction_artifact(
                    artifact_path, model_path, validation_path, sample_count=2
                )


if __name__ == "__main__":
    unittest.main()
