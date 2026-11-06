"""Generate the validation prediction artifact before starting the dashboard.

Run from the project root with:
    venv\Scripts\python.exe scripts\generate_prediction_artifact.py
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.load_dataset import load_ecg200_data
from src.tsc_model import load_rocket_model, save_prediction_artifact


MODEL_PATH = PROJECT_ROOT / "models" / "ECG200" / "miniRocket_norm.pkl"
VALIDATION_PATH = PROJECT_ROOT / "data" / "ECG200" / "ECG200_VALIDATION_normalized.npy"
ARTIFACT_PATH = PROJECT_ROOT / "data" / "ECG200" / "validation_predictions.npz"


def main():
    dataset = load_ecg200_data(PROJECT_ROOT / "data" / "ECG200")
    model = load_rocket_model(MODEL_PATH)
    save_prediction_artifact(
        model,
        dataset["validation"]["samples"],
        dataset["validation"]["labels"],
        MODEL_PATH,
        VALIDATION_PATH,
        ARTIFACT_PATH,
    )
    print(f"Saved prediction artifact to {ARTIFACT_PATH}")


if __name__ == "__main__":
    main()
