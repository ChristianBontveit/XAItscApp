# src/tsc_model.py
from pathlib import Path
import hashlib
import json

import joblib
import numpy as np

def load_rocket_model(model_path="models/ECG200/miniRocket_norm.pkl"):
    """Load the serialized miniRocket classifier from disk."""
    model_path = Path(model_path)
    return joblib.load(model_path)


def predict_time_series(model, time_series_data):
    """Run prediction on one or many time series and return integer labels."""
    data = np.asarray(time_series_data, dtype=float)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    return np.asarray(model.predict(data), dtype=int)


def accuracy_score(model, samples, labels):
    predictions = predict_time_series(model, samples)
    labels = np.asarray(labels, dtype=int)
    return float(np.mean(predictions == labels))


def label_to_text(label_idx):
    label_idx = int(label_idx)
    return "Class 1" if label_idx == 1 else "Class 0"


def binary_label_to_text(label_idx):
    """Return the dashboard's binary ECG200 category names."""
    return "Normal" if int(label_idx) == 1 else "Abnormal"


def build_prediction_rows(samples, labels, predictions):
    """Build serializable prediction rows without running model inference."""
    samples = np.asarray(samples, dtype=float)
    labels = np.asarray(labels, dtype=int)
    predictions = np.asarray(predictions, dtype=int)
    if len(samples) != len(labels) or len(samples) != len(predictions):
        raise ValueError("Samples, labels, and predictions must have equal lengths.")

    prediction_rows = []
    for idx, sample in enumerate(samples):
        true_label = int(labels[idx])
        prediction = int(predictions[idx])
        prediction_rows.append({
            "index": int(idx),
            "sample": sample,
            "true_label": true_label,
            "predicted_label": prediction,
            "true_text": label_to_text(true_label),
            "predicted_text": label_to_text(prediction),
            "is_correct": prediction == true_label,
            "rank": idx + 1,
        })
    return prediction_rows


def _file_fingerprint(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_prediction_artifact(model, samples, labels, model_path, validation_path, artifact_path):
    """Predict a validation set once and save it with input fingerprints."""
    predictions = predict_time_series(model, samples)
    metadata = {
        "model_sha256": _file_fingerprint(model_path),
        "validation_sha256": _file_fingerprint(validation_path),
        "sample_count": len(samples),
    }
    artifact_path = Path(artifact_path)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        artifact_path,
        predictions=predictions,
        metadata=np.array(json.dumps(metadata)),
    )


def load_prediction_artifact(artifact_path, model_path, validation_path, sample_count):
    """Load predictions and reject artifacts that do not match current inputs."""
    artifact_path = Path(artifact_path)
    if not artifact_path.exists():
        raise FileNotFoundError(
            f"Prediction artifact is missing: {artifact_path}. "
            "Generate it explicitly with the project's prediction generation command."
        )

    with np.load(artifact_path, allow_pickle=False) as artifact:
        predictions = np.asarray(artifact["predictions"], dtype=int)
        metadata = json.loads(str(artifact["metadata"].item()))

    expected = {
        "model_sha256": _file_fingerprint(model_path),
        "validation_sha256": _file_fingerprint(validation_path),
        "sample_count": sample_count,
    }
    if metadata != expected or len(predictions) != sample_count:
        raise ValueError(
            f"Prediction artifact is stale or incompatible: {artifact_path}. "
            "Regenerate it explicitly after changing the model or validation data."
        )
    return predictions


def prototype_predictions(model, samples, labels, mode="central", n_prototypes=5, predictions=None):
    from src.prototype_selector import select_prototypes

    indices, _ = select_prototypes(samples, n_prototypes=n_prototypes, mode=mode)
    all_rows = build_prediction_rows(samples, labels, predictions if predictions is not None else predict_time_series(model, samples))
    selected_rows = []
    for position, idx in enumerate(indices):
        row = dict(all_rows[int(idx)])
        row["rank"] = position + 1
        selected_rows.append(row)

    return indices, selected_rows