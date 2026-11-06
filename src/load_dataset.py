from pathlib import Path

import numpy as np


def _split_labels_and_features(array):
    array = np.asarray(array, dtype=float)
    if array.ndim != 2 or array.shape[1] < 2:
        raise ValueError(f"Expected a 2D time-series array with label column, got shape {array.shape}.")

    labels_raw = array[:, 0]
    labels = np.where(labels_raw == -1, 0, 1).astype(int)
    features = array[:, 1:]
    return features, labels


def load_ecg200_data(base_dir="data/ECG200"):
    base_path = Path(base_dir)

    train = np.load(base_path / "ECG200_TRAIN_normalized.npy")
    validation = np.load(base_path / "ECG200_VALIDATION_normalized.npy")
    test = np.load(base_path / "ECG200_TEST_normalized.npy")

    train_features, train_labels = _split_labels_and_features(train)
    validation_features, validation_labels = _split_labels_and_features(validation)
    test_features, test_labels = _split_labels_and_features(test)

    return {
        "train": {"samples": train_features, "labels": train_labels},
        "validation": {"samples": validation_features, "labels": validation_labels},
        "test": {"samples": test_features, "labels": test_labels},
    }


def load_dataset(data_path="data/heartbeat_dataset.npy"):
    """Backward-compatible dataset loader for older code paths."""
    data = np.load(data_path)
    if isinstance(data, np.lib.npyio.NpzFile):
        X = data["X"]
        y = data["y"]
        return X, y
    return _split_labels_and_features(data)
