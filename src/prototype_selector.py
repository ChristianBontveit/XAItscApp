import numpy as np

import numpy as np
from tslearn.metrics import cdist_dtw
from sklearn_extra.cluster import KMedoids


def select_prototypes(X, n_prototypes=5, mode="central"):
    """Select representative, diverse, or extreme time series examples.

    Parameters
    ----------
    X : np.ndarray
        2D array of shape (n_samples, n_timepoints).

    n_prototypes : int, default=5
        Number of examples to select.

    mode : {"central", "diverse", "outlier"}, default="central"
        Selection strategy:

        - "central":
            Select examples with the smallest average DTW distance
            to all other examples.

        - "diverse":
            Select examples using K-Medoids with DTW, giving
            representative examples from different regions/patterns
            of the dataset.

        - "outlier":
            Select examples with the largest average DTW distance
            to all other examples.

    Returns
    -------
    selected_indices : np.ndarray
        Indices of the selected examples in the original X array.

    selected_series : np.ndarray
        The selected time series.
    """
    X = np.asarray(X, dtype=float)

    if X.ndim != 2:
        raise ValueError(
            f"Prototype selection expects a 2D array, got shape {X.shape}."
        )

    if n_prototypes <= 0:
        raise ValueError("n_prototypes must be a positive integer.")

    if len(X) == 0:
        return np.array([], dtype=int), X

    if n_prototypes > len(X):
        n_prototypes = len(X)

    if mode not in {"central", "diverse", "outlier"}:
        raise ValueError(
            "mode must be either 'central', 'diverse', or 'outlier'."
        )

    # ------------------------------------------------------------------
    # Calculate pairwise DTW distances.
    #
    # We use the same distance matrix for central and outlier selection.
    # ------------------------------------------------------------------
    distance_matrix = cdist_dtw(X)

    if mode == "central":
        # Ignore self-distance (zero).
        np.fill_diagonal(distance_matrix, np.nan)

        # Average DTW distance to every other example.
        mean_distances = np.nanmean(distance_matrix, axis=1)

        # Smallest average distance = most central.
        selected_indices = np.argsort(mean_distances)[:n_prototypes]

    elif mode == "outlier":
        # Ignore self-distance (zero).
        np.fill_diagonal(distance_matrix, np.nan)

        # Average DTW distance to every other example.
        mean_distances = np.nanmean(distance_matrix, axis=1)

        # Largest average distance = most unusual.
        selected_indices = np.argsort(mean_distances)[
            -n_prototypes:
        ][::-1]

    elif mode == "diverse":
        # K-Medoids selects actual observations as medoids.
        #
        # Using the precomputed DTW matrix avoids calculating DTW
        # distances repeatedly inside K-Medoids.
        km = KMedoids(
            n_clusters=n_prototypes,
            metric="precomputed",
            init="random",
            random_state=42,
        )

        km.fit(distance_matrix)

        selected_indices = km.medoid_indices_

    # Ensure integer NumPy indices.
    selected_indices = np.asarray(selected_indices, dtype=int)

    return selected_indices, X[selected_indices]

'''
def select_prototypes(X, n_prototypes=5, mode="central"):
    """Select representative or extreme time series examples from a 2D sample matrix."""
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError(f"Prototype selection expects a 2D array, got shape {X.shape}.")

    if n_prototypes <= 0:
        raise ValueError("n_prototypes must be a positive integer.")

    if n_prototypes > len(X):
        n_prototypes = len(X)

    mean_series = np.mean(X, axis=0)
    distances = np.linalg.norm(X - mean_series, axis=1)

    if mode == "central":
        selected_indices = np.argsort(distances)[:n_prototypes]
    elif mode == "outlier":
        selected_indices = np.argsort(distances)[-n_prototypes:][::-1]
    else:
        raise ValueError("mode must be either 'central' or 'outlier'.")

    return selected_indices, X[selected_indices]
'''