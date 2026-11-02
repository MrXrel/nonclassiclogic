"""HDBSCAN adapter with explicit small-sample behavior."""
import numpy as np
from sklearn.cluster import HDBSCAN


def fit_density(matrix, *, min_cluster_size, min_samples, allow_single_cluster):
    for name, value, minimum in [("min_cluster_size", min_cluster_size, 2),
                                  ("min_samples", min_samples, 1)]:
        if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= 1_000_000:
            raise ValueError(f"{name}: требуется целое число от {minimum} до 1000000.")
    if not isinstance(allow_single_cluster, bool):
        raise ValueError("allow_single_cluster должен быть логическим значением.")
    settings = {"min_cluster_size": min_cluster_size, "min_samples": min_samples,
                "allow_single_cluster": allow_single_cluster,
                "cluster_selection_method": "eom", "implementation": "scikit-learn",
                "metric": "precomputed_alpha_cuts", "noise_label": 0}
    if len(matrix) < max(min_cluster_size, min_samples):
        return np.full(len(matrix), -1, dtype=int), settings
    # A zero-distance tree has no density structure. Tied zero-height branches
    # can otherwise be split arbitrarily by the backend when one group is banned.
    if not np.any(matrix):
        return np.full(len(matrix), 0 if allow_single_cluster else -1, dtype=int), settings
    estimator = HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples,
        allow_single_cluster=allow_single_cluster, cluster_selection_method="eom",
        metric="precomputed", copy=True)
    return estimator.fit_predict(matrix), settings
