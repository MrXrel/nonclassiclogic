"""Gap statistic for the one-dimensional Rayleigh scale family."""
import math
import numpy as np
from scipy.cluster.hierarchy import linkage, cut_tree


def _dispersion_curve(x, counts):
    tree = linkage(x[:, None], method="complete", metric="euclidean")
    partitions = cut_tree(tree, n_clusters=counts)
    dispersion = []
    for labels in partitions.T:
        sizes = np.bincount(labels)
        means = np.bincount(labels, weights=x) / sizes
        residuals = x - means[labels]
        dispersion.append(float(residuals @ residuals))
    # Normalized coordinates make this floor independent of sigma's units.
    return np.log(np.maximum(dispersion, np.finfo(float).eps * len(x)))


def select_cluster_count(sigmas, *, max_clusters=10, reference_samples=100, seed=42):
    """Global 1-SE gap rule, with a uniform reference on sigma.

    Distances between Rayleigh fuzzy values are C*abs(sigma_i-sigma_j).
    C cancels from the gap; squared fuzzy distances therefore give the same
    criterion as this normalized one-dimensional within-cluster SSE.
    """
    for value, low, high, name in [(max_clusters, 1, 300, "max_clusters"),
                                   (reference_samples, 2, 1000, "reference_samples"),
                                   (seed, 0, 2**32 - 1, "seed")]:
        if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
            raise ValueError(f"{name}: требуется целое число от {low} до {high}.")
    x = np.asarray(sigmas, dtype=float)
    if x.ndim != 1 or not len(x) or not np.all(np.isfinite(x)) or np.any(x <= 0):
        raise ValueError("Для автоподбора нужны конечные положительные σ.")
    unique = len(np.unique(x))
    details = {"criterion": "gap", "rule": "global_1se", "max_clusters": max_clusters,
               "reference_samples": reference_samples, "seed": seed,
               "reference": "uniform_sigma_range", "candidates": []}
    if unique == 1 or len(x) < 3 or max_clusters == 1:
        details["reason"] = ("identical" if unique == 1 else
                             "small_sample" if len(x) < 3 else "search_limit")
        return 1, details

    x = (x - x.min()) / (x.max() - x.min())
    upper = min(max_clusters, unique, len(x) - 1)
    counts = np.arange(1, upper + 1)
    observed = _dispersion_curve(x, counts)
    rng = np.random.default_rng(seed)
    reference = np.array([_dispersion_curve(rng.uniform(0, 1, len(x)), counts)
                          for _ in range(reference_samples)])
    gap = reference.mean(axis=0) - observed
    error = reference.std(axis=0, ddof=1) * math.sqrt(1 + 1 / reference_samples)
    best = int(np.argmax(gap))
    selected = int(np.flatnonzero(gap >= gap[best] - error[best])[0] + 1)
    details["reason"] = "search_limit" if selected == upper and upper < unique else "gap_rule"
    details["best_k"] = best + 1
    details["cutoff"] = float(gap[best] - error[best])
    details["candidates"] = [
        {"k": int(k), "gap": float(g), "se": float(s), "log_w": float(w),
         "reference_log_w": float(r), "eligible": bool(k <= upper)}
        for k, g, s, w, r in zip(counts, gap, error, observed, reference.mean(axis=0))]
    return selected, details
