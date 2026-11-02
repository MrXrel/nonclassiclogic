"""Complete linkage and HDBSCAN with the same fuzzy distance; no UI dependency."""
from dataclasses import dataclass, replace
import math
import numpy as np
from scipy.cluster.hierarchy import linkage, cut_tree, fcluster
from scipy.spatial.distance import pdist, squareform
from fuzzy_core import ComputationConfig, FuzzyItem, RayleighFuzzy, rank
from ._auto import select_cluster_count
from .validation import adjusted_rand_index


@dataclass(frozen=True)
class ClusterResult:
    labels: np.ndarray
    linkage: np.ndarray
    distances: np.ndarray
    mode: str
    k: int | None = None
    threshold: float | None = None
    auto: dict | None = None
    method: str = "complete"
    density: dict | None = None

    def to_dict(self):
        result = {"method": self.method, "mode": self.mode, "k": self.k,
                  "threshold": self.threshold, "labels": self.labels.tolist()}
        if self.auto is not None:
            result["auto"] = self.auto
        if self.density is not None:
            result["density"] = self.density
        return result


def cluster_hdbscan(items, config=ComputationConfig(), *, min_cluster_size=10,
                    min_samples=10, allow_single_cluster=True) -> ClusterResult:
    """Density clustering using the SAME alpha-cut distance as complete linkage.

    Positive labels are ordered by mean rank; 0 is noise, never a cluster.
    min_samples includes the point itself (scikit-learn convention).
    A sample smaller than either minimum is entirely noise.
    """
    from ._density import fit_density
    matrix, values = distance_matrix(items, config)
    raw, settings = fit_density(matrix, min_cluster_size=min_cluster_size,
        min_samples=min_samples, allow_single_cluster=allow_single_cluster)
    scores = np.array([rank(v, config) for v in values])
    order = sorted(set(raw) - {-1}, key=lambda lab: (float(scores[raw == lab].mean()), int(lab)))
    mapping = {lab: j + 1 for j, lab in enumerate(order)}
    labels = np.array([mapping.get(lab, 0) for lab in raw], dtype=int)
    # The complete-linkage dendrogram must never be presented as an HDBSCAN tree.
    return ClusterResult(labels, np.empty((0, 4)), matrix, "density", len(order),
                         method="hdbscan", density=settings)


def distance_matrix(items, config=ComputationConfig()):
    """Trapezoid-weighted Euclidean embedding of both alpha endpoints."""
    if not items:
        raise ValueError("Набор для кластеризации пуст.")
    values = [i.value if isinstance(i, FuzzyItem) else i for i in items]
    cuts = [v.to_alpha_cuts(config) for v in values]
    grid = cuts[0].alpha
    dx = np.diff(grid)
    weights = np.r_[dx[0] / 2, (dx[:-1] + dx[1:]) / 2, dx[-1] / 2]
    weights = np.sqrt(weights / (2 * (1 - config.epsilon)))
    features = np.array([np.r_[c.left * weights, c.right * weights] for c in cuts])
    return squareform(pdist(features)), values


def _cluster(items, config, mode, k=None, threshold=None):
    matrix, values = distance_matrix(items, config)
    n = len(values)
    z = linkage(squareform(matrix, checks=False), method="complete") if n > 1 else np.empty((0, 4))
    if n == 1:
        labels = np.ones(1, dtype=int)
    elif mode == "count":
        labels = cut_tree(z, n_clusters=[k]).ravel().astype(int) + 1
    else:
        labels = fcluster(z, t=threshold, criterion="distance")
    # Stable, interpretable palette: clusters ordered from small to large rank.
    scores = np.array([rank(v, config) for v in values])
    order = sorted(set(labels), key=lambda lab: (float(scores[labels == lab].mean()), int(lab)))
    mapping = {lab: j + 1 for j, lab in enumerate(order)}
    labels = np.array([mapping[lab] for lab in labels], dtype=int)
    return ClusterResult(labels, z, matrix, mode, k, threshold)


def cluster_by_count(items, k: int, config=ComputationConfig()) -> ClusterResult:
    if isinstance(k, bool) or not isinstance(k, int) or not 1 <= k <= len(items):
        raise ValueError("K должно быть целым числом от 1 до размера набора.")
    return _cluster(items, config, "count", k=k)


def cluster_by_threshold(items, threshold: float, config=ComputationConfig()) -> ClusterResult:
    if not math.isfinite(threshold) or threshold < 0:
        raise ValueError("Порог расстояния должен быть конечным неотрицательным числом.")
    return _cluster(items, config, "threshold", threshold=float(threshold))


def cluster_auto(items, config=ComputationConfig(), *, max_clusters=10,
                 reference_samples=100, seed=42) -> ClusterResult:
    """Choose K via the gap statistic, then cut the same complete-linkage tree.

    The automatic reference model is defined for original Rayleigh values.
    Manual count/threshold clustering also accepts general alpha-cut values.
    """
    values = [i.value if isinstance(i, FuzzyItem) else i for i in items]
    if not values or not all(isinstance(v, RayleighFuzzy) for v in values):
        raise ValueError("Автоподбор работает с непустым набором исходных величин Рэлея.")
    k, details = select_cluster_count([v.sigma for v in values], max_clusters=max_clusters,
                                       reference_samples=reference_samples, seed=seed)
    return replace(cluster_by_count(items, k, config), mode="auto", auto=details)
