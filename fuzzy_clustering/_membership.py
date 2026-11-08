"""A fixed, within-group score for a previously computed partition."""
from __future__ import annotations

from dataclasses import dataclass
from numbers import Real
from typing import TYPE_CHECKING

import numpy as np

from fuzzy_core import AlphaCutFuzzy, ComputationConfig, FuzzyItem

if TYPE_CHECKING:
    from . import ClusterResult


def _cut_features(items, config):
    if not items:
        raise ValueError("Набор для кластеризации пуст.")
    values = [i.value if isinstance(i, FuzzyItem) else i for i in items]
    cuts = [v.to_alpha_cuts(config) for v in values]
    dx = np.diff(cuts[0].alpha)
    weights = np.r_[dx[0] / 2, (dx[:-1] + dx[1:]) / 2, dx[-1] / 2]
    weights = np.sqrt(weights / (2 * (1 - config.epsilon)))
    features = np.array([np.r_[c.left * weights, c.right * weights] for c in cuts])
    return features, cuts, values


@dataclass(frozen=True)
class ClusterMembership:
    """Scores aligned to the original labels. Noise has NaN internally."""

    labels: np.ndarray
    scores: np.ndarray
    centers: dict[int, AlphaCutFuzzy]
    radii: dict[int, float]

    def to_dict(self):
        return {"rule": "gaussian_rms_v1",
                "scores": [float(u) if label else None for label, u in zip(self.labels, self.scores)]}


def cluster_membership(items, result: ClusterResult, config=ComputationConfig()) -> ClusterMembership:
    """Use the original assigned objects to build each center and RMS radius."""
    features, cuts, _ = _cut_features(items, config)
    labels = np.array(result.labels, copy=True)
    if (labels.shape != (len(cuts),) or not np.issubdtype(labels.dtype, np.integer)
            or np.any(labels < 0)):
        raise ValueError("Метки кластеров не соответствуют набору.")
    scores = np.full(len(cuts), np.nan)
    centers, radii = {}, {}
    for group in sorted(set(labels) - {0}):
        idx = np.flatnonzero(labels == group)
        z = features[idx]
        delta = np.linalg.norm(z - z.mean(axis=0), axis=1)
        radius = float(np.linalg.norm(delta) / np.sqrt(len(idx)))
        tolerance = 64 * np.finfo(float).eps * np.max(np.linalg.norm(z, axis=1))
        if np.all(delta <= tolerance):
            radius = 0.0
        if radius == 0:
            scores[idx] = 1.0
        else:
            u = np.exp(-0.5 * (delta / radius) ** 2)
            scores[idx] = np.where(delta <= tolerance, 1.0, np.minimum(u, np.nextafter(1., 0.)))
        centers[int(group)] = AlphaCutFuzzy(
            cuts[0].alpha, np.mean([cuts[i].left for i in idx], axis=0),
            np.mean([cuts[i].right for i in idx], axis=0), f"Центр {group}")
        radii[int(group)] = radius
    labels.flags.writeable = scores.flags.writeable = False
    return ClusterMembership(labels, scores, centers, radii)


def membership_mask(profile: ClusterMembership, alpha: float) -> np.ndarray:
    if isinstance(alpha, (bool, np.bool_)) or not isinstance(alpha, Real) or not 0 <= alpha <= 1:
        raise ValueError("Порог соответствия α должен быть числом от 0 до 1.")
    return (profile.labels > 0) & (profile.scores >= alpha)
