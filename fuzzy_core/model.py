"""Rayleigh membership functions and finite alpha-cut approximations."""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Protocol
import math
import numpy as np
from scipy.optimize import brentq


@dataclass(frozen=True)
class ComputationConfig:
    epsilon: float = 0.001
    levels: int = 401

    def __post_init__(self):
        if not math.isfinite(self.epsilon) or not 0 < self.epsilon < 1:
            raise ValueError("Порог ε должен быть между 0 и 1.")
        if isinstance(self.levels, bool) or not isinstance(self.levels, int) or not 3 <= self.levels <= 10001:
            raise ValueError("Количество уровней должно быть целым числом от 3 до 10001.")

    def grid(self) -> np.ndarray:
        return np.linspace(self.epsilon, 1.0, self.levels)


def _level(alpha: float):
    if not math.isfinite(alpha) or not 0 < alpha <= 1:
        raise ValueError("Уровень α должен принадлежать (0, 1].")


@lru_cache(maxsize=20000)
def unit_cut(alpha: float) -> tuple[float, float]:
    """Dimensionless roots, shared by all scale parameters."""
    _level(alpha)
    if alpha == 1:
        return 1.0, 1.0
    def residual(t):
        return t * math.exp((1 - t * t) / 2) - alpha
    upper = 2.0
    while residual(upper) > 0:
        upper *= 2
    return (brentq(residual, 0, 1, xtol=1e-14),
            brentq(residual, 1, upper, xtol=1e-14))


class FuzzyValue(Protocol):
    label: str
    def alpha_cut(self, alpha: float) -> tuple[float, float]: ...
    def membership(self, x): ...
    def to_alpha_cuts(self, config: ComputationConfig = ComputationConfig()) -> "AlphaCutFuzzy": ...


@dataclass(frozen=True)
class RayleighFuzzy:
    sigma: float
    label: str = ""

    def __post_init__(self):
        if isinstance(self.sigma, bool) or not math.isfinite(self.sigma) or self.sigma <= 0:
            raise ValueError("Параметр σ должен быть конечным положительным числом.")
        object.__setattr__(self, "sigma", float(self.sigma))

    def membership(self, x):
        a = np.asarray(x, dtype=float)
        if np.any(np.isnan(a)):
            raise ValueError("Аргумент функции принадлежности не может быть NaN.")
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            t = a / self.sigma
            log_mu = np.log(t) + (1 - t * t) / 2
            result = np.where((t > 0) & np.isfinite(t), np.exp(log_mu), 0.0)
        return float(result) if a.ndim == 0 else result

    def alpha_cut(self, alpha: float) -> tuple[float, float]:
        a, b = unit_cut(float(alpha))
        return self.sigma * a, self.sigma * b

    def to_alpha_cuts(self, config: ComputationConfig = ComputationConfig()) -> "AlphaCutFuzzy":
        grid = config.grid()
        bounds = np.array([self.alpha_cut(a) for a in grid])
        return AlphaCutFuzzy(grid, bounds[:, 0], bounds[:, 1], self.label)


@dataclass(frozen=True, eq=False)
class AlphaCutFuzzy:
    alpha: np.ndarray
    left: np.ndarray
    right: np.ndarray
    label: str = ""

    def __post_init__(self):
        for name in ("alpha", "left", "right"):
            a = np.array(getattr(self, name), dtype=float, copy=True)
            if a.ndim != 1 or not np.all(np.isfinite(a)):
                raise ValueError("Срезы должны быть одномерными массивами конечных чисел.")
            a.flags.writeable = False
            object.__setattr__(self, name, a)
        if len(self.alpha) < 2 or len(self.left) != len(self.alpha) or len(self.right) != len(self.alpha):
            raise ValueError("Массивы срезов должны иметь одинаковую длину не менее двух.")
        if self.alpha[0] <= 0 or self.alpha[-1] != 1 or np.any(np.diff(self.alpha) <= 0):
            raise ValueError("Уровни α должны строго возрастать от положительного порога до 1.")
        if (np.any(self.left > self.right) or np.any(np.diff(self.left) < -1e-10)
                or np.any(np.diff(self.right) > 1e-10)):
            raise ValueError("Интервалы срезов должны быть упорядочены и вложены.")

    def alpha_cut(self, alpha: float) -> tuple[float, float]:
        _level(alpha)
        if alpha < self.alpha[0] - 1e-12:
            raise ValueError("Уровень α ниже вычислительного порога ε этой величины.")
        return (float(np.interp(alpha, self.alpha, self.left)),
                float(np.interp(alpha, self.alpha, self.right)))

    def membership(self, x):
        """Piecewise-linear reconstruction, with vertical cutoff at epsilon."""
        a = np.asarray(x, dtype=float)
        if np.any(np.isnan(a)):
            raise ValueError("Аргумент функции принадлежности не может быть NaN.")
        up = np.interp(a, self.left, self.alpha, left=0, right=1)
        down = np.interp(a, self.right[::-1], self.alpha[::-1], left=1, right=0)
        result = np.minimum(up, down)
        return float(result) if a.ndim == 0 else result

    def to_alpha_cuts(self, config: ComputationConfig = ComputationConfig()) -> "AlphaCutFuzzy":
        if config.epsilon < self.alpha[0] - 1e-12:
            raise ValueError("Нельзя восстановить отсечённый хвост при меньшем ε.")
        grid = config.grid()
        return AlphaCutFuzzy(grid, np.interp(grid, self.alpha, self.left),
                             np.interp(grid, self.alpha, self.right), self.label)


@dataclass(frozen=True)
class FuzzyItem:
    id: str
    value: RayleighFuzzy | AlphaCutFuzzy

    def __post_init__(self):
        if not isinstance(self.id, str) or not self.id.strip() or len(self.id) > 100:
            raise ValueError("Идентификатор должен быть непустой строкой до 100 символов.")
        if not isinstance(self.value, (RayleighFuzzy, AlphaCutFuzzy)):
            raise ValueError("Неизвестный тип нечёткой величины.")
