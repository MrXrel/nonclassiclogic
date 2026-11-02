"""Synthetic Rayleigh-scale scenarios and separate evaluation provenance."""
from collections import Counter
from dataclasses import dataclass
import math
import hashlib
import json
import numpy as np
from .model import FuzzyItem, RayleighFuzzy
from .data import generate

SCENARIOS = {
    "demo": ("Три разделённые группы", "Исходный контрольный пример: три одинаковые группы в отдельных диапазонах σ."),
    "uniform": ("Один равномерный диапазон", "Набор без специально заложенных сгущений."),
    "random_mix": ("Случайная смесь", "От двух до шести источников с разными центрами, шириной и числом объектов. Меняйте seed."),
    "unequal": ("Разные размеры и ширина", "Четыре источника: большой и компактный, средние и маленький широкий."),
    "overlap": ("Перекрывающиеся группы", "Три источника сильно перемешаны. По одному σ их часто нельзя разделить."),
    "outliers": ("Группы и выбросы", "Три компактных источника и 8% объектов вне их диапазонов. Проверка чувствительности к выбросам."),
    "custom": ("Конструктор смеси", "Задайте число источников, разделённость, неравномерность размеров и долю выбросов."),
}


@dataclass
class GeneratedDataset:
    items: list[FuzzyItem]
    provenance: dict


def _items_digest(items):
    values = sorted((i.id, float(i.value.sigma).hex()) for i in items)
    return hashlib.sha256(json.dumps(values, ensure_ascii=False).encode("utf-8")).hexdigest()


def _positive_normal(rng, center, spread, size):
    values = rng.normal(center, spread, size)
    bad = values <= 0.000001
    while bad.any():
        values[bad] = rng.normal(center, spread, int(bad.sum()))
        bad = values <= 0.000001
    return values


def generate_scenario(n=150, seed=42, scenario="random_mix", *, groups=4,
                      separation=7.0, imbalance=0.5, outlier_fraction=0.0,
                      sigma_min=0.5, sigma_max=6.0) -> GeneratedDataset:
    if not isinstance(scenario, str) or scenario not in SCENARIOS:
        raise ValueError("Неизвестный сценарий генерации.")
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in (sigma_min, sigma_max)) or not 0 < sigma_min < sigma_max:
        raise ValueError("Границы должны удовлетворять 0 < σmin < σmax.")
    # Preserve the original generator's validation and its exact demo sequence.
    basic = generate(n, seed, "uniform" if scenario == "uniform" else "demo", sigma_min, sigma_max)
    if isinstance(groups, bool) or not isinstance(groups, int) or not 2 <= groups <= 8:
        raise ValueError("В конструкторе допускается от 2 до 8 источников.")
    for value, low, high, name in [(separation, 1, 12, "Разделённость"),
                                  (imbalance, 0, 1, "Неравномерность"),
                                  (outlier_fraction, 0, .2, "Доля выбросов")]:
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not low <= value <= high:
            raise ValueError(f"{name}: допустим диапазон от {low} до {high}.")
    parameters = {"groups": groups, "separation": float(separation), "imbalance": float(imbalance),
                  "outlier_fraction": float(outlier_fraction), "sigma_min": float(sigma_min),
                  "sigma_max": float(sigma_max)}
    if scenario in ("demo", "uniform"):
        ranges = [(sigma_min, sigma_max)] if scenario == "uniform" else [(.6, 1.2), (2.4, 3), (4.8, 5.4)]
        origins = {item.id: next(j+1 for j, (lo, hi) in enumerate(ranges) if lo <= item.value.sigma <= hi) for item in basic}
        components = [{"label": j+1, "center": (lo+hi)/2, "spread": (hi-lo)/math.sqrt(12),
                       "size": sum(v == j+1 for v in origins.values()), "distribution": "uniform"}
                      for j, (lo, hi) in enumerate(ranges)]
        parameters["groups"] = len(ranges)
        return GeneratedDataset(basic, {"schema_version": 1, "scenario": scenario, "n": n, "seed": seed,
                "parameters": parameters, "components": components, "origin_labels": origins, "outliers": 0,
                "items_digest": _items_digest(basic)})

    rng = np.random.default_rng(seed)
    if scenario == "random_mix":
        groups = int(rng.integers(2, 7))
        separation, imbalance, outlier_fraction = float(rng.uniform(4, 9)), float(rng.uniform(.1, .75)), 0.0
    elif scenario == "unequal":
        groups, separation, imbalance, outlier_fraction = 4, 7.0, .8, 0.0
    elif scenario == "overlap":
        groups, outlier_fraction = 3, 0.0
    elif scenario == "outliers":
        groups, outlier_fraction = 3, .08
    centers = .8 + np.r_[0, np.cumsum(.22 * separation * rng.uniform(.85, 1.15, groups-1))]
    spreads = .22 * rng.uniform(.6, 1.4, groups)
    weights = (1-imbalance)/groups + imbalance*rng.dirichlet(np.full(groups, .4))
    if scenario == "unequal":
        centers, spreads, weights = np.array([.9, 2.7, 4.6, 6.8]), np.array([.08, .2, .13, .34]), np.array([.5, .27, .16, .07])
    elif scenario == "overlap":
        centers, spreads, weights = np.array([1.4, 2.1, 3.0]), np.array([.5, .65, .5]), np.array([.35, .4, .25])
    elif scenario == "outliers":
        centers, spreads, weights = np.array([1.0, 3.0, 5.0]), np.array([.1, .13, .15]), np.ones(3)/3
    noise = int(round(n*outlier_fraction))
    shares = (n-noise-2*groups)*weights
    sizes = np.floor(shares).astype(int)+2
    for j in np.argsort(-(shares-np.floor(shares)))[:n-noise-int(sizes.sum())]:
        sizes[j] += 1
    sigmas = [_positive_normal(rng, float(c), float(s), int(size)) for c, s, size in zip(centers, spreads, sizes)]
    origin_arrays = [np.full(size, j+1, dtype=int) for j, size in enumerate(sizes)]
    if noise:
        # Exterior intervals keep these objects distinct from normal tails.
        right_low = float(max(centers + 4*spreads) + 1)
        left_size = noise//3
        sigmas.append(np.r_[rng.uniform(.1, .3, left_size), rng.uniform(right_low, right_low+2.5, noise-left_size)])
        origin_arrays.append(np.zeros(noise, dtype=int))
    sigmas, labels = np.concatenate(sigmas), np.concatenate(origin_arrays)
    order = rng.permutation(n)
    items = [FuzzyItem(f"R{j+1:03d}", RayleighFuzzy(float(sigmas[i]), f"R{j+1:03d}")) for j, i in enumerate(order)]
    origins = {item.id: int(labels[i]) for item, i in zip(items, order)}
    components = [{"label": j+1, "center": float(c), "spread": float(s), "size": int(size),
                   "distribution": "positive_normal"} for j, (c, s, size) in enumerate(zip(centers, spreads, sizes))]
    parameters.update(groups=groups, separation=float(separation), imbalance=float(imbalance), outlier_fraction=float(outlier_fraction))
    provenance = {"schema_version": 1, "scenario": scenario, "n": n, "seed": seed,
                  "parameters": parameters, "components": components, "origin_labels": origins, "outliers": noise,
                  "items_digest": _items_digest(items)}
    validate_provenance(provenance, items)
    return GeneratedDataset(items, provenance)


def validate_provenance(provenance, items):
    """Reject stale/misaligned labels; these are evaluation data, never features."""
    if (not isinstance(provenance, dict) or provenance.get("schema_version") != 1
            or not isinstance(provenance.get("scenario"), str) or provenance["scenario"] not in SCENARIOS):
        raise ValueError("Неверные сведения о генерации.")
    if provenance.get("n") != len(items) or not isinstance(provenance.get("parameters"), dict):
        raise ValueError("Сведения о генерации не соответствуют размеру набора.")
    if provenance.get("items_digest") != _items_digest(items):
        raise ValueError("Величины изменились: исходные метки генерации больше недействительны.")
    seed = provenance.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= 2**32-1:
        raise ValueError("Неверный seed генератора.")
    labels, components = provenance.get("origin_labels"), provenance.get("components")
    if not isinstance(labels, dict) or set(labels) != {i.id for i in items} or not isinstance(components, list) or not 1 <= len(components) <= 8:
        raise ValueError("Исходные метки не соответствуют набору.")
    if any(isinstance(v, bool) or not isinstance(v, int) or not 0 <= v <= len(components) for v in labels.values()):
        raise ValueError("Неверные номера источников.")
    counts = Counter(labels.values())
    if provenance.get("outliers") != counts[0]:
        raise ValueError("Неверное количество выбросов.")
    for j, component in enumerate(components, 1):
        if not isinstance(component, dict) or component.get("label") != j or component.get("size") != counts[j] or counts[j] < 1:
            raise ValueError("Неверное описание источника генерации.")
        for key in ("center", "spread"):
            value = component.get(key)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError("Параметры источника должны быть положительными и конечными.")
        if component.get("distribution") not in ("uniform", "positive_normal"):
            raise ValueError("Неизвестная форма источника генерации.")
