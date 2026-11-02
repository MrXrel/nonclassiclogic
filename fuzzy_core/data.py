"""Dataset generation and a validated, portable JSON session format."""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
import csv
import io
import json
import math
import numpy as np
from .model import ComputationConfig, RayleighFuzzy, AlphaCutFuzzy, FuzzyItem


def generate(n: int = 150, seed: int = 42, mode: str = "demo",
             sigma_min: float = 0.5, sigma_max: float = 6.0) -> list[FuzzyItem]:
    if isinstance(n, bool) or not isinstance(n, int) or not 100 <= n <= 300:
        raise ValueError("Генератор создаёт от 100 до 300 величин.")
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= 2**32 - 1:
        raise ValueError("Seed должен быть целым числом от 0 до 4294967295.")
    rng = np.random.default_rng(seed)
    if mode == "demo":
        ranges = [(0.6, 1.2), (2.4, 3.0), (4.8, 5.4)]
        sigmas = np.concatenate([rng.uniform(lo, hi, n // 3 + (i < n % 3))
                                 for i, (lo, hi) in enumerate(ranges)])
        rng.shuffle(sigmas)
    elif mode == "uniform":
        if not (math.isfinite(sigma_min) and math.isfinite(sigma_max) and 0 < sigma_min < sigma_max):
            raise ValueError("Границы диапазона должны удовлетворять 0 < σmin < σmax.")
        sigmas = rng.uniform(sigma_min, sigma_max, n)
    else:
        raise ValueError("Неизвестный режим генерации.")
    return [FuzzyItem(f"R{i + 1:03d}", RayleighFuzzy(s, f"R{i + 1:03d}")) for i, s in enumerate(sigmas)]


def value_dict(value):
    if isinstance(value, RayleighFuzzy):
        return {"type": "rayleigh", "sigma": value.sigma, "label": value.label}
    return {"type": "alpha_cuts", "alpha": value.alpha.tolist(), "left": value.left.tolist(),
            "right": value.right.tolist(), "label": value.label}


def parse_value(data):
    if not isinstance(data, dict) or not isinstance(data.get("label", ""), str):
        raise ValueError("Неверное описание величины.")
    if data.get("type") == "rayleigh":
        return RayleighFuzzy(data["sigma"], data.get("label", ""))
    if data.get("type") == "alpha_cuts":
        if len(data.get("alpha", [])) > 10001:
            raise ValueError("Слишком много α-срезов.")
        return AlphaCutFuzzy(data["alpha"], data["left"], data["right"], data.get("label", ""))
    raise ValueError("Неизвестный тип величины в JSON.")


@dataclass
class SessionData:
    items: list[FuzzyItem]
    results: list[FuzzyItem] = field(default_factory=list)
    config: ComputationConfig = field(default_factory=ComputationConfig)
    clustering: dict | None = None
    controls: dict = field(default_factory=dict)
    generation: dict | None = None
    comparison: dict | None = None


def validate_session(session: SessionData):
    if not 1 <= len(session.items) <= 300 or len(session.results) > 100:
        raise ValueError("Сессия должна содержать 1–300 исходных величин и до 100 результатов.")
    ids = [i.id for i in session.items + session.results]
    if len(set(ids)) != len(ids):
        raise ValueError("Идентификаторы величин должны быть уникальными.")
    if not all(isinstance(i.value, RayleighFuzzy) for i in session.items):
        raise ValueError("Исходный набор должен содержать величины Рэлея.")
    for result in session.results:
        result.value.to_alpha_cuts(session.config)
    if not isinstance(session.controls, dict):
        raise ValueError("Неверные настройки интерфейса.")
    if session.generation is not None:
        from .scenarios import validate_provenance
        validate_provenance(session.generation, session.items)
    c = session.clustering
    if session.comparison is not None:
        d = session.comparison
        if (c is None or not isinstance(d, dict) or d.get("method") != "hdbscan"
                or d.get("mode") != "density"):
            raise ValueError("Сравнение HDBSCAN должно сопровождать основную кластеризацию.")
        labels = d.get("labels", [])
        if (not isinstance(labels, list) or len(labels) != len(session.items)
                or any(isinstance(x, bool) or not isinstance(x, int) or x < 0 for x in labels)):
            raise ValueError("Неверные метки HDBSCAN: 0 означает шум, группы начинаются с 1.")
        groups = sorted(set(labels) - {0})
        if groups != list(range(1, len(groups) + 1)) or type(d.get("k")) is not int or d["k"] != len(groups):
            raise ValueError("Некорректное число групп HDBSCAN.")
        settings = d.get("density")
        if (not isinstance(settings, dict) or settings.get("implementation") != "scikit-learn"
                or settings.get("metric") != "precomputed_alpha_cuts"
                or settings.get("cluster_selection_method") != "eom"
                or settings.get("noise_label") != 0
                or not isinstance(settings.get("allow_single_cluster"), bool)):
            raise ValueError("Неверные настройки HDBSCAN.")
        for name, minimum in [("min_cluster_size", 2), ("min_samples", 1)]:
            value = settings.get(name)
            if type(value) is not int or not minimum <= value <= 1_000_000:
                raise ValueError(f"Недопустимая настройка HDBSCAN: {name}.")
        if any(labels.count(group) < settings["min_cluster_size"] for group in groups):
            raise ValueError("Группа HDBSCAN меньше заданного минимального размера.")
    if c is not None:
        if not isinstance(c, dict) or c.get("method") != "complete" or c.get("mode") not in ("count", "threshold", "auto"):
            raise ValueError("Неверные настройки кластеризации.")
        labels = c.get("labels", [])
        if len(labels) != len(session.items) or any(isinstance(x, bool) or not isinstance(x, int) or x < 1 for x in labels):
            raise ValueError("Метки кластеров не соответствуют набору.")
        if sorted(set(labels)) != list(range(1, len(set(labels)) + 1)):
            raise ValueError("Номера кластеров должны быть последовательными, начиная с 1.")
        if c["mode"] in ("count", "auto"):
            k = c.get("k")
            if isinstance(k, bool) or not isinstance(k, int) or not 1 <= k <= len(session.items) or len(set(labels)) != k:
                raise ValueError("Некорректное число кластеров.")
            if c["mode"] == "auto":
                auto = c.get("auto")
                if (not isinstance(auto, dict) or auto.get("criterion") != "gap"
                        or auto.get("rule") != "global_1se"
                        or auto.get("reference") != "uniform_sigma_range"):
                    raise ValueError("Неизвестный критерий автоподбора.")
                for name, low, high in [("max_clusters", 1, 300), ("reference_samples", 2, 1000),
                                        ("seed", 0, 2**32 - 1)]:
                    value = auto.get(name)
                    if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
                        raise ValueError(f"Недопустимая настройка автоподбора: {name}.")
                if k > auto["max_clusters"]:
                    raise ValueError("Выбранное K превышает предел автоподбора.")
        else:
            t = c.get("threshold")
            if not isinstance(t, (int, float)) or not math.isfinite(t) or t < 0:
                raise ValueError("Некорректный порог расстояния.")


def dumps_session(session: SessionData) -> str:
    validate_session(session)
    payload = {"schema_version": 1, "config": asdict(session.config),
               "items": [{"id": i.id, "value": value_dict(i.value)} for i in session.items],
               "results": [{"id": i.id, "value": value_dict(i.value)} for i in session.results],
               "clustering": session.clustering, "controls": session.controls, "generation": session.generation,
               "comparison": session.comparison}
    return json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False)


def loads_session(text: str | bytes) -> SessionData:
    if len(text) > 10_000_000:
        raise ValueError("Файл сессии должен быть меньше 10 МБ.")
    try:
        data = json.loads(text, parse_constant=lambda s: (_ for _ in ()).throw(ValueError("NaN и Infinity запрещены.")))
        if not isinstance(data, dict) or data.get("schema_version") != 1:
            raise ValueError("Поддерживается schema_version = 1.")
        if not isinstance(data.get("items"), list) or not isinstance(data.get("results", []), list):
            raise ValueError("Наборы величин должны быть массивами.")
        s = SessionData([FuzzyItem(i["id"], parse_value(i["value"])) for i in data["items"]],
                        [FuzzyItem(i["id"], parse_value(i["value"])) for i in data.get("results", [])],
                        ComputationConfig(**data["config"]), data.get("clustering"), data.get("controls", {}),
                        data.get("generation"), data.get("comparison"))
        validate_session(s)
        return s
    except (KeyError, TypeError, OverflowError, json.JSONDecodeError) as e:
        raise ValueError("Неверный формат JSON-сессии.") from e


def dataset_csv(items, labels=None, hdbscan_labels=None) -> str:
    if any(v is not None and len(v) != len(items) for v in (labels, hdbscan_labels)):
        raise ValueError("Число меток не совпадает с числом величин.")
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["id", "sigma", "cluster"] +
                    ([] if hdbscan_labels is None else ["hdbscan_cluster", "hdbscan_noise"]))
    for j, item in enumerate(items):
        row = [item.id, item.value.sigma, "" if labels is None else int(labels[j])]
        if hdbscan_labels is not None:
            row += [int(hdbscan_labels[j]), int(hdbscan_labels[j] == 0)]
        writer.writerow(row)
    return output.getvalue()
