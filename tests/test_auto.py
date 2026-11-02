import copy
import numpy as np
import pytest
from fuzzy_core import (RayleighFuzzy, AlphaCutFuzzy, generate, SessionData,
                        dumps_session, loads_session)
from fuzzy_clustering import cluster_auto, cluster_by_count


@pytest.mark.parametrize("n", [100, 150, 300])
def test_auto_recovers_demo_groups_without_generation_labels(n):
    items = generate(n, seed=42)
    result = cluster_auto(items)
    assert result.k == 3 and result.mode == "auto"
    np.testing.assert_array_equal(result.labels, cluster_by_count(items, 3).labels)
    assert sum(np.bincount(result.labels)[1:]) == n


@pytest.mark.parametrize("k", [2, 4])
def test_auto_does_not_assume_three_groups(k):
    values = [RayleighFuzzy(s) for j in range(k)
              for s in np.linspace(1 + 3*j, 1.1 + 3*j, 25)]
    assert cluster_auto(values).k == k


def test_uniform_reference_can_leave_one_group():
    assert cluster_auto(generate(100, seed=42, mode="uniform")).k == 1


def test_duplicates_and_tiny_samples():
    for n in [1, 2, 100]:
        result = cluster_auto([RayleighFuzzy(2)] * n)
        assert result.k == 1 and result.auto["reason"] == "identical"
    assert cluster_auto([RayleighFuzzy(1), RayleighFuzzy(5)]).k == 1
    result = cluster_auto([RayleighFuzzy(1)] * 20 + [RayleighFuzzy(4)] * 20)
    assert result.k == 2
    assert all(np.isfinite(row["gap"]) for row in result.auto["candidates"])


def test_global_one_se_rule_and_reproducibility():
    values = generate()
    result = cluster_auto(values)
    rows = result.auto["candidates"]
    best = max(rows, key=lambda row: row["gap"])
    expected = min(row["k"] for row in rows if row["gap"] >= best["gap"] - best["se"])
    assert result.k == expected
    assert result.to_dict() == cluster_auto(values).to_dict()
    scaled = [RayleighFuzzy(item.value.sigma * 7 + 3) for item in values]
    assert cluster_auto(scaled).k == result.k
    assert cluster_auto(values, max_clusters=2).k <= 2


def test_auto_session_round_trip_and_validation():
    items = generate()
    result = cluster_auto(items)
    restored = loads_session(dumps_session(SessionData(items, clustering=result.to_dict())))
    settings = restored.clustering["auto"]
    recalculated = cluster_auto(restored.items, **{key: settings[key] for key in
                                ("max_clusters", "reference_samples", "seed")})
    assert recalculated.to_dict() == result.to_dict()
    bad = copy.deepcopy(restored)
    bad.clustering["auto"]["reference_samples"] = 0
    with pytest.raises(ValueError):
        dumps_session(bad)


@pytest.mark.parametrize("settings", [{"max_clusters": 0}, {"max_clusters": True},
                          {"reference_samples": 1}, {"seed": -1}])
def test_bad_auto_parameters(settings):
    with pytest.raises(ValueError):
        cluster_auto(generate(100), **settings)


def test_auto_rejects_non_rayleigh_reference():
    with pytest.raises(ValueError):
        cluster_auto([])
    with pytest.raises(ValueError):
        cluster_auto([AlphaCutFuzzy([.001, 1], [0, 1], [2, 1])])
