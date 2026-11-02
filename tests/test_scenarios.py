import copy
import numpy as np
import pytest
from fuzzy_core import (SCENARIOS, FuzzyItem, RayleighFuzzy, generate_scenario, generate,
                        SessionData, dumps_session, loads_session)
from fuzzy_clustering import cluster_auto, adjusted_rand_index


@pytest.mark.parametrize("scenario", list(SCENARIOS))
@pytest.mark.parametrize("n", [100, 300])
def test_scenarios_reproducible_positive_and_exact_size(scenario, n):
    batch = generate_scenario(n, 42, scenario)
    again = generate_scenario(n, 42, scenario)
    assert len(batch.items) == n and batch.provenance == again.provenance
    assert [i.value.sigma for i in batch.items] == [i.value.sigma for i in again.items]
    assert all(i.value.sigma > 0 for i in batch.items)
    assert sum(c["size"] for c in batch.provenance["components"]) + batch.provenance["outliers"] == n
    restored = loads_session(dumps_session(SessionData(batch.items, generation=batch.provenance)))
    assert restored.generation == batch.provenance


def test_legacy_demo_still_has_identical_values():
    batch = generate_scenario(scenario="demo")
    assert [i.value.sigma for i in batch.items] == [i.value.sigma for i in generate()]


def test_custom_source_controls_and_random_scenarios_vary():
    batch = generate_scenario(100, 42, "custom", groups=6, separation=9, imbalance=.9, outlier_fraction=.2)
    assert len(batch.provenance["components"]) == 6 and batch.provenance["outliers"] == 20
    assert len(set(c["size"] for c in batch.provenance["components"])) > 1
    group_counts = {len(generate_scenario(seed=s, scenario="random_mix").provenance["components"]) for s in range(10)}
    assert len(group_counts) >= 3


def test_clear_unequal_groups_and_overlap_are_honest_checks():
    clear = generate_scenario(scenario="unequal")
    result = cluster_auto(clear.items)
    origins = [clear.provenance["origin_labels"][i.id] for i in clear.items]
    assert result.k == 4
    assert adjusted_rand_index(origins, result.labels) > .95
    mixed = generate_scenario(scenario="overlap")
    result = cluster_auto(mixed.items)
    origins = [mixed.provenance["origin_labels"][i.id] for i in mixed.items]
    assert adjusted_rand_index(origins, result.labels) < .8


def test_provenance_invalidated_if_values_change():
    batch = generate_scenario(scenario="unequal")
    batch.items[0] = FuzzyItem(batch.items[0].id, RayleighFuzzy(100))
    with pytest.raises(ValueError, match="изменились"):
        dumps_session(SessionData(batch.items, generation=batch.provenance))


@pytest.mark.parametrize("settings", [{"groups": 1}, {"groups": True}, {"separation": 0},
    {"imbalance": -1}, {"outlier_fraction": .3}, {"sigma_max": float("inf")}])
def test_invalid_scenario_controls(settings):
    with pytest.raises(ValueError):
        generate_scenario(scenario="custom", **settings)


@pytest.mark.parametrize("scenario", [[], None, True, "missing"])
def test_invalid_scenario_name(scenario):
    with pytest.raises(ValueError):
        generate_scenario(scenario=scenario)


def test_ari_is_invariant_to_numbering_and_adjusted_for_chance():
    assert adjusted_rand_index([1, 1, 2, 2], [7, 7, 3, 3]) == 1
    assert adjusted_rand_index([1, 1, 2, 2], [1, 2, 1, 2]) == pytest.approx(-.5)
    assert adjusted_rand_index([1, 1, 1], [2, 2, 2]) == 1
    assert adjusted_rand_index([1, 2, 3], [7, 8, 9]) == 1
    with pytest.raises(ValueError):
        adjusted_rand_index([1], [1])
