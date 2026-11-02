import csv
import io
import numpy as np
import pytest
from sklearn.cluster import HDBSCAN
from fuzzy_core import (RayleighFuzzy, ComputationConfig, generate_scenario,
                        SessionData, dumps_session, loads_session, dataset_csv, subtract)
from fuzzy_clustering import cluster_auto, cluster_hdbscan, adjusted_rand_index
from lab_support import comparison_figure, cluster_color, NOISE_COLOR


@pytest.mark.parametrize("n", [100, 150, 300])
def test_both_methods_use_identical_alpha_cut_distances(n):
    values = generate_scenario(n=n, scenario="demo").items
    complete, density = cluster_auto(values), cluster_hdbscan(values)
    np.testing.assert_array_equal(complete.distances, density.distances)
    assert density.method == "hdbscan" and density.mode == "density"
    assert density.k == 3 and 0 not in density.labels
    assert adjusted_rand_index(complete.labels, density.labels) == 1
    assert density.linkage.shape == (0, 4)


def test_wrapper_agrees_with_sklearn_using_the_actual_fuzzy_metric():
    items = generate_scenario(scenario="outliers").items
    result = cluster_hdbscan(items, ComputationConfig(.02, 101), min_cluster_size=12, min_samples=7)
    expected = HDBSCAN(metric="precomputed", min_cluster_size=12, min_samples=7,
                       allow_single_cluster=True, copy=True).fit_predict(result.distances)
    np.testing.assert_array_equal(result.labels == 0, expected == -1)
    assert adjusted_rand_index(result.labels, expected) == 1
    assert sorted(set(result.labels) - {0}) == list(range(1, result.k + 1))


def test_noise_is_excluded_from_count_and_all_original_objects_are_preserved():
    batch = generate_scenario(scenario="outliers")
    result = cluster_hdbscan(batch.items)
    origins = np.array([batch.provenance["origin_labels"][i.id] for i in batch.items])
    assert len(result.labels) == 150 and result.k == 3
    assert np.sum(result.labels == 0) == 8
    assert np.all(origins[result.labels == 0] == 0)
    assert sum(np.bincount(result.labels)) == 150
    assert cluster_color(0) == NOISE_COLOR


def test_duplicates_tiny_inputs_all_noise_and_single_cluster_switch():
    for n in (1, 2):
        result = cluster_hdbscan([RayleighFuzzy(2)] * n)
        assert result.k == 0 and not np.any(result.labels)
    identical = [RayleighFuzzy(2)] * 30
    assert cluster_hdbscan(identical).k == 1
    assert cluster_hdbscan(identical, allow_single_cluster=False).k == 0
    assert cluster_hdbscan(identical, min_cluster_size=31).k == 0
    assert cluster_hdbscan(identical, min_samples=31).k == 0


def test_repeatability_permutation_and_general_alpha_cut_values():
    items = generate_scenario(scenario="unequal").items
    result = cluster_hdbscan(items)
    assert result.to_dict() == cluster_hdbscan(items).to_dict()
    np.testing.assert_array_equal(cluster_hdbscan(items[::-1]).labels[::-1], result.labels)
    values = [subtract(RayleighFuzzy(i.value.sigma), RayleighFuzzy(2)) for i in items]
    assert len(cluster_hdbscan(values).labels) == len(items)


@pytest.mark.parametrize("kwargs", [{"min_cluster_size": 1}, {"min_cluster_size": True},
    {"min_cluster_size": 2.5}, {"min_samples": 0}, {"min_samples": False},
    {"min_samples": None}, {"allow_single_cluster": 1}])
def test_invalid_density_settings(kwargs):
    with pytest.raises(ValueError):
        cluster_hdbscan([RayleighFuzzy(1)], **kwargs)


def test_empty_input_is_rejected():
    with pytest.raises(ValueError):
        cluster_hdbscan([])


def test_comparison_json_csv_and_old_session_compatibility():
    batch = generate_scenario(scenario="outliers")
    primary, density = cluster_auto(batch.items), cluster_hdbscan(batch.items)
    session = SessionData(batch.items, clustering=primary.to_dict(), comparison=density.to_dict(),
                          generation=batch.provenance)
    restored = loads_session(dumps_session(session))
    assert restored.comparison == density.to_dict()
    assert cluster_hdbscan(restored.items).to_dict() == density.to_dict()
    rows = list(csv.DictReader(io.StringIO(dataset_csv(batch.items, primary.labels, density.labels))))
    assert len(rows) == 150 and sum(int(r["hdbscan_noise"]) for r in rows) == 8
    assert all(int(row["hdbscan_noise"]) == (row["hdbscan_cluster"] == "0") for row in rows)
    assert dataset_csv(batch.items).splitlines()[0] == "id,sigma,cluster"
    session.comparison = None
    assert loads_session(dumps_session(session)).comparison is None


@pytest.mark.parametrize("mutation", ["negative_label", "missing_label", "bad_k", "bad_settings", "small_group", "missing_primary"])
def test_bad_comparison_export_rejected(mutation):
    items = generate_scenario().items
    session = SessionData(items, clustering=cluster_auto(items).to_dict(), comparison=cluster_hdbscan(items).to_dict())
    d = session.comparison
    if mutation == "negative_label": d["labels"][0] = -1
    if mutation == "missing_label": d["labels"].pop()
    if mutation == "bad_k": d["k"] = True
    if mutation == "bad_settings": d["density"]["min_samples"] = 0
    if mutation == "small_group": d["density"]["min_cluster_size"] = 300
    if mutation == "missing_primary": session.clustering = None
    with pytest.raises(ValueError): dumps_session(session)


def test_comparison_chart_keeps_sigma_and_marks_noise_explicitly():
    batch = generate_scenario(scenario="outliers")
    fig = comparison_figure(batch.items, cluster_auto(batch.items), cluster_hdbscan(batch.items), batch.provenance)
    noise = [trace for trace in fig.data if trace.name == "Шум"]
    assert len(noise) == 1 and noise[0].marker.color == NOISE_COLOR
    assert noise[0].marker.symbol == "x" and len(noise[0].x) == 8
    assert sum(len(t.x) for t in fig.data) == 3 * len(batch.items)
