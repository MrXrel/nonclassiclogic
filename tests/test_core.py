import json
import sys
from pathlib import Path
import numpy as np
import pytest
from fuzzy_core import *
from fuzzy_clustering import cluster_by_count, cluster_by_threshold, distance_matrix


def test_reference_and_density_normalization():
    a = RayleighFuzzy(2)
    assert a.alpha_cut(.5) == pytest.approx((.638211348, 3.843245779), abs=1e-8)
    x = np.linspace(-2, 25, 2000)
    mu = a.membership(x)
    assert np.all((mu >= 0) & (mu <= 1))
    assert a.membership(2) == 1
    assert a.membership(0) == 0
    assert a.membership(float("inf")) == 0
    assert a.alpha_cut(1) == (2, 2)
    # The unnormalized density for a small scale really can exceed one.
    assert np.exp(-.5) / .1 > 1


@pytest.mark.parametrize("sigma", [.00001, .1, 1, 2, 100000])
def test_cut_roots_nesting_and_scale(sigma):
    a = RayleighFuzzy(sigma)
    cuts = a.to_alpha_cuts()
    assert np.all(np.diff(cuts.left) >= 0)
    assert np.all(np.diff(cuts.right) <= 0)
    assert np.max(np.abs(a.membership(cuts.left) - cuts.alpha)) < 1e-10
    assert np.max(np.abs(a.membership(cuts.right) - cuts.alpha)) < 1e-10
    assert a.alpha_cut(.9) == pytest.approx(np.array(RayleighFuzzy(1).alpha_cut(.9)) * sigma)


@pytest.mark.parametrize("sigma", [0, -1, float("nan"), float("inf"), True])
def test_bad_scales(sigma):
    with pytest.raises(ValueError): RayleighFuzzy(sigma)


def interval(lo, hi):
    return AlphaCutFuzzy([.001, 1], [lo, lo], [hi, hi])


@pytest.mark.parametrize("op, expected", [("+", (-1, 9)), ("-", (-7, 3)), ("*", (-12, 20)), ("/", (-1, 5 / 3))])
def test_interval_arithmetic_with_negative_values(op, expected):
    result = arithmetic(interval(-3, 5), interval(2, 4) if op != "/" else interval(3, 6), op)
    assert result.alpha_cut(.5) == pytest.approx(expected)


def test_arithmetic_shape_and_independent_operands():
    a, b = RayleighFuzzy(1), RayleighFuzzy(2)
    total = add(a, b)
    expected = RayleighFuzzy(3).to_alpha_cuts()
    np.testing.assert_allclose(total.left, expected.left)
    np.testing.assert_allclose(total.right, expected.right)
    subtraction = subtract(a, a)
    assert subtraction.left[0] < 0 < subtraction.right[0]
    assert subtraction.alpha_cut(1) == (0, 0)
    assert multiply(a, b).alpha_cut(1) == (2, 2)
    assert distance(multiply(a, b), RayleighFuzzy(2)) > .1
    with pytest.raises(ZeroDivisionError): divide(a, subtraction)
    with pytest.raises(ZeroDivisionError): divide(a, interval(0, 3))
    assert divide(interval(-4, -2), interval(-2, -1)).alpha_cut(.5) == pytest.approx((1, 4))


def test_reconstruction_and_config():
    a = RayleighFuzzy(2).to_alpha_cuts()
    assert a.membership(a.left[100]) == pytest.approx(a.alpha[100])
    assert a.membership(a.left[0] - 1) == 0
    assert a.membership(a.right[0] + 1) == 0
    assert a.membership(2) == 1
    with pytest.raises(ValueError): a.alpha_cut(0)
    with pytest.raises(ValueError): a.to_alpha_cuts(ComputationConfig(.0001))
    with pytest.raises(ValueError): AlphaCutFuzzy([.001, 1], [2, 1], [4, 3])
    with pytest.raises(ValueError): ComputationConfig(0)
    with pytest.raises(ValueError): ComputationConfig(.1, 1)
    with pytest.raises(ValueError): a.left[0] = 5


def test_distance_rank_and_overlap():
    a, b = RayleighFuzzy(1), RayleighFuzzy(2)
    c = compare(a, b)
    assert c.relation == "<" and c.overlap_05 is not None
    assert distance(a, a) == 0
    assert distance(a, b) == pytest.approx(distance(b, a))
    assert distance(a, RayleighFuzzy(3)) == pytest.approx(2 * distance(a, b))
    assert rank(b) == pytest.approx(2 * rank(a))
    assert compare(interval(-4, -2), interval(2, 4)).overlap_05 is None
    # Equal midpoint ranks, different uncertainty.
    same_rank = compare(interval(-1, 1), interval(-2, 2))
    assert same_rank.relation == "=" and same_rank.distance > 0


@pytest.mark.parametrize("n", [100, 150, 300])
def test_generation(n):
    x, y = generate(n), generate(n)
    assert len(x) == n and len(set(i.id for i in x)) == n
    assert [i.value.sigma for i in x] == [i.value.sigma for i in y]
    assert all(i.value.sigma > 0 for i in x)
    assert [i.value.sigma for i in x] != [i.value.sigma for i in generate(n, seed=43)]


def test_generation_errors_and_uniform():
    assert all(.2 <= i.value.sigma <= .5 for i in generate(100, mode="uniform", sigma_min=.2, sigma_max=.5))
    for n in [99, 301, True, 100.1]:
        with pytest.raises(ValueError): generate(n)
    with pytest.raises(ValueError): generate(seed=-1)
    with pytest.raises(ValueError): generate(mode="uniform", sigma_min=2, sigma_max=1)


def test_serialization_roundtrip():
    items = generate(100)
    result = multiply(items[0].value, items[1].value)
    clusters = cluster_by_count(items, 3)
    state = SessionData(items, [FuzzyItem("C001", result)], clustering=clusters.to_dict(), controls={"seed": 42})
    encoded = dumps_session(state)
    decoded = loads_session(encoded)
    assert dumps_session(decoded) == encoded
    assert distance(decoded.results[0].value, result) == 0
    assert dataset_csv(items, clusters.labels).splitlines()[0] == "id,sigma,cluster"


@pytest.mark.parametrize("mutate", [
    lambda d: d.update(schema_version=9),
    lambda d: d["items"][0]["value"].update(sigma=-1),
    lambda d: d["items"][1].update(id=d["items"][0]["id"]),
    lambda d: d.update(items=[]),
    lambda d: d.update(config={"epsilon": 0, "levels": 401}),
])
def test_invalid_json(mutate):
    data = json.loads(dumps_session(SessionData(generate())))
    mutate(data)
    with pytest.raises(ValueError): loads_session(json.dumps(data))


def test_malformed_json():
    for data in ['{"x": NaN}', '{', '[]', '{"schema_version":1,"items":false}']:
        with pytest.raises(ValueError): loads_session(data)


@pytest.mark.parametrize("k", [1, 3, 100])
def test_exact_k_and_demo_groups(k):
    items = generate(100)
    c = cluster_by_count(items, k)
    assert len(set(c.labels)) == k
    if k == 3:
        for label in set(c.labels):
            sigmas = [i.value.sigma for i, l in zip(items, c.labels) if l == label]
            assert max(sigmas) - min(sigmas) < .61
    assert c.labels.tolist() == cluster_by_count(items, k).labels.tolist()


def test_duplicates_threshold_monotonicity_and_metric_embedding():
    identical = [RayleighFuzzy(2)] * 7
    for k in [1, 2, 6, 7]: assert len(set(cluster_by_count(identical, k).labels)) == k
    assert len(set(cluster_by_threshold(identical, 0).labels)) == 1
    values = [RayleighFuzzy(s) for s in [1, 1.1, 2.5, 4, 4.1]]
    sizes = [len(set(cluster_by_threshold(values, t).labels)) for t in [0, .2, 1, 10]]
    assert sizes == sorted(sizes, reverse=True)
    assert sizes[0] == 5 and sizes[-1] == 1
    matrix, _ = distance_matrix(values)
    for i in range(len(values)):
        for j in range(len(values)):
            assert matrix[i, j] == pytest.approx(distance(values[i], values[j]), abs=1e-12)
    assert cluster_by_count([RayleighFuzzy(1)], 1).labels.tolist() == [1]
    with pytest.raises(ValueError): cluster_by_count(values, 0)
    with pytest.raises(ValueError): cluster_by_threshold(values, -1)


def test_clustering_300():
    c = cluster_by_count(generate(300), 3)
    assert c.distances.shape == (300, 300)
    assert len(set(c.labels)) == 3


def test_libraries_do_not_import_ui():
    import subprocess
    code = "import fuzzy_core, fuzzy_clustering, sys; assert 'streamlit' not in sys.modules; assert 'plotly' not in sys.modules"
    subprocess.run([sys.executable, "-c", code], check=True)
