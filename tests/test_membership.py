import csv
import io
from dataclasses import replace

import numpy as np
import pytest

from fuzzy_clustering import cluster_by_count, cluster_hdbscan, cluster_membership, membership_mask
from fuzzy_core import FuzzyItem, RayleighFuzzy, SessionData, dataset_csv, dumps_session, loads_session


def items(sigmas):
    return [FuzzyItem(f"R{i}", RayleighFuzzy(s)) for i, s in enumerate(sigmas)]


def test_within_group_scores_use_fixed_center_radius_and_inclusive_cut():
    values = items([1, 2, 3])
    result = cluster_by_count(values, 1)
    profile = cluster_membership(values, result)
    edge = np.exp(-.75)
    assert profile.scores == pytest.approx([edge, 1, edge])
    assert profile.centers[1].alpha_cut(1) == (2, 2)
    assert profile.radii[1] == pytest.approx(np.sqrt(np.mean(result.distances[1] ** 2)))
    previous = membership_mask(profile, 0)
    assert previous.all()
    assert membership_mask(profile, profile.scores[0])[0]
    for a in (.43, .5, .8, 1):
        current = membership_mask(profile, a)
        assert np.all(current <= previous)
        previous = current
    assert previous.tolist() == [False, True, False]
    np.testing.assert_array_equal(result.labels, [1, 1, 1])
    np.testing.assert_array_equal(profile.labels, result.labels)
    assert cluster_membership(values, result).scores == pytest.approx(profile.scores)


def test_no_center_sample_and_degenerate_groups():
    values = items([1, 3])
    assert not membership_mask(cluster_membership(values, cluster_by_count(values, 1)), 1).any()
    assert membership_mask(cluster_membership(values, cluster_by_count(values, 2)), 1).all()
    values = items([2.17] * 300)
    profile = cluster_membership(values, cluster_by_count(values, 1))
    assert profile.radii == {1: 0.0}
    assert membership_mask(profile, 1).all()


@pytest.mark.parametrize("alpha", [True, -.1, 1.1, np.nan, np.inf, "0.43"])
def test_invalid_cut(alpha):
    values = items([1])
    with pytest.raises(ValueError):
        membership_mask(cluster_membership(values, cluster_by_count(values, 1)), alpha)


def test_noise_and_csv_json_keep_original_labels():
    values = items([1, 2, 3])
    complete = cluster_by_count(values, 1)
    density = cluster_hdbscan(values)
    c = cluster_membership(values, complete)
    d = cluster_membership(values, density)
    assert d.centers == d.radii == {}
    assert not membership_mask(d, 0).any()
    assert d.to_dict()["scores"] == [None] * 3
    session = SessionData(values,
        clustering={**complete.to_dict(), "membership": c.to_dict()},
        comparison={**density.to_dict(), "membership": d.to_dict()},
        controls={"cluster_alpha": .5})
    restored = loads_session(dumps_session(session))
    assert restored.controls["cluster_alpha"] == .5
    assert restored.clustering == session.clustering
    assert restored.comparison == session.comparison
    output = dataset_csv(values, complete.labels, density.labels,
                         scores=c.to_dict()["scores"], hdbscan_scores=d.to_dict()["scores"],
                         cluster_alpha=.5)
    rows = list(csv.DictReader(io.StringIO(output)))
    assert [r["cluster"] for r in rows] == ["1"] * 3
    assert [r["cluster_retained"] for r in rows] == ["0", "1", "0"]
    assert all(r["hdbscan_cluster"] == "0" and r["hdbscan_noise"] == "1"
               and r["hdbscan_score"] == "" and r["hdbscan_retained"] == "0"
               and r["hdbscan_alpha"] == "0.5" for r in rows)
    with pytest.raises(ValueError):
        dumps_session(replace(session, controls={"cluster_alpha": True}))
    malformed = {**session.comparison, "membership": {"rule": "gaussian_rms_v1", "scores": [0, None, None]}}
    with pytest.raises(ValueError):
        dumps_session(replace(session, comparison=malformed))
