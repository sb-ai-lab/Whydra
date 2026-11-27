"""Opt-in sufficient-statistics scores retain guards and reference objectives."""

from itertools import combinations

import numpy as np
import pytest

from whydra.algorithms.score_based._covariance_bic import CovarianceBIC
from whydra.algorithms.score_based.ges import ges_search
from whydra.algorithms.score_based.scores import LocalScoreCache


@pytest.mark.parametrize("seed", range(8))
def test_covariance_families_agree_with_lstsq_on_regular_data(seed):
    data = np.random.default_rng(seed).normal(size=(300, 7))
    data[:, 3] += data[:, 1] + 0.7 * data[:, 2]
    reference = LocalScoreCache(data, "local_score_BIC")
    actual = LocalScoreCache(data, "local_score_BIC", bic_backend="covariance")
    for child in range(7):
        candidates = [i for i in range(7) if i != child]
        for size in range(4):
            for parents in combinations(candidates, size):
                assert actual.score(child, parents) == pytest.approx(
                    reference.score(child, parents), abs=1e-8
                )
    assert actual._covariance.fast_scores > 0


@pytest.mark.parametrize(
    "case", ["large_offset", "constant", "collinear", "perfect_fit"]
)
def test_unsafe_covariance_families_use_original_lstsq(case):
    data = np.random.default_rng(32).normal(size=(60, 4))
    if case == "large_offset":
        data += 1e10
    elif case == "constant":
        data[:, 0] = 1.0
    elif case == "collinear":
        data[:, 2] = data[:, 1]
    else:
        data[:, 0] = data[:, 1] + data[:, 2]
    reference = LocalScoreCache(data, "local_score_BIC")
    actual = LocalScoreCache(data, "local_score_BIC", bic_backend="covariance")
    assert actual.score(0, (1, 2)) == reference.score(0, (1, 2))
    assert actual._covariance.fallbacks == 1


def test_factor_memory_is_bounded_and_eviction_does_not_change_scores():
    data = np.random.default_rng(17).normal(size=(100, 8))
    bounded = CovarianceBIC(data, max_factors=2, max_bytes=128)
    reference = LocalScoreCache(data, "local_score_BIC")
    for parents in combinations(range(1, 8), 2):
        assert bounded.score(0, parents) == pytest.approx(
            reference.score(0, parents), abs=1e-8
        )
        assert len(bounded.factors) <= 2
        assert bounded.factor_bytes <= 128


@pytest.mark.parametrize("backend", ["covariance", "reference"])
def test_search_reports_reference_objective_for_selected_class(backend):
    from whydra.algorithms.score_based.ges import total_score

    data = np.random.default_rng(18).normal(size=(300, 6))
    data[:, 1] += data[:, 0]
    stats = {}
    graph, value = ges_search(
        data,
        "local_score_BIC",
        [str(i) for i in range(6)],
        max_operator_set=None,
        bic_backend=backend,
        stats=stats,
    )
    # Endpoint CPDAG encoding -> adjacency encoding.
    endpoint = graph.graph
    adjacency = ((endpoint != 0) & (endpoint.T != 0) & (endpoint != 1)).astype(np.int8)
    expected = total_score(adjacency, LocalScoreCache(data, "local_score_BIC"))
    assert value == expected
    assert graph.metadata["bic_backend"] == backend


def test_covariance_is_not_silently_used_for_bdeu():
    with pytest.raises(ValueError, match="requires local_score_BIC"):
        LocalScoreCache(np.ones((10, 3)), "local_score_BDeu", bic_backend="covariance")
