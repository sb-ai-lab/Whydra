from concurrent.futures import ThreadPoolExecutor

import networkx as nx
import numpy as np
import pytest
from scipy.stats import chi2, chi2_contingency

from whydra.algorithms.graph_core import CIT
from whydra.algorithms.standalone.fci_algo_v2 import _find_biconnected_components
from whydra.algorithms.standalone.profiler import profiler


def test_iterative_biconnected_components_chain_3000():
    matrix = np.zeros((3000, 3000), dtype=np.int8)
    i = np.arange(2999)
    matrix[i, i + 1] = matrix[i + 1, i] = 1
    result = _find_biconnected_components(matrix)
    assert {frozenset(c) for c in result} == {
        frozenset({(i, i + 1)}) for i in range(2999)
    }


def test_biconnected_components_matches_networkx():
    rng = np.random.default_rng(18)
    matrix = rng.random((20, 20)) < 0.1
    matrix |= matrix.T
    np.fill_diagonal(matrix, 0)
    expected = {
        frozenset(tuple(sorted(edge)) for edge in c)
        for c in nx.biconnected_component_edges(nx.from_numpy_array(matrix))
    }
    assert {frozenset(c) for c in _find_biconnected_components(matrix)} == expected


def test_ci_cache_canonical_bounded_and_thread_safe():
    data = np.random.default_rng(19).normal(size=(100, 6))
    test = CIT(data, cache_size=2)
    with ThreadPoolExecutor(max_workers=4) as pool:
        values = list(pool.map(lambda i: test(i % 2, 1 - i % 2, [3, 2]), range(20)))
    assert test.cache_misses == 1
    assert test.cache_hits == 19
    assert len(set(values)) == 1
    for k in range(2, 6):
        test(0, k)
    assert len(test._cache) <= 2
    assert test.cache_evictions > 0


def test_conditional_chisq_observed_strata_parity_and_no_overflow():
    data = np.random.default_rng(20).integers(0, 4, (300, 6))
    test = CIT(data, method="chisq")
    statistic = degrees = 0
    for value in np.unique(data[:, 2:4], axis=0):
        group = data[np.all(data[:, 2:4] == value, axis=1)]
        obs = np.zeros((4, 4))
        np.add.at(obs, (group[:, 0], group[:, 1]), 1)
        obs = obs[obs.sum(axis=1) > 0][:, obs.sum(axis=0) > 0]
        if min(obs.shape) >= 2:
            stat, _, df, _ = chi2_contingency(obs, correction=False)
            statistic += stat
            degrees += df
    assert test(0, 1, [2, 3]) == pytest.approx(chi2.sf(statistic, degrees), abs=1e-12)
    wide = np.random.default_rng(21).integers(0, 4, (100, 42))
    assert np.isfinite(CIT(wide, method="chisq")(0, 1, list(range(2, 42))))


def test_default_profiler_does_not_accumulate():
    if profiler.enabled:
        pytest.skip("Explicit profiling enabled")
    before = len(profiler.records)
    with profiler.time_block("disabled_check"):
        pass
    assert len(profiler.records) == before
