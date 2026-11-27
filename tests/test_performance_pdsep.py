import copy
from contextlib import closing
from itertools import combinations

import numpy as np
import pytest

from tests.test_performance_pag_family import counted, data
from whydra.algorithms.graph_core.independence_tests import _fisher_z_reference
from whydra.algorithms.standalone.cfci_algo import pdsep

from .conftest import run_scalar_queries


@pytest.mark.parametrize("size", [0, 2, 100000])
@pytest.mark.parametrize("stop", [1, 3, 12, 36])
def test_lazy_numeric_batch_commits_only_consumed_queries(size, stop):
    a, b = counted(data(), size), counted(data(), size)
    requests = [(i, j, ()) for i, j in combinations(range(9), 2)]
    expected = [a(*q) for q in requests[:stop]]
    actual = []
    with closing(b._iter_values(requests)) as stream:
        for _, value in stream:
            actual.append(value)
            if len(actual) == stop:
                break
    np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-15)
    assert a.stats == b.stats
    assert list(a.cache) == list(b.cache)
    assert list(a.cit._cache) == list(b.cit._cache)
    assert (a.cit.cache_hits, a.cit.cache_misses, a.cit.cache_evictions) == (
        b.cit.cache_hits,
        b.cit.cache_misses,
        b.cit.cache_evictions,
    )
    assert not b.cit._pending


def test_future_singular_error_is_not_published_after_early_stop():
    values = data()
    values[:, 3] = values[:, 2]
    test = counted(values)
    with closing(test._iter_values([(0, 1, ()), (2, 3, ()), (4, 5, ())])) as stream:
        next(stream)
    assert test.stats["total"] == test.cit.cache_misses == 1
    assert list(test.cache) == [(0, 1, ())]
    assert not test.cit._pending
    assert test.prefetch_stats["unused_numeric_evaluations"] == 2


@pytest.mark.parametrize("invalid", [False, True])
def test_consumed_error_preserves_original_prefix_and_counts(invalid):
    values = data()
    values[:, 3] = values[:, 2]
    a, b = counted(values), counted(values)
    queries = [(0, 1, ()), (2, 2 if invalid else 3, ()), (4, 5, ())]
    message = "disjoint" if invalid else "singular"
    with pytest.raises(ValueError, match=message) as error:
        run_scalar_queries(a, queries)
    with pytest.raises(ValueError, match=str(error.value)):
        list(b._iter_values(queries))
    assert a.stats == b.stats
    assert list(a.cit._cache) == list(b.cit._cache)
    assert a.cit.cache_misses == b.cit.cache_misses
    assert not b.cit._pending


def test_replay_rechecks_evicted_snapshot_hits_and_thresholds():
    a, b = counted(data(), 2), counted(data(), 2)
    alpha = _fisher_z_reference(a.cit.corr_matrix, len(a.cit.data), 3, 0, (1,))
    a.cit._decision_alpha = b.cit._decision_alpha = alpha
    queries = [(3, 0, (1,)), (0, 3, (1,)), (2, 4, ())]
    with closing(b._iter_values(queries)) as stream:
        assert next(stream)[1] == a(*queries[0]) == alpha
        for q in [(4, 5, ()), (6, 7, ())]:
            a(*q)
            b(*q)
        for query, value in stream:
            assert value == a(*query)
    assert a.stats == b.stats
    assert list(a.cache) == list(b.cache)
    assert a.cit.cache_evictions == b.cit.cache_evictions


@pytest.mark.parametrize("seed", range(8))
@pytest.mark.parametrize("size", [0, 3, 100000])
def test_pdsep_batched_and_scalar_ci_publish_the_same_evidence(seed, size):
    n = 9
    rng = np.random.default_rng(seed)
    edges = np.triu(rng.random((n, n)) < 0.3, 1)
    sk = ((edges | edges.T) * 2).astype(int)
    seps = {
        (x, y): set() for x in range(n) for y in range(n) if x != y and not sk[x, y]
    }
    outputs = []
    tests = []
    for batch in (False, True):
        test = counted(data(seed, n), size)
        test.cit._decision_alpha = 0.99
        test.cit.max_cond_set_size = 3

        def call(suffstat, x, y, s, *, test=test):
            return test(x, y, s)

        if batch:
            call._counted_ci = test
        call.max_cond_set_size = 3
        result = pdsep(
            {"sk": sk.copy()},
            None,
            call,
            n,
            copy.deepcopy(seps),
            0.99,
            np.full((n, n), -np.inf),
            unfVect=set(),
        )
        outputs.append(result)
        tests.append(test)
    a, b = outputs
    for key in ("G", "pMax"):
        np.testing.assert_allclose(a[key], b[key], rtol=1e-13, atol=1e-15)
    for key in ("sepset", "allPdsep", "max_ord"):
        assert a[key] == b[key]
    assert all(
        b["pdsep_stats"][key] == value for key, value in a["pdsep_stats"].items()
    )
    assert tests[0].stats == tests[1].stats
    assert list(tests[0].cache) == list(tests[1].cache)
    assert list(tests[0].cit._cache) == list(tests[1].cit._cache)


@pytest.mark.parametrize(
    ("witness", "queries"),
    [
        ((2,), [(2,)]),
        ((3,), [(2,), (3,)]),
        ((2, 3), [(2,), (3,), (2, 3)]),
    ],
)
def test_pdsep_records_first_separator_and_publishes_only_its_prefix(witness, queries):
    skeleton = np.array([[0, 2, 0, 0], [2, 0, 2, 0], [0, 2, 0, 2], [0, 0, 2, 0]])
    evidence = {(0, 2): set(), (1, 3): set()}
    calls = []

    def ci(_, x, y, conditions):
        assert (x, y) == (0, 1)
        calls.append(tuple(conditions))
        return 0.5 if calls[-1] == witness else 0.01

    result = pdsep(
        {"sk": skeleton.copy()},
        None,
        ci,
        4,
        evidence.copy(),
        0.5,
        np.full((4, 4), -np.inf),
        protected_edges=((1, 2), (2, 3)),
        unfVect=set(),
    )

    expected_graph = skeleton.copy()
    expected_graph[0, 1] = expected_graph[1, 0] = 0
    expected_peaks = np.full((4, 4), -np.inf)
    expected_peaks[0, 1] = 0.5
    np.testing.assert_array_equal(result["G"], expected_graph)
    np.testing.assert_array_equal(result["pMax"], expected_peaks)
    assert result["sepset"] == {**evidence, (0, 1): set(witness), (1, 0): set(witness)}
    assert result["allPdsep"] == [[2, 3, 1], [], [], []]
    assert result["max_ord"] == len(witness)
    assert calls == queries
    stats = result["pdsep_stats"]
    assert stats["pairs_scanned"] == stats["pairs_with_diffset"] == 1
    assert stats["ci_tests"] == len(queries)
