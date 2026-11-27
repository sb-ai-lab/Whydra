import threading
from concurrent.futures import ThreadPoolExecutor
from itertools import combinations

import numpy as np
import pytest
from scipy.linalg import hadamard

from whydra.algorithms import wrappers
from whydra.algorithms.graph_core import CIT, CausalGraph
from whydra.algorithms.graph_core.cache import ci_cache_for
from whydra.algorithms.standalone import (
    cfci_algo,
    cfci_algo_paralel_cash,
    fci_plus_algo,
    fci_plus_algo_paralel_cash,
    pag_rules,
    rfci_algo,
)
from whydra.algorithms.standalone._counted_ci import counted_fisherz, graph_batch

from .conftest import run_scalar_queries


def data(seed=21, d=9):
    rng = np.random.default_rng(seed)
    values = rng.normal(size=(250, d))
    for j in range(1, d):
        values[:, j] += 0.6 * values[:, j - 1]
    return values


def counted(values, size=100000, outer=True):
    cit = CIT(values, cache_size=size)
    cit._decision_alpha = 0.05
    cache = ci_cache_for(cit) if outer else None
    stats = {"total": 0, "skeleton": 0, "ci_cache_hits": 0, "ci_cache_misses": 0}
    lock = threading.Lock()

    def call(x, y, s):
        with lock:
            stats["total"] += 1
            stats["skeleton"] += 1
            key = (min(x, y), max(x, y), tuple(sorted(s or ())))
            if cache is not None:
                if key in cache:
                    stats["ci_cache_hits"] += 1
                    return cache[key]
                stats["ci_cache_misses"] += 1
        value = cit(x, y, s)
        if cache is not None:
            with lock:
                cache[key] = value
        return value

    return counted_fisherz(call, cit, cache, stats, {"name": "skeleton"}, lock)


@pytest.mark.parametrize("size", [0, 2, 100000])
@pytest.mark.parametrize("outer", [False, True])
def test_counted_batch_retains_both_cache_layers(size, outer):
    a, b = counted(data(), size, outer), counted(data(), size, outer)
    requests = [(0, 1, ()), (2, 3, (1,)), (1, 0, ()), (4, 5, (6,)), (2, 3, (1,))]
    expected = [a(*q) for q in requests]
    np.testing.assert_allclose(
        b._serial_batch(requests), expected, rtol=1e-13, atol=1e-15
    )
    assert a.stats == b.stats
    assert (a.cit.cache_hits, a.cit.cache_misses) == (
        b.cit.cache_hits,
        b.cit.cache_misses,
    )
    assert list(a.cit._cache) == list(b.cit._cache)
    assert a.cache is None or list(a.cache) == list(b.cache)
    assert not b.cit._pending


@pytest.mark.parametrize("invalid", [False, True])
def test_counted_batch_error_keeps_only_successful_prefix(invalid):
    values = data()
    values[:, 3] = values[:, 2]
    a, b = counted(values), counted(values)
    requests = [(0, 1, ()), (2, 2 if invalid else 3, ()), (4, 5, ())]
    message = "disjoint" if invalid else "singular"
    with pytest.raises(ValueError, match=message) as expected:
        run_scalar_queries(a, requests)
    with pytest.raises(ValueError, match=str(expected.value)):
        b._serial_batch(requests)
    assert a.stats == b.stats
    assert list(a.cit._cache) == list(b.cit._cache)
    assert a.cit.cache_misses == b.cit.cache_misses
    assert not b.cit._pending


def test_counted_concurrent_batches_and_scalar_calls():
    test = counted(data())
    requests = [(i, j, ()) for i, j in combinations(range(9), 2)]
    with ThreadPoolExecutor(3) as pool:
        futures = [
            pool.submit(test._serial_batch, requests),
            pool.submit(lambda: [test(*q) for q in requests]),
            pool.submit(test._serial_batch, requests),
        ]
        results = [f.result(timeout=10) for f in futures]
    for result in results[1:]:
        np.testing.assert_array_equal(result, results[0])
    assert test.stats["total"] == 3 * len(requests)
    assert test.stats["ci_cache_misses"] == len(requests)


def test_class_level_custom_graph_test_is_not_bypassed(monkeypatch):
    graph = CausalGraph(9, [str(i) for i in range(9)])
    graph.set_ind_test(counted(data()))
    calls = []

    def custom(self, x, y, s):
        calls.append((x, y, s))
        return 0.123

    monkeypatch.setattr(CausalGraph, "ci_test", custom)
    queries = [(0, 1, ()), (2, 3, ())]
    assert graph_batch(graph, queries) == [0.123, 0.123]
    assert calls == queries
    assert graph.test.stats["total"] == 0


def test_counted_threshold_repair_and_duplicate_members():
    from whydra.algorithms.graph_core.independence_tests import _fisher_z_reference

    a, b = counted(data()), counted(data())
    alpha = _fisher_z_reference(a.cit.corr_matrix, len(a.cit.data), 3, 0, (1,))
    a.cit._decision_alpha = b.cit._decision_alpha = alpha
    requests = [(3, 0, (1,)), (0, 3, (1,)), (0, 3, (1, 1))]
    assert [a(*q) for q in requests] == b._serial_batch(requests) == [alpha] * 3
    assert a.stats == b.stats
    assert list(a.cache) == list(b.cache)


@pytest.mark.parametrize(
    ("sepset", "options", "center_mark"),
    [
        ({}, {}, 2),
        ({(2, 0): set()}, {}, 1),
        ({(0, 2): {1}}, {}, 2),
        ({}, {"require_recorded_sepset": False}, 1),
        ({(0, 2): set()}, {"unfVect": {(2, 1, 0)}}, 2),
        ({}, {"collider_triples": {(2, 1, 0)}}, 1),
        ({(0, 2): set()}, {"collider_triples": set()}, 2),
        ({(0, 2): set()}, {"orientCollider": False}, 2),
    ],
)
def test_shared_orientation_requires_collider_evidence(sepset, options, center_mark):
    skeleton = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
    stats = {}
    result = pag_rules.udag2pag(skeleton, sepset, 3, rule_stats=stats, **options)
    np.testing.assert_array_equal(
        result, [[0, center_mark, 0], [2, 0, 2], [0, center_mark, 0]]
    )
    expected = dict.fromkeys((f"R{i}" for i in range(11)), 0)
    expected["R0"] = 2 if center_mark == 1 else 0
    assert stats == expected


@pytest.mark.parametrize("enable_r1", [False, True])
def test_shared_orientation_counts_collider_and_tail_propagation(enable_r1):
    skeleton = np.array([[0, 1, 0, 0], [1, 0, 1, 1], [0, 1, 0, 0], [0, 1, 0, 0]])
    sepsets = {(0, 2): set(), (0, 3): {1}, (2, 3): {1}}
    stats = {}
    graph = pag_rules.udag2pag(
        skeleton, sepsets, 4, rule_stats=stats, rules=[enable_r1] + [False] * 9
    )
    child, parent = (1, -1) if enable_r1 else (2, 2)
    np.testing.assert_array_equal(
        graph, [[0, 1, 0, 0], [2, 0, 2, child], [0, 1, 0, 0], [0, parent, 0, 0]]
    )
    expected = dict.fromkeys((f"R{i}" for i in range(11)), 0)
    expected.update(R0=2, R1=2 if enable_r1 else 0)
    assert stats == expected


@pytest.mark.parametrize(
    ("reverse_mark", "ambiguous", "child", "parent"),
    [(2, set(), 1, -1), (2, {(2, 1, 0)}, 2, 2), (0, set(), 2, 2)],
)
def test_rfci_r1_uses_an_unambiguous_two_sided_edge(
    reverse_mark, ambiguous, child, parent
):
    pag = np.array([[0, 1, 0], [reverse_mark, 0, 2], [0, 2, 0]])
    result = rfci_algo._apply_r1_r3(pag, unfVect=ambiguous)
    np.testing.assert_array_equal(
        result, [[0, 1, 0], [reverse_mark, 0, child], [0, parent, 0]]
    )


@pytest.mark.parametrize(("ambiguous", "mark"), [(set(), -1), ({(0, 1, 2)}, 2)])
def test_rfci_r5_requires_an_unambiguous_uncovered_circle(ambiguous, mark):
    ring = np.array([[0, 1, 0, 1], [1, 0, 1, 0], [0, 1, 0, 1], [1, 0, 1, 0]])
    result = rfci_algo._apply_r5_r10(2 * ring, 4, unfVect=ambiguous)
    np.testing.assert_array_equal(result, mark * ring)


CFCI_COUNTS = {"triple_check": 2, "pdsep": 2}
FCI_PLUS_COUNTS = {"augment": 1, "dsep": 0}


@pytest.mark.parametrize(
    ("module", "wrapper", "phase_counts", "unique_tests"),
    [
        (rfci_algo, wrappers.StandaloneRFCIStable, {"rfci_orient": 2}, 5),
        (cfci_algo, wrappers.StandaloneCFCIStable, CFCI_COUNTS, 6),
        (
            cfci_algo_paralel_cash,
            wrappers.StandaloneCFCIStableParallelCached,
            CFCI_COUNTS,
            6,
        ),
        (fci_plus_algo, wrappers.StandaloneFCIPlusStable, FCI_PLUS_COUNTS, 6),
        (
            fci_plus_algo_paralel_cash,
            wrappers.StandaloneFCIPlusStableParallelCached,
            FCI_PLUS_COUNTS,
            6,
        ),
    ],
)
@pytest.mark.parametrize("cache", [False, True])
@pytest.mark.parametrize(("n_jobs", "scalar"), [(1, False), (1, True), (2, False)])
def test_full_families_recover_collider_and_count_ci_calls(
    monkeypatch, module, wrapper, phase_counts, unique_tests, cache, n_jobs, scalar
):
    if scalar:
        monkeypatch.setattr(module, "counted_fisherz", lambda call, *args: call)
    left, right, noise = hadamard(128)[:, 1:4].T
    values = np.column_stack((left, left + right + noise, right))
    result = wrapper(n_jobs=n_jobs, use_ci_cache=cache, max_cond_set_size=2).run(values)
    np.testing.assert_array_equal(result.graph, [[0, 1, 0], [2, 0, 2], [0, 1, 0]])
    stats = result.metadata["ci_stats"]
    total = 8 + sum(phase_counts.values())
    assert stats["skeleton"] == 8
    assert stats["total"] == total
    for phase, count in phase_counts.items():
        assert stats[phase] == count
    assert result.metadata["ci_computations"] == (unique_tests if cache else total)
    cache_hits = result.metadata["ci_cache_hits"]
    if cache:
        assert stats["ci_cache_hits"] + stats["ci_cache_misses"] == total
        assert stats["ci_cache_hits"] + cache_hits == total - unique_tests
    else:
        assert cache_hits == 0
        assert "ci_cache_hits" not in stats
