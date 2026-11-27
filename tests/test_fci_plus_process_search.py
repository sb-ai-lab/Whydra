"""Coarse FCI+ work must retain separators, PAG marks and lazy CI errors."""

from importlib import import_module

import numpy as np
import pytest

from whydra.algorithms.graph_core import CIT, CausalGraph
from whydra.algorithms.graph_core.cache import BoundedCache
from whydra.algorithms.standalone import fci_plus_search, rfci_algo
from whydra.algorithms.standalone.ci_adapters import attach_counted_batch


def test_compact_naa_search_retains_candidate_filter_and_order():
    x, y, u, v, middle = range(5)
    graph = np.zeros((5, 5), dtype=int)
    for a, b in [(x, y), (u, x), (v, y)]:
        graph[a, b] = graph[b, a] = 1
    for a, b in [(y, middle), (middle, u), (x, v)]:
        graph[a, b], graph[b, a] = -1, 1

    assert fci_plus_search.naa_ancestors(y, graph) == {middle, u}
    two_components = np.kron(np.eye(2, dtype=int), graph)
    assert list(fci_plus_search.pdsep_candidates(two_components)) == [(1, 0), (6, 5)]

    graph[u, v] = graph[v, u] = 2
    assert list(fci_plus_search.pdsep_candidates(graph)) == []
    graph[u, v] = graph[v, u] = 0
    graph[middle, y] = 2
    assert list(fci_plus_search.pdsep_candidates(graph)) == []


def test_hierarchy_worklist_matches_fixed_point_and_observes_later_updates():
    from whydra.algorithms.standalone.fci_plus_algo import _get_hie

    cg = CausalGraph(8, [f"X{i}" for i in range(8)])
    cg.sepset[0][1] = [(2,)]
    cg.sepset[2][1] = [(3, 4)]
    cg.sepset[4][0] = [(6,)]

    assert _get_hie(cg, {0, 1}) == {0, 1, 2, 3, 4, 6}
    cg.sepset[6][3] = [(7,)]
    assert _get_hie(cg, {0, 1}) == {0, 1, 2, 3, 4, 6, 7}


def _augmentation_graph(data, cache_size=17, cap=None):
    p = data.shape[1]
    cg = CausalGraph(p, [f"X{i}" for i in range(p)])
    cg.set_ind_test(CIT(data, cache_size=cache_size))
    cg.max_cond_set_size = cap
    cg.test.max_cond_set_size = cap
    for x in range(p):
        for y in range(x + 1, p):
            if (x + y) % 3:
                cg.G.graph[x, y] = cg.G.graph[y, x] = 0
                cg.sepset[x][y] = cg.sepset[y][x] = [()]
    return cg


@pytest.mark.parametrize("cap", [None, 0, 1])
def test_augmentation_worker_snapshot_matches_scalar_and_bounds_cache(cap, monkeypatch):
    from whydra.algorithms.standalone import fci_plus_algo_paralel_cash as module

    data = np.random.default_rng(711).normal(size=(250, 20))
    expected = _augmentation_graph(data, cap=cap)
    scalar = CIT.fisher_z
    with monkeypatch.context() as patch:
        patch.setattr(CIT, "fisher_z", lambda self, x, y, s: scalar(self, x, y, s))
        module.AugmentGraph(expected, 20, 0.05, n_jobs=1)

    graph = _augmentation_graph(data, cap=cap)
    cit = graph.test
    stats = {"total": 0, "augment": 0, "ci_cache_hits": 0, "ci_cache_misses": 0}
    outer = BoundedCache(13)

    def counted(x, y, s):
        return cit(x, y, s)

    attach_counted_batch(counted, cit, stats, {"name": "augment"}, outer)
    graph.set_ind_test(counted)
    # The process boundary itself has separate integration coverage. Execute
    # every task descriptor here, recording that a coarse dispatch was chosen.
    dispatched = []

    def process_map(function, arguments, *, n_jobs, **options):
        arguments = list(arguments)
        dispatched.append((n_jobs, len(arguments)))
        return [function(*args) for args in arguments]

    monkeypatch.setattr(fci_plus_search, "process_map", process_map)
    module.AugmentGraph(graph, 20, 0.05, n_jobs=4, stats=stats)
    np.testing.assert_array_equal(graph.G.graph, expected.G.graph)
    assert len(cit._cache) <= 17
    assert len(outer) <= 13
    assert stats["augment"] == stats["augment_tasks"]
    assert stats["ci_cache_hits"] + stats["ci_cache_misses"] == stats["augment"]
    if cap != 0:
        assert dispatched[0][0] == 4
        assert stats["augment_process_batches"] == 1
    else:
        assert stats["augment"] == 0


def test_rfci_native_batch_propagates_backend_without_unused_orientation_pool(
    monkeypatch,
):
    cg = CausalGraph(4, [f"X{i}" for i in range(4)])
    cg.set_ind_test(CIT(np.random.default_rng(712).normal(size=(100, 4))))
    cg._parallel_backend = "processes"
    observed = []

    def batch(test, requests, **kwargs):
        observed.append(kwargs)
        return [0.8] * len(requests)

    monkeypatch.setattr(rfci_algo, "evaluate_ci_batch", batch)
    monkeypatch.setattr(
        rfci_algo, "Parallel", lambda **kwargs: pytest.fail("unused pool created")
    )
    assert rfci_algo._ci_batch(cg, 0, 1, [(), (2,)], n_jobs=4) == [0.8, 0.8]
    assert observed == [{"n_jobs": 4, "backend": "processes"}]
    assert not hasattr(cg, "_parallel_executor")


def test_parallel_fci_plus_dsep_defers_unconsumed_singular_query(monkeypatch):
    module = import_module("whydra.algorithms.standalone.fci_plus_algo_paralel_cash")
    data = np.random.default_rng(713).normal(size=(300, 5))
    data[:, 4] = data[:, 3]
    cg = CausalGraph(5, [f"X{i}" for i in range(5)])
    cg.set_ind_test(CIT(data))
    values = module._ordered_dsep_ci(cg, 0, 1, [(2,), (3, 4)], n_jobs=2)
    assert np.isfinite(next(values)[1])
    with pytest.raises(ValueError, match="singular"):
        next(values)


@pytest.mark.parametrize("jobs", [1, 2])
@pytest.mark.parametrize(
    ("algorithm", "fields"),
    [
        ("StandaloneRFCIStable", ("skeleton_backend", "orient_backend")),
        (
            "StandaloneFCIPlusStableParallelCached",
            ("skeleton_backend", "augment_backend", "dsep_backend", "ci_backend"),
        ),
    ],
)
def test_auto_backend_labels_describe_tiny_serial_fits(algorithm, fields, jobs):
    from whydra.algorithms import wrappers

    data = np.random.default_rng(714).normal(size=(100, 3))
    graph = getattr(wrappers, algorithm)(alpha=0.05, n_jobs=jobs).run(data)
    stats = graph.metadata["ci_stats"]
    assert stats["skeleton_backend_requested"] == "auto"
    assert {stats[field] for field in fields} == {"serial"}
    assert graph.metadata["execution"]["actual_backend"] == "serial"


def test_phase_backend_does_not_count_prior_phase_dispatch(monkeypatch):
    before = {"batches": {"serial": 2, "threads": 3, "processes": 4}}
    monkeypatch.setattr(rfci_algo, "batch_execution_stats", lambda test: before)
    assert rfci_algo._observed_phase_backend(None, before) == "serial"
    assert (
        rfci_algo._observed_phase_backend(None, before, extra=["processes"])
        == "processes"
    )
