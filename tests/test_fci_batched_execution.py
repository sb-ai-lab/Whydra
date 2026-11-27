"""PAG batching must preserve scalar search results and early-exit behavior."""

from collections import deque
from importlib import import_module

import numpy as np
import pytest

from whydra.algorithms.graph_core import CIT, CausalGraph
from whydra.algorithms.graph_core.cache import BoundedCache
from whydra.algorithms.standalone.ci_adapters import (
    attach_counted_batch,
    attach_legacy_batch,
    ordered_ci_values,
)

KERNELS = (
    ("fci_algo_cash", "fci_stable"),
    ("fci_algo_paralel_cash", "fci_stable"),
    ("cfci_algo", "cfci_stable"),
    ("cfci_algo_paralel_cash", "cfci_stable"),
    ("fci_plus_algo", "fci_plus_stable"),
    ("fci_plus_algo_paralel_cash", "fci_plus_stable"),
    ("rfci_algo", "rfci_stable"),
)


def _data(seed):
    rng = np.random.default_rng(seed)
    variables = 10
    weights = np.triu(
        (rng.random((variables, variables)) < 0.24)
        * rng.uniform(0.5, 1.0, (variables, variables)),
        1,
    )
    data = rng.normal(size=(600, variables))
    for node in range(variables):
        data[:, node] += data[:, :node] @ weights[:node, node]
    return data


def _fit(kernel, data, jobs):
    cg = CausalGraph(data.shape[1], [f"X{node}" for node in range(data.shape[1])])
    # Frequent eviction also exercises synchronization of the outer memoizer.
    cg.set_ind_test(CIT(data, cache_size=17))
    return np.asarray(kernel(cg, data, alpha=0.05, n_jobs=jobs)).copy()


@pytest.mark.parametrize(("module_name", "kernel_name"), KERNELS)
@pytest.mark.parametrize("jobs", [1, 4])
@pytest.mark.parametrize("seed", [3, 9])
def test_fci_family_batch_matches_scalar_search(
    module_name, kernel_name, jobs, seed, monkeypatch
):
    kernel = getattr(
        import_module(f"whydra.algorithms.standalone.{module_name}"), kernel_name
    )
    data = _data(seed)
    original = CIT.fisher_z

    def scalar_fisher(self, x, y, conditions):
        return original(self, x, y, conditions)

    # A custom numerical method opts out of native batching without changing
    # any p-value, giving an independent scalar traversal of the same search.
    with monkeypatch.context() as patch:
        patch.setattr(CIT, "fisher_z", scalar_fisher)
        scalar = _fit(kernel, data, 1)
    batched = _fit(kernel, data, jobs)
    np.testing.assert_array_equal(batched, scalar)


def _legacy(cit, cache=None):
    stats = {"total": 0, "pdsep": 0, "ci_cache_hits": 0, "ci_cache_misses": 0}
    phase = {"name": "pdsep"}

    def counted(x, y, conditions):
        stats["total"] += 1
        stats["pdsep"] += 1
        return cit(x, y, conditions)

    attach_counted_batch(counted, cit, stats, phase, cache)

    def legacy(_suff_stat, x, y, conditions):
        return counted(x, y, conditions)

    attach_legacy_batch(legacy, counted)
    return legacy, stats


def test_counted_batch_deduplicates_symmetric_requests_and_accounts_phase():
    cit = CIT(_data(18), cache_size=4)
    legacy, stats = _legacy(cit, BoundedCache(4))
    queries = [(0, 1, (2,)), (1, 0, (2,)), (0, 3, ())]
    values = legacy.batch(queries, n_jobs=4)
    assert values[0] == values[1]
    assert stats == {"total": 3, "pdsep": 3, "ci_cache_hits": 1, "ci_cache_misses": 2}
    assert cit.cache_misses == 2
    assert legacy.batch(queries) == values
    assert stats == {"total": 6, "pdsep": 6, "ci_cache_hits": 4, "ci_cache_misses": 2}


def test_ordered_fisher_batch_does_not_raise_after_first_separator():
    data = _data(29)[:, :4]
    data[:, 3] = data[:, 2]
    cit = CIT(data)
    legacy, _stats = _legacy(cit)
    results = ordered_ci_values(legacy, None, 0, 1, [(2,), (2, 3)], n_jobs=4)
    conditions, pvalue = next(results)
    assert conditions == (2,)
    assert np.isfinite(pvalue)
    # Closing after the separator ignores a singular submatrix later in the
    # speculative batch, just as stopping scalar search would.
    results.close()
    with pytest.raises(ValueError, match="singular"):
        list(ordered_ci_values(legacy, None, 0, 1, [(2,), (2, 3)], n_jobs=4))


def test_generic_pdsep_oracle_keeps_lazy_query_order():
    calls = []

    def oracle(_suff_stat, x, y, conditions):
        calls.append((x, y, tuple(conditions)))
        if len(conditions) > 1:
            pytest.fail("An oracle must not evaluate a query after early exit")
        return 0.8

    results = ordered_ci_values(oracle, None, 0, 1, [(2,), (2, 3)], n_jobs=4)
    assert next(results) == ((2,), 0.8)
    results.close()
    assert calls == [(0, 1, (2,))]


@pytest.mark.parametrize(("module_name", "kernel_name"), KERNELS[1:2] + KERNELS[3:4])
def test_process_pdsep_batch_is_not_counted_twice(
    module_name, kernel_name, monkeypatch
):
    module = import_module(f"whydra.algorithms.standalone.{module_name}")
    common = import_module("whydra.algorithms.standalone.fci_algo_paralel_cash")
    data = _data(3)[:, :6]
    monkeypatch.setattr(common, "qreach", lambda *_: list(range(data.shape[1])))
    dispatched = []
    original_attach = module.attach_counted_batch

    def track_batch(counted, cit, stats, phase, *args):
        original_attach(counted, cit, stats, phase, *args)
        batch = counted.batch

        def record(requests, **kwargs):
            requests = list(requests)
            if phase["name"] == "pdsep":
                dispatched.append(len(requests))
            return batch(requests, **kwargs)

        counted.batch = record
        return counted

    original_pdsep = module.pdsep

    def force_process_rounds(*args, **kwargs):
        kwargs["process_min_tasks"] = 0
        return original_pdsep(*args, **kwargs)

    monkeypatch.setattr(module, "attach_counted_batch", track_batch)
    monkeypatch.setattr(module, "pdsep", force_process_rounds)
    cg = CausalGraph(data.shape[1], [f"X{node}" for node in range(data.shape[1])])
    cg.set_ind_test(CIT(data))
    getattr(module, kernel_name)(cg, data, n_jobs=2, pdsep_backend="processes")
    assert sum(dispatched) > 0
    assert cg.ci_stats["pdsep"] == sum(dispatched)


def _dsep_graph():
    data = _data(29)[:, :4]
    data[:, 3] = data[:, 2]
    cg = CausalGraph(4, ["X0", "X1", "X2", "X3"])
    cg.set_ind_test(CIT(data))
    return cg


def test_base_fci_plus_dsep_batches_and_defers_later_singular_query(monkeypatch):
    from whydra.algorithms.standalone import fci_plus_algo as module

    cg = _dsep_graph()
    batch_jobs = []
    batch = cg.test.batch

    def record(requests, **kwargs):
        batch_jobs.append(kwargs["n_jobs"])
        return batch(requests, **kwargs)

    monkeypatch.setattr(cg.test, "batch", record)
    monkeypatch.setattr(
        module, "AugmentGraph", lambda cg, *_args, **_kwargs: cg.G.graph
    )
    monkeypatch.setattr(module, "_dsep_search", lambda *_: deque())
    # The first closure is {2}, the second is {2, 3}, whose duplicate columns
    # make its Fisher submatrix singular. Search must stop at the first one.
    found, _queue = module._dsep(cg, 0.0, 0, 1, (2,), (2, 3), 1, 4, n_jobs=4)
    assert found
    assert cg.sepset[0, 1] == [(2,)]
    assert cg.G.graph[0, 1] == cg.G.graph[1, 0] == 0
    assert len(batch_jobs) >= 2  # DSEP candidate search and separator minimization
    assert set(batch_jobs) == {4}


def test_base_fci_plus_dsep_raises_singular_query_if_no_earlier_separator():
    from whydra.algorithms.standalone import fci_plus_algo as module

    cg = _dsep_graph()
    with pytest.raises(ValueError, match="singular"):
        module._dsep(cg, 1.0, 0, 1, (2,), (2, 3), 1, 4, n_jobs=4)


def test_base_fci_plus_dsep_preserves_custom_oracle_early_exit(monkeypatch):
    from whydra.algorithms.standalone import fci_plus_algo as module

    cg = _dsep_graph()
    calls = []

    def oracle(x, y, conditions):
        calls.append(tuple(conditions))
        if len(conditions) > 1:
            pytest.fail("DSEP queried an oracle after finding its separator")
        return 0.8

    cg.set_ind_test(oracle)
    monkeypatch.setattr(
        module, "AugmentGraph", lambda cg, *_args, **_kwargs: cg.G.graph
    )
    monkeypatch.setattr(module, "_dsep_search", lambda *_: deque())
    found, _queue = module._dsep(cg, 0.05, 0, 1, (2,), (2, 3), 1, 4, n_jobs=4)
    assert found
    assert calls == [(2,), (2,)]


def _frozen_reference(matrix, data, cap, alpha, protected=()):
    """Independent scalar evaluation of the original full-level search contract."""
    from itertools import combinations

    result = matrix.copy()
    pmax = np.full(matrix.shape, -np.inf)
    separators = {}
    ci = CIT(data, cache_size=0)
    states = []
    for x in range(len(matrix)):
        adjacent = tuple(np.flatnonzero(matrix[x]))
        for y in adjacent:
            if frozenset((x, y)) in protected:
                continue
            possible = tuple(node for node in range(len(matrix)) if node not in (x, y))
            diff = tuple(node for node in possible if node not in adjacent)
            if diff:
                states.append((x, y, possible, adjacent, diff))
    for order in range(1, cap + 1):
        # Materialize all values independently of commit; tests use nonsingular
        # data here. This reference deliberately does not use production helpers.
        level = []
        for x, y, possible, adjacent, diff in states:
            if result[x, y] == result[y, x] == 0:
                continue
            conditions = (
                [(node,) for node in diff]
                if order == 1
                else [
                    nodes
                    for nodes in combinations(possible, order)
                    if not all(node in adjacent for node in nodes)
                ]
            )
            level.append((x, y, [(nodes, ci(x, y, nodes)) for nodes in conditions]))
        for x, y, values in level:
            if result[x, y] == result[y, x] == 0:
                continue
            for nodes, value in values:
                pmax[x, y] = max(pmax[x, y], value)
                if value >= alpha:
                    result[x, y] = result[y, x] = 0
                    separators[x, y] = separators[y, x] = set(nodes)
                    break
    return result, pmax, separators


@pytest.mark.parametrize("jobs", [1, 2])
@pytest.mark.parametrize("cache_size", [0, 1, 4])
@pytest.mark.parametrize("alpha", [0.0, 0.05, 1.0])
def test_frozen_pdsep_coarse_level_matches_scalar_contract(
    jobs, cache_size, alpha, monkeypatch
):
    from whydra.algorithms.standalone import compact_pdsep
    from whydra.algorithms.standalone import fci_algo_paralel_cash as module

    data = _data(30)[:, :6]
    matrix = np.zeros((6, 6), dtype=int)
    for x, y in ((0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 5)):
        matrix[x, y] = matrix[y, x] = 2
    protected = {frozenset((0, 1))}
    expected = _frozen_reference(matrix, data, 3, alpha, protected)
    monkeypatch.setattr(module, "qreach", lambda *_: list(range(6)))
    monkeypatch.setattr(compact_pdsep, "MIN_ROW_TESTS", 0)
    monkeypatch.setattr(compact_pdsep, "MIN_PAIR_TESTS", 0)
    cit = CIT(data, cache_size=cache_size, max_cond_set_size=3)
    outer = BoundedCache(cache_size)
    legacy, stats = _legacy(cit, outer)
    legacy.max_cond_set_size = 3
    result = module.pdsep(
        {"sk": matrix},
        None,
        legacy,
        6,
        {},
        alpha,
        np.full((6, 6), -np.inf),
        m_max=3,
        unfVect=set(),
        n_jobs=jobs,
        parallel_backend="processes",
        process_min_tasks=0,
        protected_edges=[(0, 1)],
    )
    np.testing.assert_array_equal(result["G"], expected[0])
    np.testing.assert_allclose(result["pMax"], expected[1], atol=1e-13, rtol=1e-12)
    assert result["sepset"] == expected[2]
    assert len(cit._cache) <= cache_size
    assert len(outer) <= cache_size
    assert stats["total"] == stats["pdsep"] == result["pdsep_stats"]["ci_tests"]
    assert result["pdsep_stats"]["ci_consumed"] <= result["pdsep_stats"]["ci_tests"]
    if jobs > 1:
        assert result["pdsep_stats"]["process_pairs"] > 0


@pytest.mark.parametrize("jobs", [1, 2])
def test_frozen_pdsep_defers_speculative_error_after_separator(jobs, monkeypatch):
    from whydra.algorithms.standalone import compact_pdsep
    from whydra.algorithms.standalone import fci_algo_paralel_cash as module

    data = _data(31)[:, :5]
    data[:, 4] = data[:, 0]
    matrix = np.zeros((5, 5), dtype=int)
    matrix[0, 1] = matrix[1, 0] = 2
    monkeypatch.setattr(module, "qreach", lambda *_: list(range(5)))
    monkeypatch.setattr(compact_pdsep, "MIN_ROW_TESTS", 0)
    monkeypatch.setattr(compact_pdsep, "MIN_PAIR_TESTS", 0)

    def fit(alpha):
        legacy, _stats = _legacy(CIT(data), BoundedCache(4))
        return module.pdsep(
            {"sk": matrix.copy()},
            None,
            legacy,
            5,
            {},
            alpha,
            np.full((5, 5), -np.inf),
            m_max=1,
            unfVect=set(),
            n_jobs=jobs,
            parallel_backend="processes",
            process_min_tasks=0,
        )

    result = fit(0.0)
    assert result["sepset"][0, 1] == {2}
    with pytest.raises(ValueError, match="singular"):
        fit(1.0)


def test_frozen_pdsep_generic_oracle_is_lazy_and_backend_is_serial(monkeypatch):
    from whydra.algorithms.standalone import fci_algo_paralel_cash as module

    matrix = np.zeros((7, 7), dtype=int)
    matrix[0, 1] = matrix[1, 0] = 2
    monkeypatch.setattr(module, "qreach", lambda *_: list(range(7)))
    calls = []

    def oracle(_stats, x, y, conditions):
        calls.append((x, y, tuple(conditions)))
        if len(calls) > 1:
            pytest.fail("No callback after the first separator is consumed")
        return 0.8

    result = module.pdsep(
        {"sk": matrix},
        None,
        oracle,
        7,
        {},
        0.05,
        np.full((7, 7), -np.inf),
        unfVect=set(),
        n_jobs=4,
        parallel_backend="threads",
    )
    assert calls == [(0, 1, (2,))]
    assert result["pdsep_stats"]["backend"] == "serial"
    assert result["pdsep_stats"]["ci_tests"] == 1


def test_frozen_pdsep_process_computations_exclude_cached_requests(monkeypatch):
    from whydra.algorithms.standalone import compact_pdsep
    from whydra.algorithms.standalone import fci_algo_paralel_cash as module

    data = _data(32)[:, :6]
    matrix = np.zeros((6, 6), dtype=int)
    matrix[0, 1] = matrix[1, 0] = 2
    monkeypatch.setattr(module, "qreach", lambda *_: list(range(6)))
    monkeypatch.setattr(compact_pdsep, "MIN_ROW_TESTS", 0)
    monkeypatch.setattr(compact_pdsep, "MIN_PAIR_TESTS", 0)
    cit = CIT(data, cache_size=128)
    for node in (2, 3, 4, 5):
        cit(0, 1, (node,))
    legacy, _stats = _legacy(cit, BoundedCache(128))
    result = module.pdsep(
        {"sk": matrix},
        None,
        legacy,
        6,
        {},
        0.0,
        np.full((6, 6), -np.inf),
        m_max=1,
        unfVect=set(),
        n_jobs=2,
        parallel_backend="processes",
        process_min_tasks=0,
    )
    execution = result["pdsep_stats"]
    assert execution["process_pairs"] == 2
    assert execution["ci_tests"] > 0
    assert execution["worker_cache_hits"] > 0
    assert execution["worker_ci_computations"] == execution["process_ci_tests"] == 0


@pytest.mark.parametrize("override", ["__call__", "fisher_z"])
def test_public_parallel_fci_process_request_preserves_custom_cit(
    override, monkeypatch
):
    from whydra.algorithms.standalone import fci_algo_paralel_cash as module

    calls = []

    def custom(self, x, y, conditions=()):
        calls.append((x, y, tuple(conditions)))
        return 0.0

    monkeypatch.setattr(CIT, override, custom)
    data = _data(33)[:, :6]

    def fit(jobs):
        cg = CausalGraph(6, [f"X{node}" for node in range(6)])
        cg.set_ind_test(CIT(data))
        return module.fci_stable(
            cg,
            data,
            n_jobs=jobs,
            max_cond_set_size=1,
            skeleton_backend="processes",
            pdsep_backend="processes",
        )

    scalar = fit(1)
    calls.clear()
    parallel = fit(2)
    assert calls
    np.testing.assert_array_equal(parallel, scalar)
    assert np.count_nonzero(parallel) == 6 * 5


def test_counted_custom_cit_rejects_direct_legacy_numerical_shortcut(monkeypatch):
    from whydra.algorithms.standalone import fci_algo_paralel_cash as module

    calls = []

    def custom(self, x, y, conditions=()):
        calls.append((x, y, tuple(conditions)))
        return 0.9

    monkeypatch.setattr(CIT, "fisher_z", custom)
    monkeypatch.setattr(module, "qreach", lambda *_: list(range(5)))
    data = _data(34)[:, :5]
    legacy, _stats = _legacy(CIT(data), BoundedCache(8))
    matrix = np.zeros((5, 5), dtype=int)
    matrix[0, 1] = matrix[1, 0] = 2
    result = module.pdsep(
        {"sk": matrix},
        None,
        legacy,
        5,
        {},
        0.05,
        np.full((5, 5), -np.inf),
        m_max=1,
        unfVect=set(),
        n_jobs=2,
        parallel_backend="processes",
        process_min_tasks=0,
        process_ci_info={
            "method": "fisherz",
            "n_samples": 300,
            "corr_matrix": np.eye(5) * 0.2 + 0.8,
        },
    )
    assert calls == [(0, 1, (2,))]
    assert result["G"][0, 1] == result["G"][1, 0] == 0
    assert result["pdsep_stats"]["actual_backend"] == "serial"
    assert result["pdsep_stats"]["process_ci_tests"] == 0


def test_legacy_numerical_process_defers_singular_request_after_separator(monkeypatch):
    from whydra.algorithms.standalone import fci_algo_paralel_cash as module

    monkeypatch.setattr(module, "qreach", lambda *_: list(range(5)))
    data = _data(35)[:, :5]
    data[:, 4] = data[:, 0]
    matrix = np.zeros((5, 5), dtype=int)
    matrix[0, 1] = matrix[1, 0] = 2

    def no_parent(*_args):
        pytest.fail("Trusted numerical descriptor must run in processes")

    def fit(alpha):
        return module.pdsep(
            {"sk": matrix.copy()},
            None,
            no_parent,
            5,
            {},
            alpha,
            np.full((5, 5), -np.inf),
            m_max=1,
            unfVect=set(),
            n_jobs=2,
            parallel_backend="processes",
            process_min_tasks=0,
            process_ci_info={
                "method": "fisherz",
                "n_samples": len(data),
                "corr_matrix": np.corrcoef(data.T),
            },
        )

    result = fit(0.0)
    assert result["sepset"][0, 1] == {2}
    assert result["pdsep_stats"]["actual_backend"] == "processes"
    with pytest.raises(ValueError, match="singular"):
        fit(1.0)
