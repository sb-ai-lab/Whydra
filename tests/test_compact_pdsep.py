"""Compact PDSEP workers preserve first separators and live row dependencies."""

import math
from importlib import import_module
from itertools import combinations

import numpy as np
import pytest

from whydra.algorithms.graph_core import CIT, CausalGraph
from whydra.algorithms.graph_core.cache import BoundedCache
from whydra.algorithms.graph_core.independence_tests import fisher_z_from_corr
from whydra.algorithms.standalone import compact_pdsep
from whydra.algorithms.standalone.ci_adapters import (
    attach_counted_batch,
    attach_legacy_batch,
)


def _corr(variables=6):
    data = np.random.default_rng(83).normal(size=(600, variables))
    return np.corrcoef(data.T)


def _force_process_policy(monkeypatch):
    # These tests exercise row barriers in actual child processes. The auto
    # cost model is tested separately and may correctly keep tiny rows local.
    original = compact_pdsep.plan_process_work

    def explicit(*args, **kwargs):
        kwargs["backend"] = "processes"
        return original(*args, **kwargs)

    monkeypatch.setattr(compact_pdsep, "plan_process_work", explicit)


@pytest.mark.parametrize("cap", [0, 1, 2, 3])
def test_pair_worker_matches_scalar_order_and_bounds(cap):
    corr = _corr()
    tf = (2, 3, 4, 5)
    adjacent = (1, 2)
    diff = (3, 4, 5)
    expected = [
        nodes
        for order in range(1, cap + 1)
        for nodes in (
            ([(node,) for node in diff]) if order == 1 else combinations(tf, order)
        )
        if not all(node in adjacent for node in nodes)
    ]
    values = [fisher_z_from_corr(corr, 600, 0, 1, nodes) for nodes in expected]
    result = compact_pdsep._compact_pair(
        corr, 600, 0, 1, tf, diff, adjacent, cap, 1.0, -math.inf, {}, 2
    )
    assert result["separator"] is None
    assert result["error"] is None
    assert result["computed"] == result["consumed"] == len(expected)
    assert result["order"] == cap
    assert result["pmax"] == pytest.approx(max(values, default=-math.inf), abs=1e-14)
    assert len(result["updates"]) <= 2
    assert all(len(nodes) <= cap for nodes, _value in result["updates"])


def test_pair_worker_defers_unused_singular_query():
    data = np.random.default_rng(84).normal(size=(600, 4))
    data[:, 3] = data[:, 0]
    corr = np.corrcoef(data.T)
    options = (corr, 600, 0, 1, (2, 3), (2, 3), (1,), 1)
    first_separator = compact_pdsep._compact_pair(*options, 0.0, -math.inf, {}, 4)
    assert first_separator["separator"] == (2,)
    assert first_separator["error"] is None
    assert first_separator["consumed"] == 1
    assert first_separator["computed"] == 2
    no_separator = compact_pdsep._compact_pair(*options, 1.0, -math.inf, {}, 4)
    assert isinstance(no_separator["error"], ValueError)
    assert "singular" in str(no_separator["error"])
    assert no_separator["consumed"] == 2


def test_pair_worker_reuses_only_relevant_cache_values():
    cached = {(2,): (0.1, 0), (3,): (0.2, 1)}
    result = compact_pdsep._compact_pair(
        _corr(4), 600, 0, 1, (2, 3), (2, 3), (1,), 1, 1.0, -math.inf, cached, 4
    )
    assert result["computed"] == 0
    assert result["outer_hits"] == result["cit_hits"] == 1
    assert result["requests"] == 2
    assert result["updates"] == [((3,), 0.2)]


def _legacy(cit, outer):
    stats = {"total": 0, "pdsep": 0, "ci_cache_hits": 0, "ci_cache_misses": 0}
    phase = {"name": "pdsep"}

    def counted(x, y, conditions):
        stats["total"] += 1
        stats["pdsep"] += 1
        return cit(x, y, conditions)

    attach_counted_batch(counted, cit, stats, phase, outer)

    def legacy(_suff_stat, x, y, conditions):
        return counted(x, y, conditions)

    attach_legacy_batch(legacy, counted)
    legacy.max_cond_set_size = 1
    return legacy, stats


@pytest.mark.parametrize("module_name", ["fci_algo_cash", "cfci_algo"])
@pytest.mark.parametrize("protected", [False, True])
def test_compact_rows_preserve_live_adjacency_and_bounded_caches(
    module_name, protected, monkeypatch
):
    _force_process_policy(monkeypatch)
    module = import_module(f"whydra.algorithms.standalone.{module_name}")
    monkeypatch.setattr(compact_pdsep, "MIN_ROW_TESTS", 0)
    monkeypatch.setattr(compact_pdsep, "MIN_PAIR_TESTS", 0)
    monkeypatch.setattr(module, "qreach", lambda *_: list(range(6)))
    matrix = np.zeros((6, 6), dtype=int)
    for x, y in ((0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 5)):
        matrix[x, y] = matrix[y, x] = 2
    if protected:
        # Keep two unprotected pairs available for process execution even
        # though the first neighbour in the row is protected.
        matrix[0, 3] = matrix[3, 0] = 2
    data = np.random.default_rng(85).normal(size=(600, 6))
    protected_edges = [(0, 1)] if protected else ()

    def fit(jobs):
        cit = CIT(data, cache_size=4, max_cond_set_size=1)
        outer = BoundedCache(4)
        legacy, stats = _legacy(cit, outer)
        result = module.pdsep(
            {"sk": matrix.copy()},
            None,
            legacy,
            6,
            {},
            0.0,
            np.full((6, 6), -np.inf),
            unfVect=set(),
            n_jobs=jobs,
            protected_edges=protected_edges,
        )
        assert len(cit._cache) <= 4
        assert len(outer) <= 4
        assert stats["total"] == stats["pdsep"]
        return result

    sequential, parallel = fit(1), fit(2)
    np.testing.assert_array_equal(parallel["G"], sequential["G"])
    np.testing.assert_allclose(parallel["pMax"], sequential["pMax"], atol=1e-14)
    assert parallel["sepset"] == sequential["sepset"]
    assert parallel["allPdsep"] == sequential["allPdsep"]
    assert parallel["max_ord"] == sequential["max_ord"]
    if not protected:
        # Row zero removes both of its edges. Row one must not dispatch or
        # revisit (1, 0); only this initial row has two independent pairs.
        assert parallel["pdsep_stats"]["process_rows"] == 1
        assert parallel["pdsep_stats"]["process_pairs"] == 2
    else:
        assert parallel["G"][0, 1] != 0
        assert parallel["pdsep_stats"]["process_rows"] > 0


def test_generic_oracle_never_enters_compact_row_workers(monkeypatch):
    module = import_module("whydra.algorithms.standalone.fci_algo_cash")
    monkeypatch.setattr(compact_pdsep, "MIN_ROW_TESTS", 0)
    monkeypatch.setattr(compact_pdsep, "MIN_PAIR_TESTS", 0)
    monkeypatch.setattr(module, "qreach", lambda *_: list(range(5)))
    matrix = np.ones((5, 5), dtype=int) * 2
    np.fill_diagonal(matrix, 0)
    matrix[0, 3] = matrix[3, 0] = matrix[1, 4] = matrix[4, 1] = 0
    calls = []

    def oracle(_stats, x, y, conditions):
        calls.append((x, y, tuple(conditions)))
        return 0.8

    result = module.pdsep(
        {"sk": matrix},
        None,
        oracle,
        5,
        {},
        0.05,
        np.full((5, 5), -np.inf),
        unfVect=set(),
        n_jobs=4,
    )
    assert calls
    assert result["pdsep_stats"]["process_rows"] == 0


@pytest.mark.parametrize(
    ("module_name", "kernel_name"),
    [("fci_algo_cash", "fci_stable"), ("cfci_algo", "cfci_stable")],
)
def test_complete_base_kernel_matches_serial_with_forced_compact_rows(
    module_name, kernel_name, monkeypatch
):
    _force_process_policy(monkeypatch)
    module = import_module(f"whydra.algorithms.standalone.{module_name}")
    monkeypatch.setattr(compact_pdsep, "MIN_ROW_TESTS", 0)
    monkeypatch.setattr(compact_pdsep, "MIN_PAIR_TESTS", 0)
    monkeypatch.setattr(module, "qreach", lambda *_: list(range(8)))
    data = np.random.default_rng(86).normal(size=(600, 8))
    for node in range(1, 8):
        data[:, node] += 0.8 * data[:, node - 1]

    def fit(jobs):
        cg = CausalGraph(8, [f"X{node}" for node in range(8)])
        cg.set_ind_test(CIT(data, cache_size=7))
        graph = getattr(module, kernel_name)(cg, data, n_jobs=jobs)
        return graph, cg.ci_stats

    sequential, _seq_stats = fit(1)
    parallel, stats = fit(2)
    np.testing.assert_array_equal(parallel, sequential)
    assert stats["pdsep_process_rows"] > 0
    assert stats["pdsep_worker_ci_computations"] > 0


@pytest.mark.parametrize("initial_hit", [False, True])
def test_bounded_worker_tail_preserves_cache_eviction_counts(initial_hit):
    data = np.random.default_rng(87).normal(size=(600, 7))
    cit = CIT(data, cache_size=2)
    outer = BoundedCache(2)
    legacy, _stats = _legacy(cit, outer)
    cached = {}
    if initial_hit:
        value = cit(0, 1, (2,))
        cached[(2,)] = (value, 1)  # CIT hit, but a new outer-cache insertion
    worker = compact_pdsep._compact_pair(
        cit.corr_matrix,
        cit.n_samples,
        0,
        1,
        (2, 3, 4, 5, 6),
        (2, 3, 4, 5, 6),
        (1,),
        1,
        2.0,
        -math.inf,
        cached,
        2,
    )
    assert worker["discarded_cit_insertions"] == (2 if initial_hit else 3)
    assert worker["discarded_outer_insertions"] == 3
    publisher = compact_pdsep.CompactPDSEP(legacy, 2)
    publisher._refresh_indices()
    publisher.publish(worker)
    assert cit.cache_misses == 5
    assert cit.cache_evictions == 3
    assert outer.evictions == 3
    assert len(cit._cache) == len(outer) == 2
