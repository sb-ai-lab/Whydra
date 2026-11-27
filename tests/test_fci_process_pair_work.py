"""Search-level process work preserves separators, deferred errors and evidence."""

import copy
import math
from itertools import combinations

import numpy as np
import pytest

from whydra.algorithms.graph_core import CIT
from whydra.algorithms.graph_core.cache import BoundedCache
from whydra.algorithms.graph_core.ci_batch import configure_fisher_alpha
from whydra.algorithms.graph_core.independence_tests import fisher_z_from_corr
from whydra.algorithms.standalone import compact_pdsep as compact
from whydra.algorithms.standalone import fci_algo_paralel_cash as fci
from whydra.algorithms.standalone.ci_adapters import (
    attach_counted_batch,
    attach_legacy_batch,
)
from whydra.algorithms.standalone.conservative_triples import classify


def _counted(data, capacity=7, cap=3):
    cit = CIT(data, cache_size=capacity, max_cond_set_size=cap)
    configure_fisher_alpha(cit, 0.05)
    cache = BoundedCache(capacity)
    stats = {"total": 0, "pdsep": 0, "ci_cache_hits": 0, "ci_cache_misses": 0}
    phase = {"name": "pdsep"}

    def counted(x, y, nodes):
        stats["total"] += 1
        stats["pdsep"] += 1
        return cit(x, y, nodes)

    attach_counted_batch(counted, cit, stats, phase, cache)

    def legacy(_suff, x, y, nodes):
        return counted(x, y, nodes)

    attach_legacy_batch(legacy, counted)
    legacy.max_cond_set_size = cap
    return legacy, cit, cache, stats


@pytest.mark.parametrize("alpha", [0.0, 0.05, 1.0])
def test_combined_pair_windows_preserve_individual_consumed_prefix(alpha):
    data = np.random.default_rng(190).normal(size=(180, 8))
    corr = np.corrcoef(data.T)
    descriptors = [
        (
            i,
            (
                0,
                y,
                (2, 3, 4, 5, 6, 7),
                (4, 5, 6, 7),
                (y, 2, 3),
                3,
                alpha,
                -math.inf,
                {},
                3,
                1,
            ),
        )
        for i, y in enumerate((1, 1, 1, 1, 1))
    ]
    expected = {
        i: compact._compact_pair(corr, len(data), *args) for i, args in descriptors
    }
    actual = dict(compact._compact_pair_chunk(corr, len(data), descriptors))
    assert actual == expected


def test_combined_pair_windows_keep_unused_singular_errors_local():
    data = np.random.default_rng(191).normal(size=(180, 5))
    data[:, 4] = data[:, 0]
    corr = np.corrcoef(data.T)
    descriptors = [
        (0, (0, 1, (2, 4), (2, 4), (1,), 1, 0.0, -math.inf, {}, 3, 1)),
        (1, (0, 1, (4,), (4,), (1,), 1, 0.0, -math.inf, {}, 3, 1)),
    ]
    # Independent first separators and later error envelopes stay with their
    # owning pair, even when both pairs share a single numerical evaluation.
    actual = dict(compact._compact_pair_chunk(corr, len(data), descriptors))
    assert actual[0]["separator"] == (2,)
    assert actual[0]["error"] is None
    assert isinstance(actual[1]["error"], ValueError)


def test_incremental_cache_index_reuses_own_publication_and_refreshes_external_writes():
    data = np.random.default_rng(192).normal(size=(180, 6))
    legacy, cit, outer, _stats = _counted(data, capacity=2)
    runner = compact.CompactPDSEP(legacy, 2)
    runner._refresh_indices()
    result = compact._compact_pair(
        cit.corr_matrix,
        len(data),
        0,
        1,
        (2, 3, 4),
        (2, 3, 4),
        (1,),
        1,
        1.0,
        -math.inf,
        {},
        2,
    )
    runner.publish(result)
    runner._refresh_indices()
    assert runner.stats["cache_index_rebuilds"] == 1
    assert runner.cit_index == compact._cache_index(cit._cache)
    assert runner.outer_index == compact._cache_index(outer)
    cit(2, 3, (4,))
    runner._refresh_indices()
    assert runner.stats["cache_index_rebuilds"] == 2
    assert runner.cit_index == compact._cache_index(cit._cache)


@pytest.mark.parametrize("majority", [False, True])
@pytest.mark.parametrize("cap", [0, 1, 3])
@pytest.mark.parametrize("alpha", [0.0, 0.05, 1.0])
def test_triple_sufficient_summary_matches_full_separator_evidence(
    majority, cap, alpha
):
    data = np.random.default_rng(193).normal(size=(180, 7))
    corr = np.corrcoef(data.T)
    left, right = (2, 3, 4), (3, 5)
    known_a, known_c = {6}, None
    conditions = sorted(
        {
            nodes
            for side in (left, right)
            for order in range(cap + 1)
            for nodes in combinations(side, order)
        },
        key=lambda s: (len(s), s),
    )
    assert list(fci._triple_conditions(left, right, cap)) == conditions
    evidence = [
        nodes
        for nodes in conditions
        if fisher_z_from_corr(corr, len(data), 0, 1, nodes, decision_alpha=alpha)
        >= alpha
    ]
    result = fci._triple_pair_worker(
        corr, len(data), 0, 1, left, right, cap, alpha, {tuple(sorted(known_a))}, {}, 2
    )
    assert len(result["updates"]) <= 2
    assert result["count"] == len(evidence)
    for b in (2, 3, 5, 6):
        expected, sepsets = classify(b, evidence, known_a, known_c, majority)
        if cap < 3:
            expected = 3
        assert fci._classify_triple_summary(
            b, result, known_a, known_c, majority, cap < 3
        ) == (expected, sepsets)


@pytest.mark.parametrize("jobs", [1, 2])
def test_native_triples_share_pair_work_and_match_custom_serial(jobs, monkeypatch):
    monkeypatch.setattr(fci, "MIN_TRIPLE_PROCESS_TESTS", 0)
    data = np.random.default_rng(194).normal(size=(180, 5))
    graph = np.zeros((5, 5), dtype=int)
    for x, y in ((0, 2), (0, 3), (0, 4), (1, 2), (1, 3), (1, 4)):
        graph[x, y] = graph[y, x] = 2
    initial = {"sk": graph, "sepset": {}, "unfTriples": set()}
    legacy, _cit, _outer, stats = _counted(data)
    actual = fci.pc_cons_intern(
        copy.deepcopy(initial),
        None,
        0.05,
        legacy,
        n_jobs=jobs,
        parallel_backend="processes",
    )
    reference_cit = CIT(data)
    configure_fisher_alpha(reference_cit, 0.05)
    calls = []

    def oracle(_suff, x, y, nodes):
        calls.append((x, y, tuple(nodes)))
        return reference_cit(x, y, nodes)

    expected = fci.pc_cons_intern(
        copy.deepcopy(initial), None, 0.05, oracle, n_jobs=jobs
    )
    assert actual["triple_classification"] == expected["triple_classification"]
    assert actual["sepset"] == expected["sepset"]
    assert actual["unfTriples"] == expected["unfTriples"]
    assert actual["triple_stats"]["pair_tasks"] < actual["triple_stats"]["triples"]
    assert actual["triple_stats"]["backend"] == ("processes" if jobs > 1 else "serial")
    assert expected["triple_stats"]["backend"] == "serial"
    assert stats["total"] < len(calls)


@pytest.mark.parametrize("module_name", ["cfci_algo", "fci_algo_paralel_cash"])
def test_overridden_triple_hook_remains_authoritative(module_name, monkeypatch):
    from importlib import import_module

    module = import_module("whydra.algorithms.standalone." + module_name)
    data = np.random.default_rng(195).normal(size=(120, 3))
    legacy, _cit, _outer, _stats = _counted(data)
    graph = {
        "sk": np.array([[0, 2, 0], [2, 0, 2], [0, 2, 0]]),
        "sepset": {},
        "unfTriples": set(),
    }
    calls = []

    def hook(a, b, c, *_args, **_kwargs):
        calls.append((a, b, c))
        return 3, {"sepsetA": {b}, "sepsetC": {b}}

    monkeypatch.setattr(module, "checkTriple", hook)
    actual = module.pc_cons_intern(graph, None, 0.05, legacy, n_jobs=2)
    assert calls == [(0, 1, 2)]
    assert actual["triple_classification"] == {(0, 1, 2): 3}
