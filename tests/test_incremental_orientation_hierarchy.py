"""Incremental searches retain full-scan rule order and separator semantics."""

from collections import deque
from importlib import import_module

import numpy as np
import pytest

from whydra.algorithms.graph_core import CausalGraph
from whydra.algorithms.standalone import pag_rules, rfci_algo
from whydra.algorithms.standalone.hierarchy_cache import (
    hierarchy_cache,
    hierarchy_changed,
)
from whydra.algorithms.standalone.orientation_queue import OrientationQueue
from whydra.background_knowledge import (
    BackgroundKnowledge,
    BKOrientationGuard,
    MatrixEncoding,
)


class FullScan:
    """Independent old scheduling oracle; the rule bodies remain identical."""

    def __init__(self, *args, **kwargs):
        pass

    def guard(self, guard):
        return guard

    def candidates(self, rule, pag, mask, *, paths=False, row_order=False):
        pairs = [(a, b) for a in range(len(pag)) for b in range(len(pag)) if mask[a, b]]
        return (
            pairs if row_order else sorted(pairs, key=lambda pair: (pair[1], pair[0]))
        )


def _guard():
    return BKOrientationGuard(None, MatrixEncoding.STANDARD)


def test_queue_skips_unchanged_rules_and_observes_remote_path_changes():
    pag = np.zeros((8, 8), dtype=int)
    for a, b in [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (6, 7)]:
        pag[a, b] = pag[b, a] = 2
    mask = np.zeros_like(pag, dtype=bool)
    mask[0, 1] = mask[6, 7] = True
    queue = OrientationQueue(pag)
    guard = queue.guard(_guard())
    assert list(queue.candidates("local", pag, mask)) == [(0, 1), (6, 7)]
    assert list(queue.candidates("path", pag, mask, paths=True)) == [(0, 1), (6, 7)]
    assert list(queue.candidates("local", pag, mask)) == []
    assert list(queue.candidates("path", pag, mask, paths=True)) == []
    guard.set_mark(pag, 4, 5, 1)
    assert list(queue.candidates("local", pag, mask)) == []
    assert list(queue.candidates("path", pag, mask, paths=True)) == [(0, 1)]
    # Removing an edge changes nonadjacency premises even for local R3.
    pag[3, 4] = pag[4, 3] = 0
    assert list(queue.candidates("local", pag, mask)) == [(0, 1), (6, 7)]


def test_queue_activates_later_candidate_in_same_snapshot_and_defers_earlier():
    pag = np.full((5, 5), 2, dtype=int)
    np.fill_diagonal(pag, 0)
    mask = np.zeros_like(pag, dtype=bool)
    mask[0, 1] = mask[2, 3] = True
    queue = OrientationQueue(pag)
    guard = queue.guard(_guard())
    assert list(queue.candidates("R1", pag, mask)) == [(0, 1), (2, 3)]
    guard.set_mark(pag, 0, 4, 1)
    candidates = queue.candidates("R1", pag, mask)
    assert next(candidates) == (0, 1)
    guard.set_mark(pag, 1, 2, 1)
    assert next(candidates) == (2, 3)
    guard.set_mark(pag, 0, 4, -1)
    assert list(candidates) == []
    assert list(queue.candidates("R1", pag, mask)) == [(0, 1)]


@pytest.mark.parametrize("seed", [1, 8, 17])
@pytest.mark.parametrize("with_bk", [False, True])
def test_pag_incremental_matches_full_scan_with_background_knowledge(
    seed, with_bk, monkeypatch
):
    rng = np.random.default_rng(seed)
    p = 7
    adjacency = np.triu(rng.random((p, p)) < 0.4, 1)
    adjacency[0, 1] = True
    pag = (adjacency | adjacency.T).astype(int) * 2
    separators = {
        (a, b): set(rng.choice(p, size=2, replace=False).tolist())
        for a in range(p)
        for b in range(p)
        if a != b and not pag[a, b]
    }
    bk = BackgroundKnowledge().require("X0", "X1") if with_bk else None
    options = {
        "sepset": separators,
        "p": p,
        "background_knowledge": bk,
        "node_names": [f"X{i}" for i in range(p)],
    }
    expected_stats, actual_stats = {}, {}
    with monkeypatch.context() as patch:
        patch.setattr(pag_rules, "OrientationQueue", FullScan)
        expected = pag_rules.udag2pag(pag, rule_stats=expected_stats, **options)
    actual = pag_rules.udag2pag(pag, rule_stats=actual_stats, **options)
    np.testing.assert_array_equal(actual, expected)
    assert actual_stats == expected_stats


def test_pag_long_uncovered_circle_path_matches_full_scan(monkeypatch):
    pag = np.zeros((7, 7), dtype=int)
    for a, b in [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0)]:
        pag[a, b] = pag[b, a] = 2
    with monkeypatch.context() as patch:
        patch.setattr(pag_rules, "OrientationQueue", FullScan)
        expected = pag_rules.udag2pag(pag, {}, 7, orientCollider=False)
    actual = pag_rules.udag2pag(pag, {}, 7, orientCollider=False)
    np.testing.assert_array_equal(actual, expected)
    assert np.count_nonzero(actual == -1) == 12


@pytest.mark.parametrize("seed", [3, 9])
def test_rfci_full_fit_incremental_orientation_matches_full_scan(seed, monkeypatch):
    from whydra.algorithms.wrappers import StandaloneRFCIStable

    data = np.random.default_rng(seed).normal(size=(160, 6))
    data[:, 2] += data[:, 0] + data[:, 1]
    data[:, 3] += data[:, 2]
    data[:, 5] += data[:, 3] + data[:, 4]
    with monkeypatch.context() as patch:
        patch.setattr(rfci_algo, "OrientationQueue", FullScan)
        expected = StandaloneRFCIStable(alpha=0.05, n_jobs=1).run(data)
    actual = StandaloneRFCIStable(alpha=0.05, n_jobs=1).run(data)
    np.testing.assert_array_equal(actual.graph, expected.graph)


def test_hierarchy_reuses_unaffected_closures_and_detects_external_edits():
    from whydra.algorithms.standalone.fci_plus_algo import _sepset_vars

    cg = CausalGraph(8, [f"X{i}" for i in range(8)])
    cg.sepset[0][1] = [(2,)]
    cache = hierarchy_cache(cg, _sepset_vars)
    assert cache.closure({0, 1}) == {0, 1, 2}
    assert cache.closure({5, 6}) == {5, 6}
    cg.sepset[5][6] = [(7,)]
    hierarchy_changed(cg, ((5, 6),))
    before_hits = cache.hits
    assert cache.closure({0, 1}) == {0, 1, 2}
    assert cache.hits == before_hits + 1
    assert cache.closure({5, 6}) == {5, 6, 7}
    # A dependency which was empty during the first closure must be tracked.
    cg.sepset[2][1] = [(3,)]
    assert hierarchy_cache(cg, _sepset_vars) is cache
    assert cache.closure({0, 1}) == {0, 1, 2, 3}
    graph_version, before_hits = cache.graph_version, cache.hits
    cg.G.graph[0, 1] = cg.G.graph[1, 0] = 0
    hierarchy_cache(cg, _sepset_vars)
    assert cache.graph_version > graph_version
    assert cache.closure({0, 1}) == {0, 1, 2, 3}
    assert cache.hits == before_hits + 1
    # In-place edits are caught by the next public snapshot check as well.
    cg.sepset[2][1].append((4,))
    hierarchy_cache(cg, _sepset_vars)
    assert cache.closure({0, 1}) == {0, 1, 2, 3, 4}


@pytest.mark.parametrize("name", ["fci_plus_algo", "fci_plus_algo_paralel_cash"])
def test_active_dsep_reuses_hierarchy_across_searches_then_invalidates(
    name, monkeypatch
):
    module = import_module("whydra.algorithms.standalone." + name)
    cg = CausalGraph(5, [f"X{i}" for i in range(5)])
    independent = [False]
    cg.set_ind_test(lambda x, y, conditions: 0.8 if independent[0] else 0.001)
    monkeypatch.setattr(module, "AugmentGraph", lambda graph, *a, **k: graph.G.graph)
    monkeypatch.setattr(module, "_dsep_search", lambda *a: deque())
    arguments = (cg, 0.05, 0, 1, (2,), (3,), 1, 5)
    assert not module._dsep(*arguments, n_jobs=1)[0]
    cache = cg._fci_plus_hierarchy_cache
    misses = cache.misses
    assert not module._dsep(*arguments, n_jobs=1)[0]
    assert cache.misses == misses
    assert cache.hits > 0
    independent[0] = True
    assert module._dsep(*arguments, n_jobs=1)[0]
    assert cg.G.graph[0, 1] == cg.G.graph[1, 0] == 0
    assert cache.invalidations > 0
    assert cache.pair_versions[0, 1] > 1
    assert module._get_hie(cg, {0, 1}) == {0, 1, 2}
