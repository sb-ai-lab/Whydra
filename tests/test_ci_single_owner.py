"""CIT and PAG cache facades share values while retaining cache contracts."""

import gc
import pickle
import random
import threading
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from whydra.algorithms.graph_core import CIT
from whydra.algorithms.graph_core.cache import (
    BoundedCache,
    SharedCacheView,
    ci_cache_for,
)
from whydra.algorithms.standalone.ci_adapters import attach_counted_batch


def _cit(capacity=8):
    return CIT(np.random.default_rng(2606).normal(size=(120, 6)), cache_size=capacity)


def _counted(cit, outer):
    stats = {
        "total": 0,
        "skeleton": 0,
        "pdsep": 0,
        "ci_cache_hits": 0,
        "ci_cache_misses": 0,
    }
    phase = {"name": "skeleton"}
    lock = threading.RLock()

    def counted(x, y, conditions):
        key = (min(x, y), max(x, y), tuple(sorted(set(conditions))))
        with lock:
            stats["total"] += 1
            stats[phase["name"]] += 1
            value = outer.get(key)
            if value is not None:
                stats["ci_cache_hits"] += 1
                return value
            stats["ci_cache_misses"] += 1
        value = cit(x, y, conditions)
        with lock:
            outer[key] = value
        return value

    attach_counted_batch(counted, cit, stats, phase, outer, lock)
    return counted, stats, phase


def test_one_value_index_preserves_distinct_cit_and_outer_membership():
    cit = _cit(1)
    key1, key2 = (0, 1, ()), (0, 2, ())
    value1 = cit(*key1)
    outer = ci_cache_for(cit)
    owner = cit._cache.owner
    assert isinstance(outer, SharedCacheView)
    assert outer.owner is owner
    assert len(outer) == 0
    assert list(cit._cache) == [key1]
    outer[key1] = value1
    assert len(owner.entries) == 1
    assert owner.entries[key1].references == 2
    value2 = cit(*key2)
    # Evicting CIT membership does not erase still-valid outer evidence.
    assert key1 not in cit._cache
    assert outer[key1] == value1
    assert len(owner.entries) == 2
    outer[key2] = value2
    assert list(owner.entries) == [key2]
    assert owner.entries[key2].references == 2
    assert cit.cache_evictions == outer.evictions == 1


def test_shared_views_match_legacy_order_capacity_and_evictions():
    cit = _cit(3)
    outer = ci_cache_for(cit)
    expected_cit, expected_outer = OrderedDict(), BoundedCache(3)
    rng = random.Random(2606)
    keys = [(0, i, ()) for i in range(1, 6)]
    for _ in range(180):
        key = rng.choice(keys)
        op = rng.randrange(5)
        value = key[1] / 10
        if op == 0:
            cit._cache[key] = expected_cit[key] = value
            if len(expected_cit) > 3:
                assert cit._cache.popitem(last=False) == expected_cit.popitem(
                    last=False
                )
        elif op == 1:
            outer[key] = expected_outer[key] = value
        elif op == 2 and key in expected_cit:
            cit._cache.move_to_end(key)
            expected_cit.move_to_end(key)
        elif op == 3:
            assert outer.get(key) == expected_outer.get(key)
        elif op == 4 and key in expected_outer:
            del outer[key]
            del expected_outer[key]
        assert list(cit._cache.items()) == list(expected_cit.items())
        assert list(outer.items()) == list(expected_outer.items())
        assert list(reversed(outer)) == list(reversed(expected_outer))
        assert outer.evictions == expected_outer.evictions
        assert set(outer.owner.entries) == set(expected_cit) | set(expected_outer)


@pytest.mark.parametrize("capacity", [0, 1, 8])
def test_batch_phase_and_cache_counters_match_separate_cache(capacity):
    shared_cit, legacy_cit = _cit(capacity), _cit(capacity)
    shared_outer, legacy_outer = ci_cache_for(shared_cit), BoundedCache(capacity)
    shared, shared_stats, shared_phase = _counted(shared_cit, shared_outer)
    legacy, legacy_stats, legacy_phase = _counted(legacy_cit, legacy_outer)
    requests = [(0, 1, (2,)), (1, 0, (2,)), (0, 3, ()), (0, 1, (2,))]
    for phase in ["skeleton", "pdsep", "pdsep"]:
        shared_phase["name"] = legacy_phase["name"] = phase
        assert shared.batch(requests) == legacy.batch(requests)
        assert shared_stats == legacy_stats
        assert (
            shared_cit.cache_hits,
            shared_cit.cache_misses,
            shared_cit.cache_evictions,
        ) == (
            legacy_cit.cache_hits,
            legacy_cit.cache_misses,
            legacy_cit.cache_evictions,
        )
        assert list(shared_cit._cache.items()) == list(legacy_cit._cache.items())
        assert list(shared_outer.items()) == list(legacy_outer.items())
        assert shared_outer.evictions == legacy_outer.evictions
    if capacity == 0:
        assert len(shared_cit._cache.owner.entries) == 0
        assert shared_cit.cache_misses == 12


def test_alpha_refinement_shares_one_value_and_keeps_both_lru_orders():
    from whydra.algorithms.graph_core.ci_batch import configure_fisher_alpha

    cit = _cit()
    outer = ci_cache_for(cit)
    counted, _, _ = _counted(cit, outer)
    keys = [(0, 1, ()), (0, 2, (3,))]
    values = counted.batch(keys)
    before_cit, before_outer = list(cit._cache), list(outer)
    counters = cit.cache_hits, cit.cache_misses, cit.cache_evictions, outer.evictions
    configure_fisher_alpha(counted, values[0])
    assert list(cit._cache) == before_cit
    assert list(outer) == before_outer
    assert counters == (
        cit.cache_hits,
        cit.cache_misses,
        cit.cache_evictions,
        outer.evictions,
    )
    assert all(
        cit._cache[key] == outer[key] == outer.owner.entries[key].value for key in keys
    )
    assert len(outer.owner.entries) == len(keys)


def test_shared_batch_uses_one_owner_lookup_for_both_cache_layers():
    import sys

    cit = _cit()
    outer = ci_cache_for(cit)
    counted, stats, _ = _counted(cit, outer)
    outer_hit, cit_hit, missing = (0, 1, ()), (0, 2, ()), (0, 3, ())
    counted(*outer_hit)
    cit(*cit_hit)

    class LookupCounts(dict):
        def __init__(self, values):
            super().__init__(values)
            self.reads = []

        def get(self, key, default=None):
            caller = sys._getframe(1).f_code.co_name
            if caller != "_set":
                self.reads.append((caller, key))
            return super().get(key, default)

    owner = outer.owner
    owner.entries = LookupCounts(owner.entries)
    requests = [outer_hit, cit_hit, missing, (3, 0, ())]
    counted.batch(requests)
    assert owner.entries.reads == [
        ("cit_batch", key) for key in [outer_hit, cit_hit, missing, missing]
    ]
    assert stats["ci_cache_hits"] == 2
    assert stats["ci_cache_misses"] == 3
    assert cit.cache_hits == 1
    assert cit.cache_misses == 3
    owner.entries.reads.clear()
    cit(*cit_hit)
    assert owner.entries.reads == [("_node", cit_hit)]


def test_outer_lifetime_releases_its_unique_values_and_reuses_slots():
    cit = _cit()
    outer = ci_cache_for(cit)
    owner, slot = outer.owner, outer._slot
    outer[(0, 1, ())] = 0.3
    assert len(owner.entries) == 1
    del outer
    gc.collect()
    assert len(owner.entries) == 0
    replacement = ci_cache_for(cit)
    assert replacement._slot == slot
    assert len(replacement) == 0


def test_cit_pickle_is_an_independent_cache_snapshot():
    cit = _cit()
    outer = ci_cache_for(cit)
    value = cit(0, 1)
    outer[(0, 1, ())] = value
    restored = pickle.loads(pickle.dumps(cit))
    assert isinstance(restored._cache, SharedCacheView)
    assert restored._cache.owner is not outer.owner
    assert list(restored._cache.items()) == list(cit._cache.items())
    restored._cache.clear()
    assert len(cit._cache) == len(outer) == 1
    snapshot = pickle.loads(pickle.dumps(outer))
    assert isinstance(snapshot, BoundedCache)
    assert list(snapshot.items()) == list(outer.items())


@pytest.mark.parametrize("fail", [False, True])
def test_shared_cache_keeps_single_flight_and_releases_failed_queries(
    monkeypatch, fail
):
    cit = _cit()
    outer = ci_cache_for(cit)
    entered, release = threading.Event(), threading.Event()
    calls = []

    def numerical(x, y, conditions):
        calls.append((x, y, conditions))
        entered.set()
        assert release.wait(5)
        if fail:
            raise ValueError("intentional CI error")
        return 0.25

    monkeypatch.setattr(cit, "fisher_z", numerical)
    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(cit, 0, 1)
        assert entered.wait(5)
        second = executor.submit(cit, 1, 0)
        # Ensure the duplicate joined the in-flight query before releasing it.
        for _ in range(10000):
            with cit._lock:
                pending = cit._inflight.get((0, 1, ()))
            if pending is not None:
                break
            threading.Event().wait(0.0001)
        assert pending is not None
        release.set()
        if fail:
            for future in [first, second]:
                with pytest.raises(ValueError, match="intentional CI error"):
                    future.result(timeout=5)
        else:
            assert first.result(timeout=5) == second.result(timeout=5) == 0.25
            outer[(0, 1, ())] = 0.25
    assert len(calls) == 1
    assert not cit._inflight
    assert len(outer.owner.entries) == (0 if fail else 1)
    assert cit.cache_misses == 1
    assert cit.cache_hits == (0 if fail else 1)


@pytest.mark.parametrize("fail", [False, True])
def test_concurrent_counted_batches_share_flight_without_losing_phase_counts(
    monkeypatch, fail
):
    from whydra.algorithms.graph_core import ci_batch

    cit = _cit()
    outer = ci_cache_for(cit)
    counted, stats, _ = _counted(cit, outer)
    entered, release = threading.Event(), threading.Event()
    requests_computed = []

    def numerical(corr, n_samples, queries, **kwargs):
        if not queries:
            return []
        requests_computed.extend(queries)
        entered.set()
        assert release.wait(5)
        return [ValueError("batch CI error") if fail else 0.25 for _ in queries]

    monkeypatch.setattr(ci_batch, "fisher_z_batch_from_corr", numerical)
    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(counted.batch, [(0, 1, ())])
        assert entered.wait(5)
        second = executor.submit(counted.batch, [(1, 0, ())])
        for _ in range(10000):
            with cit._lock:
                pending = cit._inflight.get((0, 1, ()))
            if pending is not None:
                break
            threading.Event().wait(0.0001)
        assert pending is not None
        release.set()
        if fail:
            for future in [first, second]:
                with pytest.raises(ValueError, match="batch CI error"):
                    future.result(timeout=5)
        else:
            assert first.result(timeout=5) == second.result(timeout=5) == [0.25]
    assert len(requests_computed) == 1
    assert not cit._inflight
    assert stats == {
        "total": 2,
        "skeleton": 2,
        "pdsep": 0,
        "ci_cache_hits": 0,
        "ci_cache_misses": 2,
    }
    assert cit.cache_misses == 1
    assert cit.cache_hits == (0 if fail else 1)
    assert len(outer.owner.entries) == (0 if fail else 1)


def test_custom_and_categorical_outer_caches_retain_legacy_fallback(monkeypatch):
    categorical = CIT(
        np.random.default_rng(2608).integers(0, 3, (40, 3)), method="chisq"
    )
    assert isinstance(ci_cache_for(categorical), BoundedCache)
    custom = _cit()
    monkeypatch.setattr(custom, "fisher_z", lambda *args: 0.5)
    assert isinstance(ci_cache_for(custom), BoundedCache)


@pytest.mark.parametrize(
    ("module_name", "wrapper_name"),
    [
        ("fci_algo_cash", "StandaloneFCIStableCached"),
        ("fci_algo_paralel_cash", "StandaloneFCIStableParallelCached"),
        ("cfci_algo", "StandaloneCFCIStable"),
        ("cfci_algo_paralel_cash", "StandaloneCFCIStableParallelCached"),
        ("rfci_algo", "StandaloneRFCIStable"),
        ("fci_plus_algo", "StandaloneFCIPlusStable"),
        ("fci_plus_algo_paralel_cash", "StandaloneFCIPlusStableParallelCached"),
    ],
)
def test_default_pag_fit_uses_single_owner_without_graph_change(
    monkeypatch, module_name, wrapper_name
):
    import importlib

    from whydra.algorithms import wrappers

    module = importlib.import_module("whydra.algorithms.standalone." + module_name)
    captured = []

    def capture(cit):
        outer = ci_cache_for(cit)
        captured.append((cit, outer))
        return outer

    monkeypatch.setattr(module, "ci_cache_for", capture)
    data = np.random.default_rng(2607).normal(size=(120, 6))
    data[:, 1] += 0.8 * data[:, 0]
    data[:, 3] += 0.7 * data[:, 1] - 0.8 * data[:, 2]
    data[:, 4] += 0.5 * data[:, 3]
    actual = getattr(wrappers, wrapper_name)(n_jobs=1).run(data)
    assert captured
    assert all(isinstance(outer, SharedCacheView) for _, outer in captured)
    for cit, outer in captured:
        assert cit._cache.owner is outer.owner
        assert len(outer.owner.entries) == len(set(cit._cache) | set(outer))
    monkeypatch.setattr(
        module, "ci_cache_for", lambda cit: BoundedCache(cit.cache_size)
    )
    expected = getattr(wrappers, wrapper_name)(n_jobs=1).run(data)
    np.testing.assert_array_equal(actual.graph, expected.graph)
    assert actual.metadata["ci_computations"] == expected.metadata["ci_computations"]
    for counter in ["total", "ci_cache_hits", "ci_cache_misses"]:
        assert (
            actual.metadata["ci_stats"][counter]
            == expected.metadata["ci_stats"][counter]
        )


def test_concurrent_fits_keep_dataset_owners_separate(monkeypatch):
    from whydra.algorithms.standalone import fci_algo_cash
    from whydra.algorithms.wrappers import StandaloneFCIStableCached

    captured = []

    def capture(cit):
        outer = ci_cache_for(cit)
        captured.append((cit, outer))
        return outer

    monkeypatch.setattr(fci_algo_cash, "ci_cache_for", capture)
    inputs = [np.random.default_rng(seed).normal(size=(90, 5)) for seed in [2609, 2610]]
    with ThreadPoolExecutor(max_workers=2) as executor:
        graphs = list(
            executor.map(
                lambda data: StandaloneFCIStableCached(n_jobs=1).run(data), inputs
            )
        )
    assert len(captured) == 2
    assert captured[0][1].owner is not captured[1][1].owner
    monkeypatch.setattr(
        fci_algo_cash, "ci_cache_for", lambda cit: BoundedCache(cit.cache_size)
    )
    for data, actual in zip(inputs, graphs, strict=False):
        expected = StandaloneFCIStableCached(n_jobs=1).run(data)
        np.testing.assert_array_equal(actual.graph, expected.graph)


@pytest.mark.parametrize("jobs", [1, 2])
@pytest.mark.parametrize("capacity", [0, 3])
def test_worker_publication_keeps_shared_cache_tails_and_counts(jobs, capacity):
    from whydra.algorithms.graph_core import CausalGraph
    from whydra.algorithms.standalone.skeleton import SkeletonDiscovery

    data = np.random.default_rng(2611).normal(size=(120, 12))

    def level(shared):
        cit = CIT(data, cache_size=capacity)
        outer = ci_cache_for(cit) if shared else BoundedCache(capacity)
        counted, stats, _ = _counted(cit, outer)
        cg = CausalGraph(12, [str(i) for i in range(12)])
        cg.set_ind_test(counted)
        cg.pMax = np.full((12, 12), -np.inf)
        discovery = SkeletonDiscovery(cg, n_jobs=jobs)
        result = discovery._process_depth_fci_native(1, None, {})
        return cit, outer, stats, cg, result

    actual, expected = level(True), level(False)
    cit, outer, stats, graph, result = actual
    old_cit, old_outer, old_stats, old_graph, old_result = expected
    assert result == old_result
    assert stats == old_stats
    np.testing.assert_array_equal(graph.sepset, old_graph.sepset)
    np.testing.assert_array_equal(graph.pMax, old_graph.pMax)
    assert list(cit._cache.items()) == list(old_cit._cache.items())
    assert list(outer.items()) == list(old_outer.items())
    assert (cit.cache_hits, cit.cache_misses, cit.cache_evictions, outer.evictions) == (
        old_cit.cache_hits,
        old_cit.cache_misses,
        old_cit.cache_evictions,
        old_outer.evictions,
    )
    assert len(outer.owner.entries) == len(set(cit._cache) | set(outer))
