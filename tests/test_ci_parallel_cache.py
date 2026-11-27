"""The shared CI cache must deduplicate work without serializing other keys."""

import pickle
import threading
from concurrent.futures import Future, ThreadPoolExecutor

import numpy as np
import pytest

from whydra.algorithms.graph_core import CIT, independence_tests


@pytest.mark.parametrize("cache_size", [0, 4])
@pytest.mark.parametrize("method", ["fisherz", "chisq"])
def test_different_ci_keys_compute_concurrently(monkeypatch, cache_size, method):
    rng = np.random.default_rng(93)
    data = (
        rng.normal(size=(100, 4))
        if method == "fisherz"
        else rng.integers(0, 3, (100, 4))
    )
    cit = CIT(data, method=method, cache_size=cache_size)
    computing = threading.Barrier(2)

    def statistic(self, x, y, conditions):
        computing.wait(timeout=5)
        return 0.25

    monkeypatch.setattr(
        CIT, "fisher_z" if method == "fisherz" else "chi_square", statistic
    )
    with ThreadPoolExecutor(max_workers=2) as pool:
        tasks = [pool.submit(cit, 0, 1), pool.submit(cit, 2, 3)]
        assert [task.result(timeout=10) for task in tasks] == [0.25, 0.25]
    assert cit.cache_misses == 2
    assert cit.cache_hits == 0


def _observe_waiters(monkeypatch, count):
    """Let a test release a computation after its duplicate callers wait."""
    waiting = threading.Barrier(count + 1)

    class ObservedFuture(Future):
        def result(self, timeout=None):
            waiting.wait(timeout=5)
            return super().result(timeout=timeout)

    monkeypatch.setattr(independence_tests, "Future", ObservedFuture)
    return waiting


@pytest.mark.parametrize("value", [0.25, float("nan")])
def test_same_ci_key_computes_once_including_nan(monkeypatch, value):
    cit = CIT(np.random.default_rng(94).normal(size=(100, 4)))
    started, release = threading.Event(), threading.Event()
    waiting = _observe_waiters(monkeypatch, 3)
    calls = []

    def statistic(self, x, y, conditions):
        calls.append((x, y, conditions))
        started.set()
        assert release.wait(timeout=5)
        return value

    monkeypatch.setattr(CIT, "fisher_z", statistic)
    with ThreadPoolExecutor(max_workers=4) as pool:
        owner = pool.submit(cit, 0, 1, [3, 2])
        try:
            assert started.wait(timeout=5)
            duplicates = [pool.submit(cit, 1, 0, [2, 3]) for _ in range(3)]
            waiting.wait(timeout=5)
        finally:
            release.set()
        results = [task.result(timeout=5) for task in [owner, *duplicates]]
    assert len(calls) == 1
    assert cit.cache_misses == 1
    assert cit.cache_hits == 3
    if np.isnan(value):
        assert np.isnan(results).all()
        assert np.isnan(cit(0, 1, [2, 3]))
    else:
        assert results == [value] * 4
        assert cit(0, 1, [2, 3]) == value
    assert cit.cache_hits == 4


def test_failed_ci_releases_waiters_and_can_be_retried(monkeypatch):
    cit = CIT(np.random.default_rng(95).normal(size=(100, 4)))
    started, release = threading.Event(), threading.Event()
    waiting = _observe_waiters(monkeypatch, 2)
    calls = []

    def statistic(self, x, y, conditions):
        calls.append(None)
        if len(calls) == 1:
            started.set()
            assert release.wait(timeout=5)
            raise ValueError("statistic failed")
        return 0.5

    monkeypatch.setattr(CIT, "fisher_z", statistic)
    with ThreadPoolExecutor(max_workers=3) as pool:
        owner = pool.submit(cit, 0, 1)
        try:
            assert started.wait(timeout=5)
            duplicates = [pool.submit(cit, 1, 0) for _ in range(2)]
            waiting.wait(timeout=5)
        finally:
            release.set()
        for task in [owner, *duplicates]:
            with pytest.raises(ValueError, match="statistic failed"):
                task.result(timeout=5)
    assert len(calls) == cit.cache_misses == 1
    assert cit.cache_hits == 0
    assert not cit._cache
    assert cit(0, 1) == 0.5
    assert len(calls) == cit.cache_misses == 2


def test_ci_pickle_omits_pending_computations(monkeypatch):
    cit = CIT(np.random.default_rng(96).normal(size=(100, 4)))
    cit(2, 3)
    started, release = threading.Event(), threading.Event()

    def statistic(self, x, y, conditions):
        if self is cit:
            started.set()
            assert release.wait(timeout=5)
        return 0.25

    monkeypatch.setattr(CIT, "fisher_z", statistic)
    with ThreadPoolExecutor(max_workers=1) as pool:
        pending = pool.submit(cit, 0, 1)
        try:
            assert started.wait(timeout=5)
            restored = pickle.loads(pickle.dumps(cit))
            assert restored(0, 1) == 0.25
            assert restored(3, 2) == cit._cache[(2, 3, ())]
        finally:
            release.set()
        assert pending.result(timeout=5) == 0.25


def test_disabled_ci_cache_does_not_deduplicate_calls(monkeypatch):
    cit = CIT(np.random.default_rng(97).normal(size=(100, 4)), cache_size=0)
    computing = threading.Barrier(2)

    def statistic(self, x, y, conditions):
        computing.wait(timeout=5)
        return 0.25

    monkeypatch.setattr(CIT, "fisher_z", statistic)
    with ThreadPoolExecutor(max_workers=2) as pool:
        tasks = [pool.submit(cit, 0, 1), pool.submit(cit, 1, 0)]
        assert [task.result(timeout=10) for task in tasks] == [0.25, 0.25]
    assert cit.cache_misses == 2
    assert cit.cache_hits == 0
    assert not cit._cache
