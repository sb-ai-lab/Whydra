import threading
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack

import numpy as np
import pytest

from whydra.algorithms.graph_core import CIT, ci_batch, fisher_z_from_corr


def _queries():
    rng = np.random.default_rng(419)
    result = []
    for order in range(7):
        for _ in range(40):
            variables = rng.choice(16, order + 2, replace=False)
            result.append(
                (int(variables[0]), int(variables[1]), tuple(map(int, variables[2:])))
            )
    return result


@pytest.mark.parametrize("jobs", [1, 4])
def test_native_fisher_batch_matches_scalar_values_and_order(jobs):
    data = np.random.default_rng(418).normal(size=(400, 16))
    corr = np.corrcoef(data.T)
    queries = _queries()
    expected = [fisher_z_from_corr(corr, len(data), *query) for query in queries]
    actual = ci_batch.fisher_z_batch_from_corr(
        corr, len(data), queries, n_jobs=jobs, batch_size=16
    )
    np.testing.assert_array_equal(actual, expected)


def test_batch_ci_deduplicates_symmetric_requests_and_respects_disabled_cache():
    data = np.random.default_rng(420).normal(size=(200, 6))
    requests = [(0, 1, (3, 2)), (1, 0, (2, 3)), (2, 4, ()), (0, 1, (2, 3))]
    for cache_size, misses, hits in [(10, 2, 2), (0, 4, 0)]:
        cit = CIT(data, cache_size=cache_size)
        actual = cit.batch(requests)
        expected = [CIT(data)(*query) for query in requests]
        np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-15)
        assert (cit.cache_misses, cit.cache_hits) == (misses, hits)
        assert not cit._inflight
    bounded = CIT(data, cache_size=1)
    bounded.batch(requests, n_jobs=1)
    assert len(bounded._cache) == 1
    assert bounded.cache_evictions == 1


def test_batch_keeps_individual_nan_singularity_dof_and_input_errors():
    data = np.random.default_rng(421).normal(size=(12, 12))
    cit = CIT(data)
    cit.corr_matrix[4, :] = cit.corr_matrix[:, 4] = np.nan
    cit.corr_matrix[0, 1] = cit.corr_matrix[1, 0] = 1.0
    requests = [
        (2, 3, ()),
        (0, 1, ()),
        (2, 4, ()),
        (0, 2, tuple(range(3, 12))),
        (0, 0, ()),
        (0, 99, ()),
    ]
    results = cit.batch(requests, return_exceptions=True)
    assert np.isfinite(results[0])
    assert isinstance(results[1], ValueError)
    assert "singular" in str(results[1])
    assert np.isnan(results[2])
    assert isinstance(results[3], ValueError)
    assert "degrees of freedom" in str(results[3])
    assert isinstance(results[4], ValueError)
    assert "disjoint" in str(results[4])
    assert isinstance(results[5], ValueError)
    assert "bounds" in str(results[5])
    assert not cit._inflight
    with pytest.raises(ValueError, match="singular"):
        cit.batch(requests)


def test_batch_respects_conditioning_limit_without_computing_forbidden_queries():
    cit = CIT(np.random.default_rng(422).normal(size=(100, 5)), max_cond_set_size=1)
    with pytest.raises(ValueError, match="max_cond_set_size"):
        cit.batch([(0, 1, (2, 3))])
    assert cit.cache_misses == 0
    assert not cit._inflight


@pytest.mark.parametrize("override", ["__call__", "fisher_z"])
def test_batch_honors_patched_custom_tests(monkeypatch, override):
    cit = CIT(np.random.default_rng(423).normal(size=(100, 4)))
    monkeypatch.setattr(CIT, override, lambda self, x, y, conditions=None: 0.75)
    assert not ci_batch.supports_fisher_batch(cit)
    assert ci_batch.evaluate_ci_batch(cit, [(0, 1, ()), (2, 3, ())], n_jobs=4) == [
        0.75,
        0.75,
    ]


def test_batch_and_scalar_share_one_inflight_computation(monkeypatch):
    cit = CIT(np.random.default_rng(424).normal(size=(100, 4)))
    original = ci_batch._fisher_chunk
    started, release = threading.Event(), threading.Event()

    def paused(*args):
        started.set()
        assert release.wait(5)
        return original(*args)

    monkeypatch.setattr(ci_batch, "_fisher_chunk", paused)
    with ThreadPoolExecutor(max_workers=2) as pool:
        batch = pool.submit(cit.batch, [(0, 1, ())], n_jobs=1)
        try:
            assert started.wait(5)
            scalar = pool.submit(cit, 1, 0)
        finally:
            release.set()
        assert scalar.result(5) == batch.result(5)[0]
    assert (cit.cache_misses, cit.cache_hits) == (1, 1)
    assert not cit._inflight


@pytest.mark.parametrize("condition", [2.0, 2.5])
def test_batch_does_not_silently_truncate_condition_indices(condition):
    data = np.random.default_rng(425).normal(size=(100, 4))
    with pytest.raises(IndexError):
        CIT(data)(0, 1, [condition])
    with pytest.raises(IndexError):
        CIT(data).batch([(0, 1, [condition])])


def test_batch_honors_replaced_free_fisher_kernel(monkeypatch):
    from whydra.algorithms.graph_core import independence_tests

    cit = CIT(np.random.default_rng(426).normal(size=(100, 4)))
    monkeypatch.setattr(independence_tests, "fisher_z_from_corr", lambda *args: 0.6)
    assert not ci_batch.supports_fisher_batch(cit)
    assert cit.batch([(0, 1, ())]) == [0.6]


def test_auto_small_matrix_batches_avoid_process_startup(monkeypatch):
    from whydra.algorithms.graph_core import process_execution

    def unexpected_process_pool(*args, **kwargs):
        raise AssertionError(
            "Small-matrix batches must amortize process dispatch first"
        )

    monkeypatch.setattr(process_execution, "process_map", unexpected_process_pool)
    corr = np.corrcoef(np.random.default_rng(427).normal(size=(100, 3)).T)
    values = ci_batch.fisher_z_batch_from_corr(corr, 100, [(0, 1, ())] * 4096, n_jobs=4)
    assert len(values) == 4096
    assert all(value == values[0] for value in values)


def test_explicit_threads_dispatch_moderate_batches_and_reuse_pool(monkeypatch):
    cit = CIT(np.random.default_rng(428).normal(size=(200, 8)), cache_size=0)
    queries = [(0, 1, (2, 3, 4, 5))] * 2048
    submissions = []
    with ci_batch.thread_executor(4) as pool:
        submit = pool.submit

        def observed_submit(*args, **kwargs):
            submissions.append(1)
            return submit(*args, **kwargs)

        monkeypatch.setattr(pool, "submit", observed_submit)
        for _ in range(2):
            values = cit.batch(queries, n_jobs=4, backend="threads")
            expected = fisher_z_from_corr(cit.corr_matrix, cit.n_samples, *queries[0])
            np.testing.assert_array_equal(values, [expected] * len(queries))
            with ci_batch.thread_executor(4) as reused:
                assert reused is pool
    stats = ci_batch.batch_execution_stats(cit)
    assert len(submissions) == stats["tasks"]["threads"] == 16
    assert stats["batches"] == {"serial": 0, "threads": 2, "processes": 0}
    assert stats["queries"] == stats["queries_by_backend"]["threads"] == 4096
    assert stats["actual_backend"] == "threads"
    assert stats["requested_backends"] == {"threads": 2}
    assert stats["fallback_reasons"] == {}


def test_thread_pool_eviction_waits_for_concurrent_active_leases(monkeypatch):
    # Pause between acquiring the pool and submitting work. Another caller
    # can trigger eviction here while this executor still has an active lease.
    registry = OrderedDict()
    monkeypatch.setattr(ci_batch, "_THREAD_EXECUTORS", registry)
    acquired, submit = threading.Event(), threading.Event()

    def paused_batch():
        with ci_batch.thread_executor(2) as executor:
            acquired.set()
            assert submit.wait(5)
            return list(executor.map(lambda value: value + 1, (1, 2)))

    try:
        with ThreadPoolExecutor(max_workers=1) as callers:
            result = callers.submit(paused_batch)
            try:
                assert acquired.wait(5)
                with ExitStack() as leases:
                    pools = [
                        leases.enter_context(ci_batch.thread_executor(workers))
                        for workers in (3, 4, 5, 6)
                    ]
                    assert len(registry) == 5
                    assert all(not pool._shutdown for pool in pools)
                assert len(registry) <= 4
            finally:
                submit.set()
            assert result.result(5) == [2, 3]
        assert len(registry) <= 4
        assert all(entry["users"] == 0 for entry in registry.values())
    finally:
        for entry in registry.values():
            entry["executor"].shutdown(wait=True)


def test_dispatch_stats_report_serial_fallback_and_cached_batches():
    cit = CIT(np.random.default_rng(429).normal(size=(100, 4)))
    requests = [(0, 1, ()), (2, 3, ())]
    cit.batch(requests, n_jobs=4, backend="threads")
    cit.batch(requests, n_jobs=4, backend="threads")
    stats = ci_batch.batch_execution_stats(cit)
    assert stats["actual_backend"] == "serial"
    assert stats["batches"] == {"serial": 1, "threads": 0, "processes": 0}
    assert stats["queries"] == stats["queries_by_backend"]["serial"] == 2
    assert stats["fallback_reasons"] == {"single_task": 1, "no_misses": 1}
    assert stats["requested_backends"] == {"threads": 2}
    assert stats["build_seconds"] >= 0
    assert stats["compute_seconds"] >= 0
    stats["batches"]["serial"] = 99
    assert ci_batch.batch_execution_stats(cit)["batches"]["serial"] == 1


def test_process_dispatch_uses_common_reusable_executor(monkeypatch):
    from whydra.algorithms.graph_core import process_execution

    observed = []

    def observed_process_map(function, arguments, *, n_jobs, plan, shared_arg_indices):
        assert plan.jobs == n_jobs
        assert shared_arg_indices == (0,)
        observed.append((function, n_jobs))
        return [function(*args) for args in arguments]

    monkeypatch.setattr(process_execution, "process_map", observed_process_map)
    corr = np.corrcoef(np.random.default_rng(430).normal(size=(100, 4)).T)
    stats = {}
    result = ci_batch.fisher_z_batch_from_corr(
        corr,
        100,
        [(0, 1, ())] * 2048,
        n_jobs=4,
        backend="processes",
        execution_stats=stats,
    )
    assert len(result) == 2048
    assert observed == [(ci_batch._fisher_chunk, 4)]
    assert stats["actual_backend"] == "processes"
    assert stats["queries_by_backend"] == {"serial": 0, "threads": 0, "processes": 2048}
