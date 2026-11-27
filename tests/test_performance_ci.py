"""Differential contracts for CPU CI acceleration (no timing assertions)."""

import pickle
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from itertools import combinations

import numpy as np
import pytest

from whydra.algorithms.graph_core import CIT, CausalGraph, fisher_z_from_corr_batch
from whydra.algorithms.graph_core import independence_tests as ci
from whydra.algorithms.standalone import (
    parallel_pc_stable,
    skeleton,
    standalone_pc_stable,
)

from .conftest import run_scalar_queries


def data(seed=61, variables=7):
    values = np.random.default_rng(seed).normal(size=(250, variables))
    values[:, 1] += 0.5 * values[:, 0]
    values[:, 2] += 0.7 * values[:, 1]
    return values


def graph(values=None, test=None):
    values = data() if values is None else values
    result = CausalGraph(values.shape[1], [f"X{i}" for i in range(values.shape[1])])
    result.set_ind_test(CIT(values) if test is None else test)
    return result


@pytest.mark.parametrize("size", [0, 1, 2, 3])
def test_scalar_and_bounded_batch_agree_with_reference(size, monkeypatch):
    corr = np.corrcoef(data().T)
    # Preserve actual corrcoef diagonal and asymmetric rounding, rather than
    # relying on a manufactured exactly symmetric/unit-diagonal matrix.
    corr[0, 1] = np.nextafter(corr[0, 1], np.inf)
    queries = [
        (x, y, tuple(s))
        for x, y in combinations(range(7), 2)
        for s in combinations([v for v in range(7) if v not in (x, y)], size)
    ]
    expected = [ci._fisher_z_reference(corr, 250, x, y, s) for x, y, s in queries]
    scalar = [ci.fisher_z_from_corr(corr, 250, x, y, s) for x, y, s in queries]
    monkeypatch.setattr(ci, "_CI_BATCH_QUERIES", 11)
    monkeypatch.setattr(ci, "_CI_BATCH_BYTES", 4096)
    batched = fisher_z_from_corr_batch(corr, 250, iter(queries))
    np.testing.assert_allclose(scalar, expected, rtol=1e-12, atol=1e-14)
    np.testing.assert_array_equal(batched, scalar)


def test_loworder_admission_keeps_reference_errors_and_nan():
    singular = np.ones((3, 3))
    with pytest.raises(ValueError, match="singular or ill-conditioned"):
        ci.fisher_z_from_corr(singular, 100, 0, 1, ())
    with pytest.raises(ValueError, match="degrees of freedom"):
        ci.fisher_z_from_corr(np.eye(3), 4, 0, 1, (2,))
    for corr in (
        np.array([[1.0, np.nan], [np.nan, np.nan]]),
        np.array([[1.0, 0.9999999999999], [0.9999999999999, 1.0]]),
    ):
        try:
            expected = ci._fisher_z_reference(corr, 100, 0, 1, ())
        except ValueError as expected_error:
            with pytest.raises(ValueError, match=re.escape(str(expected_error))):
                ci.fisher_z_from_corr(corr, 100, 0, 1, ())
        else:
            assert np.isnan(expected)
            assert np.isnan(ci.fisher_z_from_corr(corr, 100, 0, 1, ()))


@pytest.mark.parametrize("boolean", [True, np.bool_(True)])
def test_scalar_boolean_condition_indices_keep_integer_indexing(boolean):
    values = np.random.default_rng(4).normal(size=(100, 5))
    test = CIT(values)
    expected = ci._fisher_z_reference(test.corr_matrix, 100, 2, 3, (boolean,))
    assert test(2, 3, (boolean,)) == pytest.approx(expected, abs=1e-14)
    assert test(2, 3, (boolean,)) == test(2, 3, (1,))


@pytest.mark.parametrize("cache_size", [0, 1, 2, 100])
def test_batch_replays_lru_duplicates_and_canonical_validation(cache_size):
    requests = [
        (0, 1, (3, 2, 3)),
        (1, 0, (2, 3)),
        (0, 2, ()),
        (0, 3, ()),
        (0, 1, (2, 3)),
        (0, 2, ()),
    ]
    scalar, batched = (
        CIT(data(), cache_size=cache_size),
        CIT(data(), cache_size=cache_size),
    )
    expected = [scalar(x, y, s) for x, y, s in requests]
    np.testing.assert_array_equal(batched.batch(iter(requests)), expected)
    assert list(batched._cache.items()) == list(scalar._cache.items())
    assert (batched.cache_hits, batched.cache_misses, batched.cache_evictions) == (
        scalar.cache_hits,
        scalar.cache_misses,
        scalar.cache_evictions,
    )
    assert not batched._pending


@pytest.mark.parametrize(
    "bad", [(0, 0, ()), (0, 1, (0,)), (0, 1, (99,)), (0, 1, (2, 3))]
)
def test_batch_invalid_request_commits_only_prefix(bad):
    scalar, batched = CIT(data(), max_cond_set_size=1), CIT(data(), max_cond_set_size=1)
    requests = [(0, 2, ()), bad, (0, 3, ())]
    with pytest.raises(ValueError, match="CI ") as expected:
        run_scalar_queries(scalar, requests)
    with pytest.raises(ValueError, match=re.escape(str(expected.value))) as actual:
        batched.batch(requests)
    assert str(actual.value) == str(expected.value)
    assert list(batched._cache.items()) == list(scalar._cache.items())
    assert batched.cache_misses == scalar.cache_misses
    assert not batched._pending


def test_batch_numerical_failure_keeps_prefix_and_releases_waiters():
    values = data()
    values[:, 3] = values[:, 2]
    scalar, batched = CIT(values), CIT(values)
    requests = [(0, 1, ()), (2, 3, ()), (0, 4, ())]
    with pytest.raises(ValueError, match="singular"):
        run_scalar_queries(scalar, requests)
    with pytest.raises(ValueError, match="singular"):
        batched.batch(requests)
    assert list(batched._cache.items()) == list(scalar._cache.items())
    assert batched.cache_misses == scalar.cache_misses == 2
    assert not batched._pending
    assert np.isfinite(batched(0, 4))


def test_concurrent_distinct_misses_do_not_hold_global_cache_lock(monkeypatch):
    test = CIT(data())
    barrier = threading.Barrier(2)
    original = test.fisher_z

    def coordinated(x, y, s):
        barrier.wait(timeout=5)
        return original(x, y, s)

    monkeypatch.setattr(test, "fisher_z", coordinated)
    with ThreadPoolExecutor(max_workers=2) as pool:
        a = pool.submit(test, 0, 1)
        b = pool.submit(test, 0, 2)
        assert np.isfinite(a.result(timeout=10))
        assert np.isfinite(b.result(timeout=10))
    assert test.cache_misses == 2
    assert not test._pending


def test_concurrent_failed_key_never_leaves_stale_reservation(monkeypatch):
    test = CIT(data())

    def fail(*args):
        raise ValueError("expected diagnostic")

    monkeypatch.setattr(test, "fisher_z", fail)
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(test, 0, 1) for _ in range(8)]
        for future in futures:
            with pytest.raises(ValueError, match="expected diagnostic"):
                future.result(timeout=5)
    assert not test._pending
    assert not test._cache
    assert test.cache_misses == 8


def test_concurrent_batches_share_keys_without_deadlock_or_double_misses():
    test = CIT(data())
    with ThreadPoolExecutor(max_workers=2) as pool:
        left = pool.submit(test.batch, [(0, 1, ()), (0, 2, ())])
        right = pool.submit(test.batch, [(1, 0, ()), (0, 3, ())])
        assert left.result(timeout=5)[0] == right.result(timeout=5)[0]
    assert (test.cache_misses, test.cache_hits) == (3, 1)
    assert not test._pending


def test_pickle_discards_inflight_state_and_restores_cache():
    test = CIT(data())
    test.batch([(0, 1, ()), (0, 2, (3,))])
    restored = pickle.loads(pickle.dumps(test))
    assert not restored._pending
    assert restored(0, 1) == test(0, 1)


@pytest.mark.parametrize("condition", [2.0, 2.5])
def test_batch_never_truncates_noninteger_condition_indices(condition):
    test = CIT(data())
    with pytest.raises(IndexError):
        test.batch([(0, 1, (condition,)), (0, 1, (3,))])
    assert not test._pending
    assert test.cache_misses == 1
    assert not test._cache


@pytest.mark.parametrize("override", ["subclass", "instance", "class", "helper"])
def test_custom_ci_variants_keep_scalar_dispatch(monkeypatch, override):
    calls = []

    def custom(self, x, y, s):
        calls.append((x, y, tuple(s)))
        return 0.2

    if override == "subclass":

        class Custom(CIT):
            fisher_z = custom

        test = Custom(data())
    else:
        test = CIT(data())
        if override == "instance":
            monkeypatch.setattr(test, "fisher_z", lambda x, y, s: custom(test, x, y, s))
        elif override == "class":
            monkeypatch.setattr(CIT, "fisher_z", custom)
        else:
            monkeypatch.setattr(
                ci, "fisher_z_from_corr", lambda corr, n, x, y, s: custom(test, x, y, s)
            )
    test._decision_alpha = 0.2
    assert test.batch([(0, 1, ()), (0, 2, (3,))]) == [0.2, 0.2]
    assert calls == [(0, 1, ()), (0, 2, (3,))]


def test_chisq_batch_uses_existing_scalar_statistics():
    values = np.random.default_rng(7).integers(0, 3, size=(100, 4))
    scalar, batched = CIT(values, "chisq"), CIT(values, "chisq")
    requests = [(0, 1, ()), (0, 2, (3,)), (1, 0, ())]
    assert batched.batch(requests) == [scalar(*r) for r in requests]
    assert (batched.cache_hits, batched.cache_misses) == (
        scalar.cache_hits,
        scalar.cache_misses,
    )


@pytest.mark.parametrize("size", [0, 1, 2])
@pytest.mark.parametrize("side", [-1, 0, 1])
def test_algorithm_boundary_uses_reference_decision(size, side):
    cg = graph()
    conditions = tuple(range(2, 2 + size))
    expected = ci._fisher_z_reference(
        cg.test.corr_matrix, cg.test.n_samples, 0, 1, conditions
    )
    alpha = (
        expected
        if side == 0
        else np.nextafter(expected, np.inf if side > 0 else -np.inf)
    )
    results = list(skeleton._ci_values(cg, 0, 1, [conditions], alpha, allow_batch=True))
    result = results[0][1]
    assert result == expected
    assert (result > alpha) == (expected > alpha)  # PC uses >
    assert (result >= alpha) == (expected >= alpha)  # FCI uses >=


def test_boundary_repair_keeps_first_cache_orientation_without_decision_context():
    values = np.random.default_rng(61).normal(size=(250, 5))
    test = CIT(values)
    alpha = ci._fisher_z_reference(test.corr_matrix, 250, 0, 4, (3,))
    results = [
        ci._threshold_pvalue(test, x, y, (3,), test(x, y, (3,)), alpha)
        for x, y in ((0, 4), (4, 0))
    ]
    assert results == [alpha, alpha]


@pytest.mark.parametrize("mode", ["closure", "batch", "prefilled_reversed"])
def test_decision_context_repairs_wrapped_cached_and_batched_queries(mode):
    values = np.random.default_rng(61).normal(size=(250, 5))
    test = CIT(values)
    if mode == "prefilled_reversed":
        test(4, 0, (3,))
        alpha = ci._fisher_z_reference(test.corr_matrix, 250, 4, 0, (3,))
    else:
        alpha = ci._fisher_z_reference(test.corr_matrix, 250, 0, 4, (3,))
    test._decision_alpha = alpha
    if mode == "batch":
        result = test.batch([(0, 4, (3,)), (1, 2, ()), (4, 0, (3,))])
        assert result[0] == result[2] == alpha
    else:

        def wrapped(x, y, s):
            return test(x, y, s)

        assert wrapped(0, 4, (3,)) == wrapped(4, 0, (3,)) == alpha
    assert test._cache[(0, 4, (3,))] == alpha


def test_boundary_reference_metadata_is_lru_bounded():
    test = CIT(data(), cache_size=1)
    test._decision_alpha = ci._fisher_z_reference(test.corr_matrix, 250, 1, 0, (3,))
    test(1, 0, (3,))
    assert len(test._decision_reference) == len(test._cache_origins) == 1
    test(0, 2, ())
    assert not test._decision_reference
    assert not test._cache_origins
    assert len(test._cache_generations) == len(test._cache) == 1


def test_reference_repair_cannot_overwrite_reinserted_reverse_owner(monkeypatch):
    values = np.random.default_rng(61).normal(size=(250, 5))
    test = CIT(values, cache_size=1)
    key = (0, 4, (3,))
    test(*key)
    original = ci._fisher_z_reference
    forward = original(test.corr_matrix, 250, 0, 4, (3,))
    reverse = original(test.corr_matrix, 250, 4, 0, (3,))
    entered, release = threading.Event(), threading.Event()

    def blocked(corr, n, x, y, conditions):
        if (x, y) == (0, 4):
            entered.set()
            assert release.wait(timeout=5)
        return original(corr, n, x, y, conditions)

    monkeypatch.setattr(ci, "_fisher_z_reference", blocked)
    with ThreadPoolExecutor(max_workers=1) as pool:
        old_repair = pool.submit(test._reference_value, 0, 4, (3,))
        try:
            assert entered.wait(timeout=5)
            test(0, 1)  # evicts the original forward-owned key
            replacement = test(4, 0, (3,))
            generation = test._cache_generations[key]
        finally:
            release.set()
        assert old_repair.result(timeout=5) == forward
    assert test._cache[key] == replacement
    assert test._cache_generations[key] == generation
    assert key not in test._decision_reference
    # The new entry must still be repaired against its own reverse owner.
    test._decision_alpha = reverse
    assert test(0, 4, (3,)) == reverse


def test_protected_builtin_pair_still_stops_before_other_separators():
    cg = graph(data(11, 12))
    search = skeleton.SkeletonDiscovery(
        cg, alpha=0.0, stable=True, protected_edges=[(0, 1)]
    )
    assert search._find_separating_union(0, 1, tuple(range(2, 12)), 1) == (2,)
    assert cg.test.cache_misses == 1
    assert len(cg.test._cache) == 1


def test_graph_conditioning_limit_is_not_bypassed_by_numeric_batch():
    cg = graph()
    cg.max_cond_set_size = 0
    with pytest.raises(ValueError, match="search must skip"):
        list(skeleton._ci_values(cg, 0, 1, [(2,)], 0.05, True))
    assert cg.test.cache_misses == 0


@pytest.mark.parametrize("family", ["pc", "fci"])
def test_builtin_batch_skeleton_matches_scalar_reference(monkeypatch, family):
    values = data(17, 6)
    accelerated = graph(values)
    scalar = graph(values)
    # An exact custom wrapper requests the original call sequence and disables
    # batching, providing a differential graph/sepset oracle on these data.
    scalar.test.fisher_z = lambda x, y, s: ci._fisher_z_reference(
        scalar.test.corr_matrix, scalar.test.n_samples, x, y, s
    )
    fast = skeleton.SkeletonDiscovery(accelerated, n_jobs=1)
    reference = skeleton.SkeletonDiscovery(scalar, n_jobs=1)
    if family == "fci":
        actual, expected = fast.run_fci(), reference.run_fci()
        assert actual["sepset"] == expected["sepset"]
        np.testing.assert_allclose(actual["pMax"], expected["pMax"], atol=1e-14)
    else:
        fast.run()
        reference.run()
        np.testing.assert_array_equal(accelerated.sepset, scalar.sepset)
    np.testing.assert_array_equal(accelerated.G.graph, scalar.G.graph)


def test_graph_ci_override_is_never_bypassed():
    cg = graph()
    calls = []
    cg.ci_test = lambda x, y, s: calls.append((x, y, tuple(s))) or 0.25
    expected = [(2,), (3,)]
    assert list(skeleton._ci_values(cg, 0, 1, expected, 0.05, True)) == [
        (s, 0.25) for s in expected
    ]
    assert calls == [(0, 1, s) for s in expected]
    assert cg.test.cache_misses == 0


@pytest.mark.parametrize("family", ["pc", "fci", "standalone_pc", "parallel_pc"])
def test_stable_marginals_batch_different_pairs_and_match_scalar_cache(
    monkeypatch, family
):
    batches = []
    numeric = ci._fisher_batch_results

    def capture(corr, n, requests):
        batches.append(list(requests))
        return numeric(corr, n, requests)

    monkeypatch.setattr(ci, "_fisher_batch_results", capture)
    values = data(32, 6)
    fast, reference = graph(values), graph(values)
    reference.test.fisher_z = lambda x, y, s: ci._fisher_z_reference(
        reference.test.corr_matrix, reference.test.n_samples, x, y, s
    )

    def run(cg):
        if family == "pc":
            return skeleton.SkeletonDiscovery(cg).run()
        if family == "fci":
            return skeleton.SkeletonDiscovery(cg).run_fci()
        if family == "standalone_pc":
            return standalone_pc_stable.skeleton_discovery(cg, 0.05, True, False, False)
        return parallel_pc_stable.skeleton_discovery(cg, 0.05, True, False, False, 1)

    run(fast)
    run(reference)
    assert any(len({(x, y) for x, y, s in batch if not s}) > 1 for batch in batches)
    np.testing.assert_array_equal(fast.G.graph, reference.G.graph)
    np.testing.assert_array_equal(fast.sepset, reference.sepset)
    assert (
        fast.test.cache_hits,
        fast.test.cache_misses,
        fast.test.cache_evictions,
    ) == (
        reference.test.cache_hits,
        reference.test.cache_misses,
        reference.test.cache_evictions,
    )


@pytest.mark.parametrize("family", ["pc", "fci", "parallel_pc"])
def test_small_builtin_passes_keep_requested_workers_without_starting_pool(
    monkeypatch, family
):
    def forbidden_pool(*args, **kwargs):
        pytest.fail("small built-in stable pass started a worker pool")

    monkeypatch.setattr(skeleton, "ThreadPoolExecutor", forbidden_pool)
    monkeypatch.setattr(skeleton, "Parallel", forbidden_pool)
    monkeypatch.setattr(parallel_pc_stable, "Parallel", forbidden_pool)
    cg = graph(data(18, 4))
    if family == "parallel_pc":
        parallel_pc_stable.skeleton_discovery(cg, 0.05, True, False, False, 4)
    else:
        search = skeleton.SkeletonDiscovery(cg, n_jobs=4)
        search.run_fci() if family == "fci" else search.run()
        assert search.n_jobs == 4
    assert cg.execution_stats["passes"]
    assert all(
        row["requested_workers"] == 4
        and row["effective_workers"] == 1
        and row["backend"] == "serial"
        for row in cg.execution_stats["passes"]
    )
