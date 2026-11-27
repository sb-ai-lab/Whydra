"""Closed forms retain reference decisions and public CI validation."""

from concurrent.futures import Future

import numpy as np
import pytest

from whydra.algorithms.graph_core import CIT, ci_batch
from whydra.algorithms.graph_core import independence_tests as ci
from whydra.algorithms.graph_core.cache import BoundedCache
from whydra.algorithms.standalone import ci_adapters


def _safe_corr():
    return np.array(
        [
            [1.0, 0.25, -0.3, 0.1],
            [0.25, 1.0, 0.2, -0.15],
            [-0.3, 0.2, 1.0, 0.05],
            [0.1, -0.15, 0.05, 1.0],
        ]
    )


@pytest.mark.parametrize("order", [0, 1])
def test_well_conditioned_low_order_uses_no_lapack_and_scalar_batch_are_exact(
    monkeypatch, order
):
    corr = _safe_corr()
    requests = [(0, 1, (2,) if order else ()), (2, 3, (0,) if order else ())] * 8
    reference = [
        ci._fisher_z_from_corr_reference(corr, 400, *request) for request in requests
    ]

    def unexpected(*_args, **_kwargs):
        pytest.fail("Well-conditioned low-order CI must not call LAPACK")

    monkeypatch.setattr(np.linalg, "eigvalsh", unexpected)
    monkeypatch.setattr(np.linalg, "inv", unexpected)
    scalar = [ci.fisher_z_from_corr(corr, 400, *request) for request in requests]
    batched = ci_batch.fisher_z_batch_from_corr(corr, 400, requests)
    np.testing.assert_array_equal(batched, scalar)
    np.testing.assert_allclose(batched, reference, rtol=5e-13, atol=1e-15)


@pytest.mark.parametrize("conditions", [(), (2,)])
def test_near_alpha_uses_exact_legacy_probability_for_strict_and_nonstrict_decisions(
    conditions,
):
    corr = _safe_corr()
    request = (0, 1, conditions)
    legacy = ci._fisher_z_from_corr_reference(corr, 400, *request)
    for alpha in (legacy, np.nextafter(legacy, 0.0), np.nextafter(legacy, 1.0)):
        scalar = ci.fisher_z_from_corr(corr, 400, *request, decision_alpha=alpha)
        batch = ci_batch.fisher_z_batch_from_corr(
            corr, 400, [request] * 16, decision_alpha=alpha
        )
        assert scalar == legacy
        assert batch == [legacy] * 16
        assert (scalar > alpha) == (legacy > alpha)
        assert (scalar >= alpha) == (legacy >= alpha)


@pytest.mark.parametrize("rho", [1.0 - 1e-13, 1.0 - 1e-10, -1.0 + 1e-10, 1.01])
def test_unsafe_low_order_preserves_reference_singular_boundary(rho):
    corr = np.array([[1.0, rho, 0.0], [rho, 1.0, 0.001], [0.0, 0.001, 1.0]])
    requests = [(0, 1, ()), (0, 1, (2,))]
    for request in requests:
        try:
            legacy = ci._fisher_z_from_corr_reference(corr, 100, *request)
        except ValueError as exc:
            expected_error = exc
        else:
            assert ci.fisher_z_from_corr(corr, 100, *request) == legacy
            assert (
                ci_batch.fisher_z_batch_from_corr(corr, 100, [request] * 2)
                == [legacy] * 2
            )
            continue
        with pytest.raises(ValueError, match="singular"):
            ci.fisher_z_from_corr(corr, 100, *request)
        values = ci_batch.fisher_z_batch_from_corr(
            corr, 100, [request] * 2, return_exceptions=True
        )
        assert all(
            type(value) is type(expected_error) and str(value) == str(expected_error)
            for value in values
        )


def test_nonfinite_and_bad_dof_still_use_reference_semantics():
    corr = _safe_corr()
    corr[0, 1] = np.nan
    for conditions in ((), (2,)):
        request = (0, 1, conditions)
        assert np.isnan(ci.fisher_z_from_corr(corr, 100, *request))
        assert np.isnan(
            ci_batch.fisher_z_batch_from_corr(corr, 100, [request] * 2)
        ).all()
        with pytest.raises(ValueError, match="degrees of freedom"):
            ci.fisher_z_from_corr(corr, 3 + len(conditions), *request)


def test_low_order_accepts_numpy_conditioning_array_containing_node_zero():
    corr = _safe_corr()
    expected = ci.fisher_z_from_corr(corr, 200, 1, 2, (0,))
    assert ci.fisher_z_from_corr(corr, 200, 1, 2, np.array([0])) == expected
    assert ci.fisher_z_from_corr(
        corr, 200, 1, 2, np.array([], dtype=int)
    ) == ci.fisher_z_from_corr(corr, 200, 1, 2, ())


@pytest.mark.parametrize("conditions_count", [0, 1])
def test_vector_low_order_matches_scalar_exactly_across_random_pairs(conditions_count):
    rng = np.random.default_rng(186)
    data = rng.normal(size=(300, 12))
    data[:, 1:] += 0.7 * data[:, :1]
    corr = np.corrcoef(data.T)
    requests = []
    for _ in range(200):
        x, y, z = map(int, rng.choice(12, 3, replace=False))
        requests.append((x, y, (z,) if conditions_count else ()))
    expected = [
        ci.fisher_z_from_corr(corr, len(data), *query, decision_alpha=0.05)
        for query in requests
    ]
    actual = ci_batch.fisher_z_batch_from_corr(
        corr, len(data), requests, decision_alpha=0.05
    )
    np.testing.assert_array_equal(actual, expected)


def test_configure_alpha_preserves_cache_evidence_and_counters():
    cit = CIT(np.random.default_rng(182).normal(size=(200, 4)))
    cit(0, 1)
    before = (cit.cache_hits, cit.cache_misses, cit.cache_evictions)
    ci_batch.configure_fisher_alpha(cit, 0.05)
    assert cit.decision_alpha == 0.05
    assert cit._cache
    assert (cit.cache_hits, cit.cache_misses, cit.cache_evictions) == before
    cit(0, 1)
    ci_batch.configure_fisher_alpha(cit, 0.05)
    assert cit._cache
    cit._inflight[(2, 3, ())] = Future()
    with pytest.raises(ValueError, match="pending"):
        ci_batch.configure_fisher_alpha(cit, 0.1)
    assert cit.decision_alpha == 0.05
    assert cit._cache


def test_configure_alpha_refines_cached_boundary_without_lru_or_counter_changes():
    cit = CIT(np.random.default_rng(185).normal(size=(200, 4)))
    cit(0, 1)
    cit(2, 3)
    reference = ci._fisher_z_from_corr_reference(
        cit.corr_matrix, cit.n_samples, 0, 1, ()
    )
    keys = list(cit._cache)
    counters = (cit.cache_hits, cit.cache_misses, cit.cache_evictions)
    ci_batch.configure_fisher_alpha(cit, reference)
    assert list(cit._cache) == keys
    assert cit._cache[(0, 1, ())] == reference
    assert (cit.cache_hits, cit.cache_misses, cit.cache_evictions) == counters


def test_counted_adapter_validates_once_and_keeps_per_request_errors(monkeypatch):
    cit = CIT(np.random.default_rng(183).normal(size=(200, 4)))
    stats = {"total": 0, "pdsep": 0, "ci_cache_hits": 0, "ci_cache_misses": 0}
    phase = {"name": "pdsep"}

    def counted(x, y, conditions):
        return cit(x, y, conditions)

    ci_adapters.attach_counted_batch(counted, cit, stats, phase, BoundedCache(10))
    normalized = []
    original = ci_adapters._normalize_request

    def once(test, request):
        normalized.append(request)
        return original(test, request)

    def unexpected(*_args):
        pytest.fail("Validated adapter requests must not be normalized again")

    monkeypatch.setattr(ci_adapters, "_normalize_request", once)
    monkeypatch.setattr(ci_batch, "_normalize_request", unexpected)
    queries = [(0, 1, (2, 2)), (1, 0, (2,)), (0, 0, ()), (0, 1, (2.5,))]
    values = counted.batch(queries, return_exceptions=True)
    assert len(normalized) == len(queries)
    assert values[0] == values[1]
    assert isinstance(values[2], ValueError)
    assert isinstance(values[3], IndexError)
    assert not cit._inflight
    assert stats["total"] == stats["pdsep"] == 4
    assert stats["ci_cache_hits"] == 1


def test_counted_adapter_keeps_public_batch_override(monkeypatch):
    cit = CIT(np.random.default_rng(184).normal(size=(100, 4)))
    stats = {"total": 0, "pdsep": 0, "ci_cache_hits": 0, "ci_cache_misses": 0}
    called = []
    original = cit.batch

    def observed(requests, **kwargs):
        called.append(1)
        return original(requests, **kwargs)

    monkeypatch.setattr(cit, "batch", observed)

    def counted(x, y, conditions):
        return cit(x, y, conditions)

    ci_adapters.attach_counted_batch(
        counted, cit, stats, {"name": "pdsep"}, BoundedCache(10)
    )
    assert np.isfinite(counted.batch([(0, 1, ())])[0])
    assert called == [1]


def test_auto_numerical_dispatch_never_chooses_threads(monkeypatch):
    from whydra.algorithms.graph_core import process_execution

    corr = np.eye(8)
    observed = []

    def unexpected(*_args, **_kwargs):
        pytest.fail("auto may use only processes or serial")

    def processes(function, arguments, *, n_jobs, plan, shared_arg_indices):
        assert plan.jobs == n_jobs
        assert shared_arg_indices == (0,)
        observed.append(n_jobs)
        return [function(*args) for args in arguments]

    monkeypatch.setattr(ci_batch, "thread_executor", unexpected)
    monkeypatch.setattr(process_execution, "process_map", processes)
    small = {}
    ci_batch.fisher_z_batch_from_corr(
        corr, 100, [(0, 1, ())] * 2048, n_jobs=4, execution_stats=small
    )
    assert small["actual_backend"] == "serial"
    large = {}
    ci_batch.fisher_z_batch_from_corr(
        corr, 100, [(0, 1, (2, 3, 4, 5, 6, 7))] * 2048, n_jobs=4, execution_stats=large
    )
    selected = large["process_policy"][-1]["selected_jobs"]
    assert large["actual_backend"] == ("processes" if selected > 1 else "serial")
    assert observed == ([selected] if selected > 1 else [])
