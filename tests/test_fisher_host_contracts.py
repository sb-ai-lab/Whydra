"""Fisher-Z host fallback and decision boundaries."""

from math import erfc, log, sqrt

import numpy as np
import pytest

from whydra.algorithms.graph_core.cpu_fisher_z import (
    NumpyFisherZ,
    _p_from_r,
    _p_from_r_vec,
    fisher_z_schur_chol,
    fisher_z_schur_chol_fused,
    fisher_z_threshold,
)
from whydra.algorithms.graph_core.independence_tests import fisher_z_from_corr


@pytest.fixture(params=["schur", "inv", "scalar", "fused", "canonical"])
def host_pvalue(request):
    def evaluate(corr, n_samples, conditions, eps=0.0):
        if request.param in ("schur", "inv", "canonical"):
            batcher = NumpyFisherZ(corr, n_samples, method=request.param, eps=eps)
            return float(
                batcher.batch([0], [1], np.asarray([conditions], dtype=int))[0]
            )
        function = (
            fisher_z_schur_chol
            if request.param == "scalar"
            else fisher_z_schur_chol_fused
        )
        return function(corr, n_samples, 0, 1, conditions, eps=eps)

    return evaluate


@pytest.mark.parametrize("conditions", [(), (2,), (2, 3)])
def test_host_fisher_rejects_nonpositive_dof(host_pvalue, conditions):
    with pytest.raises(ValueError, match="positive degrees of freedom"):
        host_pvalue(np.eye(4), 3 + len(conditions), conditions)


@pytest.mark.parametrize("rho", [1.0, 1.0 - 1e-13, 1.01])
def test_production_fallback_rejects_invalid_conditioning_blocks(rho):
    corr = np.eye(4)
    corr[2, 3] = corr[3, 2] = rho
    with pytest.raises(ValueError, match="singular or ill-conditioned"):
        NumpyFisherZ.for_cuda_fallback(corr, 100).batch([0], [1], [[2, 3]])


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_host_fisher_propagates_nonfinite_input(host_pvalue, bad):
    corr = np.eye(3)
    corr[0, 2] = corr[2, 0] = bad
    with np.errstate(divide="ignore", invalid="ignore"):
        assert np.isnan(host_pvalue(corr, 100, (2,)))


def test_host_fisher_preserves_tiny_tail_independence_decision(host_pvalue):
    corr = np.array([[1.0, 0.8], [0.8, 1.0]])
    expected = fisher_z_from_corr(corr, 100, 0, 1, ())
    actual = host_pvalue(corr, 100, ())
    # Computing 2 * (1 - cdf) rounds this small tail probability to zero.
    assert expected > 1e-30
    assert actual > 1e-30
    assert actual == pytest.approx(expected, rel=1e-12, abs=0.0)


@pytest.mark.parametrize("method", ["inv", "schur", "canonical"])
def test_nonfinite_batch_row_does_not_contaminate_valid_queries(method):
    corr = np.eye(5)
    corr[0, 1] = corr[1, 0] = np.nan
    corr[2, 3] = corr[3, 2] = 0.2
    result = NumpyFisherZ(corr, 100, method=method).batch([0, 2], [1, 3], [[4], [4]])
    assert np.isnan(result[0])
    assert result[1] == pytest.approx(fisher_z_from_corr(corr, 100, 2, 3, (4,)))


@pytest.mark.parametrize("function", [_p_from_r, _p_from_r_vec])
def test_shared_host_tail_rejects_bad_dof_and_nonfinite_correlations(function):
    with pytest.raises(ValueError, match="positive degrees of freedom"):
        function(0.2, 4, 1)
    for r in (np.nan, np.inf, -np.inf):
        assert np.isnan(function(r, 100, 0))
    expected = fisher_z_from_corr(np.array([[1.0, 0.8], [0.8, 1.0]]), 100, 0, 1, ())
    assert function(0.8, 100, 0) == pytest.approx(expected, rel=1e-12, abs=0.0)


@pytest.mark.parametrize("alpha", [0.05, 0.99, 1e-20, 1e-30, 1e-300, 5e-324])
def test_fused_host_threshold_is_finite_and_inverts_the_tail(alpha):
    threshold = fisher_z_threshold(alpha)
    assert np.isfinite(threshold)
    assert erfc(threshold / sqrt(2.0)) == pytest.approx(alpha, rel=1e-12, abs=0.0)


@pytest.mark.parametrize("alpha", [1e-20, 1e-30])
def test_fused_host_threshold_preserves_extreme_alpha_decisions(alpha):
    corr = np.array([[1.0, 0.8], [0.8, 1.0]])
    expected = fisher_z_from_corr(corr, 100, 0, 1, ()) > alpha
    statistic = sqrt(97) * abs(0.5 * log((1 + 0.8) / (1 - 0.8)))
    assert (statistic < fisher_z_threshold(alpha)) == expected


def test_experimental_schur_keeps_singular_nan_and_explicit_ridge():
    corr = np.array([[1.0, 0.5, 0.3], [0.5, 1.0, 0.4], [0.3, 0.4, 0.0]])
    with np.errstate(divide="ignore", invalid="ignore"):
        assert np.isnan(NumpyFisherZ(corr, 100).batch([0], [1], [[2]])).all()
        assert np.isnan(fisher_z_schur_chol(corr, 100, 0, 1, (2,)))
    with pytest.raises(ValueError, match="singular or ill-conditioned"):
        NumpyFisherZ.for_cuda_fallback(corr, 100).batch([0], [1], [[2]])
    result = NumpyFisherZ(corr, 100, eps=1e-4).batch([0], [1], [[2]])
    assert np.isfinite(result).all()
    np.testing.assert_array_equal(
        NumpyFisherZ.for_cuda_fallback(corr, 100, eps=1e-4).batch([0], [1], [[2]]),
        result,
    )
    assert np.isfinite(fisher_z_schur_chol(corr, 100, 0, 1, (2,), eps=1e-4))


def test_canonical_mode_cannot_silently_ignore_explicit_ridge():
    with pytest.raises(ValueError, match="does not support ridge"):
        NumpyFisherZ(np.eye(3), 100, method="canonical", eps=1e-4)


@pytest.mark.parametrize("rho", [0.01, 0.2, 0.8])
def test_production_fallback_matches_configured_cpu_boundary_decisions(rho):
    corr = np.array([[1.0, rho], [rho, 1.0]])
    alpha = fisher_z_from_corr(corr, 100, 0, 1, ())
    batcher = NumpyFisherZ.for_cuda_fallback(corr, 100)
    batcher.decision_alpha = alpha
    expected = fisher_z_from_corr(corr, 100, 0, 1, (), decision_alpha=alpha)
    actual = batcher.batch([0], [1], np.empty((1, 0), dtype=int))[0]
    assert actual == expected
    assert (actual > alpha) == (expected > alpha)
    assert (actual >= alpha) == (expected >= alpha)
