from math import erf, sqrt
from types import SimpleNamespace

import numpy as np
import pytest

from whydra.scmgen import sample_scm, scenarios, truth
from whydra.scmgen.betafn import betainc, betaincinv
from whydra.scmgen.outcomes import BETA_EDGE, beta_mean_preserving_noise, beta_quantile


@pytest.mark.parametrize(
    ("a", "b", "x", "cdf"),
    [
        (0.125, 5.75, 1e-40, 1.308604841276559071268401722819967e-5),
        (4.25, 8.5, 1e-4, 5.164512460225951224301967690544536e-15),
        (120.125, 250.5, 0.2, 1.085590842903986237556284076491464e-8),
        (0.0078125, 0.0625, 0.8, 0.89813166986891441690105238989337116),
        (65.25, 2.5, 0.985, 0.84985421345523567291836865981474651),
    ],
)
def test_beta_matches_high_precision_reference(a, b, x, cdf):
    """References rounded from mpmath.betainc(a, b, 0, x, regularized=True), 100 digits."""
    np.testing.assert_allclose(betainc(a, b, x), cdf, rtol=2e-13, atol=0)
    np.testing.assert_allclose(betaincinv(a, b, cdf), x, rtol=2e-13, atol=0)


@pytest.mark.parametrize("fn", [betainc, betaincinv])
def test_beta_broadcast_and_float64(fn):
    a = np.array([[1], [2], [4]], dtype=np.int32)
    u = np.array([0, 0.25, 0.5, 1], dtype=np.float32)
    expected = u.astype(np.float64) ** (a if fn is betainc else 1.0 / a)
    got = fn(a, 1, u)
    assert got.dtype == np.float64
    np.testing.assert_allclose(got, expected, rtol=1e-14)
    scalar = fn(1, 1, np.float32(0.75))
    assert isinstance(scalar, np.ndarray)
    assert scalar.shape == ()
    assert scalar.dtype == np.float64
    assert scalar == 0.75
    assert fn(1, 1, np.empty((0, 3))).shape == (0, 3)


def test_cdf_accepts_a_precomputed_log_beta():
    a = np.array([[1.0], [2.0], [5.0]])
    x = [0.125, 0.5, 0.875]
    np.testing.assert_array_equal(betainc(a, 1, x, -np.log(a)), betainc(a, 1, x))
    with pytest.raises(ValueError, match=r"shape|broadcast"):
        betainc(a, 1, x, np.zeros(2))


@pytest.mark.parametrize(
    ("fn", "a", "b", "expected"),
    [
        (betainc, 1, 1, [0, 0, 0.5, 1, 1, 0]),
        (betaincinv, 1, 1, [0, 0, 0.5, 1, 1, 0]),
        (betainc, np.nan, 2, [0, 0, np.nan, 1, 1, 0]),
        (betaincinv, -1, 2, [0, 0, np.nan, 1, 1, 0]),
        (betainc, 0, 2, [0, 1, 1, 1, 1, 0]),
        (betainc, 2, 0, [0, 0, 0, 1, 1, 0]),
        (betaincinv, 0, 2, [0, 0, 0, 0, 1, 0]),
        (betaincinv, 2, 0, [0, 1, 1, 1, 1, 0]),
        (betainc, 0, 0, [0, np.nan, np.nan, np.nan, 1, 0]),
        (betaincinv, 0, 0, [0, np.nan, np.nan, np.nan, 1, 0]),
    ],
)
def test_beta_boundaries(fn, a, b, expected):
    np.testing.assert_array_equal(fn(a, b, [-1, 0, 0.5, 1, 2, np.nan]), expected)


def test_saturated_beta_keeps_mixture_masses_and_continuous_clip():
    u = [0, 0.25, 0.5, 0.75, 1]
    m = np.array([[0], [1]])
    expected = [
        [0, BETA_EDGE, BETA_EDGE, 1, 1],
        [0, 1 - BETA_EDGE, 1 - BETA_EDGE, 1, 1],
    ]
    np.testing.assert_array_equal(beta_quantile(u, 0.25, 0.25, m, 10), expected)


def test_truth_integral_for_saturated_beta_predictors():
    dataset = SimpleNamespace(config=scenarios.load("rct-beta-clean"), latent={})
    mu = np.array([-1000.0, 1000.0])
    weights = np.ones((len(mu), truth.NU_TABLE_N))
    with np.errstate(over="ignore"):
        (effect,) = truth._nu_ite_expectation(
            dataset, [], [lambda sl: weights[sl]], mu, -2 * mu, 0.0, 0.0
        )
    normal_mass_on_grid = erf(truth._NU_GRID[-1] / sqrt(2))
    np.testing.assert_allclose(
        effect, [normal_mass_on_grid, -normal_mass_on_grid], rtol=0, atol=1e-13
    )


def test_beta_measurement_noise_uniform_law():
    u = np.array([0, 0.25, 0.5, 0.75, 1])
    np.testing.assert_array_equal(
        beta_mean_preserving_noise(0.5, 2, u),
        [BETA_EDGE, 0.25, 0.5, 0.75, 1 - BETA_EDGE],
    )


@pytest.mark.parametrize(
    "name", ["rct-beta-clean", "rct-beta-zoi-both", "rct-beta-indep", "rct-noisey-beta"]
)
def test_beta_scm_sample_bitwise_prefix(name):
    model = sample_scm(scenarios.load(name), model_seed=28, n_calib=512)
    small = model.sample(55, sample_seed=172)
    large = model.sample(151, sample_seed=172)
    assert (
        small.to_frame().to_numpy().tobytes()
        == large.to_frame().iloc[:55].to_numpy().tobytes()
    )
    for field in ("y0", "y1", "mu0", "mu1", "tau", "tau_cate"):
        assert (
            getattr(small, field).tobytes() == getattr(large, field)[:55].tobytes()
        ), field
