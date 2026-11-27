"""Derive ground truth from the latent state of a generated dataset.

Dataset.latent describes the data-generating process (DGP) for each row. It contains the
structural terms mu0_x and tau_x, mediator coefficients and noise, collider noise, the
assignment draw u_t, the SCM assignment propensity e_assign, and each axis's additive
contribution to the assignment logit, e_logit_extra. It also stores outcome-family noise
(eps or u, and ln), the hidden modifier (tau_hidden, var_hidden), the cluster intercept
mu0_cluster, noncompliance parameters nc, and the retention probabilities sel_q1 and
sel_q0. DGP axes update this state and call refresh_truth(ds) to recompute all dependent
fields. They must not write tau, y0, y1, tau_itt, or e directly.

Conditioning sets: W contains the true pretreatment covariates except colliders,
plus cluster when present and period under effect_drift. The hidden modifier is u_tau.
For Gaussian mediators at fixed W, weighted mediator noise contributes N(0, ‖med_w‖²)
to the outcome. Binary mediators use shared uniform draws. The variable ν is
standardized outcome noise, and compliance identifies the unit's
compliance type. The fields tau and tau_itt condition on (W, u_tau, eps_med). The fields
tau_cate, cvte, and tau_itt_cate condition only on W: integrate over u_tau and eps_med
using 64-point Gauss–Hermite quadrature for one active component or a 32 × 32 grid for
two. Fields ending in _selected refer to the selected population and weight ν by
selection probability.

All families use ν ~ N(0, 1). Gaussian and lognormal outcomes use eps/sd_eps for normal
noise. For heavy tails, the t₃ CDF gives a probability that Φ⁻¹ maps to ν. Binary,
zero-inflated lognormal (ZILN), and beta outcomes use Φ⁻¹(u). The noncompliance score uses ν as its
hidden outcome component.

Version 0.8.0 added the beta family and outcome.coupling. With independent coupling,
Y(1) has its own noise (latent eps1 or u1, and ln_z1), and ν is the noise for Y(0). For
beta outcomes or independent coupling, _nu_ite_expectation computes the fields
integrated over ν (tau_itt, tau_itt_cate, and *_selected). The existing shared-noise
paths for Gaussian, binary, and ZILN outcomes are unchanged. Dataset.vte_w evaluates
vte_w(ds) lazily to obtain Var[Y(1) − Y(0) | W].

Version 0.9.0 added the natural Gaussian outcome scale s(x) = exp(scale_sigma · z(x_j)).
At each refresh, y_scale(ds) recomputes it from the true X and calibration constants and
stores it in latent["y_scale"]. Both outcomes._outcomes_gaussian and the Gaussian
branches here multiply outcome-scale fields by s and variances cvte and vte_w by s².

Version 0.11.0 added lognormal outcomes. The fields tau_cate, cvte, and
vte_w have closed forms (outcomes.lognormal_cate and lognormal_ite_second_moment). Other
fields integrate over ν by Gauss–Hermite quadrature. Outcome lags y_pre1 through y_preK
are observed columns in W. Integration over ν uses its posterior given the lags
(pre_nu_m, pre_nu_v). The oracle tau_cate_x excludes the lags. Under block assignment
(cluster.assign = block), latent["e_block"] is the treated fraction of the block and
equals P(Z = 1 | W).
"""

import functools
import math

import numpy as np

from .betafn import betainc
from .config import CATE_MODIFYING, PathologyConfig
from .graph import ROLE_COLLIDER
from .graph import standardize as _standardize
from .mechanisms import (
    _rowsum,
    mediator_contributions,
    mediator_family,
    mediator_potentials,
    mediator_probs,
)
from .outcomes import (
    arm_moments,
    beta_arm,
    beta_mean,
    beta_quantile,
    family_outcomes,
    family_probs,
    lognormal_cate,
    lognormal_ite_second_moment,
    lognormal_log_level,
    nu_posterior,
    outcome_scale,
    tau_scale_mean,
    ziln_second_moment,
)

# Fixed quadrature nodes. The node counts are implementation details. Tests check accuracy.
_GH_NODES, _GH_WEIGHTS = np.polynomial.hermite.hermgauss(64)  # ∫ f(x) e^{-x²} dx
_GH2_NODES, _GH2_WEIGHTS = np.polynomial.hermite.hermgauss(
    32
)  # Tensor grid of 32 × 32 nodes over (eps_med, u_tau).
_GL_NODES, _GL_WEIGHTS = np.polynomial.legendre.leggauss(32)  # ∫_{-1}^{1} f(x) dx
GH_N, GH2_N, GL_N = _GH_NODES.size, _GH2_NODES.size, _GL_NODES.size


# Integrate beta quantities on (0, 1) with 16-point Gauss–Legendre quadrature. The change of
# variable s -> s³(10 − 15s + 6s²) smooths power-law singularities of the beta quantile or CDF at
# the endpoints. Tests check a relative error of about 1e-6.
def _unit_rule(k: int) -> tuple[np.ndarray, np.ndarray]:
    x, w = np.polynomial.legendre.leggauss(k)
    s = 0.5 * (x + 1.0)
    return s**3 * (10.0 - 15.0 * s + 6.0 * s * s), 0.5 * w * 30.0 * s * s * (
        1.0 - s
    ) ** 2


_UNIT_NODES, _UNIT_WEIGHTS = _unit_rule(16)
UNIT_N = _UNIT_NODES.size
_INV_SQRT_PI = 1.0 / np.sqrt(np.pi)


def _sigmoid(v):
    return 1.0 / (1.0 + np.exp(-v))


def _logit(p):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return np.log(p / (1.0 - p))


_A = (
    3.3871328727963666080e0,
    1.3314166789178437745e2,
    1.9715909503065514427e3,
    1.3731693765509461125e4,
    4.5921953931549871457e4,
    6.7265770927008700853e4,
    3.3430575583588128105e4,
    2.5090809287301226727e3,
)
_B = (
    1.0,
    4.2313330701600911252e1,
    6.8718700749205790830e2,
    5.3941960214247511077e3,
    2.1213794301586595867e4,
    3.9307895800092710610e4,
    2.8729085735721942674e4,
    5.2264952788528545610e3,
)
_C = (
    1.42343711074968357734e0,
    4.63033784615654529590e0,
    5.76949722146069140550e0,
    3.64784832476320460504e0,
    1.27045825245236838258e0,
    2.41780725177450611770e-1,
    2.27238449892691845833e-2,
    7.74545014278341407640e-4,
)
_D = (
    1.0,
    2.05319162663775882187e0,
    1.67638483018380384940e0,
    6.89767334985100004550e-1,
    1.48103976427480074590e-1,
    1.51986665636164571966e-2,
    5.47593808499534494600e-4,
    1.05075007164441684324e-9,
)
_E = (
    6.65790464350110377720e0,
    5.46378491116411436990e0,
    1.78482653991729133580e0,
    2.96560571828504891230e-1,
    2.65321895265761230930e-2,
    1.24266094738807843860e-3,
    2.71155556874348757815e-5,
    2.01033439929228813265e-7,
)
_F = (
    1.0,
    5.99832206555887937690e-1,
    1.36929880922735805310e-1,
    1.48753612908506148525e-2,
    7.86869131145613259100e-4,
    1.84631831751005468180e-5,
    1.42151175831644588870e-7,
    2.04426310338993978564e-15,
)


def _poly(coef, r):
    out = np.zeros_like(r)
    for c in reversed(coef):
        out = out * r + c
    return out


def ndtri(p) -> np.ndarray:
    """Return standard normal quantiles, including infinities at 0 and 1 and NaN outside that interval."""
    p = np.asarray(p, dtype=float)
    q = p - 0.5
    out = np.full(p.shape, np.nan)
    central = np.abs(q) <= 0.425
    r = 0.180625 - q[central] ** 2
    out[central] = q[central] * _poly(_A, r) / _poly(_B, r)
    tail = ~central & (p > 0) & (p < 1)
    r = np.sqrt(-np.log(np.where(q[tail] < 0, p[tail], 1.0 - p[tail])))
    near = r <= 5.0
    val = np.empty(r.shape)
    val[near] = _poly(_C, r[near] - 1.6) / _poly(_D, r[near] - 1.6)
    val[~near] = _poly(_E, r[~near] - 5.0) / _poly(_F, r[~near] - 5.0)
    out[tail] = np.where(q[tail] < 0, -val, val)
    out[p == 0] = -np.inf
    out[p == 1] = np.inf
    return out


def t3_cdf(t) -> np.ndarray:
    """Return the closed-form CDF of Student's t distribution with three degrees of freedom."""
    s = np.asarray(t, dtype=float) / np.sqrt(3.0)
    return 0.5 + (np.arctan(s) + s / (1.0 + s * s)) / np.pi


def outcome_noise_normal(ds) -> np.ndarray:
    """Return each row's standardized outcome noise ν ~ N(0, 1) for any outcome family."""
    lat, cfg = ds.latent, ds.config.outcome
    if "nu" in lat:
        return lat["nu"]
    if cfg.family in ("gaussian", "lognormal"):
        eps = lat["eps"]
        if cfg.heavy_tails:
            nu = ndtri(t3_cdf(eps * np.sqrt(3.0) / lat["sd_eps"]))
        else:
            nu = eps / lat["sd_eps"]
    else:
        nu = ndtri(np.clip(lat["u"], 1e-300, 1 - 1e-16))
    lat["nu"] = nu
    return nu


def _phi_cdf(v: float) -> float:
    return 0.5 * math.erfc(-v / math.sqrt(2.0))


def _t3_ppf(p: float) -> float:
    """Find a scalar t₃ quantile by bisecting the closed-form CDF."""
    lo, hi = -1.0, 1.0
    while float(t3_cdf(lo)) > p:
        lo *= 2.0
    while float(t3_cdf(hi)) < p:
        hi *= 2.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if float(t3_cdf(mid)) < p:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


@functools.cache
def _gh_eps_nodes(heavy_tails: bool) -> tuple:
    """Return ε(ν)/sd_eps at the Gauss–Hermite nodes ν = √2·x.

    This is ν for normal noise and T₃⁻¹(Φ(ν))/√3 for heavy tails.
    """
    if not heavy_tails:
        return tuple(np.sqrt(2.0) * x for x in _GH_NODES)
    return tuple(
        _t3_ppf(_phi_cdf(float(np.sqrt(2.0) * x))) / np.sqrt(3.0) for x in _GH_NODES
    )


def _expect_g_eps(g, sd_eps: float, heavy_tails: bool) -> np.ndarray:
    """Compute the expected product of a noise function and the outcome noise.

    Use 64-point Gauss–Hermite quadrature for E_ν[g(ν)·ε(ν)]. For Gaussian outcomes ε =
    sd_eps·ν with normal noise, or sd_eps/√3 · T₃⁻¹(Φ(ν)) with heavy tails.
    """
    acc = None
    for x, w, eps in zip(
        _GH_NODES, _GH_WEIGHTS, _gh_eps_nodes(bool(heavy_tails)), strict=False
    ):
        v = w * eps * g(np.sqrt(2.0) * x)
        acc = v if acc is None else acc + v
    return acc * (_INV_SQRT_PI * sd_eps)


# Quadrature rules


def expect_normal(fn, sigma) -> np.ndarray:
    """Compute E[fn(U)] for U ~ N(0, sigma²) with 64-point Gauss–Hermite quadrature.

    sigma can be a scalar or an array with one entry per row. fn receives a scalar or an
    array of the same shape and returns values for each row.
    """
    sigma = np.asarray(sigma, dtype=float)
    acc = None
    for x, w in zip(_GH_NODES, _GH_WEIGHTS, strict=False):
        v = fn(np.sqrt(2.0) * sigma * x)
        acc = w * v if acc is None else acc + w * v
    return acc * _INV_SQRT_PI


def expect_normal_moments(fn, sigma) -> tuple[np.ndarray, np.ndarray]:
    """Compute E[fn(U)] and Var[fn(U)] for U ~ N(0, sigma²) in one pass over the same nodes."""
    sigma = np.asarray(sigma, dtype=float)
    m1 = m2 = None
    for x, w in zip(_GH_NODES, _GH_WEIGHTS, strict=False):
        v = fn(np.sqrt(2.0) * sigma * x)
        m1 = w * v if m1 is None else m1 + w * v
        m2 = w * v * v if m2 is None else m2 + w * v * v
    mean = m1 * _INV_SQRT_PI
    var = np.maximum(m2 * _INV_SQRT_PI - mean * mean, 0.0)
    return mean, var


def expect_normal2_moments(fn, sigma1, sigma2) -> tuple[np.ndarray, np.ndarray]:
    """Compute the mean and variance of fn(A, B) for independent Gaussian inputs.

    A ~ N(0, sigma1²) and B ~ N(0, sigma2²). Each scale is a scalar or an array of
    row-specific values. Use a 32 × 32 Gauss–Hermite grid when both scales are active
    and 64 nodes when only one is active. When both scales are zero, return fn(0, 0) and
    zero variance.
    """
    s1, s2 = np.asarray(sigma1, dtype=float), np.asarray(sigma2, dtype=float)
    a1, a2 = bool(np.any(s1 > 0)), bool(np.any(s2 > 0))
    if a1 and a2:
        m1 = m2 = None
        for x, wx in zip(_GH2_NODES, _GH2_WEIGHTS, strict=False):
            a = np.sqrt(2.0) * s1 * x
            for y, wy in zip(_GH2_NODES, _GH2_WEIGHTS, strict=False):
                v = fn(a, np.sqrt(2.0) * s2 * y)
                w = wx * wy
                m1 = w * v if m1 is None else m1 + w * v
                m2 = w * v * v if m2 is None else m2 + w * v * v
        mean = m1 * _INV_SQRT_PI**2
        return mean, np.maximum(m2 * _INV_SQRT_PI**2 - mean * mean, 0.0)
    if a1:
        return expect_normal_moments(lambda a: fn(a, 0.0), s1)
    if a2:
        return expect_normal_moments(lambda b: fn(0.0, b), s2)
    v = fn(0.0, 0.0)
    return v, np.zeros_like(v)


def integrate_normal_interval(fn, lo, hi) -> np.ndarray:
    """Compute ∫_{lo}^{hi} fn(ν) φ(ν) dν per row using 32 Gauss–Legendre nodes on each row's interval."""
    lo, hi = np.asarray(lo, dtype=float), np.asarray(hi, dtype=float)
    half, mid = 0.5 * (hi - lo), 0.5 * (hi + lo)
    acc = None
    for x, w in zip(_GL_NODES, _GL_WEIGHTS, strict=False):
        nu = mid + half * x
        v = w * fn(nu) * np.exp(-0.5 * nu * nu) / np.sqrt(2.0 * np.pi)
        acc = v if acc is None else acc + v
    assert acc is not None
    return acc * half


# Tabulate G(v) = ∫_{−∞}^{v} g(ν) φ(ν) dν for the outer quadrature. Each of the 32 ×
# 32 outer nodes over (eps_med, u_tau) needs an inner integral over [Φ⁻¹(p_lo), Φ⁻¹(p_hi)]. The
# integrand g(ν), such as the complier probability or selection weight ω, is independent of the
# outer nodes. Build G once per row on [−6.5, 6.5] at spacing 0.025. Use cumulative Simpson
# integration and cubic Hermite interpolation with the exact derivative G′ = g·φ. The error is
# about 5e-9. Process rows in blocks so the b × K table fits in memory.
_NU_H = 0.025
_NU_GRID = np.linspace(
    -6.5, 6.5, 521
)  # φ(6.5) ≈ 3e-10. Omit the tails outside the grid.
_NU_PHI = np.exp(-0.5 * _NU_GRID * _NU_GRID) / np.sqrt(2.0 * np.pi)
NU_TABLE_N = _NU_GRID.size
_ROW_BLOCK = 4096


class _CdfTable:
    """Tabulate G_row(v) for a block of rows from a b × K grid of g(ν).

    Accept v of shape (b,) or (b, m).
    """

    def __init__(self, g: np.ndarray):
        f = g * _NU_PHI[None, :]
        h = _NU_H
        pairs = h / 3.0 * (f[:, 0:-2:2] + 4.0 * f[:, 1:-1:2] + f[:, 2::2])
        even = np.concatenate(
            [np.zeros((f.shape[0], 1)), np.cumsum(pairs, axis=1)], axis=1
        )
        odd = even[:, :-1] + h / 12.0 * (
            5.0 * f[:, 0:-2:2] + 8.0 * f[:, 1:-1:2] - f[:, 2::2]
        )
        G = np.empty_like(f)
        G[:, 0::2], G[:, 1::2] = even, odd
        self.f, self.G = f, G
        self.rows = np.arange(f.shape[0])

    def __call__(self, v: np.ndarray) -> np.ndarray:
        v = np.clip(v, _NU_GRID[0], _NU_GRID[-1])
        i = np.clip(
            ((v - _NU_GRID[0]) * (1.0 / _NU_H)).astype(np.int64), 0, NU_TABLE_N - 2
        )  # Uniformly spaced grid.
        t = (v - _NU_GRID[i]) / _NU_H
        t2, t3 = t * t, t * t * t
        r = self.rows if v.ndim == 1 else self.rows[:, None]
        flat = r * NU_TABLE_N + i
        g0, g1 = self.G.take(flat), self.G.take(flat + 1)
        f0, f1 = self.f.take(flat), self.f.take(flat + 1)
        return (
            (2 * t3 - 3 * t2 + 1) * g0
            + (t3 - 2 * t2 + t) * _NU_H * f0
            + (-2 * t3 + 3 * t2) * g1
            + (t3 - t2) * _NU_H * f1
        )


def _gh_axis(active: bool, tensor: bool):
    """Return the Gauss–Hermite nodes and normalized weights for one dimension.

    Use 32 nodes in a tensor grid and 64 for a single active dimension. An inactive
    dimension has just one node at zero.
    """
    if not active:
        return np.zeros(1), np.ones(1)
    nodes, weights = (_GH2_NODES, _GH2_WEIGHTS) if tensor else (_GH_NODES, _GH_WEIGHTS)
    return np.sqrt(2.0) * nodes, weights * _INV_SQRT_PI


def _outer_nu_expectation(
    grid_fns, mu0_w, tau_c, sd_med, sd_h, kernel, block: int = 4096
) -> list[np.ndarray]:
    """Average a kernel over the latent variables separately for each row.

    Compute E_{η,U}[kernel] for each g in grid_fns. For a block of b rows, grid_fn(sl)
    returns g on a b × K grid in ν. Its table is G(v) = ∫_{−∞}^{v} g φ. kernel(tables,
    mu0_e, u_vals) returns a list of b × K_u arrays of inner integrals over ν, evaluated
    at mu0 = mu0_w + η (b × 1) and tau = tau_c + U (b × K_u). As in
    expect_normal2_moments, use 32 × 32 outer nodes for two active scales or 64 nodes
    for one. Loop over η and evaluate the nodes of U together.
    """
    n = mu0_w.size
    sd_h_arr = np.asarray(sd_h, dtype=float)
    a_med, a_h = sd_med > 0.0, bool(np.any(sd_h_arr > 0))
    tensor = a_med and a_h
    x_e, w_e = _gh_axis(a_med, tensor)
    x_u, w_u = _gh_axis(a_h, tensor)
    outs = [np.empty(n) for _ in grid_fns]
    for lo in range(0, n, block):
        sl = slice(lo, lo + block)
        tables = [_CdfTable(fn(sl)) for fn in grid_fns]
        mu0_b, tau_b = mu0_w[sl][:, None], tau_c[sl][:, None]
        sd_h_b = sd_h_arr[sl][:, None] if sd_h_arr.ndim else sd_h_arr
        u_vals = tau_b + sd_h_b * x_u[None, :]  # b × K_u
        acc = [np.zeros(mu0_b.shape[0]) for _ in grid_fns]
        for xe, we in zip(x_e, w_e, strict=False):
            for k, val in enumerate(kernel(tables, mu0_b + sd_med * xe, u_vals)):
                acc[k] += we * np.einsum("bk,k->b", val, w_u)
        for k in range(len(grid_fns)):
            outs[k][sl] = acc[k]
    return outs


def _outer_interval_expectation(
    fam, oc, mu0_w, tau_c, sd_med, sd_h, grid_fns
) -> list[np.ndarray]:
    """Integrate the effect over the interval in shared outcome noise.

    For binary and ZILN outcomes with shared noise, the ITE is a scaled, signed interval
    indicator in ν. Compute E_{η,U}[scale·(G(Φ⁻¹(p₁(η,U))) − G(Φ⁻¹(p₀(η))))].
    Subtracting the two CDF values accounts for both the interval and its sign:
    sign·∫_{lo}^{hi} = G(ν₁) − G(ν₀).
    """

    def kernel(tables, mu0_e, u_vals):
        p0, p1, scale = family_probs(
            fam, mu0_e, u_vals, oc
        )  # p0, scale — b × 1; p1 — b × K_u
        nu0 = ndtri(np.clip(p0, 1e-15, 1 - 1e-15))[:, 0]
        nu1 = ndtri(np.clip(p1, 1e-15, 1 - 1e-15))
        return [scale * (table(nu1) - table(nu0)[:, None]) for table in tables]

    return _outer_nu_expectation(grid_fns, mu0_w, tau_c, sd_med, sd_h, kernel)


_BETA_BLOCK = 256


def _nu_ite_expectation(ds, fns, grid_fns, mu0, tau, sd_med, sd_h) -> list[np.ndarray]:
    """Compute the expected product of a noise function and the individual treatment effect.

    For each g, compute E_{η,U} E[g(ν)·ite] for each row under the beta family or
    independent coupling (0.8.0). Center the integral over (η, U) at (mu0_w,
    tau_c) for fields conditional on W, or at (mu0_eff, tau_eff) with sd_med = sd_h = 0
    for the latent-state oracle. Each g has two forms: fns evaluates a scalar ν for all
    rows, while grid_fn(sl) in grid_fns returns a b × K grid for one block.

    Under independent coupling Y(1) is independent of ν = ν₀, so E[g·ite] = m₁·E[g] −
    E[g(ν)·y₀(ν)]. For Gaussian outcomes this is tau·E[g] − E[g·ε(ν)]: η cancels and U
    enters linearly. For binary and ZILN outcomes the second term is scale·G(Φ⁻¹(p₀)),
    where _CdfTable tabulates G(v) = ∫_{−∞}^{v} g φ.

    For beta outcomes, E[g·Q(Φ(ν))] = ∫₀¹ E[g·1(Y > y)] dy = ∫₀¹ (E[g] − G(Φ⁻¹(F_Y(y))))
    dy. Here F_Y(y) = π₀ + (1 − π₀ − π₁)·I_y. This requires only the forward incomplete
    beta function. Integrate over y using _UNIT_NODES.
    """
    lat, oc = ds.latent, ds.config.outcome
    fam, shared = oc.family, oc.coupling == "shared"
    if fam == "gaussian":
        s = (
            1.0 if lat.get("y_scale") is None else lat["y_scale"]
        )  # Outcome scale (0.9.0).
        return [
            s
            * (
                tau * expect_normal(g, 1.0)
                - _expect_g_eps(g, lat["sd_eps"], oc.heavy_tails)
            )
            for g in fns
        ]
    if fam == "lognormal":
        return _lognormal_nu_ite(oc, lat, fns, mu0, tau, sd_h)
    phi = oc.beta_phi

    def cdf_integral(tables, arm):
        """Integrate G(Φ⁻¹(F_Y(y))) over y in (0, 1) for arm = (π₀, π₁, m) of shape b × k.

        Return a list containing one b × k array per table.
        """
        pi0, pi1, m = (v[..., None] for v in arm)
        a, b_ = m * phi, (1.0 - m) * phi
        f = pi0 + (1.0 - pi0 - pi1) * betainc(a, b_, _UNIT_NODES)
        nu = ndtri(np.clip(f, 1e-300, 1 - 1e-16))
        b, k = arm[0].shape
        return [
            np.einsum(
                "bky,y->bk", t(nu.reshape(b, -1)).reshape(b, k, UNIT_N), _UNIT_WEIGHTS
            )
            for t in tables
        ]

    def kernel(tables, mu0_e, u_vals):
        egs = [t.G[:, -1][:, None] for t in tables]  # E[g], b × 1
        if fam == "beta":
            arm1 = beta_arm(mu0_e, u_vals, oc)
            h0 = cdf_integral(tables, beta_arm(mu0_e, 0.0, oc))  # b × 1
            if shared:
                return [
                    a - b_
                    for a, b_ in zip(h0, cdf_integral(tables, arm1), strict=False)
                ]
            m1 = beta_mean(*arm1)
            return [m1 * eg - (eg - a) for eg, a in zip(egs, h0, strict=False)]
        p0, p1, scale = family_probs(
            fam, mu0_e, u_vals, oc
        )  # binary / ziln, "independent"
        nu0 = ndtri(np.clip(p0, 1e-15, 1 - 1e-15))[:, 0]
        return [
            scale * (p1 * eg - t(nu0)[:, None])
            for eg, t in zip(egs, tables, strict=False)
        ]

    return _outer_nu_expectation(
        grid_fns,
        mu0,
        tau,
        sd_med,
        sd_h,
        kernel,
        _BETA_BLOCK if fam == "beta" else _ROW_BLOCK,
    )


def _lognormal_nu_ite(oc, lat, fns, mu0, tau, sd_h) -> list[np.ndarray]:
    """Compute the noise-weighted lognormal individual treatment effect.

    For shared noise, E_U E[g(ν)·ite] uses ite = Z·exp(L + σ_ε·ν)·(exp(tau+U) − 1),
    where Z indicates a draw outside the point mass at zero and is independent of ν
    (0.11.0). With independent coupling, E[g·ite] = E[g]·E[Y₁] − E[g·Y₀]. Given lagged
    outcomes, ν has posterior N(m, v), or N(0, 1) without lags. Use 64-point
    Gauss–Hermite quadrature for E[g·exp(σ_ε·ν)] and E[g]. Integrate U ~ N(0, sd_h²)
    analytically: E[exp(tau+U)] = exp(tau + sd_h²/2).
    """
    se, pi = float(oc.noise_sigma), float(oc.zero_mass_ln)
    m, v = nu_posterior(lat)
    sv = float(np.sqrt(v))
    a = (1.0 - pi) * np.exp(lognormal_log_level(mu0, oc))
    e1 = np.exp(tau + 0.5 * np.square(np.asarray(sd_h, dtype=float)))
    out = []
    for g in fns:
        eg_exp = expect_normal(
            lambda z, g=g: g(m + sv * z) * np.exp(se * (m + sv * z)), 1.0
        )
        if oc.coupling == "shared":
            out.append(a * eg_exp * (e1 - 1.0))
        else:
            eg = expect_normal(lambda z, g=g: g(m + sv * z), 1.0)
            out.append(a * (eg * np.exp(0.5 * se * se) * e1 - eg_exp))
    return out


def _ite_closed_in_nu(ds) -> bool:
    """Return whether the shared-noise integral is constant or reduces to an interval.

    For shared ν, the Gaussian ITE is constant and the binary or ZILN ITE is an interval
    indicator. Compute the corresponding fields in closed form or by integrating over
    the interval. For beta, lognormal, or independent coupling, use _nu_ite_expectation.
    """
    oc = ds.config.outcome
    return oc.coupling == "shared" and oc.family not in ("beta", "lognormal")


def _beta_shared_sq(mu0, tau, oc) -> np.ndarray:
    """Compute the second moment of the outcome difference under shared beta noise.

    For E_u[(Q₁(u) − Q₀(u))²], split the unit interval at both arms' jump points π₀,t
    and 1 − π₁,t. Integrate each piece with _UNIT_NODES. Outside the extreme jump points
    both outcomes are equal, either both zero or both one.
    """
    a0, a1 = beta_arm(mu0, 0.0, oc), beta_arm(mu0, tau, oc)
    cuts = np.sort(np.stack([a0[0], a1[0], 1.0 - a0[1], 1.0 - a1[1]]), axis=0)
    out = np.zeros(np.shape(a0[0]))
    for k in range(3):
        lo, width = cuts[k], cuts[k + 1] - cuts[k]
        rows = width > 0
        if not rows.any():
            continue
        u = lo[rows, None] + width[rows, None] * _UNIT_NODES[None, :]
        pi01, pi11, m1 = (v[rows, None] for v in a1)
        pi00, pi10, m0 = (v[rows, None] for v in a0)
        d = beta_quantile(u, pi01, pi11, m1, oc.beta_phi) - beta_quantile(
            u, pi00, pi10, m0, oc.beta_phi
        )
        out[rows] += width[rows] * np.einsum("by,y->b", d * d, _UNIT_WEIGHTS)
    return out


def vte_w(ds) -> np.ndarray:
    """Compute the variance of the realized treatment effect conditional on W.

    Var[Y(1) − Y(0) | W] includes outcome noise (0.8.0). Like tau_cate, it
    uses the underlying DGP before noncompliance or selection. Compute E[ite² | W] −
    tau_cate². E_lat uses the same quadrature over (eps_med, u_tau) as tau_cate.

    With independent coupling the second moment is E_lat[s₁ + s₀ − 2·m₁·m₀], where m_t
    and s_t are the first two arm moments given the latents. For Gaussian outcomes with
    no lags or additional scale, the variance is cvte + 2·sd_eps². Under shared Gaussian
    noise the variance is cvte. The shared-noise second moments are E_lat|p₁ − p₀| for
    binary, E_lat[|p₁ − p₀|·E ln²] for ZILN, and E_lat E_u[(Q₁ − Q₀)²] for beta.
    Evaluating beta quantiles at every node is expensive, so the field is computed
    lazily.
    """
    lat, oc = ds.latent, ds.config.outcome
    fam, shared = oc.family, oc.coupling == "shared"
    if fam == "lognormal":  # Closed form added in 0.11.0.
        m2 = lognormal_ite_second_moment(
            oc, lat, _mu0_w(ds), _tau_cond(ds), _var_hidden(ds)
        )
        return np.maximum(m2 - ds.tau_cate**2, 0.0)
    if fam == "gaussian":
        s2 = (
            1.0 if lat.get("y_scale") is None else lat["y_scale"] ** 2
        )  # Outcome scale (0.9.0).
        if shared:
            out = ds.cvte.copy()
        elif "pre_nu_m" in lat:  # With outcome lags, Var[ν₀ | W] = v (0.11.0).
            out = ds.cvte + (1.0 + lat["pre_nu_v"]) * lat["sd_eps"] ** 2 * s2
        else:
            out = ds.cvte + 2.0 * lat["sd_eps"] ** 2 * s2
        mt = mediator_truth(ds)
        if mt is not None:  # Binary mediators (0.10.0):
            d = np.abs(mt["tau_m"])  # (M1 − M0) ~ ±Bernoulli(|d|)
            out = out + s2 * _rowsum(mt["theta"] ** 2 * d * (1.0 - d))
        return out
    mu0_w, tau_c = _mu0_w(ds), _tau_cond(ds)

    def second(eta, u):
        mu0, tau = mu0_w + eta, tau_c + u
        if not shared:
            m0, m1, s0, s1 = arm_moments(fam, mu0, tau, oc)
            return s1 + s0 - 2.0 * m1 * m0
        if fam == "beta":
            return _beta_shared_sq(mu0, tau, oc)
        p0, p1, _ = family_probs(fam, mu0, tau, oc)
        return np.abs(p1 - p0) * (ziln_second_moment(mu0) if fam == "ziln" else 1.0)

    if shared:
        # The absolute ITE has a kink in U where tau_c + U = 0 and the effect changes sign. Split
        # the integral there.
        mean2 = _expect_outer_kink(
            second, _sd_med(ds), _sd_hidden(ds), tau_c, *_vte_outer_n(fam)
        )
    else:
        mean2, _ = expect_normal2_moments(second, _sd_med(ds), _sd_hidden(ds))
    return np.maximum(mean2 - ds.tau_cate**2, 0.0)


# Outer quadrature for shared-noise vte_w: the first count gives the Gauss–Hermite nodes for
# eps_med, the second the nodes on each side of the split in U. Each beta node requires quantiles
# at 3 × 16 points, so use fewer nodes. Relative error with mediators is about 1e-3.
VTE_OUTER_N = {"beta": (16, 8), "other": (32, 16)}


def _vte_outer_n(fam: str) -> tuple[int, int]:
    return VTE_OUTER_N["beta" if fam == "beta" else "other"]


def _expect_outer_kink(fn, sd_med, sd_h, tau_c, k_eta: int, k_u: int) -> np.ndarray:
    """Integrate a function with a kink over two independent normal variables.

    Compute E[fn(η, U)] for η ~ N(0, sd_med²) and U ~ N(0, sd_h²). sd_h can be scalar or
    row-specific. The kink in U occurs at tau_c + U = 0. Use k_eta Gauss–Hermite nodes
    for η. Substitute U = sd_h·Φ⁻¹(p), split at p* = Φ(−tau_c / sd_h), and apply
    _unit_rule(k_u) on both (0, p*) and (p*, 1).
    """
    sd_h = np.asarray(sd_h, dtype=float)
    if sd_med > 0.0:
        nodes, weights = np.polynomial.hermite.hermgauss(k_eta)
        eta_axis = list(
            zip(np.sqrt(2.0) * sd_med * nodes, weights * _INV_SQRT_PI, strict=False)
        )
    else:
        eta_axis = [(0.0, 1.0)]
    if not np.any(sd_h > 0):
        u_axis = [(0.0, 1.0)]
    else:
        sd = np.broadcast_to(sd_h, tau_c.shape)
        z = np.where(sd > 0, -tau_c / np.where(sd > 0, sd, 1.0), 0.0)
        p_star = np.array([_phi_cdf(v) for v in z])
        u_axis, rule = [], _unit_rule(k_u)
        for lo, width in ((0.0, p_star), (p_star, 1.0 - p_star)):
            for s_, w_ in zip(*rule, strict=False):
                u_axis.append(
                    (
                        sd * ndtri(np.clip(lo + width * s_, 1e-300, 1 - 1e-16)),
                        width * w_,
                    )
                )
    acc = None
    for eta, we in eta_axis:
        for u, wu in u_axis:
            v = we * wu * fn(eta, u)
            acc = v if acc is None else acc + v
    assert acc is not None
    return acc


def _interval(fam, mu0, tau, oc):
    """Return (sign·scale, Φ⁻¹(p_lo), Φ⁻¹(p_hi)) for the binary/ZILN ITE sign·scale·1[lo ≤ ν < hi]."""
    p0, p1, scale = family_probs(fam, mu0, tau, oc)
    lo = ndtri(np.clip(np.minimum(p0, p1), 1e-15, 1 - 1e-15))
    hi = ndtri(np.clip(np.maximum(p0, p1), 1e-15, 1 - 1e-15))
    return np.sign(p1 - p0) * scale, lo, hi


# --- non-compliance ---


def nc_type_probs(nc: dict, nu) -> tuple[np.ndarray, np.ndarray]:
    """Return row-wise (p_nt, p_at) for a scalar or row-specific ν.

    Both are zero outside the mask.

    Draw compliance types in sequence: P(never-taker) = p_nt, P(always-taker) = (1 −
    p_nt)·p_at, and P(complier) = (1 − p_nt)(1 − p_at).
    """
    s, h = nc["strength"], nc["hidden"]
    score = np.sqrt(1.0 - h) * nc["score_x"] + np.sqrt(h) * nu
    sig_neg = _sigmoid(-score)
    p_nt = np.clip(2.0 * s * sig_neg, 0.0, 1.0)
    p_at = (
        np.clip(s * (1.0 - sig_neg), 0.0, 1.0)
        if nc["side"] == "two"
        else np.zeros_like(p_nt)
    )
    mask = nc["mask"]
    return np.where(mask, p_nt, 0.0), np.where(mask, p_at, 0.0)


def _nc_block(nc: dict, sl: slice) -> dict:
    """Return the block's noncompliance parameters as b × 1 arrays for a b × K grid of ν values."""
    return {
        **nc,
        "score_x": nc["score_x"][sl][:, None],
        "mask": nc["mask"][sl][:, None],
    }


def _complier_grid(nc: dict, sl: slice) -> np.ndarray:
    """Return P(complier | ν) on a b × K grid for this block of rows."""
    p_nt, p_at = nc_type_probs(_nc_block(nc, sl), _NU_GRID[None, :])
    return (1.0 - p_nt) * (1.0 - p_at)


def _nc_active(ds) -> bool:
    nc = ds.latent.get("nc")
    return nc is not None and nc["strength"] > 0.0


# --- selection ---


def selection_e(e: np.ndarray, q1: np.ndarray, q0: np.ndarray) -> np.ndarray:
    """Return P(T = 1 | W, S = 1) = e q₁ / (e q₁ + (1 − e) q₀), where q_t is the retention probability given T = t."""
    num = e * q1
    den = num + (1.0 - e) * q0
    return np.where(den > 0, num / np.where(den > 0, den, 1.0), e)


# Effective latent quantities


def _terms(ds, terms=None):
    """Return precomputed mediator contributions (mechanisms.MediatorTerms or None), or derive them from latent state."""
    return mediator_contributions(ds.latent) if terms is None else terms


def _mu0_eff(ds, terms=None) -> np.ndarray:
    """Compute the control outcome level at fixed mediator latents.

    Add mu0_x, the row's mediator contribution (including ε_med for Gaussian mediators
    or realized M(0) for binary mediators, 0.10.0), and the cluster intercept (P-19).
    This conditions on W and the mediator latents.
    """
    lat, t = ds.latent, _terms(ds, terms)
    mu0 = lat["mu0_x"] if t is None or t.mu0_row is None else lat["mu0_x"] + t.mu0_row
    if "mu0_cluster" in lat:
        mu0 = mu0 + lat["mu0_cluster"]
    return mu0


def _mu0_w(ds, terms=None) -> np.ndarray:
    """Compute the control outcome level using the expected mediator contribution given W.

    Add mu0_x, the contribution determined by W (Σ w_k f_k for Gaussian mediators or Σ
    θ_k p0_k for binary mediators), and the cluster intercept. This centers the integral
    over eps_med. Cluster membership is observed in W.
    """
    lat, t = ds.latent, _terms(ds, terms)
    mu0 = lat["mu0_x"] if t is None or t.mu0_w is None else lat["mu0_x"] + t.mu0_w
    if "mu0_cluster" in lat:
        mu0 = mu0 + lat["mu0_cluster"]
    return mu0


def _sd_med(ds) -> float:
    """Return the standard deviation of the mediator noise contribution to the outcome.

    For Gaussian mediators this is ‖med_w‖. Return zero without mediators, or for binary
    mediators with a Gaussian outcome, where mediator noise is integrated out
    analytically.
    """
    lat = ds.latent
    if "med_w" not in lat:
        return 0.0
    return float(np.sqrt(np.einsum("k,k->", lat["med_w"], lat["med_w"])))


def _tau_eff(ds, terms=None) -> np.ndarray:
    """Compute the structural effect with the row's latent variables fixed.

    Add tau_x, the mediated effect, and the hidden modifier. For binary mediators, use
    the realized M(1) − M(0) in the mediated effect.
    """
    lat, t = ds.latent, _terms(ds, terms)
    tau = lat["tau_x"] if t is None else lat["tau_x"] + t.tau_row
    if "tau_hidden" in lat:
        tau = tau + lat["tau_hidden"]
    return tau


def _tau_cond(ds, terms=None) -> np.ndarray:
    """Compute the effect using the expected mediator contribution given W.

    Add tau_x and the expected mediated effect: Σ w_k a_k for Gaussian mediators, or Σ
    θ_k (p1_k − p0_k) for binary mediators. Exclude the hidden modifier to obtain the
    center of the integral over U.
    """
    t = _terms(ds, terms)
    return ds.latent["tau_x"] if t is None else ds.latent["tau_x"] + t.tau_w


def mediator_truth(ds) -> dict | None:
    """Return true probabilities, effects, and potential values for binary mediators.

    The n × m fields are p_m0, p_m1, tau_m = p1 − p0, theta, m0, and m1 (0.10.0). Return
    None for Gaussian mediators or no mediators. Like vte_w, these fields are cached
    until ground truth is refreshed, here in latent["mediator_truth"]. Selection and
    duplication carry the relevant rows of that cache along with the latent state.
    """
    lat = ds.latent
    if mediator_family(lat) != "binary":
        return None
    if "mediator_truth" not in lat:
        p0, p1 = mediator_probs(lat)
        m0, m1 = mediator_potentials(lat, (p0, p1))
        lat["mediator_truth"] = {
            "p_m0": p0,
            "p_m1": p1,
            "tau_m": p1 - p0,
            "theta": lat["med_theta"],
            "m0": m0,
            "m1": m1,
        }
    return lat["mediator_truth"]


def _var_hidden(ds):
    """Return the hidden modifier's variance by row, or 0.0 when it is absent."""
    var_h = ds.latent.get("var_hidden")
    return 0.0 if var_h is None else np.asarray(var_h, dtype=float)


def _sd_hidden(ds):
    var_h = ds.latent.get("var_hidden")
    if var_h is None or not np.any(var_h > 0):
        return 0.0
    return np.sqrt(np.asarray(var_h, dtype=float))


def e_assign_effective(ds) -> np.ndarray:
    """Return P(Z = 1 | W) = σ(logit(e_SCM) + sum of axis contributions).

    Rows without contributions retain e_SCM bit for bit. Under block assignment
    (cluster.assign = block, 0.11.0), use the assigned fraction in each row's block,
    stored in latent["e_block"].
    """
    lat = ds.latent
    if "e_block" in lat:
        return lat["e_block"]
    e = lat["e_assign"]
    extra = lat.get("e_logit_extra")
    if extra is None:
        return e
    return np.where(extra != 0.0, _sigmoid(_logit(e) + extra), e)


def y_scale(ds):
    """Return the natural outcome scale for each row.

    Compute s(x) from the current true X and the SCM calibration constants
    (0.9.0). Return None when scale_sigma = 0. refresh_truth recomputes s(x)
    because covariate_shift can change X, so it need not be a fixed row parameter. For a
    dataset outside an SCM, use the saved latent["y_scale"].
    """
    lat, oc = ds.latent, ds.config.outcome
    if oc.scale_sigma <= 0:
        return None
    scm = ds.scm
    if scm is None:
        return lat.get("y_scale")
    j = scm.params["scale_feature"]
    return outcome_scale(
        lat["x_true"][:, j],
        scm.calib["x_raw_mean"][j],
        scm.calib["x_raw_sd"][j],
        oc.scale_sigma,
    )


# Refresh all dependent fields together


def refresh_truth(ds) -> None:
    """Recompute all dependent ground-truth fields from ds.latent (contract 0.7.0).

    Compute effective mu0 and tau, then potential outcomes using saved noise. Derive
    assignment z and received treatment t from u_t, e_assign, and compliance. Integrate
    over u_tau, eps_med, and when needed ν for the CATE fields given W. Apply selection
    to obtain e_source, e = P(T=1 | W,S=1), and *_selected. Update mediators and
    colliders with saved noise. Finally set y = y_true and t = t_true: the DGP phase
    introduces no measurement error.
    """
    lat, cfg = ds.latent, ds.config
    fam, oc = cfg.outcome.family, cfg.outcome
    n = lat["tau_x"].size

    # 1–2. Compute effective row values and potential outcomes. The saved latent noise needs no
    # new random draws.
    s = y_scale(ds)  # Outcome scale (0.9.0): use 1 when scale_sigma = 0.
    if s is None:
        lat.pop("y_scale", None)
    else:
        lat["y_scale"] = s
    lat.pop("mediator_truth", None)  # Cached binary mediator fields (0.10.0).
    terms = mediator_contributions(
        lat
    )  # Compute mediator contributions once per refresh.
    mu0_eff, tau_eff = _mu0_eff(ds, terms), _tau_eff(ds, terms)
    y0, y1, tau_s, mu0_o, mu1_o = family_outcomes(fam, mu0_eff, tau_eff, oc, lat, None)
    ds.y0, ds.y1, ds.tau, ds.mu0, ds.mu1 = y0, y1, tau_s, mu0_o, mu1_o
    if "pre_nu" in lat:  # Pretreatment outcome lags (0.11.0).
        _lagged_outcomes(ds, mu0_eff, s)

    # 3. Compute assignment and received treatment T.
    e_z = e_assign_effective(ds)
    z = (lat["u_t"] < e_z).astype(np.int8)
    has_nc = ds.compliance is not None
    if has_nc:
        t_true = z.copy()
        t_true[ds.compliance == 1] = 0
        t_true[ds.compliance == 2] = 1
        ds.z, ds.e_assign = z, e_z.copy()
    else:
        t_true = z
        ds.z = ds.e_assign = None
    ds.t_true = t_true
    ds.t = t_true.copy()

    # 4. For fields conditional on W, integrate tau_cate and cvte over eps_med and u_tau. The
    # noncompliance calculations for tau_itt, e, and tau_itt_cate also use ν, with the
    # conditioning set appropriate to each field.
    mu0_w, tau_c = _mu0_w(ds, terms), _tau_cond(ds, terms)
    sd_med, sd_h = _sd_med(ds), _sd_hidden(ds)
    ds.tau_cate_x = None
    if (
        fam == "gaussian"
    ):  # eps_med cancels in the difference. U enters additively, and s multiplies the result.
        ds.tau_cate = tau_c.copy() if s is None else tau_c * s
        var_h = lat.get("var_hidden")
        ds.cvte = (
            np.array(var_h, dtype=float)
            if np.any(np.asarray(sd_h) > 0)
            else np.zeros(n)
        )
        if s is not None:
            ds.cvte = ds.cvte * (s * s)
        if (
            "pre_nu" in lat
        ):  # With lags and independent coupling, E[ite | W] = s·(tau − sd_eps·E[ν₀ | W]) (0.11.0).
            ds.tau_cate_x = ds.tau_cate.copy()
            if oc.coupling == "independent":
                ds.tau_cate = (
                    ds.tau_cate
                    - (1.0 if s is None else s) * lat["sd_eps"] * lat["pre_nu_m"]
                )
    elif (
        fam == "lognormal"
    ):  # Closed form over U and ν, using the posterior of ν when lags are present (0.11.0).
        ds.tau_cate, ds.cvte = lognormal_cate(oc, lat, mu0_w, tau_c, _var_hidden(ds))
        if "pre_nu" in lat:
            ds.tau_cate_x, _ = lognormal_cate(oc, {}, mu0_w, tau_c, _var_hidden(ds))
    elif sd_med == 0.0 and not np.any(np.asarray(sd_h) > 0):
        ds.tau_cate = tau_scale_mean(fam, mu0_w, tau_c, oc)
        ds.cvte = np.zeros(n)
    else:
        ds.tau_cate, ds.cvte = expect_normal2_moments(
            lambda eta, u: tau_scale_mean(fam, mu0_w + eta, tau_c + u, oc), sd_med, sd_h
        )
    lat.pop("vte_w", None)  # Cache for the lazily computed Dataset.vte_w.

    if has_nc and _nc_active(ds):
        _noncompliance_fields(ds, e_z, mu0_eff, tau_eff, mu0_w, tau_c, sd_med, sd_h)
    else:
        ds.tau_itt = (
            ds.tau.copy() if has_nc else None
        )  # All units comply. Preserve the original values bit for bit.
        ds.tau_itt_cate = ds.tau_cate.copy() if has_nc else None
        ds.e = e_z.copy()

    # 5. Selection: keep e_source = P(T=1 | W) before selection, update e = P(T=1 | W,S=1),
    # and derive the selected-population fields.
    _selection_fields(ds, e_z)

    # 6. Update mediators under received T and compute colliders with saved noise.
    if "med_a" in lat:
        ds.mediators = (
            lat["med_a"][None, :] * t_true[:, None].astype(float) + lat["med_eps"]
        )
        if "med_f" in lat:
            ds.mediators = ds.mediators + lat["med_f"]
    elif mediator_family(lat) == "binary":  # Binary mediators (0.10.0): M = M(t_true).
        mt = mediator_truth(ds)
        assert mt is not None
        ds.mediators = np.where(t_true[:, None] == 1, mt["m1"], mt["m0"]).astype(
            np.int8
        )
    y_true = np.where(t_true == 1, y1, y0)
    coll = ds.graph.covariate_indices(ROLE_COLLIDER)
    if coll.size:
        # Standardize y with the SCM calibration constants (0.6.0), or with sample statistics
        # outside an SCM.
        if "coll_y_mean" in lat:
            y_std = (y_true.astype(float) - lat["coll_y_mean"]) / lat["coll_y_sd"]
        else:
            y_std = _standardize(y_true.astype(float))
        c = 0.7 * t_true[:, None] + 0.7 * y_std[:, None] + 0.5 * lat["eps_c"]  # n × k
        lat["x_true"][:, coll] = c
        for k, i in enumerate(coll):
            ds.x[ds.graph.node_names[i]] = c[:, k]

    # 7. The observed y equals y_true.
    ds.y = y_true


def _lagged_outcomes(ds, mu0_eff, s) -> None:
    """Compute observed pretreatment outcomes.

    The lag y_pre1 through y_preK (0.11.0) is Y(0) in period −k, with the same x,
    its own ν_k from latent["pre_nu"], and a level shift of −k·δ. The Gaussian value is
    s·(mu0 − kδ + sd_eps·ν_k). The lognormal positive value is exp(log b + σ_μ·mu0 − kδ
    + σ_ε·ν_k). These observed X columns belong to W. Each truth refresh recomputes
    them, using the new X after covariate shift and the same cluster intercept across
    periods.
    """
    lat, oc = ds.latent, ds.config.outcome
    nu = lat["pre_nu"]
    shift = oc.pre_drift * np.arange(1, nu.shape[1] + 1, dtype=float)
    if oc.family == "lognormal":
        y = np.exp(
            lognormal_log_level(mu0_eff, oc)[:, None]
            - shift[None, :]
            + lat["sd_eps"] * nu
        )
    else:
        y = mu0_eff[:, None] - shift[None, :] + lat["sd_eps"] * nu
        if s is not None:
            y = y * s[:, None]
    for k in range(nu.shape[1]):
        ds.x[pre_column(k)] = y[:, k]


def pre_column(k: int) -> str:
    """Return the lag column name for zero-based index k. y_pre1 is the period closest to treatment."""
    return f"y_pre{k + 1}"


def pre_columns(ds) -> list[str]:
    """Return outcome-lag columns in period order, or an empty list when there are no lags."""
    return [pre_column(k) for k in range(int(ds.config.outcome.pre_periods))]


def _noncompliance_fields(
    ds, e_z, mu0_eff, tau_eff, mu0_w, tau_c, sd_med, sd_h
) -> None:
    """Compute tau_itt conditional on (W, u_tau, eps_med), e = P(T_true = 1 | W), and tau_itt_cate conditional on W.

    e = E_ν[(1 − p_nt)(p_at + (1 − p_at) e_z)], and complier(ν) = (1 − p_nt)(1 − p_at).
    With shared Gaussian noise, ite is independent of ν: tau_itt = tau · E_ν[complier]
    and tau_itt_cate = tau_cate · E_ν[complier]. For shared binary or ZILN noise, ite(ν)
    = ±scale·1[Φ⁻¹(p_lo) ≤ ν < Φ⁻¹(p_hi)]. Thus tau_itt = sign·scale·∫ complier φ dν
    over the interval. For tau_itt_cate evaluate the same integral at (mu0_w + η, tau_c
    + U), then integrate over (η, U).
    """
    nc, fam, oc = ds.latent["nc"], ds.config.outcome.family, ds.config.outcome

    def e_received(nu):
        p_nt, p_at = nc_type_probs(nc, nu)
        return (1.0 - p_nt) * (p_at + (1.0 - p_at) * e_z)

    def complier(nu):
        p_nt, p_at = nc_type_probs(nc, nu)
        return (1.0 - p_nt) * (1.0 - p_at)

    ds.e = expect_normal(e_received, 1.0)

    def complier_grid(sl):
        return _complier_grid(nc, sl)

    # Beta outcomes and independent coupling require integration over nu (0.8.0).
    if not _ite_closed_in_nu(ds):
        (ds.tau_itt,) = _nu_ite_expectation(
            ds, [complier], [complier_grid], mu0_eff, tau_eff, 0.0, 0.0
        )
        if sd_med == 0.0 and not np.any(np.asarray(sd_h) > 0):
            ds.tau_itt_cate = ds.tau_itt.copy()
        else:
            (ds.tau_itt_cate,) = _nu_ite_expectation(
                ds, [complier], [complier_grid], mu0_w, tau_c, sd_med, sd_h
            )
        return
    if fam == "gaussian":
        c = expect_normal(complier, 1.0)
        ds.tau_itt = ds.tau * c
        ds.tau_itt_cate = ds.tau_cate * c
        return

    def itt(mu0, tau):
        sgn, lo, hi = _interval(fam, mu0, tau, oc)
        return sgn * integrate_normal_interval(complier, lo, hi)

    ds.tau_itt = itt(mu0_eff, tau_eff)
    if sd_med == 0.0 and not np.any(np.asarray(sd_h) > 0):
        ds.tau_itt_cate = ds.tau_itt.copy()  # With no latent variables outside W, the assignment CATE equals the latent-state oracle.
        return

    (ds.tau_itt_cate,) = _outer_interval_expectation(
        fam, oc, mu0_w, tau_c, sd_med, sd_h, [complier_grid]
    )


def selection_truth(ds) -> None:
    """Update the ground-truth fields affected by selection.

    The selection stage leaves the DGP latent state intact, so only selection fields
    need a refresh. Use e_source from the first stage, or the current e if selection has
    not yet occurred.
    """
    e_src = ds.e_source if ds.e_source is not None else ds.e
    _selection_fields(ds, e_assign_effective(ds), e_src)


def _selection_fields(ds, e_z, e_src=None) -> None:
    """Compute ground truth for the selected population.

    sel_q1 and sel_q0 are the products of the stage-specific retention probabilities for
    T = 1 and T = 0. Before selection, e_source = P(T=1|W). Update e by Bayes' rule
    using those retention probabilities.

    Let ω(ν) = P(S=1|W,ν) = P(T=1|W,ν) q₁ + P(T=0|W,ν) q₀. Then E_ν[ω] = e_source q₁ +
    (1 − e_source) q₀. Because ite does not depend on compliance type, tau_cate_selected
    = E_ν[ite ω] / E_ν[ω].

    tau_itt_selected = E[1[complier]·ite | W,S=1] = ω_c · E_ν[complier(ν)·ite(ν)] /
    E_ν[ω], where ω_c = P(S=1 | complier,W) = e_z q₁ + (1 − e_z) q₀. A complier has T =
    Z, independently of ν.

    e_assign_selected = e_z P(S=1|W,Z=1) / E_ν[ω]. For Z=1, retention is E_ν[(1−p_nt) q₁
    + p_nt q₀]. For Z=0 it is E_ν[(1−p_nt) p_at q₁ + (1 − (1−p_nt) p_at) q₀]. Their
    weighted sum e_z P(S|Z=1) + (1−e_z) P(S|Z=0) equals E_ν[ω].

    Without active noncompliance ω is independent of ν. Copy the source fields so the
    selected values are exactly equal.
    """
    lat = ds.latent
    if "sel_q1" not in lat:
        ds.e_source = ds.tau_cate_selected = ds.tau_itt_selected = (
            ds.e_assign_selected
        ) = None
        return
    q1, q0 = lat["sel_q1"], lat["sel_q0"]
    e_src = ds.e if e_src is None else e_src
    ds.e_source = e_src.copy()
    ds.e = selection_e(e_src, q1, q0)
    has_nc = ds.compliance is not None
    if not (has_nc and _nc_active(ds)):
        ds.tau_cate_selected = ds.tau_cate.copy()
        ds.tau_itt_selected = ds.tau_itt.copy() if has_nc else None
        ds.e_assign_selected = e_z.copy() if has_nc else None
        return
    nc, fam, oc = lat["nc"], ds.config.outcome.family, ds.config.outcome
    denom = e_src * q1 + (1.0 - e_src) * q0
    denom = np.where(denom > 0, denom, 1.0)

    def omega(nu):
        p_nt, p_at = nc_type_probs(nc, nu)
        p1 = (1.0 - p_nt) * (p_at + (1.0 - p_at) * e_z)
        return p1 * q1 + (1.0 - p1) * q0

    def complier(nu):
        p_nt, p_at = nc_type_probs(nc, nu)
        return (1.0 - p_nt) * (1.0 - p_at)

    omega_c = e_z * q1 + (1.0 - e_z) * q0  # P(S = 1 | complier, W)

    def s_given_z1(nu):
        p_nt, _ = nc_type_probs(nc, nu)
        return (1.0 - p_nt) * q1 + p_nt * q0

    ds.e_assign_selected = e_z * expect_normal(s_given_z1, 1.0) / denom

    def omega_grid(sl):
        p_nt, p_at = nc_type_probs(_nc_block(nc, sl), _NU_GRID[None, :])
        p1 = (1.0 - p_nt) * (p_at + (1.0 - p_at) * e_z[sl][:, None])
        return p1 * q1[sl][:, None] + (1.0 - p1) * q0[sl][:, None]

    def complier_grid(sl):
        return _complier_grid(nc, sl)

    # Beta outcomes and independent coupling require integration over nu (0.8.0).
    if not _ite_closed_in_nu(ds):
        num_cate, num_itt = _nu_ite_expectation(
            ds,
            [omega, complier],
            [omega_grid, complier_grid],
            _mu0_w(ds),
            _tau_cond(ds),
            _sd_med(ds),
            _sd_hidden(ds),
        )
        ds.tau_cate_selected, ds.tau_itt_selected = (
            num_cate / denom,
            num_itt * omega_c / denom,
        )
        return
    if fam == "gaussian":
        ds.tau_cate_selected = ds.tau_cate.copy()  # ite ⊥ ν
        ds.tau_itt_selected = (
            ds.tau_cate * expect_normal(complier, 1.0) * omega_c / denom
        )
        return
    mu0_w, tau_c = _mu0_w(ds), _tau_cond(ds)
    sd_med, sd_h = _sd_med(ds), _sd_hidden(ds)

    if sd_med == 0.0 and not np.any(
        np.asarray(sd_h) > 0
    ):  # With no latent variables outside W, apply Gauss–Legendre quadrature directly.
        sgn, lo, hi = _interval(fam, mu0_w, tau_c, oc)
        num_cate = sgn * integrate_normal_interval(omega, lo, hi)
        num_itt = sgn * integrate_normal_interval(complier, lo, hi)
    else:
        num_cate, num_itt = _outer_interval_expectation(
            fam, oc, mu0_w, tau_c, sd_med, sd_h, [omega_grid, complier_grid]
        )
    ds.tau_cate_selected, ds.tau_itt_selected = (
        num_cate / denom,
        num_itt * omega_c / denom,
    )


# Target definitions for meta.json


def w_columns(ds) -> tuple[list[str], list[str]]:
    """Separate the conditioning covariates W from optional covariates.

    W contains the named X nodes except colliders, plus cluster for P-19 and period for
    effect_drift. Under covariate_shift or policy_drift, period belongs to w_optional
    because the CATE function stays the same.
    """
    g = ds.graph
    w = [g.node_names[i] for i in range(g.d) if g.roles[i] != ROLE_COLLIDER]
    w.extend(pre_columns(ds))  # Pretreatment outcome lags (0.11.0) belong to W.
    if ds.cluster is not None:
        w.append("cluster")
    kinds = {e.get("kind") for e in ds.pathology_log if e.get("active", True)}
    optional: list[str] = []
    if ds.period is not None:
        if "effect_drift" in kinds:
            w.append("period")
        else:
            optional.append("period")
    return w, optional


def w_affected_nodes(ds) -> list[str]:
    """Find W columns changed during measurement.

    Use the actual measurement results: affected_columns intersected with W,
    columns hidden by hidden_confounder, and noise added to copies by duplicate_rows.
    """
    w, _ = w_columns(ds)
    out: list[str] = []
    for entry in ds.pathology_log:
        if not entry.get("active", True) or entry.get("phase") == "dgp":
            continue
        cols = list(entry.get("affected_columns", [])) + list(
            entry.get("removed_columns", [])
        )
        out.extend(c for c in cols if c in w and c not in out)
    return out


def cate_dgp(ds) -> str:
    """Return "modified" when an applied DGP mechanism changes CATE(W) for this family, otherwise "source"."""
    fam = ds.config.outcome.family
    for e in ds.pathology_log:
        kind = e.get("kind")
        if not e.get("active", True) or fam not in CATE_MODIFYING.get(kind, ()):
            continue
        if kind == "cluster" and float(e.get("icc_y", 0.0)) <= 0:
            continue
        if kind == "hidden_modifier" and float(e.get("strength", 0.0)) <= 0:
            continue
        return "modified"
    return "source"


_W_PRE, _U, _EPS_MED, _NU, _U_MED = "w_pre", "u_tau", "eps_med", "nu", "u_med"


def truth_targets(ds) -> dict:
    """Describe each ground-truth field and how it is computed.

    Record its definition, conditioning latents, integrated latents, population, scale,
    and integration method.
    """
    kinds = [e.get("kind") for e in ds.pathology_log]
    selected = "selection" in kinds
    pop = "selected" if selected else "source"
    fam = ds.config.outcome.family
    closed = fam in (
        "gaussian",
        "lognormal",
    )  # Closed forms for tau_cate, cvte, vte_w, and fields integrated over U.
    has_nc = ds.compliance is not None
    nc_active = has_nc and _nc_active(ds)
    has_hidden = bool(np.any(ds.cvte > 0)) or bool(
        np.any(np.asarray(ds.latent.get("var_hidden", 0.0)) > 0)
    )
    med_fam = mediator_family(ds.latent)
    has_med = (
        med_fam == "gaussian"
    )  # Gaussian mediators require outer quadrature over eps_med.
    has_bin = (
        med_fam == "binary"
    )  # For binary mediators with Gaussian Y, integrate u_med in closed form (0.10.0).
    n_outer = int(has_hidden) + int(has_med)
    outer = (
        "closed_form"
        if n_outer == 0
        else f"quadrature:gauss_hermite_{GH_N}"
        if n_outer == 1
        else f"quadrature:gauss_hermite_{GH2_N}x{GH2_N}"
    )
    gh = f"quadrature:gauss_hermite_{GH_N}"
    gl = f"quadrature:gauss_legendre_{GL_N}"
    table = f"cdf_table_{NU_TABLE_N}"  # Inner integral over ν from _CdfTable.
    coupling = ds.config.outcome.coupling
    vte_n = _vte_outer_n(fam)
    unit = f"quadrature:gauss_legendre_{UNIT_N}_unit"  # Integrate y or u over (0, 1) for beta outcomes.
    if not _ite_closed_in_nu(
        ds
    ):  # Beta outcomes and independent coupling use fields integrated over ν (0.8.0).
        gl = gh if closed else f"{table};{unit}" if fam == "beta" else table
    integrated_w = [
        x for x, on in ((_U, has_hidden), (_EPS_MED, has_med), (_U_MED, has_bin)) if on
    ]
    affected = w_affected_nodes(ds)
    w, w_optional = w_columns(ds)
    dup_noise = any(
        e.get("kind") == "duplicate_rows"
        and e.get("active", True)
        and float(e.get("noise", 0.0)) > 0
        for e in ds.pathology_log
    )

    def row(
        definition,
        conditioning,
        integrated,
        scale="outcome",
        integration="none",
        population="source",
    ):
        return {
            "definition": definition,
            "conditioning": list(conditioning),
            "latents_integrated": list(integrated),
            "population": population,
            "scale": scale,
            "integration": integration,
        }

    med_lat = _U_MED if has_bin else _EPS_MED
    cond_row = [_W_PRE, _U, med_lat, _NU]
    cond_struct = [_W_PRE, _U, med_lat]
    # These formulas and their English descriptions are stored in the dataset metadata.
    targets = {
        "y0": row(
            "Y(T=0) with exogenous noise held fixed",
            cond_row,
            [],
            population="rows",
        ),
        "y1": row(
            "Y(T=1) with exogenous noise held fixed",
            cond_row,
            [],
            population="rows",
        ),
        "ite": row(
            "y1 - y0 is the realized individual effect. The generator assumes a joint distribution for (y0, y1) "
            f"through the potential-outcome coupling = {coupling!r} (shared: common outcome noise, "
            "comonotonic outcomes; independent: separate noise for Y(1))",
            cond_row,
            [],
            population="rows",
        ),
        "tau": row(
            "structural effect on the outcome scale at fixed (W, u_tau, "
            + med_lat
            + "): gaussian = ite "
            "(for binary mediators, use the realized M(1) − M(0)); binary: p1-p0; ziln, beta, lognormal: mu1-mu0 "  # noqa: RUF001
            "(latent-state oracle, distinct from CATE)",
            cond_struct,
            [_NU],
            integration="closed_form",
        ),
        "tau_cate": row(
            "E[tau | W], integrating over u_tau and "
            + med_lat
            + " (CATE given W)"
            + ("; binary mediators: tau_x + Σ θ_k (p1_k − p0_k)" if has_bin else ""),  # noqa: RUF001
            [_W_PRE],
            [*integrated_w, _NU],
            integration=("closed_form" if closed else outer),
        ),
        "cvte": row(
            "Var[tau | W] over u_tau and eps_med (excluding the variance of ν)",  # noqa: RUF001
            [_W_PRE],
            integrated_w,
            integration=("closed_form" if closed else outer),
        ),
        "vte_w": row(
            "Var[Y(1) - Y(0) | W] includes outcome noise in the variance of the realized effect. cvte excludes it. "
            f"This variance depends on the potential-outcome coupling = {coupling!r}",
            [_W_PRE],
            [*integrated_w, _NU],
            integration=(
                "closed_form"
                if closed
                else outer
                if coupling != "shared"
                else ";".join(
                    ([f"quadrature:gauss_hermite_{vte_n[0]}"] if has_med else [])
                    + (
                        [f"quadrature:gauss_legendre_{vte_n[1]}_unit_x2_split_at_tau_0"]
                        if has_hidden
                        else []
                    )
                    + ([unit] if fam == "beta" else [])
                )
                or "closed_form"
            ),
        ),
        "e": row(
            "P(T_true = 1 | W" + (", S = 1" if selected else "") + ")",
            [_W_PRE],
            [_NU] if nc_active else [],
            scale="probability",
            integration=(gh if nc_active else "closed_form"),
            population=pop,
        ),
        "ate": row(
            "mean(tau) over exported rows (source-population structural tau evaluated on the selected rows)",
            ["population"],
            [],
            population=pop,
        ),
        "cate": {
            "w": w,
            "w_optional": w_optional,
            "conditioning": "w_pre = true pretreatment X (all covariates except colliders)"
            + (" + outcome lags y_pre" if pre_columns(ds) else "")
            + (" + cluster" if ds.cluster is not None else "")
            + (" + period" if "period" in w else ""),
            "dgp": cate_dgp(ds),
            "observed_equivalent": not affected and not dup_noise,
            "affected_nodes": affected,
            "affected_rows": "copies" if dup_noise else None,
            "population": pop,
        },
    }
    if has_nc:
        targets["itt"] = row(
            "1[complier]·ite is the realized effect of assignment",
            cond_row,
            [],
            population="rows",
        )
        targets["tau_itt"] = row(
            "E[Y(T(Z=1)) - Y(T(Z=0)) | W, u_tau, eps_med] (latent-state oracle)",
            cond_struct,
            [_NU],
            integration=(
                "closed_form" if closed and not nc_active else gh if closed else gl
            ),
        )
        targets["tau_itt_cate"] = row(
            "E[Y(T(Z=1)) - Y(T(Z=0)) | W], the conditional average effect of assignment",
            [_W_PRE],
            [*integrated_w, _NU],
            integration=(
                "closed_form"
                if not nc_active
                else gh
                if closed
                else gl
                if n_outer == 0
                else f"{outer};{table}"
            ),
        )
        targets["e_assign"] = row(
            "P(Z = 1 | W), the assignment propensity", [_W_PRE], [], scale="probability"
        )
    if selected:
        targets["e_source"] = row(
            "P(T_true = 1 | W) before selection",
            [_W_PRE],
            [_NU] if nc_active else [],
            scale="probability",
            integration=(gh if nc_active else "closed_form"),
        )
        targets["ate_source"] = row(
            "mean(tau) over the source sample", ["population"], []
        )
        targets["ate_selected"] = row(
            "mean(tau_cate_selected) over selected rows",
            ["population"],
            [],
            population="selected",
        )
        sel_int = (
            "copy_of_source"
            if not nc_active
            else f"{gh}"
            if closed
            else gl
            if n_outer == 0
            else f"{outer};{table}"
        )
        targets["tau_cate_selected"] = row(
            "E[Y(1) - Y(0) | W, S = 1] = E_ν[ite ω] / E_ν[ω], ω = P(S=1|W,ν)",  # noqa: RUF001
            [_W_PRE],
            [*integrated_w, _NU],
            integration=sel_int,
            population="selected",
        )
        if has_nc:
            targets["tau_itt_selected"] = row(
                "E_ν[complier·ite·ω] / E_ν[ω]",  # noqa: RUF001
                [_W_PRE],
                [*integrated_w, _NU],
                integration=sel_int,
                population="selected",
            )
            targets["e_assign_selected"] = row(
                "P(Z = 1 | W, S = 1)",
                [_W_PRE],
                [_NU] if nc_active else [],
                scale="probability",
                integration=sel_int,
                population="selected",
            )
    if has_bin:  # Binary mediators (0.10.0).
        targets["p_m0"] = row(
            "P(M_k = 1 | W, T = 0) = σ(c_k + f_k(x)), n × m",  # noqa: RUF001
            [_W_PRE],
            [],
            scale="probability",
        )
        targets["p_m1"] = row(
            "P(M_k = 1 | W, T = 1) = σ(c_k + f_k(x) + effect_k(x))",  # noqa: RUF001
            [_W_PRE],
            [],
            scale="probability",
        )
        targets["tau_m"] = row(
            "p_m1 − p_m0 is the effect of T on the mediator (mediator CATE given W)",  # noqa: RUF001
            [_W_PRE],
            [],
            scale="probability",
        )
        targets["theta"] = row(
            "θ_k(x) is the effect M_k -> Y on the outcome scale before multiplying by s",
            [_W_PRE],
            [],
        )
        targets["m0"] = row(
            "M_k(T=0) = 1[u_med < p_m0] with shared uniform noise",
            [_W_PRE, _U_MED],
            [],
            scale="indicator",
            population="rows",
        )
        targets["m1"] = row(
            "M_k(T=1) = 1[u_med < p_m1]; when tau_m ≥ 0, m1 ≥ m0 in each row",
            [_W_PRE, _U_MED],
            [],
            scale="indicator",
            population="rows",
        )
    if ds.latent.get("y_scale") is not None:
        j = (
            ds.config.outcome.scale_feature
            if ds.scm is None
            else ds.scm.params["scale_feature"]
        )
        targets["y_scale"] = row(
            f"s(x) = exp(scale_sigma · z(x_{j})) is the natural outcome scale (0.9.0). Multiply y0, y1, ite, tau, "
            "mu0, mu1, tau_cate, tau_itt*, and *_selected by s, and cvte and vte_w by s²",
            [_W_PRE],
            [],
            population="rows",
        )
    if ds.tau_cate_x is not None:  # Outcome lags (0.11.0).
        targets["tau_cate_x"] = row(
            "E[ite | x] is the CATE given X without outcome lags y_pre (the lag-free oracle). "
            "tau_cate also conditions on the lags.",
            [_W_PRE + "_without_y_pre"],
            [*integrated_w, _NU],
            integration=("closed_form" if closed else outer),
        )
    if ds.period is not None:
        targets["period"] = row(
            "drift period (0 or 1) in the original row order",
            ["row"],
            [],
            population="rows",
        )
    if ds.cluster is not None:
        targets["cluster"] = row(
            "cluster identifier for the row (P-19; observed and included in W)",
            ["row"],
            [],
            population="rows",
        )
    if ds.dup_index is not None:
        targets["dup_index"] = row(
            "0 for the original row and 1..k for its copies (P-24)",
            ["row"],
            [],
            population="rows",
        )
    return targets


def late_realized(ds) -> float | None:
    """Return mean(ite | complier), the realized LATE, or None without noncompliance or compliers."""
    if ds.compliance is None:
        return None
    m = ds.compliance == 0
    return float((ds.y1 - ds.y0)[m].mean()) if m.any() else None


def new_axis_rng(seed: int, index: int, kind: str) -> np.random.Generator:
    """Return a reproducible random number generator for one axis.

    default_rng((seed, 0x9A70, index, code(kind))) gives the axis its own stream. Adding
    or removing the last axis, or setting its strength to zero, leaves other streams
    unchanged.
    """
    import hashlib

    code = int.from_bytes(hashlib.sha256(kind.encode()).digest()[:4], "little")
    return np.random.default_rng((seed, 0x9A70, index, code))


def make_nc_state(p: PathologyConfig, score_x: np.ndarray, mask: np.ndarray) -> dict:
    """Collect the scores, row mask and parameters for the noncompliance mechanism."""
    return {
        "score_x": score_x,
        "hidden": float(np.clip(p.params.get("hidden", 0.5), 0.0, 1.0)),
        "side": p.params.get("side", "one"),
        "strength": float(p.strength),
        "mask": mask,
    }
